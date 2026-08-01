import os
import glob
import copy
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import optuna

import features_label_from_wsi

# =====================================================================
# DIRECTORIES & PATHS SETUP
# =====================================================================
BASE_DIR = Path("/scratch.hpc/leonardo.meloni/HER2")
WSI_DIR = BASE_DIR / "data" / "raw" / "BRCA"
LABELS_CSV = WSI_DIR / "her2_labels.csv"
RUNS_DIR = BASE_DIR / "runs"

# Create a unique directory for the current run
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = RUNS_DIR / f"run_{TIMESTAMP}"
CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
METRICS_LOG_FILE = RUNS_DIR / "all_runs_metrics.csv"

print(f"=== Starting New Run: {TIMESTAMP} ===")
print(f"Run outputs will be saved in: {CURRENT_RUN_DIR}")

LEVEL = 2
PATCH_SIZE = 1024

'''
==========================================================================================================
1. DATASET DEFINITION
==========================================================================================================
'''
class WSIDataset(Dataset):
    def __init__(self, feature_dimension=2048, device='cpu'):
        self.all_wsi_features = []
        self.all_wsi_labels = []

        file_labels = pd.read_csv(LABELS_CSV)
        file_labels['cases.submitter_id'] = file_labels['cases.submitter_id'].str.strip()
        file_labels['her2_result'] = file_labels['her2_result'].str.strip()

        label_map = {"Negative" : 1, "Equivocal": 1, "Positive" : 0} 

        # Initialize Feature Extractor
        feature_extractor = create_feature_extractor(
            resnet50(weights=ResNet50_Weights.DEFAULT), 
            return_nodes={'avgpool' : 'features'}
        )
        feature_extractor.eval() 
        feature_extractor = feature_extractor.to(device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # glob ensures we only pick .svs files (ignoring .filepart)
        pattern = os.path.join(WSI_DIR, "*.svs")
        
        total_slides = 0
        equivocal_slides = 0
        total_discarded_patches = 0
        total_saved_patches = 0
        all_patches_data = []
        
        print("\n--- Extracting Features from WSIs ---")
        for wsi in glob.glob(pattern, recursive=True):
            total_slides += 1
            slide_name = Path(wsi).stem 
            wsi_id = slide_name[0:12] # Standard TCGA ID length
            
            patient_row = file_labels[file_labels['cases.submitter_id'].str.startswith(wsi_id)]
            if patient_row.empty:
                equivocal_slides += 1
                continue
                
            her2_label = patient_row['her2_result'].iloc[0]
            if her2_label not in ['Negative', 'Equivocal', 'Positive']:
                equivocal_slides += 1
                continue
                
            label_number = label_map[her2_label]      
            
            features, patch_metadata, saved_patches, discarded_patches = features_label_from_wsi.extract_features_from_wsi(
                wsi, 
                patch_size=PATCH_SIZE, 
                level=LEVEL, 
                feature_extractor=feature_extractor, 
                device=device, 
                transform=transform
            )
            
            all_patches_data.append(patch_metadata)
            total_discarded_patches += discarded_patches
            total_saved_patches += saved_patches

            if features is None or len(features) == 0:
                continue

            self.all_wsi_features.append(features)
            self.all_wsi_labels.append(label_number)
            
        total_extracted = total_saved_patches + total_discarded_patches
        print(f"Total WSIs processed: {total_slides} | Discarded (Equivocal): {equivocal_slides}")
        if total_extracted > 0:
            print(f"Patches Saved: {total_saved_patches} ({total_saved_patches/total_extracted*100:.2f}%) | "
                  f"Discarded: {total_discarded_patches} ({total_discarded_patches/total_extracted*100:.2f}%)")
        
        # Flatten patches metadata and save to CSV
        flat_data = [item for sublist in all_patches_data for item in sublist]
        df_mapping = pd.DataFrame(flat_data)
        mapping_file = CURRENT_RUN_DIR / f"mapping_patches_level{LEVEL}_patch{PATCH_SIZE}.csv"
        df_mapping.to_csv(mapping_file, index=False)
        
    def __len__(self):
        return len(self.all_wsi_features)

    def __getitem__(self, index):
        features_of_one_wsi = self.all_wsi_features[index]
        label_of_one_wsi = torch.tensor(self.all_wsi_labels[index], dtype=torch.long)
        return features_of_one_wsi, label_of_one_wsi

'''
==========================================================================================================
2. COLLATE FUNCTION
==========================================================================================================
'''
def mil_collate_function(batch):
    features_of_one_wsi, label_of_one_wsi = batch[0] 
    return features_of_one_wsi, label_of_one_wsi

'''
==========================================================================================================
3. MINIMAL ATTENTION MIL MODEL
==========================================================================================================
'''
class MinimalAttentionMIL(nn.Module): 
    def __init__(self, input_feature_dimension=2048, hidden_feature_dimension=64, number_of_output_classes=2, dropout=0.4):
        super().__init__() 

        self.patch_feature_projection = nn.Sequential(
            nn.Linear(input_feature_dimension, hidden_feature_dimension),
            nn.Dropout(p=dropout) 
        )
        
        self.patch_attention_scoring = nn.Sequential( 
            nn.Linear(hidden_feature_dimension, hidden_feature_dimension),
            nn.Tanh(),
            nn.Linear(hidden_feature_dimension, 1)
        )

        self.slide_level_classifier = nn.Linear(hidden_feature_dimension, number_of_output_classes)
    
    def forward(self, patch_features_of_one_wsi): 
        projected_patch_features = self.patch_feature_projection(patch_features_of_one_wsi)
        raw_attention_scores = self.patch_attention_scoring(projected_patch_features)
        normalized_attention_weigth = torch.softmax(raw_attention_scores, dim=0)
        slide_representation = torch.sum(normalized_attention_weigth * projected_patch_features, dim=0)
        slide_logits = self.slide_level_classifier(slide_representation.unsqueeze(0))
        return slide_logits, normalized_attention_weigth

'''
==========================================================================================================
4. SETUP & DATALOADER
==========================================================================================================
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

dataset = WSIDataset(feature_dimension=2048, device=device)

labels = np.array(dataset.all_wsi_labels)
indices = np.arange(len(dataset))

train_indices, val_test_indices = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
val_test_labels = labels[val_test_indices]
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.50, stratify=val_test_labels, random_state=42)

training_set = Subset(dataset, train_indices)
validation_set = Subset(dataset, val_indices)
test_set = Subset(dataset, test_indices)

train_id = training_set.indices
target = np.array([dataset.all_wsi_labels[i] for i in train_id]) 
class_label, class_sample_count = np.unique(target, return_counts=True) 
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])
samples_weight = torch.from_numpy(samples_weight).double()

sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight, num_samples=len(target), replacement=True)

training_dataloader = DataLoader(training_set, batch_size=1, sampler=sampler, collate_fn=mil_collate_function)
validation_dataloader = DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=mil_collate_function)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=mil_collate_function)

'''
==========================================================================================================
5. OPTUNA HYPERPARAMETER TUNING
==========================================================================================================
'''
def objective(trial):
    hidden_layers = trial.suggest_int("hidden_layers", 64, 128) 
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.4)
    loss_weight = trial.suggest_float("loss_weight", 0.1, 0.3, log=True)

    mil_model = MinimalAttentionMIL(
        input_feature_dimension=2048,
        hidden_feature_dimension=hidden_layers,
        number_of_output_classes=1,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(mil_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    number_of_epochs = 50
    best_val_f1 = 0.0
    patience = 7
    epochs_without_improvement = 0
    best_model_weights = None

    for epoch_index in range(number_of_epochs):
        mil_model.train() 
        for patch_features, slide_label in training_dataloader:
            patch_features = patch_features.to(device)
            slide_label = slide_label.to(device).view(1)

            optimizer.zero_grad() 
            slide_logits, _ = mil_model(patch_features) 
            loss = sigmoid_focal_loss(slide_logits.view(-1), slide_label.float(), alpha=loss_weight) 
            loss.backward() 
            optimizer.step()
        
        val_label = []
        val_pred = []
        mil_model.eval() 
        with torch.no_grad(): 
            for patch_features, slide_label in validation_dataloader:
                patch_features = patch_features.to(device)
                slide_label = slide_label.to(device).view(1)
                slide_logits, _ = mil_model(patch_features) 
                predicted_class = (torch.sigmoid(slide_logits) > 0.5).long().view(-1) 
                
                val_label.append(slide_label.item())
                val_pred.append(predicted_class.item())

        value_f1_score = f1_score(val_label, val_pred, average='macro', zero_division=0)
        trial.report(value_f1_score, epoch_index)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned() 

        if value_f1_score > best_val_f1:
            best_val_f1 = value_f1_score
            epochs_without_improvement = 0
            best_model_weights = copy.deepcopy(mil_model.state_dict())
        else:
            epochs_without_improvement += 1 
            if epochs_without_improvement >= patience:
                break   

    if best_model_weights is not None:
        mil_model.load_state_dict(best_model_weights)
    else:
        raise optuna.exceptions.TrialPruned()

    return best_val_f1

print("\n--- Starting Optuna Optimization ---")
study = optuna.create_study(
    direction="maximize",
    study_name=f"mil_her2_{TIMESTAMP}",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    sampler=optuna.samplers.TPESampler(seed=42)
)

optuna.logging.set_verbosity(optuna.logging.WARNING) 
study.optimize(objective, n_trials=50)

best = study.best_params
print("\n=== BEST HYPERPARAMETERS FOUND ===")
for k, v in best.items():
    print(f"  {k}: {v}")

'''
==========================================================================================================
6. FINAL TRAINING LOOP WITH BEST PARAMS
==========================================================================================================
'''
model = MinimalAttentionMIL(
    input_feature_dimension=2048,
    hidden_feature_dimension=best["hidden_layers"],
    number_of_output_classes=1,
    dropout=best["dropout"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=best["learning_rate"], weight_decay=best["weight_decay"])

number_of_epochs = 50
best_val_f1 = 0.0
patience = 7
epochs_without_improvement = 0
best_model_weights = None

train_loss, train_acc, val_loss, val_acc = [], [], [], []

print("\n--- Starting Final Training ---")
for epoch_index in range(number_of_epochs):
    # Training
    total_train_loss = 0.0
    correct_train = 0
    seen_train = 0
    model.train() 

    for patch_features, slide_label in training_dataloader:
        patch_features = patch_features.to(device)
        slide_label = slide_label.to(device).view(1)

        optimizer.zero_grad() 
        slide_logits, _ = model(patch_features) 
        loss = sigmoid_focal_loss(slide_logits.view(-1), slide_label.float(), alpha=best["loss_weight"]) 
        loss.backward() 
        optimizer.step()

        predicted_class = (torch.sigmoid(slide_logits) > 0.5).long().view(-1) 
        total_train_loss += loss.item()
        correct_train += (predicted_class == slide_label).sum().item()
        seen_train += 1

    train_loss.append(total_train_loss / len(training_dataloader))
    train_acc.append(correct_train / seen_train)
    
    # Validation
    total_val_loss = 0.0
    correct_val = 0
    seen_val = 0
    val_label, val_pred, val_probs = [], [], []

    model.eval() 
    with torch.no_grad(): 
        for patch_features, slide_label in validation_dataloader:
            patch_features = patch_features.to(device)
            slide_label = slide_label.to(device).view(1)

            slide_logits, _ = model(patch_features) 
            loss = sigmoid_focal_loss(slide_logits.view(-1), slide_label.float(), alpha=best["loss_weight"]) 
            predicted_class = (torch.sigmoid(slide_logits) > 0.5).long().view(-1) 

            prob_neg = torch.sigmoid(slide_logits).item()
            prob_pos = 1.0 - prob_neg

            val_probs.append(prob_pos)
            val_label.append(slide_label.item())
            val_pred.append(predicted_class.item())

            total_val_loss += loss.item()
            correct_val += (predicted_class == slide_label).sum().item()
            seen_val += 1

    val_loss.append(total_val_loss / len(validation_dataloader))
    val_acc.append(correct_val / seen_val)
    val_f1_score = f1_score(val_label, val_pred, average='macro', zero_division=0)

    print(f"Epoch {epoch_index + 1}/{number_of_epochs} | "
          f"Train Loss: {train_loss[-1]:.4f} | Acc: {train_acc[-1]:.4f} | "
          f"Val Loss: {val_loss[-1]:.4f} | Acc: {val_acc[-1]:.4f} | F1: {val_f1_score:.4f}")

    if val_f1_score > best_val_f1:
        best_val_f1 = val_f1_score
        epochs_without_improvement = 0
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        epochs_without_improvement += 1 
        if epochs_without_improvement >= patience:
            print("Early stopping activated")
            break   

model.load_state_dict(best_model_weights)

# Threshold Optimization
y_true = np.array(val_label)
y_probs = np.array(val_probs)

def threshold_objective(trial):
    threshold = trial.suggest_float('threshold', 0.2, 0.5)
    preds = (y_probs < threshold).astype(int)
    return f1_score(y_true, preds, average='macro', zero_division=0)

study_thresh = optuna.create_study(direction='maximize')
study_thresh.optimize(threshold_objective, n_trials=20)
best_threshold = study_thresh.best_params['threshold']
print(f"Optimized Threshold found: {best_threshold:.4f}")

'''
==========================================================================================================
7. PLOTS, TESTING & LOGGING
==========================================================================================================
'''
# Training Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('MIL Training - Loss & Accuracy', fontsize=14, fontweight='bold')
epochs = range(1, len(train_loss) + 1)
ax1.plot(epochs, train_loss, 'r-o', label='Train Loss')
ax1.plot(epochs, val_loss, 'b-o', label='Validation Loss')
ax1.set_title('Average Loss')
ax1.legend()
ax2.plot(epochs, train_acc, 'g-o', label='Train Accuracy')
ax2.plot(epochs, val_acc, 'orange', marker='o', label='Validation Accuracy')
ax2.set_title('Accuracy')
ax2.legend()
plt.tight_layout()
plt.savefig(CURRENT_RUN_DIR / "training_graph.png", dpi=150, bbox_inches='tight')
plt.close()

# Testing
print("\n=== TESTING ===")
test_label, test_pred = [], []
total_test_loss = 0
with torch.no_grad():
    for patch_features, slide_label in test_dataloader:
        patch_features = patch_features.to(device)
        slide_label = slide_label.to(device).view(1)
        slide_logits, _ = model(patch_features)
        
        loss = sigmoid_focal_loss(slide_logits.view(-1), slide_label.float(), alpha=best['loss_weight'])
        total_test_loss += loss.item()

        pred_pos = 1.0 - torch.sigmoid(slide_logits)
        predicted_class = (pred_pos < best_threshold).long().view(-1)
        
        test_label.append(slide_label.item())
        test_pred.append(predicted_class.item())

test_loss = total_test_loss / len(test_dataloader)
test_acc = (np.array(test_pred) == np.array(test_label)).mean()
test_f1_score = f1_score(test_label, test_pred, average='macro', zero_division=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1_score:.4f}")

# Confusion Matrix
cm = confusion_matrix(test_label, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])
disp.plot(cmap=plt.cm.Blues)
plt.savefig(CURRENT_RUN_DIR / "confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

# === SALVATAGGIO METRICHE NEL CSV GLOBALE ===
run_metrics = {
    "Run_ID": TIMESTAMP,
    "Level": LEVEL,
    "Patch_Size": PATCH_SIZE,
    "Hidden_Layers": best["hidden_layers"],
    "Learning_Rate": best["learning_rate"],
    "Weight_Decay": best["weight_decay"],
    "Dropout": best["dropout"],
    "Loss_Weight": best["loss_weight"],
    "Opt_Threshold": best_threshold,
    "Test_Loss": test_loss,
    "Test_Accuracy": test_acc,
    "Test_F1_Score": test_f1_score
}

df_metrics = pd.DataFrame([run_metrics])

# Se il file esiste, appende senza riga di intestazione, altrimenti lo crea
if METRICS_LOG_FILE.exists():
    df_metrics.to_csv(METRICS_LOG_FILE, mode='a', header=False, index=False)
else:
    df_metrics.to_csv(METRICS_LOG_FILE, mode='w', header=True, index=False)

print(f"\n✅ Finito! I file di questa esecuzione sono in: {CURRENT_RUN_DIR}")
print(f"✅ Metriche registrate nel log globale: {METRICS_LOG_FILE}")