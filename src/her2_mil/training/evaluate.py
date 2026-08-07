"""Threshold search, test-set evaluation, and diagnostic plots."""
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

from her2_mil.models.base import MILModel
from her2_mil.utils.logging_utils import get_logger

logger = get_logger()


def optimize_threshold(val_true: List[int], val_probs: List[float], n_trials: int = 20) -> float:
    y_true = np.array(val_true)
    y_probs = np.array(val_probs)

    def objective(trial: optuna.Trial) -> float:
        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        preds = (y_probs < threshold).astype(int)
        return f1_score(y_true, preds, average="macro", zero_division=0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params["threshold"]


def evaluate_on_test_set(
    model: MILModel, test_dataloader, threshold: float, device: str
) -> Tuple[float, float, float, float, List[int], List[int]]:
    model.eval()
    test_true, test_pred, test_probs_neg = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for patch_features, slide_label in test_dataloader:
            patch_features = patch_features.to(device)
            slide_label = slide_label.to(device).view(1)
            logits, _ = model(patch_features)

            #loss = sigmoid_focal_loss(logits.view(-1), slide_label.float(), alpha=loss_weight)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), slide_label.float())
            total_loss += loss.item()

            prob_pos = 1.0 - torch.sigmoid(logits)
            pred = (prob_pos < threshold).long().view(-1)

            test_true.append(slide_label.item())
            test_pred.append(pred.item())
            test_probs_neg.append(torch.sigmoid(logits).item())

    test_loss = total_loss / len(test_dataloader)
    test_acc = float((np.array(test_pred) == np.array(test_true)).mean())
    test_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    try:
      test_auc = roc_auc_score(test_true, test_probs_neg)
    except ValueError:
      test_auc = 0.5
    
    return test_loss, test_acc, test_f1, test_auc, test_true, test_pred


def plot_training_curves(history: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MIL Training - Loss & Accuracy", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "r-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "b-o", label="Validation Loss")
    ax1.set_title("Average Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "g-o", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], color="orange", marker="o", label="Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(test_true: List[int], test_pred: List[int], out_path: Path) -> None:
    cm = confusion_matrix(test_true, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
