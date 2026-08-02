"""Final training loop, run once with the best hyperparameters found by Optuna."""
import copy
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torchvision.ops import sigmoid_focal_loss

from her2_mil.config import Config
from her2_mil.models.base import MILModel
from her2_mil.utils.logging_utils import get_logger

logger = get_logger()


def train_final_model(
    cfg: Config,
    model: MILModel,
    best_params: dict,
    training_dataloader,
    validation_dataloader,
    device: str,
) -> Tuple[MILModel, Dict[str, List[float]], List[int], List[float]]:
    optimizer = optim.Adam(
        model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"]
    )
    loss_weight = best_params["loss_weight"]

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    best_weights = None
    val_true_final, val_probs_final = [], []

    for epoch in range(cfg.training.max_epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for patch_features, slide_label in training_dataloader:
            patch_features = patch_features.to(device)
            slide_label = slide_label.to(device).view(1)

            optimizer.zero_grad()
            logits, _ = model(patch_features)
            loss = sigmoid_focal_loss(logits.view(-1), slide_label.float(), alpha=loss_weight)
            loss.backward()
            optimizer.step()

            pred = (torch.sigmoid(logits) > 0.5).long().view(-1)
            total_loss += loss.item()
            correct += (pred == slide_label).sum().item()
            seen += 1

        history["train_loss"].append(total_loss / len(training_dataloader))
        history["train_acc"].append(correct / seen)

        model.eval()
        total_val_loss, correct_val, seen_val = 0.0, 0, 0
        val_true, val_pred, val_probs = [], [], []
        with torch.no_grad():
            for patch_features, slide_label in validation_dataloader:
                patch_features = patch_features.to(device)
                slide_label = slide_label.to(device).view(1)

                logits, _ = model(patch_features)
                loss = sigmoid_focal_loss(logits.view(-1), slide_label.float(), alpha=loss_weight)
                pred = (torch.sigmoid(logits) > 0.5).long().view(-1)

                prob_neg = torch.sigmoid(logits).item()
                val_probs.append(1.0 - prob_neg)
                val_true.append(slide_label.item())
                val_pred.append(pred.item())

                total_val_loss += loss.item()
                correct_val += (pred == slide_label).sum().item()
                seen_val += 1

        history["val_loss"].append(total_val_loss / len(validation_dataloader))
        history["val_acc"].append(correct_val / seen_val)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f F1: %.4f",
            epoch + 1, cfg.training.max_epochs,
            history["train_loss"][-1], history["train_acc"][-1],
            history["val_loss"][-1], history["val_acc"][-1], val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_weights = copy.deepcopy(model.state_dict())
            val_true_final, val_probs_final = val_true, val_probs
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.patience:
                logger.info("Early stopping activated")
                break

    model.load_state_dict(best_weights)
    return model, history, val_true_final, val_probs_final
