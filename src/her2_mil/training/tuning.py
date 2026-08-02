"""Optuna objective: searches MIL hyperparameters for a fixed, already
cached, feature set. Model-agnostic -- `cfg.model.name` decides which
architecture is built via the registry, the search loop itself doesn't
change.
"""
import copy

import optuna
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torchvision.ops import sigmoid_focal_loss

from her2_mil.config import Config
from her2_mil.models.registry import build_model


def make_objective(cfg: Config, feature_dim: int, training_dataloader, validation_dataloader, device: str):
    space = cfg.optuna.search_space

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_int("hidden_dim", *space.hidden_dim)
        learning_rate = trial.suggest_float("learning_rate", *space.learning_rate, log=True)
        weight_decay = trial.suggest_float("weight_decay", *space.weight_decay, log=True)
        dropout = trial.suggest_float("dropout", *space.dropout)
        loss_weight = trial.suggest_float("loss_weight", *space.loss_weight, log=True)

        model = build_model(
            cfg.model.name,
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=cfg.model.num_classes,
            dropout=dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_f1 = 0.0
        epochs_without_improvement = 0
        best_weights = None

        for epoch in range(cfg.training.max_epochs):
            model.train()
            for patch_features, slide_label in training_dataloader:
                patch_features = patch_features.to(device)
                slide_label = slide_label.to(device).view(1)

                optimizer.zero_grad()
                logits, _ = model(patch_features)
                loss = sigmoid_focal_loss(logits.view(-1), slide_label.float(), alpha=loss_weight)
                loss.backward()
                optimizer.step()

            val_true, val_pred = [], []
            model.eval()
            with torch.no_grad():
                for patch_features, slide_label in validation_dataloader:
                    patch_features = patch_features.to(device)
                    slide_label = slide_label.to(device).view(1)
                    logits, _ = model(patch_features)
                    pred = (torch.sigmoid(logits) > 0.5).long().view(-1)
                    val_true.append(slide_label.item())
                    val_pred.append(pred.item())

            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.training.patience:
                    break

        if best_weights is None:
            raise optuna.exceptions.TrialPruned()

        return best_val_f1

    return objective
