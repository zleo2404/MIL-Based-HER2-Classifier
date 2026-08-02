#!/usr/bin/env python
"""
STEP 2 of the pipeline: hyperparameter search + final training + test
evaluation, reading features that were already cached by
scripts/extract_features.py.

Usage:
    python scripts/train.py --config configs/default.yaml \
        --features-run 20260801_120000_resnet50_level2_patch1024

To try a different model or feature extractor, edit `model.name` /
`features.extractor_name` in the config (or point --config at a different
YAML) -- no code changes needed.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from her2_mil.config import load_config
from her2_mil.data.dataset import CachedWSIDataset, mil_collate_fn
from her2_mil.models.registry import build_model
from her2_mil.training.evaluate import (
    evaluate_on_test_set,
    optimize_threshold,
    plot_confusion_matrix,
    plot_training_curves,
)
from her2_mil.training.trainer import train_final_model
from her2_mil.training.tuning import make_objective
from her2_mil.utils.logging_utils import setup_logging
from her2_mil.utils.repro import save_run_metadata
from her2_mil.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--features-run", required=True,
        help="Feature cache directory name (printed by extract_features.py)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{cfg.model.name}"
    run_dir = Path(cfg.paths.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("=== Training run: %s ===", run_id)
    logger.info("Model: %s | Device: %s | Features: %s", cfg.model.name, device, args.features_run)

    save_run_metadata(run_dir, cfg, args.config)

    feature_run_dir = Path(cfg.paths.features_dir) / args.features_run
    dataset = CachedWSIDataset(feature_run_dir)
    feature_dim = dataset[0][0].shape[-1]
    logger.info("Loaded %d slides | feature_dim=%d", len(dataset), feature_dim)

    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    train_idx, val_test_idx = train_test_split(
        indices, test_size=cfg.training.val_size, stratify=labels, random_state=cfg.seed
    )
    val_test_labels = labels[val_test_idx]
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=cfg.training.test_size_of_val,
        stratify=val_test_labels, random_state=cfg.seed,
    )

    training_set = Subset(dataset, train_idx)
    validation_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_targets = labels[train_idx]
    class_labels, class_counts = np.unique(train_targets, return_counts=True)
    class_weight = 1.0 / class_counts
    sample_weight = torch.from_numpy(np.array([class_weight[t] for t in train_targets])).double()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weight, num_samples=len(train_targets), replacement=True
    )

    training_dataloader = DataLoader(training_set, batch_size=1, sampler=sampler, collate_fn=mil_collate_fn)
    validation_dataloader = DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    # --- Hyperparameter search (persisted study: resumable if the job dies) ---
    logger.info("Starting Optuna search (%d trials)...", cfg.optuna.n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"mil_her2_{run_id}",
        storage=f"sqlite:///{run_dir / 'optuna.db'}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.optuna.n_startup_trials, n_warmup_steps=cfg.optuna.n_warmup_steps
        ),
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
    )

    objective = make_objective(cfg, feature_dim, training_dataloader, validation_dataloader, device)
    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    best_params = study.best_params
    logger.info("Best hyperparameters: %s", best_params)

    # --- Final training with best hyperparameters ---
    model = build_model(
        cfg.model.name,
        input_dim=feature_dim,
        hidden_dim=best_params["hidden_dim"],
        num_classes=cfg.model.num_classes,
        dropout=best_params["dropout"],
    ).to(device)

    model, history, val_true, val_probs = train_final_model(
        cfg, model, best_params, training_dataloader, validation_dataloader, device
    )
    plot_training_curves(history, run_dir / "training_curves.png")

    best_threshold = optimize_threshold(val_true, val_probs, n_trials=cfg.optuna.threshold_trials)
    logger.info("Optimized decision threshold: %.4f", best_threshold)

    test_loss, test_acc, test_f1, test_true, test_pred = evaluate_on_test_set(
        model, test_dataloader, best_threshold, best_params["loss_weight"], device
    )
    plot_confusion_matrix(test_true, test_pred, run_dir / "confusion_matrix.png")

    logger.info("Test Loss: %.4f | Test Accuracy: %.4f | Test F1: %.4f", test_loss, test_acc, test_f1)

    torch.save(model.state_dict(), run_dir / "model.pt")

    # --- Append to the global metrics log ---
    metrics_log_file = Path(cfg.paths.runs_dir) / "all_runs_metrics.csv"
    run_metrics = {
        "Run_ID": run_id,
        "Model": cfg.model.name,
        "Feature_Extractor": args.features_run,
        "Level": cfg.patching.level,
        "Patch_Size": cfg.patching.patch_size,
        "Hidden_Dim": best_params["hidden_dim"],
        "Learning_Rate": best_params["learning_rate"],
        "Weight_Decay": best_params["weight_decay"],
        "Dropout": best_params["dropout"],
        "Loss_Weight": best_params["loss_weight"],
        "Opt_Threshold": best_threshold,
        "Test_Loss": test_loss,
        "Test_Accuracy": test_acc,
        "Test_F1_Score": test_f1,
    }
    df_metrics = pd.DataFrame([run_metrics])
    df_metrics.to_csv(
        metrics_log_file,
        mode="a" if metrics_log_file.exists() else "w",
        header=not metrics_log_file.exists(),
        index=False,
    )

    logger.info("Done. Outputs in: %s", run_dir)


if __name__ == "__main__":
    main()
