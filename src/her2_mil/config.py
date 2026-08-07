"""
Configuration management for the HER2 MIL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class PathsConfig:
    base_dir: str
    wsi_dir: Optional[str] = None
    labels_csv: Optional[str] = None
    runs_dir: Optional[str] = None
    features_dir: Optional[str] = None

    def __post_init__(self) -> None:
        base = Path(self.base_dir)
        self.wsi_dir = "/scratch.hpc/sabrina.tassinari/ProgettoTesi/wsi_organizzate"
        self.labels_csv = "/scratch.hpc/sabrina.tassinari/ProgettoTesi/dataset/label_her2.csv"
        self.runs_dir = self.runs_dir or str(base / "runs")
        self.features_dir = self.features_dir or str(base / "features")


@dataclass
class PatchingConfig:
    level: int = 2
    patch_size: int = 1024
    tissue_ratio_threshold: float = 0.4
    artifact_ratio_threshold: float = 0.05


@dataclass
class FeaturesConfig:
    # registry key, see her2_mil/features/registry.py -> "resnet50" | "uni" | ...
    extractor_name: str = "resnet50"


@dataclass
class LabelsConfig:
    mapping: dict = field(default_factory=lambda: {"Negative": 1, "Equivocal": 1, "Positive": 0})


@dataclass
class ModelConfig:
    # registry key, see her2_mil/models/registry.py -> "abmil" | "transmil" | ...
    name: str = "abmil"
    num_classes: int = 1
    hidden_dim: int = 512
    num_heads: int = 8


@dataclass
class OptunaSearchSpace:
    hidden_dim: tuple = (64, 128)
    learning_rate: tuple = (1e-5, 1e-2)
    weight_decay: tuple = (1e-5, 5e-3)
    dropout: tuple = (0.2, 0.4)
    loss_weight: tuple = (0.1, 0.3)


@dataclass
class OptunaConfig:
    n_trials: int = 50
    n_startup_trials: int = 5
    n_warmup_steps: int = 5
    threshold_trials: int = 20
    search_space: OptunaSearchSpace = field(default_factory=OptunaSearchSpace)


@dataclass
class TrainingConfig:
    max_epochs: int = 50
    patience: int = 7
    val_size: float = 0.30
    test_size_of_val: float = 0.50


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    paths: PathsConfig = None
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build_optuna_config(raw: dict) -> OptunaConfig:
    raw = raw or {}
    search_raw = raw.get("search_space", {})
    defaults = OptunaSearchSpace()
    search_space = OptunaSearchSpace(
        hidden_dim=tuple(search_raw.get("hidden_dim", defaults.hidden_dim)),
        learning_rate=tuple(search_raw.get("learning_rate", defaults.learning_rate)),
        weight_decay=tuple(search_raw.get("weight_decay", defaults.weight_decay)),
        dropout=tuple(search_raw.get("dropout", defaults.dropout)),
        loss_weight=tuple(search_raw.get("loss_weight", defaults.loss_weight)),
    )
    return OptunaConfig(
        n_trials=raw.get("n_trials", 50),
        n_startup_trials=raw.get("n_startup_trials", 5),
        n_warmup_steps=raw.get("n_warmup_steps", 5),
        threshold_trials=raw.get("threshold_trials", 20),
        search_space=search_space,
    )


def load_config(path: Union[str, Path]) -> Config:
    """Load a Config from a YAML file, filling in defaults for anything omitted."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    paths_raw = raw.get("paths", {})
    if "base_dir" not in paths_raw:
        raise ValueError("config YAML must define paths.base_dir")

    return Config(
        seed=raw.get("seed", 42),
        device=raw.get("device", "cuda"),
        paths=PathsConfig(**paths_raw),
        patching=PatchingConfig(**raw.get("patching", {})),
        features=FeaturesConfig(**raw.get("features", {})),
        labels=LabelsConfig(**raw.get("labels", {})) if raw.get("labels") else LabelsConfig(),
        model=ModelConfig(**raw.get("model", {})),
        optuna=_build_optuna_config(raw.get("optuna", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
