"""Dataset that reads pre-extracted, cached per-slide features. 
That is a separate, cacheable step, so it's possible to
try many models/hyperparameters against the same cached features
without re-running the expensive part.
"""
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedWSIDataset(Dataset):
    def __init__(self, feature_run_dir: Union[str, Path]):
        self.feature_run_dir = Path(feature_run_dir)
        manifest_path = self.feature_run_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.csv in {self.feature_run_dir}. "
                "Run scripts/extract_features.py first."
            )
        manifest = pd.read_csv(manifest_path)
        self.slide_ids: List[str] = manifest["slide_id"].tolist()
        self.labels: List[int] = manifest["label"].tolist()

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int):
        slide_id = self.slide_ids[index]
        payload = torch.load(self.feature_run_dir / f"{slide_id}.pt")
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return payload["features"], label


def mil_collate_fn(batch):
    """Bags have variable size (different patch counts per slide), so
    batch_size must stay 1 -- this just unwraps that single-item batch."""
    features, label = batch[0]
    return features, label
