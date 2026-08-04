"""Abstract interface every MIL aggregator must implement.

A MIL model takes the bag of patch-level features for a single WSI and
returns slide-level logits (plus attention weights when available, useful
for interpretability / heatmaps).
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MILModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_features: (N, input_dim) tensor, N patches for one slide.
        Returns:
            logits: (1, num_classes) tensor.
            attention_weights: (N, 1) tensor, or None for architectures that
                don't expose a per-patch attention readout.
        """
        raise NotImplementedError
