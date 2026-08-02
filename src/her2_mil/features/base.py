"""Abstract interface every feature extractor must implement."""

from abc import ABC, abstractmethod
from typing import Callable

import torch


class FeatureExtractor(ABC):
    """Wraps a pretrained backbone used to embed a single RGB patch."""

    #: dimensionality of the embedding this extractor produces
    feature_dim: int

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def get_transform(self) -> Callable:
        """Return the torchvision/timm transform expected by this backbone."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tensor: a single transformed patch, shape (1, C, H, W),
                already moved to `self.device`.
        Returns:
            1D feature tensor of length `self.feature_dim`.
        """
        raise NotImplementedError
