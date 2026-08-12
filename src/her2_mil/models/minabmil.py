"""Minimal Attention MIL -- vanilla (non-gated) attention pooling.

Baseline piu' semplice rispetto ad ABMIL: usa un solo ramo Tanh per calcolare
gli attention score (Ilse et al., 2018, versione base senza gating), invece
del doppio ramo Tanh/Sigmoid di ABMIL. Utile come riferimento per misurare
quanto il gating di ABMIL migiora effettivamente le performance.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from her2_mil.models.base import MILModel


class MinimalAttentionMIL(MILModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 1, dropout: float = 0.4):
        super().__init__()

        self.patch_feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.patch_attention_scoring = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.slide_level_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # patch_features shape: [N, input_dim] (N patch della singola WSI)

        projected = self.patch_feature_projection(patch_features)  # [N, hidden_dim]

        raw_scores = self.patch_attention_scoring(projected)  # [N, 1]
        attention_weights = torch.softmax(raw_scores, dim=0)  # [N, 1]

        slide_repr = torch.sum(attention_weights * projected, dim=0)  # [hidden_dim]

        logits = self.slide_level_classifier(slide_repr.unsqueeze(0))  # [1, num_classes]

        return logits, attention_weights