"""Attention-based MIL (Ilse et al., 2018) -- gated single-head attention pooling.

Implemented with a projection block (to reduce dimensionality from modern 
extractors like UNI/CTransPath) followed by the Gated Attention mechanism.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from her2_mil.models.base import MILModel


class ABMIL(MILModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 1, dropout: float = 0.4):
        super().__init__()

        self.patch_feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.slide_level_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # patch_features shape: [N, input_dim] (where N is number of patches in WSI)
        
        projected = self.patch_feature_projection(patch_features)  # [N, hidden_dim]
        
        # Gated attention scoring
        A_V = self.attention_V(projected)  # [N, hidden_dim]
        A_U = self.attention_U(projected)  # [N, hidden_dim]
        
        # Element-wise multiplication of the two branches
        raw_scores = self.attention_weights(A_V * A_U)  # [N, 1]
        attention_weights = torch.softmax(raw_scores, dim=0)  # [N, 1]
        
        # Pool patches into a single slide representation
        slide_repr = torch.sum(attention_weights * projected, dim=0)  # [hidden_dim]
        
        # Classify the WSI
        logits = self.slide_level_classifier(slide_repr.unsqueeze(0))  # [1, num_classes]
        
        return logits, attention_weights