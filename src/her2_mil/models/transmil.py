"""
TransMIL (Shao et al., NeurIPS 2021) -- Correlated Multiple Instance Learning.

This implementation features the core PPEG (Pyramid Position Encoding Generator)
which injects spatial relationships into the patches using depth-wise 
convolutions, without requiring explicit (x,y) slide coordinates.

Note on memory: PyTorch 2.0+ automatically utilizes FlashAttention in 
`nn.TransformerEncoderLayer`, making the vanilla O(N^2) attention highly 
efficient and capable of handling 10k+ patches without requiring Nystromformer.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from her2_mil.models.base import MILModel


def _resolve_num_heads(hidden_dim: int, preferred_num_heads: int) -> int:
    """Fall back to the largest valid divisor for Optuna compatibility."""
    for heads in range(min(preferred_num_heads, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator.
    Dynamically reshapes a 1D sequence into a 2D pseudo-grid to apply 
    spatial convolutions, then flattens back to 1D.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Depth-wise convolutions (groups=dim) capture spatial info independently per channel
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, N, C]
        B, N, C = x.shape
        
        # Calculate grid size (ceiling of square root)
        H = int(math.ceil(math.sqrt(N)))
        W = H
        pad_size = H * W - N
        
        # Pad sequence with zeros if N is not a perfect square
        if pad_size > 0:
            pad = torch.zeros(B, pad_size, C, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x
            
        # Reshape to 2D image-like tensor: [B, C, H, W]
        x_2d = x_padded.transpose(1, 2).view(B, C, H, W)
        
        # Apply pyramidal spatial convolutions
        pos_enc = self.conv3x3(x_2d) + self.conv5x5(x_2d) + self.conv7x7(x_2d)
        
        # Flatten back to sequence and remove padding: [B, N, C]
        pos_enc = pos_enc.flatten(2).transpose(1, 2)
        
        # Add positional encoding back to input
        return x + pos_enc[:, :N, :]


class TransMIL(MILModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 1,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        resolved_num_heads = _resolve_num_heads(hidden_dim, num_heads)

        # 1. Feature Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU() # Standard activation for Vision Transformers
        )
        
        # 2. Position Encoding (PPEG)
        self.ppeg = PPEG(dim=hidden_dim)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=resolved_num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu", # Better than default ReLU for ViTs
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Readout
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # patch_features shape: [N, input_dim]
        
        # Unsqueeze to simulate Batch Size = 1 (Required by Transformer)
        x = patch_features.unsqueeze(0)        # [1, N, input_dim]
        
        # Projection & Spatial Positional Encoding
        x = self.projection(x)                 # [1, N, hidden_dim]
        x = self.ppeg(x)                       # [1, N, hidden_dim]
        
        # Append [CLS] Token
        cls = self.cls_token.expand(x.size(0), -1, -1)   # [1, 1, hidden_dim]
        x = torch.cat([cls, x], dim=1)                   # [1, N+1, hidden_dim]
        
        # Self-Attention
        x = self.transformer(x)                # [1, N+1, hidden_dim]
        x = self.norm(x)
        
        # Read out from [CLS] token (index 0)
        logits = self.classifier(x[:, 0])      # [1, num_classes]
        
        return logits, None