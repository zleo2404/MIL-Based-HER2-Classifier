# src/her2_mil/data/stain_norm.py
"""Macenko stain normalization for H&E patches.

Normalizza ogni patch verso un unico riferimento di colore, riducendo la
variabilita' di colorazione tra slide/siti diversi -- una delle cause note
di domain shift nelle coorti TCGA multi-centro.
"""
from typing import Optional

import numpy as np


class MacenkoStainNormalizer:
    """Wrapper attorno al normalizzatore Macenko (backend numpy) di
    torchstain, fittato una volta su una patch di riferimento fissa."""

    def __init__(self, reference_patch: np.ndarray):
        import torchstain
        self._normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        self._normalizer.fit(reference_patch)

    @classmethod
    def from_reference_image(cls, reference_path: str) -> "MacenkoStainNormalizer":
        from PIL import Image
        reference_patch = np.array(Image.open(reference_path).convert("RGB"))
        return cls(reference_patch)

    def normalize(self, patch_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            patch_rgb: array (H, W, 3) uint8 RGB.
        Returns:
            Array normalizzato, oppure None se la normalizzazione fallisce
            (patch quasi uniforme -> matrice di stain non invertibile).
            Il chiamante deve trattare None come patch da scartare.
        """
        try:
            norm, _, _ = self._normalizer.normalize(I=patch_rgb, stains=True)
        except Exception:
            return None
        return np.clip(np.asarray(norm), 0, 255).astype(np.uint8)