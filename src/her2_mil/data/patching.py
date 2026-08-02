"""
WSI patch extraction: tiling, tissue/artifact filtering and feature
embedding for a single slide.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from her2_mil.features.base import FeatureExtractor


def is_valid_tissue_patch(
    patch_rgb: np.ndarray,
    tissue_ratio_threshold: float = 0.4,
    artifact_ratio_threshold: float = 0.05,
) -> bool:
    """Heuristic HSV-based tissue/artifact filter for H&E-stained patches.

    Args:
        patch_rgb: (H, W, 3) uint8 RGB array.
    """
    patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    h, s = patch_hsv[:, :, 0], patch_hsv[:, :, 1]

    pink = (h > 160) & (h < 180)
    purple = (h > 120) & (h < 160)
    tissue = (pink & (s > 10)) | (purple & (s > 8))
    has_enough_tissue = np.mean(tissue) > tissue_ratio_threshold

    other_color = (~tissue) & (s > 30)
    has_few_artifacts = np.mean(other_color) < artifact_ratio_threshold

    return bool(has_enough_tissue and has_few_artifacts)


def compute_patch_grid(dimension: int, patch_size: int) -> List[int]:
    """Non-overlapping tile origins along one axis, always covering the last edge."""
    coords = list(range(0, dimension - patch_size, patch_size))
    coords.append(dimension - patch_size)
    return coords


def extract_patches_and_features(
    file_path: str,
    patch_size: int,
    level: int,
    feature_extractor: FeatureExtractor,
    transform: Callable,
    device: str = "cuda",
    tissue_ratio_threshold: float = 0.4,
    artifact_ratio_threshold: float = 0.05,
) -> Tuple[Optional[torch.Tensor], List[dict], int, int]:
    """
    Tile a WSI, keep only tissue patches, embed each with `feature_extractor`.

    Returns:
        feature_tensor: (num_saved_patches, feature_dim), or None if no
            patch passed the tissue filter.
        patch_metadata: list of dicts (slide_id, level, height, width,
            origin_x, origin_y), one per saved patch.
        saved_patches, discarded_patches: counts.
    """
    import openslide

    slide_name = Path(file_path).stem
    slide = openslide.OpenSlide(str(file_path))
    downsample = slide.level_downsamples
    slide_dimension = slide.level_dimensions[level]

    coord_x = compute_patch_grid(slide_dimension[0], patch_size)
    coord_y = compute_patch_grid(slide_dimension[1], patch_size)

    features, patch_metadata = [], []
    saved_patches, discarded_patches = 0, 0

    for x in coord_x:
        for y in coord_y:
            patch = slide.read_region(
                (int(x * downsample[level]), int(y * downsample[level])),
                level,
                (patch_size, patch_size),
            )
            patch_rgb = patch.convert("RGB")
            patch_np = np.array(patch_rgb)

            if not is_valid_tissue_patch(patch_np, tissue_ratio_threshold, artifact_ratio_threshold):
                discarded_patches += 1
                continue

            patch_tensor = transform(patch_rgb).unsqueeze(0).to(device)
            feature = feature_extractor.extract(patch_tensor)
            features.append(feature)

            patch_metadata.append({
                "slide_id": slide_name,
                "level": level,
                "height": patch_size,
                "width": patch_size,
                "origin_x": x,
                "origin_y": y,
            })
            saved_patches += 1

    slide.close()

    if saved_patches == 0:
        return None, [], saved_patches, discarded_patches

    return torch.stack(features), patch_metadata, saved_patches, discarded_patches
