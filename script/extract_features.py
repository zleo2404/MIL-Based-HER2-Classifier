#!/usr/bin/env python
"""
STEP 1 of the pipeline: tile every WSI, keep tissue patches, embed them
with the configured feature extractor, and cache the result to disk.

Run this once per (patching config, feature extractor) combination. The
expensive part (openslide I/O + backbone forward passes) never has to be
repeated when you later only want to try a different MIL model or
hyperparameters -- see scripts/train.py.

Usage:
    python scripts/extract_features.py --config configs/default.yaml
"""
import argparse
import datetime
import glob
import os
from pathlib import Path

import pandas as pd
import torch

from her2_mil.config import load_config
from her2_mil.data.labels import get_slide_label, load_label_lookup
from her2_mil.data.patching import extract_patches_and_features
from her2_mil.features.registry import build_feature_extractor
from her2_mil.utils.logging_utils import setup_logging
from her2_mil.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_run_id = (
        f"{timestamp}_{cfg.features.extractor_name}"
        f"_level{cfg.patching.level}_patch{cfg.patching.patch_size}"
    )
    feature_run_dir = Path(cfg.paths.features_dir) / feature_run_id
    feature_run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(feature_run_dir)
    logger.info("=== Feature extraction run: %s ===", feature_run_id)
    logger.info("Device: %s | Extractor: %s", device, cfg.features.extractor_name)

    feature_extractor = build_feature_extractor(cfg.features.extractor_name, device=device)
    transform = feature_extractor.get_transform()

    label_df = load_label_lookup(cfg.paths.labels_csv)
    wsi_paths = glob.glob(os.path.join(cfg.paths.wsi_dir, "*.svs"))

    manifest_rows = []
    all_patch_metadata = []
    total_saved, total_discarded, skipped_slides = 0, 0, 0

    for wsi_path in wsi_paths:
        slide_id = Path(wsi_path).stem
        label = get_slide_label(slide_id, label_df, cfg.labels)
        if label is None:
            skipped_slides += 1
            continue

        features, patch_metadata, saved, discarded = extract_patches_and_features(
            wsi_path,
            patch_size=cfg.patching.patch_size,
            level=cfg.patching.level,
            feature_extractor=feature_extractor,
            transform=transform,
            device=device,
            tissue_ratio_threshold=cfg.patching.tissue_ratio_threshold,
            artifact_ratio_threshold=cfg.patching.artifact_ratio_threshold,
        )
        total_saved += saved
        total_discarded += discarded
        all_patch_metadata.extend(patch_metadata)

        if features is None:
            logger.warning("Slide %s produced 0 valid patches, skipping.", slide_id)
            continue

        torch.save({"features": features, "label": label}, feature_run_dir / f"{slide_id}.pt")
        manifest_rows.append({"slide_id": slide_id, "label": label, "num_patches": saved})
        logger.info("Slide %s: %d patches saved", slide_id, saved)

    pd.DataFrame(manifest_rows).to_csv(feature_run_dir / "manifest.csv", index=False)
    pd.DataFrame(all_patch_metadata).to_csv(
        feature_run_dir / f"patch_mapping_level{cfg.patching.level}_patch{cfg.patching.patch_size}.csv",
        index=False,
    )

    total = total_saved + total_discarded
    logger.info("Slides found: %d | Skipped (no/invalid label): %d", len(wsi_paths), skipped_slides)
    if total > 0:
        logger.info(
            "Patches saved: %d (%.2f%%) | discarded: %d (%.2f%%)",
            total_saved, total_saved / total * 100, total_discarded, total_discarded / total * 100,
        )
    logger.info("Feature cache written to: %s", feature_run_dir)
    logger.info("=> pass --features-run %s to scripts/train.py", feature_run_id)


if __name__ == "__main__":
    main()
