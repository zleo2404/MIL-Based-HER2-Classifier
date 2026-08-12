# script/save_stain_reference.py
"""Estrae una patch di tessuto pulita da una WSI da usare come riferimento
per la normalizzazione Macenko.

Usage:
    python script/save_stain_reference.py --wsi path/to/slide.svs \
        --out assets/stain_reference.png --level 2 --patch-size 1024
"""
import argparse
from pathlib import Path

import numpy as np
import openslide

from her2_mil.data.patching import compute_patch_grid, is_valid_tissue_patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=1024)
    args = parser.parse_args()

    slide = openslide.OpenSlide(args.wsi)
    downsample = slide.level_downsamples
    dim = slide.level_dimensions[args.level]

    for x in compute_patch_grid(dim[0], args.patch_size):
        for y in compute_patch_grid(dim[1], args.patch_size):
            patch = slide.read_region(
                (int(x * downsample[args.level]), int(y * downsample[args.level])),
                args.level, (args.patch_size, args.patch_size),
            ).convert("RGB")
            if is_valid_tissue_patch(np.array(patch)):
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                patch.save(args.out)
                print(f"Salvata reference patch in {args.out} (origin=({x},{y}))")
                slide.close()
                return
    slide.close()
    print("Nessuna patch di tessuto valida trovata.")


if __name__ == "__main__":
    main()