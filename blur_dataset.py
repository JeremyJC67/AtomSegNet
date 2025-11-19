#!/usr/bin/env python3
"""Generate heavily blurred versions of dataset images for outlining atom clusters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


def iter_images(root: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            yield path


def blur_image(image_path: Path, sigma: float, normalize: bool) -> Image.Image:
    image = Image.open(image_path).convert("L")
    array = np.array(image, dtype=np.float32)
    blurred = ndimage.gaussian_filter(array, sigma=sigma)
    if normalize:
        blurred = blurred - blurred.min()
        max_val = blurred.max()
        if max_val > 0:
            blurred = blurred / max_val
        blurred = blurred * 255.0
    return Image.fromarray(blurred.astype(np.uint8), mode="L")


def save_blurred(image_path: Path, output_root: Path, blurred_img: Image.Image, suffix: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_name = f"{stem}_{suffix}.png" if suffix else f"{stem}.png"
    blurred_img.save(output_root / out_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blur dataset images to highlight atom cluster outlines.")
    parser.add_argument("dataset_root", type=Path, help="Root directory containing original images.")
    parser.add_argument("output_dir", type=Path, help="Directory where blurred images will be saved.")
    parser.add_argument("--pattern", default="*.jpg", help="Glob pattern for dataset images (default: *.jpg).")
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian blur sigma (default: 8.0).")
    parser.add_argument("--normalize", action="store_true",
                        help="Rescale blurred output to full 0-255 range.")
    parser.add_argument("--suffix", default="heavyblur", help="Suffix appended to output filenames.")
    parser.add_argument("--head", type=int, default=50, help="Number of leading images to process (default: 50).")
    parser.add_argument("--tail", type=int, default=50, help="Number of trailing images to process (default: 50).")
    parser.add_argument("--dry-run", action="store_true", help="List files without writing outputs.")
    parser.add_argument("--verbose", action="store_true", help="Print per-image status.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    files = list(iter_images(args.dataset_root, args.pattern))
    if not files:
        print("No images found.")
        return

    head = max(0, args.head)
    tail = max(0, args.tail)
    if head or tail:
        selection = []
        if head:
            selection.extend(files[:head])
        if tail:
            selection.extend(files[max(len(files) - tail, 0):])
        seen = set()
        unique_selection = []
        for path in selection:
            if path not in seen:
                unique_selection.append(path)
                seen.add(path)
        files = unique_selection if unique_selection else files

    processed = 0
    for image_path in files:
        blurred = blur_image(image_path, sigma=args.sigma, normalize=args.normalize)
        if args.verbose:
            print(f"Blurred {image_path} (sigma={args.sigma})")
        if not args.dry_run:
            save_blurred(image_path, args.output_dir, blurred, args.suffix)
        processed += 1

    print(f"Processed {processed} images.")


if __name__ == "__main__":  # pragma: no cover
    main()
