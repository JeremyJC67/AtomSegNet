#!/usr/bin/env python3
"""Generate enhanced denoised images for AtomSegNet preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from PIL import ImageOps


def list_images(root: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in root.iterdir() if p.suffix.lower() in supported])


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_v) / (max_v - min_v)
    return (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)


def enhance_image(
    image: np.ndarray,
    *,
    sigma_bg: float,
    sigma_fg: float,
    clahe: bool,
) -> np.ndarray:
    work = image.astype(np.float32)
    if clahe:
        # PIL equalize operates on uint8
        eq = ImageOps.equalize(Image.fromarray(work.astype(np.uint8), mode="L"))
        work = np.array(eq, dtype=np.float32)
    work = work / 255.0
    if sigma_bg > 0:
        background = gaussian_filter(work, sigma=sigma_bg)
        work = np.clip(work - background, 0.0, 1.0)
    if sigma_fg > 0:
        work = gaussian_filter(work, sigma=sigma_fg)
    return normalize_uint8(work)


def select_subset(files: list[Path], head: int, tail: int) -> Iterable[Path]:
    n = len(files)
    if head <= 0 and tail <= 0:
        return files
    selected: list[Path] = []
    if head > 0:
        selected.extend(files[: min(head, n)])
    if tail > 0:
        selected.extend(files[max(n - tail, 0):])
    seen = set()
    unique: list[Path] = []
    for path in selected:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced denoise preprocessing for AtomSegNet.")
    parser.add_argument("dataset_root", type=Path, help="Directory containing raw images.")
    parser.add_argument("output_root", type=Path, help="Destination directory for enhanced denoised images.")
    parser.add_argument("--head", type=int, default=50, help="Process first N images (default: 50).")
    parser.add_argument("--tail", type=int, default=50, help="Process last N images (default: 50).")
    parser.add_argument("--sigma-bg", type=float, default=18.0,
                        help="Gaussian sigma used for background estimation (default: 18.0).")
    parser.add_argument("--sigma-fg", type=float, default=1.8,
                        help="Gaussian sigma used for foreground smoothing (default: 1.8).")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE before filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_images = list_images(args.dataset_root)
    if not all_images:
        print("No images found to process.")
        return
    selection = list(select_subset(all_images, args.head, args.tail))
    print(f"Processing {len(selection)} images (head={args.head}, tail={args.tail}).")

    for path in selection:
        image = np.array(Image.open(path).convert("L"))
        enhanced = enhance_image(
            image,
            sigma_bg=args.sigma_bg,
            sigma_fg=args.sigma_fg,
            clahe=args.clahe,
        )
        out_path = args.output_root / f"{path.stem}.png"
        Image.fromarray(enhanced, mode="L").save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
