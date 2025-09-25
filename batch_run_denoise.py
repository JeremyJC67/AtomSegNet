#!/usr/bin/env python3
"""Batch apply AtomSegNet's denoise models to a folder of images."""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from utils.utils import load_model, map01

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTS and path.is_file():
            yield path


def run_batch(input_dir: Path, output_dir: Path, model_name: str, use_cuda: bool, iterations: int,
              overwrite: bool) -> None:
    model_path = Path("model_weights") / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weight not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(input_dir):
        output_path = output_dir / f"{image_path.stem}_{model_name}.png"
        if output_path.exists() and not overwrite:
            print(f"→ Skip existing {output_path}")
            continue

        print(f"Processing {image_path} → {output_path}")
        image = Image.open(image_path).convert("L")
        result = load_model(str(model_path), image, use_cuda, iterations)
        denoised = (map01(result) * 255).astype(np.uint8)
        Image.fromarray(denoised, mode="L").save(output_path)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run AtomSegNet denoise models")
    parser.add_argument("input_dir", type=Path, help="Folder containing source images")
    parser.add_argument("output_dir", type=Path, help="Folder to store denoised outputs")
    parser.add_argument("--model", default="denoise", help="Model name from model_weights (default: denoise)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--iter", type=int, default=1, help="Inference iterations (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(args.input_dir, args.output_dir, args.model, args.cuda, args.iter, args.overwrite)


if __name__ == "__main__":  # pragma: no cover
    main()
