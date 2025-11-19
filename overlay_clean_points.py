#!/usr/bin/env python3
"""Overlay cleaned detection points on AtomSegNet origin images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


def list_result_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def select_dirs(dirs: Sequence[Path], head: int, tail: int) -> list[Path]:
    selection: list[Path] = []
    total = len(dirs)
    if head > 0:
        selection.extend(dirs[: min(head, total)])
    if tail > 0:
        selection.extend(dirs[max(total - tail, 0) :])
    seen = set()
    unique_selection: list[Path] = []
    for path in selection:
        if path not in seen:
            unique_selection.append(path)
            seen.add(path)
    return unique_selection


def load_positions(path: Path) -> np.ndarray:
    data = []
    with path.open("r") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                raise ValueError(f"Unexpected column count in {path} line {line_no}")
            cy = float(parts[0])
            cx = float(parts[1])
            data.append((cy, cx))
    return np.array(data, dtype=np.float32)


def find_origin_image(directory: Path) -> Path:
    matches = sorted(directory.glob("*_origin_*.png"))
    if not matches:
        raise FileNotFoundError(f"No origin image found in {directory}")
    return matches[0]


def resolve_base_image(directory: Path, base_root: Optional[Path], image_ext: str,
                       name_mode: str) -> Path:
    if base_root is None:
        return find_origin_image(directory)

    if name_mode == "full":
        stem = directory.name
    else:
        stem = directory.name.split("_")[0]
    ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
    candidate = base_root / f"{stem}{ext}"
    if not candidate.exists():
        raise FileNotFoundError(f"Base image not found: {candidate}")
    return candidate


def find_clean_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No position file matching {pattern} in {directory}")
    return matches[0]


def draw_points(image: Image.Image, points: np.ndarray, radius: int, color: str) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    r = max(radius, 1)
    for cy, cx in points:
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=color, outline=color)
    return image


def output_name(origin_path: Optional[Path], directory: Path, overlay_on_dataset: bool) -> Path:
    if not overlay_on_dataset and origin_path is not None:
        name = origin_path.name.replace("_origin_", "_origin_clean_")
        return origin_path.with_name(name)

    if origin_path is not None:
        name = origin_path.name.replace("_origin_", "_origin_clean_")
        return directory / name

    stem = directory.name
    name = f"{stem}_origin_clean.png"
    return directory / name


def process_directory(directory: Path, radius: int, color: str, dry_run: bool,
                      base_root: Optional[Path], image_ext: str, positions_pattern: str,
                      name_mode: str) -> Path | None:
    base_path = resolve_base_image(directory, base_root, image_ext, name_mode)
    overlay_on_dataset = base_root is not None
    origin_path: Optional[Path] = None
    if overlay_on_dataset:
        try:
            origin_path = find_origin_image(directory)
        except FileNotFoundError:
            origin_path = None
    else:
        origin_path = base_path
    clean_path = find_clean_file(directory, positions_pattern)
    points = load_positions(clean_path)
    if points.size == 0:
        return None
    image = Image.open(base_path)
    image = draw_points(image, points, radius, color)
    out_path = output_name(origin_path, directory, overlay_on_dataset)
    if not dry_run:
        image.save(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay cleaned detections on origin images.")
    parser.add_argument("input_root", type=Path, help="Root directory with AtomSegNet per-image folders.")
    parser.add_argument("--head", type=int, default=50, help="Number of leading folders to process (default: 50).")
    parser.add_argument("--tail", type=int, default=50, help="Number of trailing folders to process (default: 50).")
    parser.add_argument("--radius", type=int, default=3, help="Radius of the drawn markers in pixels.")
    parser.add_argument("--color", default="red", help="Marker color (Pillow format).")
    parser.add_argument("--base-image-root", type=Path, default=None,
                        help="Directory containing original images (overrides origin PNGs).")
    parser.add_argument("--image-ext", default=".jpg",
                        help="Extension of base images when using --base-image-root (default: .jpg).")
    parser.add_argument("--base-name-mode", choices=["prefix", "full"], default="prefix",
                        help="How to derive dataset filename from directory name when using --base-image-root.")
    parser.add_argument("--positions-pattern", default="*_pos_*_clean.txt",
                        help="Glob pattern for position files inside each directory.")
    parser.add_argument("--dry-run", action="store_true", help="List files without writing output.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file status.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_root
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    dirs = list_result_dirs(root)
    if not dirs:
        print("No result directories found.")
        return

    selection = select_dirs(dirs, args.head, args.tail)
    if not selection:
        selection = dirs

    written = 0
    for directory in selection:
        try:
            out_path = process_directory(
                directory,
                args.radius,
                args.color,
                args.dry_run,
                args.base_image_root,
                args.image_ext,
                args.positions_pattern,
                args.base_name_mode,
            )
        except FileNotFoundError as exc:
            if args.verbose:
                print(f"Skip {directory}: {exc}")
            continue
        if out_path is None:
            if args.verbose:
                print(f"{directory}: no points after cleaning")
            continue
        written += 1
        if args.verbose:
            print(f"Wrote {out_path}")

    if args.dry_run:
        print(f"Would process {len(selection)} directories (matched {len(dirs)} total).")
    else:
        print(f"Processed {len(selection)} directories, generated {written} overlays.")


if __name__ == "__main__":  # pragma: no cover
    main()
