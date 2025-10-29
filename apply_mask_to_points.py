#!/usr/bin/env python3
"""Filter AtomSegNet detections using precomputed masks and render overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


DTYPE = np.dtype(
    [
        ("cy", np.float64),
        ("cx", np.float64),
        ("min_row", np.int64),
        ("min_col", np.int64),
        ("max_row", np.int64),
        ("max_col", np.int64),
        ("score", np.float64),
    ]
)


def iter_mask_files(root: Path, pattern: str) -> list[Path]:
    files = sorted([p for p in root.glob(pattern) if p.is_file()])
    return files


def select_head_tail(files: list[Path], head: int, tail: int) -> list[Path]:
    n = len(files)
    selected: list[Path] = []
    if head > 0:
        selected.extend(files[: min(head, n)])
    if tail > 0:
        selected.extend(files[max(n - tail, 0) :])
    # remove duplicates while preserving order
    seen = set()
    result: list[Path] = []
    for path in selected:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result if result else files


def load_points(path: Path) -> np.ndarray:
    rows = []
    with path.open("r") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 7:
                raise ValueError(f"Unexpected column count in {path} line {line_no}")
            cy = float(fields[0])
            cx = float(fields[1])
            min_row = int(round(float(fields[2])))
            min_col = int(round(float(fields[3])))
            max_row = int(round(float(fields[4])))
            max_col = int(round(float(fields[5])))
            score = float(fields[6])
            rows.append((cy, cx, min_row, min_col, max_row, max_col, score))
    if not rows:
        return np.zeros(0, dtype=DTYPE)
    return np.asarray(rows, dtype=DTYPE)


def save_points(path: Path, data: np.ndarray) -> None:
    lines = []
    for row in data:
        values = [
            f"{row['cy']}",
            f"{row['cx']}",
            str(int(row["min_row"])),
            str(int(row["min_col"])),
            str(int(row["max_row"])),
            str(int(row["max_col"])),
            f"{row['score']}",
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def filter_points(points: np.ndarray, mask: np.ndarray, dilate: int) -> np.ndarray:
    if points.size == 0 or mask.size == 0:
        return np.zeros(0, dtype=DTYPE)
    work_mask = mask.astype(bool)
    if dilate > 0:
        work_mask = binary_dilation(work_mask, iterations=dilate)
    h, w = work_mask.shape
    ys = np.clip(np.rint(points["cy"]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points["cx"]).astype(int), 0, w - 1)
    keep = work_mask[ys, xs]
    return points[keep]


def draw_points(image: Image.Image, points: np.ndarray, radius: int, color: str) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    r = max(radius, 1)
    for cy, cx in zip(points["cy"], points["cx"]):
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=color, outline=color)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter detection points using contour masks.")
    parser.add_argument("--mask-root", type=Path, default=Path("atom_cluster_out"),
                        help="Directory containing *_outline_mask.png files.")
    parser.add_argument("--mask-pattern", default="*_outline_mask.png",
                        help="Glob pattern for mask files.")
    parser.add_argument("--detection-root", type=Path, default=Path("atomsegnet1742_results"),
                        help="Root directory containing *_denoise folders.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset/1742"),
                        help="Directory containing original dataset images.")
    parser.add_argument("--head", type=int, default=50, help="Number of leading masks to process.")
    parser.add_argument("--tail", type=int, default=50, help="Number of trailing masks to process.")
    parser.add_argument("--dilate", type=int, default=1,
                        help="Dilate mask by this many iterations to include boundary points.")
    parser.add_argument("--radius", type=int, default=3, help="Radius for drawing points.")
    parser.add_argument("--color", default="red", help="Color for overlay points.")
    parser.add_argument("--output-suffix", default="gaussian_blur",
                        help="Suffix inserted into point filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_root = args.mask_root
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")
    detection_root = args.detection_root
    if not detection_root.exists():
        raise FileNotFoundError(f"Detection root not found: {detection_root}")
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    masks = iter_mask_files(mask_root, args.mask_pattern)
    if not masks:
        print("No mask files found.")
        return
    selection = select_head_tail(masks, args.head, args.tail)

    for mask_path in selection:
        base = mask_path.stem.replace("_outline_mask", "")
        detection_dir = detection_root / f"{base}_denoise"
        if not detection_dir.exists():
            print(f"Skip {base}: detection directory missing.")
            continue
        detection_file = detection_dir / f"{base}_denoise_pos_Gen1-gaussianMask.txt"
        if not detection_file.exists():
            print(f"Skip {base}: detection file missing.")
            continue
        dataset_image = dataset_root / f"{base}.jpg"
        if not dataset_image.exists():
            print(f"Skip {base}: dataset image missing.")
            continue

        mask_img = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask_img) > 0
        points = load_points(detection_file)
        filtered = filter_points(points, mask_arr, args.dilate)

        suffix = args.output_suffix
        out_points = detection_dir / f"{base}_denoise_pos_{suffix}_Gen1-gaussianMask.txt"
        save_points(out_points, filtered)

        base_img = Image.open(dataset_image)
        overlay = draw_points(base_img, filtered, args.radius, args.color)
        out_overlay = detection_dir / f"{base}_denoise_origin_{suffix}_Gen1-gaussianMask.png"
        overlay.save(out_overlay)

        print(f"{base}: raw={points.size} kept={filtered.size}")


if __name__ == "__main__":
    main()
