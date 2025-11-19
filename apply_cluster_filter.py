#!/usr/bin/env python3
"""Filter enhanced detections by precomputed cluster masks and redraw overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


POINT_DTYPE = np.dtype(
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


def load_points(path: Path) -> np.ndarray:
    rows = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 7:
                continue
            cy, cx, min_row, min_col, max_row, max_col, score = map(float, fields)
            rows.append(
                (
                    cy,
                    cx,
                    int(round(min_row)),
                    int(round(min_col)),
                    int(round(max_row)),
                    int(round(max_col)),
                    score,
                )
            )
    if not rows:
        return np.zeros(0, dtype=POINT_DTYPE)
    return np.asarray(rows, dtype=POINT_DTYPE)


def save_points(path: Path, data: np.ndarray) -> None:
    lines = [
        ",".join(
            [
                f"{row['cy']}",
                f"{row['cx']}",
                str(int(row["min_row"])),
                str(int(row["min_col"])),
                str(int(row["max_row"])),
                str(int(row["max_col"])),
                f"{row['score']}",
            ]
        )
        for row in data
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def filter_points(points: np.ndarray, mask: np.ndarray, dilate: int) -> np.ndarray:
    if points.size == 0:
        return points
    valid_mask = mask.astype(bool)
    if dilate > 0:
        valid_mask = binary_dilation(valid_mask, iterations=dilate)
    h, w = valid_mask.shape
    ys = np.clip(np.rint(points["cy"]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points["cx"]).astype(int), 0, w - 1)
    keep = valid_mask[ys, xs]
    return points[keep]


def draw_points(base: Image.Image, points: np.ndarray, radius: int, color: str) -> Image.Image:
    img = base.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(radius, 1)
    for cy, cx in points[["cy", "cx"]]:
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=color, outline=color)
    return img


def build_ids(head: int, tail: int, total: int = 1175) -> list[str]:
    ids = [f"{i:04d}" for i in range(head)]
    tail_ids = [f"{i:04d}" for i in range(total - tail, total)]
    return ids + tail_ids


def parse_ids_from_masks(mask_root: Path, template: str) -> list[str]:
    if "{}" not in template:
        raise ValueError("mask template must contain '{}' placeholder to infer ids.")
    prefix, suffix = template.split("{}", 1)
    pattern = f"{prefix}*{suffix}"
    ids = []
    for path in sorted(mask_root.glob(pattern)):
        name = path.name
        if not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        end = -len(suffix) if suffix else None
        idx = name[len(prefix):end]
        if idx:
            ids.append(idx)
    return ids


def select_from_list(items: list[str], head: int, tail: int) -> list[str]:
    selected: list[str] = []
    if head > 0:
        selected.extend(items[: min(head, len(items))])
    if tail > 0:
        selected.extend(items[max(len(items) - tail, 0):])
    if not selected:
        selected = items
    seen = set()
    ordered: list[str] = []
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter enhanced detections by cluster masks.")
    parser.add_argument("--mask-root", type=Path, default=Path("atom_cluster_out"))
    parser.add_argument("--mask-template", default="{}_outline_mask.png",
                        help="Format string for mask filenames (default: '{}_outline_mask.png').")
    parser.add_argument("--detection-root", type=Path, default=Path("atomsegnet_enhanced_results"))
    parser.add_argument("--dir-template", default="{}",
                        help="Format string for detection subdirectories (default: '{}').")
    parser.add_argument("--points-template",
                        default="{}_pos_Gen1-gaussianMask_gaussian_blur_clean.txt",
                        help="Format string for input detection filenames.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jicwang/atom-research/dataset/1742"))
    parser.add_argument("--head", type=int, default=50)
    parser.add_argument("--tail", type=int, default=50)
    parser.add_argument("--use-mask-names", action="store_true",
                        help="Derive ids by scanning mask filenames instead of numeric ranges.")
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--color", default="red")
    parser.add_argument("--suffix", default="gaussian_blur", help="Suffix for saved coordinates/overlays.")
    args = parser.parse_args()

    if not args.mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {args.mask_root}")
    if not args.detection_root.exists():
        raise FileNotFoundError(f"Detection root not found: {args.detection_root}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    if args.use_mask_names:
        all_ids = parse_ids_from_masks(args.mask_root, args.mask_template)
        if not all_ids:
            raise FileNotFoundError("No mask files found to derive ids.")
        ids = select_from_list(all_ids, args.head, args.tail)
    else:
        ids = build_ids(args.head, args.tail)

    for idx in ids:
        mask_path = args.mask_root / args.mask_template.format(idx)
        detection_dir = args.detection_root / args.dir_template.format(idx)
        dataset_img = args.dataset_root / f"{idx}.jpg"

        if not (mask_path.exists() and detection_dir.exists() and dataset_img.exists()):
            print(f"Skip {idx}: missing resources.")
            continue

        points_path = detection_dir / args.points_template.format(idx)
        if not points_path.exists():
            print(f"Skip {idx}: cleaned points missing.")
            continue

        mask = np.array(Image.open(mask_path).convert("L")) > 0
        points = load_points(points_path)
        filtered = filter_points(points, mask, args.dilate)

        out_points = detection_dir / f"{idx}_pos_{args.suffix}_Gen1-gaussianMask.txt"
        save_points(out_points, filtered)

        base = Image.open(dataset_img).convert("RGB")
        overlay_path = detection_dir / f"{idx}_origin_{args.suffix}_Gen1-gaussianMask.png"
        draw_points(base, filtered, args.radius, args.color).save(overlay_path)

        print(f"{idx}: kept {filtered.size} points -> {overlay_path.name}")


if __name__ == "__main__":
    main()
