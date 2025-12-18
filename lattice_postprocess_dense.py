#!/usr/bin/env python3
"""Dense lattice-style augmentation of AtomSegNet detections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as ndi_label


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


def list_result_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def select_dirs(dirs: list[Path], head: int, tail: int) -> list[Path]:
    total = len(dirs)
    selected: list[Path] = []
    if head > 0:
        selected.extend(dirs[: min(head, total)])
    if tail > 0:
        selected.extend(dirs[max(total - tail, 0) :])
    if not selected:
        selected = dirs
    seen = set()
    ordered: list[Path] = []
    for d in selected:
        if d in seen:
            continue
        seen.add(d)
        ordered.append(d)
    return ordered


def load_points(path: Path) -> np.ndarray:
    rows = []
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
            min_row = int(round(float(parts[2])))
            min_col = int(round(float(parts[3])))
            max_row = int(round(float(parts[4])))
            max_col = int(round(float(parts[5])))
            score = float(parts[6])
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


def load_mask(mask_root: Path | None, mask_template: str, stem: str) -> np.ndarray | None:
    if mask_root is None:
        return None
    mask_path = mask_root / mask_template.format(stem)
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if mask.ndim != 2:
        return None
    return mask


def split_by_mask(points: np.ndarray, mask: np.ndarray | None) -> list[np.ndarray]:
    if points.size == 0 or mask is None:
        return [points]
    labeled, n_lab = ndi_label(mask.astype(np.int32))
    if n_lab <= 1:
        return [points]
    h, w = labeled.shape
    ys = np.clip(np.rint(points["cy"]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points["cx"]).astype(int), 0, w - 1)
    labels_at_points = labeled[ys, xs]

    groups: list[np.ndarray] = []
    for lab in range(1, n_lab + 1):
        mask_lab = labels_at_points == lab
        if not np.any(mask_lab):
            continue
        groups.append(points[mask_lab])

    outside = labels_at_points == 0
    if np.any(outside):
        groups.append(points[outside])

    if not groups:
        return [points]
    return groups


def densify_group(points: np.ndarray, resid_mult: float, step_frac: float, min_new_dist: float) -> np.ndarray:
    if points.size < 10:
        return points

    coords = np.column_stack([points["cx"], points["cy"]])
    mean = coords.mean(axis=0, keepdims=True)
    C = coords - mean
    try:
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
    except np.linalg.LinAlgError:
        return points

    u = Vt[0]
    v = Vt[1]
    s = C @ u
    d = C @ v

    s_sorted = np.sort(s)
    ds = np.diff(s_sorted)
    ds = ds[ds > 0]
    if ds.size == 0:
        return points
    delta = float(np.median(ds))
    if not np.isfinite(delta) or delta <= 0:
        return points

    omega = 2.0 * np.pi / delta
    M = np.column_stack([np.sin(omega * s), np.cos(omega * s), np.ones_like(s)])
    try:
        params, _, _, _ = np.linalg.lstsq(M, d, rcond=None)
    except np.linalg.LinAlgError:
        return points
    A_sin, A_cos, c = params
    A = float(np.hypot(A_sin, A_cos))
    phi = float(np.arctan2(A_cos, A_sin))
    d_fit = A * np.sin(omega * s + phi) + c
    resid = np.abs(d - d_fit)

    base_thr = max(2.0, 0.5 * A)
    thr = resid_mult * base_thr
    support_mask = resid <= thr
    if not np.any(support_mask):
        return points

    s_support = s[support_mask]
    s_min = float(s_support.min())
    s_max = float(s_support.max())

    h_med = float(np.median(points["max_row"] - points["min_row"]))
    w_med = float(np.median(points["max_col"] - points["min_col"]))
    score_med = float(np.median(points["score"]))

    step = max(delta * step_frac, 1e-3)
    new_rows = []
    coords_existing = coords.copy()

    s_val = s_min
    while s_val <= s_max + 1e-6:
        d_val = A * np.sin(omega * s_val + phi) + c
        coord = mean[0] + s_val * u + d_val * v
        cx_new = float(coord[0])
        cy_new = float(coord[1])
        dist = np.hypot(coords_existing[:, 0] - cx_new, coords_existing[:, 1] - cy_new)
        if np.any(dist < min_new_dist):
            s_val += step
            continue
        min_row = int(round(cy_new - h_med / 2.0))
        max_row = int(round(cy_new + h_med / 2.0))
        min_col = int(round(cx_new - w_med / 2.0))
        max_col = int(round(cx_new + w_med / 2.0))
        new_rows.append(
            (
                cy_new,
                cx_new,
                min_row,
                min_col,
                max_row,
                max_col,
                score_med,
            )
        )
        coords_existing = np.vstack([coords_existing, [cx_new, cy_new]])
        s_val += step

    if not new_rows:
        return points
    new_pts = np.asarray(new_rows, dtype=DTYPE)
    return np.concatenate([points, new_pts], axis=0)


def apply_lattice_dense(
    points: np.ndarray,
    mask: np.ndarray | None,
    resid_mult: float,
    step_frac: float,
    min_new_dist: float,
) -> np.ndarray:
    if points.size == 0:
        return points
    groups = split_by_mask(points, mask)
    augmented = [
        densify_group(g, resid_mult=resid_mult, step_frac=step_frac, min_new_dist=min_new_dist)
        for g in groups
    ]
    return np.concatenate(augmented, axis=0) if augmented else points


def draw_overlay(base_img: Path, points: np.ndarray, radius: int, color: str, out_path: Path) -> None:
    img = Image.open(base_img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(radius, 1)
    for cy, cx in zip(points["cy"], points["cx"]):
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=color, outline=color)
    img.save(out_path)


def process_directory(directory: Path, args: argparse.Namespace) -> Tuple[int, int]:
    stem = directory.name
    matches = sorted(directory.glob(args.pattern))
    if not matches:
        if args.verbose:
            print(f"{stem}: no points matching {args.pattern}")
        return 0, 0
    pts_path = matches[0]
    pts = load_points(pts_path)
    raw = pts.size

    mask = load_mask(args.mask_root, args.mask_template, stem) if args.mask_root is not None else None
    augmented = apply_lattice_dense(
        pts,
        mask,
        resid_mult=args.resid_mult,
        step_frac=args.step_frac,
        min_new_dist=args.min_new_dist,
    )
    kept = augmented.size

    if args.output_suffix:
        out_name = pts_path.stem + f"_{args.output_suffix}" + pts_path.suffix
    else:
        out_name = pts_path.name
    out_path = pts_path.with_name(out_name)
    save_points(out_path, augmented)

    if args.dataset_root is not None and kept > 0:
        base_img = args.dataset_root / f"{stem}.jpg"
        if base_img.exists():
            overlay_name = f"{stem}_origin_gaussian_blur_{args.output_suffix}_Gen1-gaussianMask.png"
            overlay_path = directory / overlay_name
            draw_overlay(base_img, augmented, args.radius, args.color, overlay_path)
            if args.verbose:
                print(f"{stem}: wrote {overlay_path.name}")

    if args.verbose:
        print(f"{stem}: raw={raw} after_dense={kept}")
    return raw, kept


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense lattice augmentation for AtomSegNet detections.")
    p.add_argument("input_root", type=Path, help="Root directory containing per-image result folders.")
    p.add_argument(
        "--pattern",
        default="*_pos_gaussian_blur_Gen1-gaussianMask.txt",
        help="Glob pattern for input coordinate files.",
    )
    p.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Optional root of *_heavyblur_mask.png files.",
    )
    p.add_argument(
        "--mask-template",
        default="{}_heavyblur_mask.png",
        help="Mask filename template formatted with directory name.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional base image root containing <dir>.jpg files.",
    )
    p.add_argument("--head", type=int, default=50, help="Number of leading folders to process.")
    p.add_argument("--tail", type=int, default=50, help="Number of trailing folders to process.")
    p.add_argument(
        "--output-suffix",
        default="lattice_dense",
        help="Suffix appended to augmented coordinate filenames.",
    )
    p.add_argument(
        "--resid-mult",
        type=float,
        default=3.0,
        help="Residual multiplier for support point selection (higher -> more support).",
    )
    p.add_argument(
        "--step-frac",
        type=float,
        default=0.33,
        help="Sampling step as fraction of lattice spacing Δs (default ~3 points per period).",
    )
    p.add_argument(
        "--min-new-dist",
        type=float,
        default=2.0,
        help="Minimum distance (pixels) from existing points to accept a new point.",
    )
    p.add_argument("--radius", type=int, default=3, help="Overlay marker radius.")
    p.add_argument("--color", default="red", help="Overlay marker color.")
    p.add_argument("--verbose", action="store_true", help="Print per-directory details.")
    return p.parse_args()


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
    total_raw = total_kept = 0
    for d in selection:
        raw, kept = process_directory(d, args)
        total_raw += raw
        total_kept += kept
    print(f"Processed {len(selection)} folders; raw={total_raw}, after_dense={total_kept}.")


if __name__ == "__main__":  # pragma: no cover
    main()

