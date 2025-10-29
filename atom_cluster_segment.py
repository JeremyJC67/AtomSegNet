#!/usr/bin/env python3
"""Segment blurred atom clusters, export contours, masks, and overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, distance_transform_edt, median_filter
from skimage import exposure, measure, morphology
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.measure import approximate_polygon
from skimage.segmentation import watershed


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    return np.clip(arr, 0.0, 1.0)


def polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def smooth_contour(points: np.ndarray, kernel: int) -> np.ndarray:
    if points.shape[0] < kernel or kernel <= 1:
        return points
    xs = median_filter(points[:, 0], size=kernel, mode="wrap")
    ys = median_filter(points[:, 1], size=kernel, mode="wrap")
    return np.column_stack([xs, ys])


def simplify_contour(points: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0:
        return points
    simplified = approximate_polygon(points, tolerance=epsilon)
    return simplified if simplified.shape[0] >= 3 else points


def segment_atom_clusters(
    image: np.ndarray,
    invert: bool,
    sigma_bg: float,
    sigma_fg: float,
    use_local: bool,
    block_size: int,
    offset: float,
    min_area: int,
    close_radius: int,
    open_radius: int,
    split_touching: bool,
    keep_k: int,
    smooth_k: int,
    simplify_eps: float,
    use_clahe: bool,
    dark_object: bool,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    work = image.astype(np.float32)
    if use_clahe:
        work = exposure.equalize_adapthist(work, clip_limit=0.01)
    work = normalize01(work)
    if invert:
        work = 1.0 - work

    bg = gaussian(work, sigma=sigma_bg) if sigma_bg > 0 else 0.0
    diff = np.clip(work - bg, 0.0, 1.0)
    proc = gaussian(diff, sigma=sigma_fg) if sigma_fg > 0 else diff

    if use_local:
        if block_size % 2 == 0:
            block_size += 1
        thresh = threshold_local(proc, block_size=block_size, offset=offset)
        mask = proc < thresh if dark_object else proc > thresh
    else:
        t_val = threshold_otsu(proc)
        mask = proc < t_val if dark_object else proc > t_val

    if close_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(close_radius))
    if open_radius > 0:
        mask = morphology.binary_opening(mask, morphology.disk(open_radius))
    mask = binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=max(1, int(min_area)))

    if split_touching and mask.any():
        dist = distance_transform_edt(mask)
        if dist.max() > 0:
            peaks = dist > (0.5 * dist.max())
            markers, _ = morphology.label(peaks, return_num=True)
            if markers.max() >= 2:
                labels = watershed(-dist, markers, mask=mask)
                mask = labels > 0

    contours_rc = measure.find_contours(mask.astype(np.float32), 0.5)
    contours_xy = [c[:, ::-1] for c in contours_rc if c.shape[0] >= 6]
    ranked = sorted(contours_xy, key=lambda c: polygon_area(c), reverse=True)

    kept: list[np.ndarray] = []
    for contour in ranked[:keep_k]:
        smooth = smooth_contour(contour, smooth_k)
        simple = simplify_contour(smooth, simplify_eps)
        kept.append(simple)

    return mask, proc, kept


def overlay_contours(image: np.ndarray, contours: list[np.ndarray]) -> Image.Image:
    base = Image.fromarray((normalize01(image) * 255).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(base)
    for contour in contours:
        pts = [(float(x), float(y)) for x, y in contour]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=3)
    return base


def save_contours(contours: list[np.ndarray], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not contours:
        return
    combined = np.vstack(contours)
    np.savetxt(out_dir / f"{stem}_boundary_coords.txt", combined, fmt="%.3f", delimiter=",")
    for idx, contour in enumerate(contours):
        np.savetxt(out_dir / f"{stem}_boundary_{idx}.txt", contour, fmt="%.3f", delimiter=",")


def save_cluster_crops(image: np.ndarray, contours: list[np.ndarray], out_dir: Path, stem: str) -> None:
    if not contours:
        return
    for idx, contour in enumerate(contours):
        min_x = max(0, int(np.floor(contour[:, 0].min())) - 5)
        max_x = min(image.shape[1], int(np.ceil(contour[:, 0].max())) + 5)
        min_y = max(0, int(np.floor(contour[:, 1].min())) - 5)
        max_y = min(image.shape[0], int(np.ceil(contour[:, 1].max())) + 5)
        region = image[min_y:max_y, min_x:max_x]
        crop = Image.fromarray((normalize01(region) * 255).astype(np.uint8), mode="L").convert("RGB")
        draw = ImageDraw.Draw(crop)
        local = contour - np.array([min_x, min_y])
        pts = [(float(x), float(y)) for x, y in local]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
        crop.save(out_dir / f"{stem}_cluster_{idx}.png")


def iter_images(root: Path, pattern: str) -> Iterable[Path]:
    if root.is_dir():
        yield from sorted(root.glob(pattern))
    else:
        yield root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract atom cluster contours from blurred images.")
    parser.add_argument("input", type=Path, help="Image file or directory containing images.")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern when input is a directory.")
    parser.add_argument("--output", type=Path, default=Path("atom_cluster_out"), help="Output directory.")
    parser.add_argument("--invert", action="store_true", help="Use if clusters appear brighter than background.")
    parser.add_argument("--sigma-bg", type=float, default=12.0, help="Gaussian sigma for background estimation.")
    parser.add_argument("--sigma-fg", type=float, default=2.0, help="Gaussian sigma for foreground smoothing.")
    parser.add_argument("--use-local", action="store_true", help="Use adaptive local threshold.")
    parser.add_argument("--block-size", type=int, default=121, help="Block size for local thresholding.")
    parser.add_argument("--offset", type=float, default=0.02, help="Offset for local thresholding.")
    parser.add_argument("--min-area", type=int, default=1500, help="Minimum object area in pixels.")
    parser.add_argument("--close", type=int, default=5, help="Closing radius (disk).")
    parser.add_argument("--open", type=int, default=3, help="Opening radius (disk).")
    parser.add_argument("--no-split", action="store_true", help="Disable watershed splitting for touching clusters.")
    parser.add_argument("--keep-k", type=int, default=2, help="Number of largest contours to keep.")
    parser.add_argument("--smooth-k", type=int, default=5, help="Median filter kernel for contour smoothing.")
    parser.add_argument("--simplify-eps", type=float, default=2.0, help="Douglas-Peucker tolerance for simplification.")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE before segmentation.")
    parser.add_argument("--dark-object", action="store_true", help="Treat darker regions as foreground.")
    parser.add_argument("--save-crops", action="store_true", help="Save per-cluster cropped visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")
    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(args.input, args.pattern):
        if not image_path.is_file():
            continue
        stem = image_path.stem
        img = Image.open(image_path).convert("L")
        img_array = np.array(img, dtype=np.float32)

        mask, proc, contours = segment_atom_clusters(
            img_array,
            invert=args.invert,
            sigma_bg=args.sigma_bg,
            sigma_fg=args.sigma_fg,
            use_local=args.use_local,
            block_size=args.block_size,
            offset=args.offset,
            min_area=args.min_area,
            close_radius=args.close,
            open_radius=args.open,
            split_touching=not args.no_split,
            keep_k=args.keep_k,
            smooth_k=args.smooth_k,
            simplify_eps=args.simplify_eps,
            use_clahe=args.clahe,
            dark_object=args.dark_object,
        )

        overlay = overlay_contours(img_array, contours)
        overlay.save(out_root / f"{stem}_overlay.png")
        Image.fromarray((normalize01(proc) * 255).astype(np.uint8), mode="L").save(out_root / f"{stem}_proc.png")
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(out_root / f"{stem}_mask.png")
        save_contours(contours, out_root, stem)
        if args.save_crops:
            save_cluster_crops(img_array, contours, out_root, stem)


if __name__ == "__main__":
    main()
