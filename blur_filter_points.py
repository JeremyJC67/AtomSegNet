#!/usr/bin/env python3
"""Filter AtomSegNet detection points with a Gaussian-blur mask."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import threshold_local, threshold_otsu
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt


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


def iter_detection_files(root: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            yield path


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
    return np.array(rows, dtype=DTYPE)


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


def polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def infer_polarity_from_points(image01: np.ndarray, points: Optional[np.ndarray], radius: int = 3) -> str:
    """Infer whether detections correspond to darker ('dark') or brighter ('bright') regions."""
    if points is None or points.size == 0:
        return "dark"
    h, w = image01.shape
    ys = np.clip(np.rint(points["cy"]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points["cx"]).astype(int), 0, w - 1)
    yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    disk_mask = (yy * yy + xx * xx) <= radius * radius
    means: list[float] = []
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        patch = image01[y0:y1, x0:x1]
        local = disk_mask[(y0 - y + radius):(y1 - y + radius), (x0 - x + radius):(x1 - x + radius)]
        if patch.size and local.any():
            means.append(float(patch[local].mean()))
    if not means:
        return "dark"
    image_med = float(np.median(image01))
    point_med = float(np.median(means))
    return "dark" if point_med < image_med else "bright"

def keep_components_with_points(mask: np.ndarray,
                                points: Optional[np.ndarray],
                                min_points: int,
                                min_density: float) -> np.ndarray:
    """Keep connected components that contain enough detection points and density."""
    if mask.sum() == 0:
        return mask
    labels = label(mask)
    if labels.max() == 0 or points is None or points.size == 0:
        return mask
    h, w = mask.shape
    ys = np.clip(np.rint(points["cy"]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points["cx"]).astype(int), 0, w - 1)
    comp_ids = labels[ys, xs]
    kept = np.zeros_like(mask, dtype=bool)
    for region in regionprops(labels):
        area = float(region.area)
        if area <= 0:
            continue
        pts_in = int((comp_ids == region.label).sum())
        density = pts_in / area
        if pts_in >= min_points and density >= min_density:
            kept[labels == region.label] = True
    return kept


def extract_topk_contours(mask: np.ndarray, k: int = 2) -> list[np.ndarray]:
    """Return the largest k contours (in x, y order)."""
    contours = measure.find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return []
    ranked: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        contour_xy = contour[:, ::-1]
        area = polygon_area(contour_xy.astype(np.float32))
        ranked.append((area, contour_xy))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [c for _, c in ranked[:k] if c.size]


def save_boundary_coords_multi(output_dir: Path, base_name: str, contours_xy: list[np.ndarray]) -> None:
    if not contours_xy:
        return
    combined = np.vstack(contours_xy)
    np.savetxt(output_dir / f"{base_name}_boundary_coords.txt", combined, fmt="%.4f", delimiter=",")
    for idx, contour_xy in enumerate(contours_xy):
        np.savetxt(output_dir / f"{base_name}_boundary_{idx}.txt", contour_xy, fmt="%.4f", delimiter=",")


def draw_boundary_overlay(image: Image.Image,
                          contours_xy: list[np.ndarray],
                          points: np.ndarray,
                          point_radius: int = 3,
                          contour_color: tuple = (255, 0, 0),
                          contour_width: int = 3) -> Image.Image:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    for contour_xy in contours_xy:
        if contour_xy is not None and contour_xy.shape[0] >= 2:
            coords = [(float(x), float(y)) for x, y in contour_xy]
            if len(coords) >= 2:
                draw.line(coords + [coords[0]], fill=contour_color, width=contour_width)
    if points.size:
        r = max(point_radius, 1)
        for cy, cx in zip(points["cy"], points["cx"]):
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=(0, 255, 0), outline=(0, 255, 0))
    return img


def build_mask(image: np.ndarray,
               sigma: float,
               threshold_scale: float,
               radius_frac: float,
               open_radius_frac: Optional[float],
               points: Optional[np.ndarray] = None,
               use_local: bool = True,
               local_block: int = 91,
               local_offset: float = 5.0,
               min_obj_area: int = 1200,
               split_touching: bool = True,
               sigma_bg: float = 25.0,
               inside_ratio_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    img = image.astype(np.float32)
    vmin, vmax = np.percentile(img, (1, 99))
    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    blurred = ndi.gaussian_filter(img, sigma=sigma) if sigma > 0 else img
    background = ndi.gaussian_filter(blurred, sigma=sigma_bg)
    diff_dark = background - blurred
    diff_bright = blurred - background

    polarity = infer_polarity_from_points(blurred, points)
    diff = diff_dark if polarity == 'dark' else diff_bright

    def threshold_and_postprocess(diff_map: np.ndarray) -> np.ndarray:
        if use_local:
            block = local_block + 1 if local_block % 2 == 0 else local_block
            thr = threshold_local(diff_map, block_size=block, offset=local_offset)
            mask_local = diff_map > thr
        else:
            t = threshold_otsu(diff_map)
            mask_local = diff_map > (t * threshold_scale)
        close_radius = max(1, int(round(1.2 * sigma)))
        open_radius = max(1, int(round(0.6 * sigma))) if open_radius_frac is None else max(1, int(round(open_radius_frac * sigma)))
        mask_local = ndi.binary_closing(mask_local, structure=disk(close_radius))
        mask_local = ndi.binary_opening(mask_local, structure=disk(open_radius))
        mask_local = ndi.binary_fill_holes(mask_local)
        mask_local = remove_small_objects(mask_local, min_size=int(min_obj_area))
        return mask_local

    mask = threshold_and_postprocess(diff)

    if points is not None and points.size and mask.any():
        h, w = mask.shape
        ys = np.clip(np.rint(points['cy']).astype(int), 0, h - 1)
        xs = np.clip(np.rint(points['cx']).astype(int), 0, w - 1)
        inside_ratio = float(mask[ys, xs].mean()) if mask.any() else 0.0
        if inside_ratio < inside_ratio_threshold:
            diff_alt = diff_bright if polarity == 'dark' else diff_dark
            mask = threshold_and_postprocess(diff_alt)
            diff = diff_alt

    if split_touching and mask.any():
        dist = ndi.distance_transform_edt(mask)
        if dist.max() > 0:
            grad_y = ndi.sobel(diff, axis=0)
            grad_x = ndi.sobel(diff, axis=1)
            grad_mag = np.hypot(grad_x, grad_y)
            if mask.any():
                grad_thresh = np.percentile(grad_mag[mask], 60) if np.any(mask) else grad_mag.max()
            else:
                grad_thresh = grad_mag.max()
            peak_mask = (dist > 0.45 * dist.max()) & (grad_mag < grad_thresh)
            markers, _ = ndi.label(peak_mask)
            if markers.max() >= 2:
                labels_ws = watershed(-dist, markers, mask=mask)
                mask = labels_ws > 0

    return mask, blurred


def apply_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if points.size == 0 or mask.size == 0:
        return np.zeros(0, dtype=DTYPE)
    ys = np.clip(np.rint(points["cy"]).astype(int), 0, mask.shape[0] - 1)
    xs = np.clip(np.rint(points["cx"]).astype(int), 0, mask.shape[1] - 1)
    keep = mask[ys, xs]
    return points[keep]


def infer_image_path(dataset_root: Path, detection_path: Path, image_ext: str) -> Path:
    ext = image_ext if image_ext.startswith(".") else f".{image_ext}"

    candidates = []
    dir_name = detection_path.parent.name
    stem = detection_path.stem.split("_pos")[0]
    for token in (stem, dir_name):
        if token and token not in candidates:
            candidates.append(token)
        parts = token.split("_")
        if parts and parts[0] not in candidates:
            candidates.append(parts[0])

    for base in candidates:
        candidate_path = dataset_root / f"{base}{ext}"
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Dataset image not found for {detection_path}")


def save_intermediate(base_name: str,
                      blur_img: np.ndarray,
                      mask: np.ndarray,
                      args: argparse.Namespace) -> None:
    if args.save_blur_dir is None:
        return
    out_root = args.save_blur_dir
    out_root.mkdir(parents=True, exist_ok=True)

    Image.fromarray(blur_img, mode="L").save(out_root / f"{base_name}_blur.png")

    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(out_root / f"{base_name}_mask.png")


def save_blur_outline(base_name: str,
                      blur_img: np.ndarray,
                      contours_xy: list[np.ndarray],
                      args: argparse.Namespace) -> None:
    if args.save_blur_dir is None:
        return
    out_root = args.save_blur_dir
    out_root.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(blur_img, mode="L").convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for contour_xy in contours_xy:
        if contour_xy is None or contour_xy.size == 0:
            continue
        coords = [(float(x), float(y)) for x, y in contour_xy]
        if len(coords) >= 2:
            draw.line(coords + [coords[0]], fill=(255, 0, 0), width=3)
    overlay.save(out_root / f"{base_name}_outline_overlay.png")


def save_individual_contour_images(base_name: str,
                                   image: np.ndarray,
                                   contours_xy: list[np.ndarray],
                                   args: argparse.Namespace) -> None:
    """Save individual images for each detected atomic cluster with its contour highlighted."""
    if args.save_blur_dir is None:
        return
    out_root = args.save_blur_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, contour_xy in enumerate(contours_xy):
        if contour_xy is None or contour_xy.size == 0:
            continue

        # Get bounding box for the contour
        min_x = max(0, int(np.floor(contour_xy[:, 0].min())) - 5)
        max_x = min(image.shape[1], int(np.ceil(contour_xy[:, 0].max())) + 5)
        min_y = max(0, int(np.floor(contour_xy[:, 1].min())) - 5)
        max_y = min(image.shape[0], int(np.ceil(contour_xy[:, 1].max())) + 5)

        # Extract region
        region = image[min_y:max_y, min_x:max_x].copy()
        region_img = Image.fromarray(region, mode="L").convert("RGB")
        draw = ImageDraw.Draw(region_img)

        # Adjust contour coordinates to region space
        contour_local = contour_xy - np.array([min_x, min_y])
        coords = [(float(x), float(y)) for x, y in contour_local]
        if len(coords) >= 2:
            draw.line(coords + [coords[0]], fill=(255, 0, 0), width=2)

        region_img.save(out_root / f"{base_name}_cluster_{idx}.png")


def resolve_outline_source(base_name: str, args: argparse.Namespace) -> Optional[Path]:
    if args.outline_root is None:
        return None
    outline_root = args.outline_root
    candidates: list[Path] = []

    tokens = [base_name]
    if "_denoise" in base_name:
        prefix = base_name.split("_denoise")[0]
        if prefix:
            tokens.append(prefix)
    primary = base_name.split("_")[0]
    if primary and primary not in tokens:
        tokens.append(primary)

    seen: set[Path] = set()
    for token in tokens:
        for suffix in ("_outline.png", "_outline_overlay.png", ".png"):
            candidate = outline_root / f"{token}{suffix}"
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_file(detection_path: Path, dataset_root: Path, args: argparse.Namespace) -> tuple[int, int]:
    points = load_points(detection_path)
    image_path = infer_image_path(dataset_root, detection_path, args.image_ext)
    if not image_path.exists():
        raise FileNotFoundError(f"Dataset image not found: {image_path}")
    base_name = detection_path.stem.split("_pos")[0]

    outline_source = resolve_outline_source(base_name, args)
    mask_source_path = outline_source if outline_source is not None else image_path
    mask_source = Image.open(mask_source_path).convert("L")

    mask, blurred = build_mask(np.array(mask_source, dtype=np.float32),
                               sigma=args.sigma,
                               threshold_scale=args.threshold_scale,
                               radius_frac=args.radius_frac,
                               open_radius_frac=args.open_radius_frac,
                               points=points,
                               use_local=not args.use_global_threshold,
                               local_block=args.local_block,
                               local_offset=args.local_offset,
                               min_obj_area=args.min_obj_area,
                               split_touching=not args.no_split_touching,
                               sigma_bg=args.sigma_bg,
                               inside_ratio_threshold=args.inside_ratio_threshold)
    mask = keep_components_with_points(mask,
                                       points,
                                       min_points=args.min_pts_in_blob,
                                       min_density=args.min_density_in_blob)
    contours_xy = extract_topk_contours(mask, k=args.max_contours)
    blur_norm = blurred - blurred.min()
    if blur_norm.max() > 0:
        blur_norm = blur_norm / blur_norm.max()
    blur_img = (blur_norm * 255).astype(np.uint8)
    save_intermediate(base_name, blur_img, mask, args)
    save_blur_outline(base_name, blur_img, contours_xy, args)
    save_individual_contour_images(base_name, blur_img, contours_xy, args)
    filtered = apply_mask(points, mask)

    if args.output_suffix:
        out_path = detection_path.with_name(
            detection_path.name.replace("_pos_", f"_pos_{args.output_suffix}_"))
    else:
        out_path = detection_path
    if not args.dry_run:
        save_points(out_path, filtered)
        save_boundary_coords_multi(detection_path.parent, base_name, contours_xy)
        origin_image = None
        origin_candidates = sorted(detection_path.parent.glob(f"{base_name}_origin_*.png"))
        if origin_candidates:
            origin_image = Image.open(origin_candidates[0]).convert("L")
        else:
            origin_image = Image.open(image_path).convert("L")
        overlay = draw_boundary_overlay(origin_image, contours_xy, filtered)
        if origin_candidates:
            overlay_name = origin_candidates[0].name.replace("_origin_", "_origin_gaussian_blur_")
        else:
            overlay_name = f"{base_name}_origin_gaussian_blur.png"
        overlay.save(detection_path.parent / overlay_name)

    return points.size, filtered.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter AtomSegNet detection points using Gaussian-blur masks."
    )
    parser.add_argument("detection_root", type=Path,
                        help="Root containing *_pos_*.txt detection files.")
    parser.add_argument("dataset_root", type=Path,
                        help="Root containing original dataset images.")
    parser.add_argument("--pattern", default="*_pos_Gen1-gaussianMask.txt",
                        help="Glob pattern for detection files.")
    parser.add_argument("--image-ext", default=".jpg",
                        help="Extension of dataset images (default: .jpg).")
    parser.add_argument("--output-suffix", default="blurmask",
                        help="Suffix inserted after _pos_ for filtered outputs.")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Gaussian blur sigma (pixels).")
    parser.add_argument("--threshold-scale", type=float, default=0.95,
                        help="Scale applied to Otsu threshold (default: 0.95).")
    parser.add_argument("--radius-frac", type=float, default=0.10,
                        help="Morphological closing radius as fraction of min(H, W). Defaults to 0.10.")
    parser.add_argument("--open-radius-frac", type=float, default=None,
                        help="Morphological opening radius fraction (defaults to radius-frac).")
    parser.add_argument("--save-blur-dir", type=Path, default=None,
                        help="Optional directory to save blurred images and masks.")
    parser.add_argument("--outline-root", type=Path, default=None,
                        help="Directory containing pre-blurred outline images (e.g., *_outline.png).")
    parser.add_argument("--max-contours", type=int, default=2,
                        help="Keep at most this many largest contours per mask (default: 2).")
    parser.add_argument("--local-block", type=int, default=91,
                        help="Block size for local thresholding (odd integer, default: 91).")
    parser.add_argument("--local-offset", type=float, default=5.0,
                        help="Offset for local thresholding (default: 5.0).")
    parser.add_argument("--min-obj-area", type=int, default=1200,
                        help="Minimum connected component area to keep (default: 1200).")
    parser.add_argument("--sigma-bg", type=float, default=25.0,
                        help="Gaussian sigma for background estimation (default: 25.0).")
    parser.add_argument("--min-pts-in-blob", type=int, default=30,
                        help="Minimum detection points required inside a blob (default: 30).")
    parser.add_argument("--min-density-in-blob", type=float, default=1e-4,
                        help="Minimum detection point density per blob (default: 1e-4).")
    parser.add_argument("--inside-ratio-threshold", type=float, default=0.5,
                        help="Minimum fraction of detections that must fall inside mask before polarity flip (default: 0.5).")
    parser.add_argument("--use-global-threshold", action="store_true",
                        help="Use global Otsu threshold instead of adaptive local threshold.")
    parser.add_argument("--no-split-touching", action="store_true",
                        help="Disable watershed splitting of touching blobs.")
    parser.add_argument("--head", type=int, default=50,
                        help="Number of leading files to process (default: 50).")
    parser.add_argument("--tail", type=int, default=50,
                        help="Number of trailing files to process (default: 50).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute statistics without writing outputs.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-file details.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.detection_root.exists():
        raise FileNotFoundError(f"Detection root not found: {args.detection_root}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    files = list(iter_detection_files(args.detection_root, args.pattern))
    if not files:
        print("No detection files found.")
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

    total_raw = total_kept = 0
    for path in files:
        raw, kept = process_file(path, args.dataset_root, args)
        total_raw += raw
        total_kept += kept
        if args.verbose:
            print(f"{path}: raw={raw} kept={kept}")

    print(f"Processed {len(files)} files.")
    print(f"Total detections: {total_raw}")
    print(f"Kept after blur mask: {total_kept}")


if __name__ == "__main__":  # pragma: no cover
    main()
