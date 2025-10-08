#!/usr/bin/env python3
"""Manual/automatic polygon-based post-processing for AtomSegNet results."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOUNDARY_REF_IMAGE = Path(
    "atomsegnet_results/0000_denoise/0000_denoise_origin_Gen1-gaussianMask.png"
)
BOUNDARY_PATH = Path("boundary_0000.txt")
POLYGON_PATH = Path("boundary_0000_polygon.txt")
BOUNDARY_PREVIEW = Path("boundary_preview.png")
MANUAL_MASK_IMAGE = Path("manual_mask.png")  # optional user-defined mask

RESULT_ROOT = Path("atomsegnet_results")
MODEL_SUFFIX = "Gen1-gaussianMask"
DATASET_ROOT = Path("/Users/jichengwang/Documents/Stanford_Intern/dataset/39frames")
SOLID_MASK_PATH = Path("mask/solid_polygon_from_red.npy")

LINE_THRESHOLD = 80  # darker-than-threshold pixels treated as hand-drawn line
DILATION_STEPS = 3   # thicken the line to ensure there are no gaps
MIN_POLYGON_AREA_RATIO = 0.002  # discard tiny regions (relative to image area)
KEEP_IF_Y_LESS_THAN = 540  # keep any detection with y < threshold (None to disable)
REMOVE_REGION_MIN_X = 100.0  # lower bound of cleanup range
REMOVE_REGION_SPLIT_X = 410.0  # split between low/high y thresholds
REMOVE_REGION_Y_LOW = 590.0  # y threshold for 100 < x < 410
REMOVE_REGION_Y_HIGH = 630.0  # y threshold for x >= 410

_MASK_CACHE: Dict[Path, np.ndarray] = {}
_SOLID_MASK: Optional[np.ndarray] = None

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Simple 3x3 binary dilation using only numpy operations."""

    result = mask.copy()
    height, width = mask.shape

    for _ in range(iterations):
        expanded = result.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.zeros_like(result, dtype=bool)

                src_y = slice(max(0, -dy), height - max(0, dy))
                src_x = slice(max(0, -dx), width - max(0, dx))
                dst_y = slice(max(0, dy), height - max(0, -dy))
                dst_x = slice(max(0, dx), width - max(0, -dx))

                shifted[dst_y, dst_x] = result[src_y, src_x]
                expanded |= shifted
        result = expanded
    return result


def _flood_fill_from_border(barrier: np.ndarray) -> np.ndarray:
    """Flood fill from all image borders while treating `barrier` pixels as walls."""

    height, width = barrier.shape
    filled = np.zeros_like(barrier, dtype=bool)
    queue: deque[Tuple[int, int]] = deque()

    def enqueue(y: int, x: int) -> None:
        if barrier[y, x] or filled[y, x]:
            return
        filled[y, x] = True
        queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(height):
        enqueue(y, 0)
        enqueue(y, width - 1)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if barrier[ny, nx] or filled[ny, nx]:
                continue
            filled[ny, nx] = True
            queue.append((ny, nx))
    return filled


def _mask_to_polygon(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    thresholds = np.full(width, height, dtype=float)
    for x in range(width):
        column = np.flatnonzero(mask[:, x])
        if column.size:
            thresholds[x] = float(column[0])

    polygon = [(0.0, float(height - 1))]
    polygon.append((0.0, thresholds[0]))
    for x in range(1, width):
        polygon.append((float(x), thresholds[x]))
    polygon.append((float(width - 1), float(height - 1)))
    return thresholds, np.array(polygon, dtype=float)


def _mask_to_overlay(mask: np.ndarray) -> Image.Image:
    height, width = mask.shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask] = (255, 0, 0, 90)
    return Image.fromarray(overlay, mode="RGBA")


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    cleaned = np.zeros_like(mask, dtype=bool)
    queue: deque[Tuple[int, int]] = deque()

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            component_pixels = []
            visited[y, x] = True
            queue.append((y, x))

            while queue:
                cy, cx = queue.popleft()
                component_pixels.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            if len(component_pixels) >= min_pixels:
                for cy, cx in component_pixels:
                    cleaned[cy, cx] = True

    return cleaned


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[Tuple[int, int]] = deque()
    largest_component: list[Tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            component_pixels: list[Tuple[int, int]] = []
            visited[y, x] = True
            queue.append((y, x))

            while queue:
                cy, cx = queue.popleft()
                component_pixels.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            if len(component_pixels) > len(largest_component):
                largest_component = component_pixels

    result = np.zeros_like(mask, dtype=bool)
    for cy, cx in largest_component:
        result[cy, cx] = True
    return result


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def _extract_polygon_mask(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Polygon reference image missing: {image_path}")

    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)

    line_mask = np.all(arr < LINE_THRESHOLD, axis=2)
    line_mask = _binary_dilation(line_mask, DILATION_STEPS)

    accessible = _flood_fill_from_border(line_mask)
    interior_raw = (~accessible) & (~line_mask)

    if not interior_raw.any():
        raise RuntimeError(
            "Unable to derive polygon interior. Adjust thresholds or provide a "
            "manual mask."
        )

    min_pixels = max(1, int(interior_raw.size * MIN_POLYGON_AREA_RATIO))
    interior = _remove_small_components(interior_raw, min_pixels)

    if not interior.any():
        interior = _keep_largest_component(interior_raw)

    return interior


def _load_manual_mask(width: int, height: int) -> Optional[np.ndarray]:
    if not MANUAL_MASK_IMAGE.exists():
        return None
    manual = Image.open(MANUAL_MASK_IMAGE).convert("L").resize((width, height))
    arr = np.array(manual)
    return arr > 0


def _get_remove_mask(image_path: Path) -> np.ndarray:
    resolved = image_path.resolve()
    if resolved in _MASK_CACHE:
        return _MASK_CACHE[resolved]

    mask = _extract_polygon_mask(image_path)
    manual_mask = _load_manual_mask(mask.shape[1], mask.shape[0])
    if manual_mask is not None:
        mask |= manual_mask

    _MASK_CACHE[resolved] = mask
    return mask


def _load_solid_mask(expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    global _SOLID_MASK
    if not SOLID_MASK_PATH.exists():
        return None
    if _SOLID_MASK is None:
        solid = np.load(SOLID_MASK_PATH)
        if solid.ndim != 2:
            raise ValueError("Solid polygon mask must be 2D")
        solid = solid.astype(bool)
        _SOLID_MASK = solid
    if _SOLID_MASK.shape != expected_shape:
        raise ValueError(
            f"Solid mask shape {_SOLID_MASK.shape} does not match expected {expected_shape}"
        )
    return _SOLID_MASK


# ---------------------------------------------------------------------------
# Result filtering
# ---------------------------------------------------------------------------


def load_positions(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 7))
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1 and data.size:
        data = data[None, :]
    return data


def _frame_id_from_folder(folder: Path) -> Optional[str]:
    parts = folder.name.split("_")
    if not parts:
        return None
    maybe_id = parts[0]
    return maybe_id if maybe_id.isdigit() else None


def _dataset_image(frame_id: str) -> Optional[Path]:
    candidate = DATASET_ROOT / f"{frame_id}.jpg"
    return candidate if candidate.exists() else None


def filter_positions(data: np.ndarray, remove_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if data.size == 0:
        return data, np.array([], dtype=bool)

    height, width = remove_mask.shape

    keep_flags = []
    for y, x, *_ in data:
        if KEEP_IF_Y_LESS_THAN is not None and y < KEEP_IF_Y_LESS_THAN:
            keep_flags.append(True)
            continue
        if REMOVE_REGION_MIN_X is not None and x > REMOVE_REGION_MIN_X:
            within_lower = (
                REMOVE_REGION_SPLIT_X is not None
                and x < REMOVE_REGION_SPLIT_X
                and REMOVE_REGION_Y_LOW is not None
                and y >= REMOVE_REGION_Y_LOW
            )
            within_upper = (
                (REMOVE_REGION_SPLIT_X is None or x >= REMOVE_REGION_SPLIT_X)
                and REMOVE_REGION_Y_HIGH is not None
                and y >= REMOVE_REGION_Y_HIGH
            )
            if within_lower or within_upper:
                keep_flags.append(False)
                continue
        xi = int(np.clip(np.floor(x), 0, width - 1))
        yi = int(np.clip(np.floor(y), 0, height - 1))
        keep_flags.append(not remove_mask[yi, xi])

    keep_mask = np.array(keep_flags, dtype=bool)
    return data[keep_mask], keep_mask


def draw_overlays(base_path: Path, all_points: np.ndarray, keep_mask: np.ndarray, suffix: str, output_path: Optional[Path] = None) -> None:
    if not base_path.exists():
        return
    base = Image.open(base_path).convert("RGB")
    draw = ImageDraw.Draw(base)

    # Keep only points that survive filtering
    kept = all_points[keep_mask]

    for y, x, *_ in kept:
        bbox = [x - 2, y - 2, x + 2, y + 2]
        draw.ellipse(bbox, outline="red", fill="red", width=1)

    if output_path is None:
        output_path = base_path.with_name(base_path.stem + suffix + base_path.suffix)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_path)


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------


def _find_reference_image(folder: Path) -> Optional[Path]:
    candidates = (
        folder / f"{folder.name}_origin_{MODEL_SUFFIX}.png",
        folder / f"{folder.name}_denoised_{MODEL_SUFFIX}.png",
        folder / f"{folder.name}.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_result_folder(folder: Path) -> None:
    pos_path = folder / f"{folder.name}_pos_{MODEL_SUFFIX}.txt"
    data = load_positions(pos_path)
    if data.size == 0:
        return

    ref_image = _find_reference_image(folder)
    if ref_image is None:
        print(f"No image found to derive polygon for {folder}")
        return

    remove_mask = _get_remove_mask(ref_image)
    filtered, keep_mask = filter_positions(data, remove_mask)

    solid_mask = _load_solid_mask(remove_mask.shape)
    if solid_mask is not None and filtered.size:
        y = filtered[:, 0]
        x = filtered[:, 1]
        yi = np.clip(np.floor(y).astype(int), 0, solid_mask.shape[0] - 1)
        xi = np.clip(np.floor(x).astype(int), 0, solid_mask.shape[1] - 1)
        keep_after_solid = ~solid_mask[yi, xi]
        if not np.all(keep_after_solid):
            filtered = filtered[keep_after_solid]
            keep_indices = np.where(keep_mask)[0]
            keep_mask[keep_indices[~keep_after_solid]] = False

    out_txt = pos_path.with_name(pos_path.stem + "_filtered.txt")
    np.savetxt(out_txt, filtered, fmt="%.6f", delimiter=",")

    inside_txt = pos_path.with_name(pos_path.stem + "_inside_polygon.txt")
    outside_txt = pos_path.with_name(pos_path.stem + "_outside_polygon.txt")
    np.savetxt(inside_txt, data[~keep_mask], fmt="%.6f", delimiter=",")
    np.savetxt(outside_txt, filtered, fmt="%.6f", delimiter=",")

    raw_png = folder / f"{folder.name}.png"
    origin_png = folder / f"{folder.name}_origin_{MODEL_SUFFIX}.png"
    denoised_png = folder / f"{folder.name}_denoised_{MODEL_SUFFIX}.png"

    base_for_overlay = raw_png if raw_png.exists() else origin_png
    if base_for_overlay.exists():
        draw_overlays(base_for_overlay, data, keep_mask, "_filtered")
    if denoised_png.exists():
        draw_overlays(denoised_png, data, keep_mask, "_filtered")

    frame_id = _frame_id_from_folder(folder)
    if frame_id is not None:
        dataset_img = _dataset_image(frame_id)
        if dataset_img is not None:
            output_path = folder / f"{folder.name}_mask_filtered.png"
            draw_overlays(dataset_img, data, keep_mask, "_mask_filtered", output_path=output_path)

    print(f"Filtered {pos_path} → {out_txt} ({len(data)} → {len(filtered)})")


def main() -> None:
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(f"Result directory not found: {RESULT_ROOT}")

    for folder in sorted(RESULT_ROOT.iterdir()):
        if folder.is_dir():
            process_result_folder(folder)


if __name__ == "__main__":
    main()
