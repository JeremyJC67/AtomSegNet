#!/usr/bin/env python3
"""Batch CLI for AtomSegNet inference using Gen1 models and morphology post-processing."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
try:
    from scipy import ndimage as ndi
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for AtomSegNet CLI. Please install scipy>=1.0.") from exc

try:
    from scipy.io import savemat  # optional
except ImportError:  # pragma: no cover
    savemat = None
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import disk, erosion, opening
from skimage.segmentation import watershed

from utils.utils import GetIndexRangeOfBlk, load_model, map01

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
RESIZE_OPTIONS = {
    "do_nothing": None,
    "down2": ("down", 2),
    "up2": ("up", 2),
    "down3": ("down", 3),
    "up3": ("up", 3),
    "down4": ("down", 4),
    "up4": ("up", 4),
}


@dataclass
class PipelineConfig:
    model: str = "Gen1-gaussianMask"
    resize: Optional[str] = None
    split: bool = True
    iteration: int = 1
    disconnect_method: str = "opening"
    disconnect_level: int = 0
    threshold_percent: Optional[float] = None
    cuda: bool = False
    overwrite: bool = False
    save_all: bool = True


@dataclass
class PipelineOutputs:
    ori_image: Image.Image
    model_output: Image.Image
    denoised_image: Image.Image
    ori_markers: Image.Image
    out_markers: Image.Image
    result_array: np.ndarray
    imarray_original: np.ndarray
    props: list


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def apply_resize(image: Image.Image, resize_key: Optional[str]) -> Image.Image:
    if not resize_key or resize_key not in RESIZE_OPTIONS:
        return image
    direction, factor = RESIZE_OPTIONS[resize_key]
    width, height = image.size
    if direction == "down":
        return image.resize((width // factor, height // factor), Image.BILINEAR)
    else:
        return image.resize((width * factor, height * factor), Image.BICUBIC)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    min_v = float(array.min())
    max_v = float(array.max())
    if math.isclose(max_v, min_v):
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - min_v) / (max_v - min_v)
    return (norm * 255).astype(np.uint8)


def run_network(ori_image: Image.Image, cfg: PipelineConfig) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    ori_content = apply_resize(ori_image, cfg.resize)
    width, height = ori_content.size

    blk_col = blk_row = 1
    if cfg.split:
        if height > 1024:
            blk_row = 4
        elif height > 512:
            blk_row = 2
        if width > 1024:
            blk_col = 4
        elif width > 512:
            blk_col = 2

    result = np.full((height, width), -100.0, dtype=np.float32)
    model_path = Path("model_weights") / f"{cfg.model}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weight not found: {model_path}")

    for r in range(blk_row):
        for c in range(blk_col):
            _, outer_blk = GetIndexRangeOfBlk(height, width, blk_row, blk_col, r, c, over_lap=int(width * 0.01))
            temp_image = ori_content.crop((outer_blk[0], outer_blk[1], outer_blk[2], outer_blk[3]))
            temp_result = load_model(str(model_path), temp_image, cfg.cuda, cfg.iteration)
            temp_slice = result[outer_blk[1]:outer_blk[3], outer_blk[0]:outer_blk[2]]
            result[outer_blk[1]:outer_blk[3], outer_blk[0]:outer_blk[2]] = np.maximum(temp_result, temp_slice)

    result[result < 0] = 0
    model_output_uint8 = normalize_to_uint8(result)
    model_output_image = Image.fromarray(model_output_uint8, mode="L")
    return ori_content, result, model_output_image


def apply_disconnect(image: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    radius = max(cfg.disconnect_level, 0)
    if radius == 0:
        return image.copy()
    kernel = disk(radius)
    if kernel.size == 0:
        return image.copy()
    if cfg.disconnect_method.lower() == "opening":
        return opening(image, kernel)
    return erosion(image, kernel)


def detect_atoms(denoised: np.ndarray, ori_content: Image.Image, result_array: np.ndarray,
                 threshold_percent: Optional[float]):
    elevation_map = sobel(denoised)
    denoised = denoised.astype(np.uint8)
    markers = np.zeros_like(denoised, dtype=np.uint8)

    if threshold_percent is not None and threshold_percent > 0:
        max_thre = threshold_percent * 2.55
    else:
        max_thre = 100

    min_thre = 30
    markers[denoised < min_thre] = 1
    markers[denoised > max_thre] = 2

    seg_1 = watershed(elevation_map, markers)
    filled_regions = ndi.binary_fill_holes(seg_1 - 1)
    label_objects, _ = ndi.label(filled_regions)
    props = regionprops(label_objects)

    denoised_rgb = np.dstack([denoised] * 3)
    out_markers = Image.fromarray(denoised_rgb.astype(np.uint8), mode="RGB")

    ori_array = np.array(ori_content)
    if ori_array.ndim == 2:
        ori_rgb = np.dstack([ori_array] * 3)
    else:
        ori_rgb = ori_array
    ori_markers = Image.fromarray(ori_rgb.astype(np.uint8), mode="RGB")

    draw_out = ImageDraw.Draw(out_markers)
    draw_ori = ImageDraw.Draw(ori_markers)

    height, width = denoised.shape
    for p in props:
        c_y, c_x = p.centroid
        x0 = int(min(max(c_x - 2, 0), width - 1))
        y0 = int(min(max(c_y - 2, 0), height - 1))
        x1 = int(min(max(c_x + 2, 0), width - 1))
        y1 = int(min(max(c_y + 2, 0), height - 1))
        bbox = [x0, y0, x1, y1]
        draw_out.ellipse(bbox, fill="red", outline="red")
        draw_ori.ellipse(bbox, fill="red", outline="red")

    return props, ori_markers, out_markers


def save_results(image_path: Path, output_root: Path, cfg: PipelineConfig, outputs: PipelineOutputs) -> None:
    base_name = image_path.stem
    if cfg.resize:
        base_name = f"{base_name}_{cfg.resize}"
    out_dir = output_root / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_all:
        outputs.ori_image.save(out_dir / f"{base_name}.png")
        outputs.model_output.save(out_dir / f"{base_name}_output_{cfg.model}.png")
        outputs.ori_markers.save(out_dir / f"{base_name}_origin_{cfg.model}.png")
        outputs.out_markers.save(out_dir / f"{base_name}_denoised_{cfg.model}.png")

    pos_file = out_dir / f"{base_name}_pos_{cfg.model}.txt"
    with open(pos_file, "w") as f:
        for p in outputs.props:
            c_y, c_x = p.centroid
            min_row, min_col, max_row, max_col = p.bbox
            c_y_int = int(min(max(round(c_y), 0), outputs.result_array.shape[0] - 1))
            c_x_int = int(min(max(round(c_x), 0), outputs.result_array.shape[1] - 1))
            score = outputs.result_array[c_y_int, c_x_int]
            f.write(",".join(map(str, [c_y, c_x, min_row, min_col, max_row, max_col, score])))
            f.write("\n")

    if cfg.save_all and savemat is not None:
        savemat(out_dir / f"{base_name}_output_{cfg.model}.mat", {"result": outputs.result_array})
        savemat(out_dir / f"{base_name}_ori_{cfg.model}.mat", {"origin": outputs.imarray_original})


def process_image(image_path: Path, output_root: Path, cfg: PipelineConfig) -> None:
    base_name = image_path.stem
    target_dir = output_root / (f"{base_name}_{cfg.resize}" if cfg.resize else base_name)
    if target_dir.exists() and not cfg.overwrite:
        print(f"→ Skip existing results for {image_path}")
        return

    print(f"Processing {image_path}")
    image = Image.open(image_path).convert("L")
    imarray_original = np.array(image)

    ori_content, result_array, model_output_image = run_network(image, cfg)
    denoised_uint8 = apply_disconnect(np.array(model_output_image), cfg)
    denoised_image = Image.fromarray(denoised_uint8, mode="L")
    props, ori_markers, out_markers = detect_atoms(denoised_uint8, ori_content, result_array,
                                                   cfg.threshold_percent)

    outputs = PipelineOutputs(
        ori_image=ori_content,
        model_output=model_output_image,
        denoised_image=denoised_image,
        ori_markers=ori_markers,
        out_markers=out_markers,
        result_array=result_array,
        imarray_original=imarray_original,
        props=props,
    )
    save_results(image_path, output_root, cfg, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run AtomSegNet inference with post-processing")
    parser.add_argument("input_dir", type=Path, help="Folder containing denoised images")
    parser.add_argument("output_dir", type=Path, help="Destination for results")
    parser.add_argument("--model", default="Gen1-gaussianMask", help="Model name (default: Gen1-gaussianMask)")
    parser.add_argument("--resize", choices=sorted(RESIZE_OPTIONS.keys()), default="do_nothing",
                        help="Resize option (default: do_nothing)")
    parser.add_argument("--no-split", action="store_true", help="Disable automatic tiling")
    parser.add_argument("--iteration", type=int, default=1, help="Inference iterations (default: 1)")
    parser.add_argument("--disconnect-method", choices=["opening", "erosion"], default="opening",
                        help="Morphological operation for disconnecting atoms")
    parser.add_argument("--disconnect-level", type=int, default=0,
                        help="Radius for morphological kernel (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Optional threshold percentage (0 disables custom threshold)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for model inference")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--no-save-all", action="store_true", help="Skip saving images/mats (coords only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        model=args.model,
        resize=None if args.resize == "do_nothing" else args.resize,
        split=not args.no_split,
        iteration=args.iteration,
        disconnect_method=args.disconnect_method,
        disconnect_level=args.disconnect_level,
        threshold_percent=None if args.threshold <= 0 else args.threshold,
        cuda=args.cuda,
        overwrite=args.overwrite,
        save_all=not args.no_save_all,
    )

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(input_dir):
        process_image(image_path, output_dir, cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
