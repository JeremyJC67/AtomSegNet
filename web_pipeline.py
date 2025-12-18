"""Web-friendly pipeline wrapper for AtomSegNet.

This module exposes single-image helpers for:
- UDVD denoise (placeholder fallback to AtomSegNet denoise if UDVD is unavailable)
- AtomSegNet built-in denoise
- AtomSegNet inference (Gen1 models)
- Combined pipelines A/B/C as defined for the web MVP

Existing CLI scripts remain untouched; this file simply reuses their helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from batch_run_atomsegnet import (
    PipelineConfig,
    apply_disconnect,
    detect_atoms,
    run_network,
)
from utils.utils import load_model, map01

logger = logging.getLogger(__name__)

# Default model names used across the project
AS_DENOISE_MODEL = "denoise"
AS_SEG_MODEL = "Gen1-gaussianMask"
BASE_DIR = Path(__file__).resolve().parent


def _to_pil_gray(image: np.ndarray) -> Image.Image:
    """Ensure an input numpy array is converted to 8-bit grayscale PIL Image."""
    if image.ndim == 3:
        # If RGB, convert to luminance
        image = image[..., 0]
    array = np.asarray(image, dtype=np.float32)
    # Normalize to 0-255 range if outside
    min_v = float(array.min())
    max_v = float(array.max())
    if max_v > min_v:
        array = (array - min_v) / (max_v - min_v)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def run_udvd_denoise(image: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
    """Run UDVD denoise stage.

    NOTE: If a real UDVD model is not available locally, this falls back to the
    AtomSegNet denoise model to keep the pipeline functional for the web MVP.
    """
    params = params or {}
    # Placeholder: reuse AtomSegNet denoise until a UDVD checkpoint is wired in.
    logger.info("Running UDVD denoise (fallback to AtomSegNet denoise).")
    return run_as_denoise(image, params)


def run_as_denoise(image: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
    """Run AtomSegNet's built-in denoise model on a single image."""
    params = params or {}
    model_name = params.get("model", AS_DENOISE_MODEL)
    use_cuda = bool(params.get("cuda", False))
    iterations = int(params.get("iterations", 1))

    pil_img = _to_pil_gray(image)
    model_path = BASE_DIR / "model_weights" / f"{model_name}.pth"
    result = load_model(str(model_path), pil_img, use_cuda, iterations)
    denoised_uint8 = (map01(result) * 255).astype(np.uint8)
    return denoised_uint8


def run_atomsegnet(image: np.ndarray, params: Optional[Dict] = None) -> Dict:
    """Run AtomSegNet segmentation (Gen1) on a single denoised image."""
    params = params or {}
    # Ensure model weights resolve relative to repo root
    import os

    prev_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    cfg = PipelineConfig(
        model=params.get("model", AS_SEG_MODEL),
        resize=params.get("resize", None),
        split=bool(params.get("split", True)),
        iteration=int(params.get("iteration", 1)),
        disconnect_method=str(params.get("disconnect_method", "opening")),
        disconnect_level=int(params.get("disconnect_level", 0)),
        threshold_percent=params.get("threshold_percent", None),
        cuda=bool(params.get("cuda", False)),
        overwrite=True,
        save_all=False,
    )

    try:
        pil_img = _to_pil_gray(image)
        ori_content, result_array, model_output_image = run_network(pil_img, cfg)
    finally:
        os.chdir(prev_cwd)

    # Post-process model output to get markers and atoms
    denoised_uint8 = apply_disconnect(np.array(model_output_image), cfg)
    props, ori_markers, out_markers = detect_atoms(
        denoised_uint8, ori_content, result_array, cfg.threshold_percent
    )

    atoms: List[Dict[str, float]] = []
    height, width = result_array.shape
    for p in props:
        c_y, c_x = p.centroid
        c_y_int = int(min(max(round(c_y), 0), height - 1))
        c_x_int = int(min(max(round(c_x), 0), width - 1))
        score = float(result_array[c_y_int, c_x_int])
        atoms.append({"x": float(c_x), "y": float(c_y), "score": score})

    return {
        "atoms": atoms,
        "prob_map": result_array,
        "model_output": np.array(model_output_image),
        "overlay": ori_markers,  # PIL Image with red dots over original/resize
        "denoised_model": denoised_uint8,
        "ori_image": np.array(ori_content),
    }


def run_pipeline(image: np.ndarray, pipeline_id: str, params: Optional[Dict] = None) -> Dict:
    """Run one of the three predefined pipelines on a single image."""
    params = params or {}
    pipeline_id = pipeline_id.lower()
    if pipeline_id not in {
        "udvd_plus_as",
        "as_denoise_plus_as",
        "udvd_plus_as_denoise_plus_as",
    }:
        raise ValueError(f"Unsupported pipeline_id: {pipeline_id}")

    raw_array = np.asarray(image)

    if pipeline_id == "udvd_plus_as":
        denoised_stage = run_udvd_denoise(raw_array, params.get("udvd"))
    elif pipeline_id == "as_denoise_plus_as":
        denoised_stage = run_as_denoise(raw_array, params.get("as_denoise"))
    else:  # udvd_plus_as_denoise_plus_as
        first = run_udvd_denoise(raw_array, params.get("udvd"))
        denoised_stage = run_as_denoise(first, params.get("as_denoise"))

    seg_result = run_atomsegnet(denoised_stage, params.get("atomsegnet"))

    return {
        "pipeline_id": pipeline_id,
        "image_raw": raw_array,
        "image_denoised": denoised_stage,
        "atoms": seg_result["atoms"],
        "prob_map": seg_result["prob_map"],
        "overlay": seg_result["overlay"],
        "model_output": seg_result["model_output"],
        "denoised_model": seg_result["denoised_model"],
    }


__all__ = [
    "run_udvd_denoise",
    "run_as_denoise",
    "run_atomsegnet",
    "run_pipeline",
]
