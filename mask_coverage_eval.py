#!/usr/bin/env python3
"""Mask coverage evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def compute_mask_coverage(coords: np.ndarray, mask_path: str | Path) -> Dict[str, object]:
    """Compute fraction of points inside a binary mask.

    Args:
        coords: Array-like of shape (N, 2), interpreted as (x, y).
        mask_path: Path to mask image (nonzero = inside).

    Returns:
        {
          "inside_frac": float,
          "inside_count": int,
          "n_points": int
        }
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be an array of shape (N, 2)")
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = np.array(Image.open(mask_path).convert("L")) > 0
    h, w = mask.shape
    n_points = coords.shape[0]
    if n_points == 0:
        return {"inside_frac": 0.0, "inside_count": 0, "n_points": 0}

    xs = np.clip(np.rint(coords[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(coords[:, 1]).astype(int), 0, h - 1)
    inside = mask[ys, xs]
    inside_count = int(inside.sum())
    return {
        "inside_frac": float(inside.mean()),
        "inside_count": inside_count,
        "n_points": n_points,
    }
