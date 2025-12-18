#!/usr/bin/env python3
"""Point-set overlap evaluation utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import KDTree


def compute_overlap(a: np.ndarray, b: np.ndarray, radius: float = 3.0) -> Dict[str, float]:
    """Compute mutual coverage within a radius between two point sets.

    Args:
        a: Array-like of shape (N, 2) as (x, y).
        b: Array-like of shape (M, 2) as (x, y).
        radius: Matching radius in pixels.

    Returns:
        {
          "a_to_b": float,  # fraction of A points matched by B
          "b_to_a": float   # fraction of B points matched by A
        }
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("a must be an array of shape (N, 2)")
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("b must be an array of shape (M, 2)")

    if a.size == 0 or b.size == 0:
        return {"a_to_b": 0.0, "b_to_a": 0.0}

    ta = KDTree(a)
    tb = KDTree(b)
    d_ab, _ = tb.query(a, k=1, distance_upper_bound=radius)
    d_ba, _ = ta.query(b, k=1, distance_upper_bound=radius)

    a_to_b = float(np.mean(np.isfinite(d_ab)))
    b_to_a = float(np.mean(np.isfinite(d_ba)))
    return {"a_to_b": a_to_b, "b_to_a": b_to_a}
