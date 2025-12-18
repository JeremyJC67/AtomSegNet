#!/usr/bin/env python3
"""Nearest-neighbor (NN) distance evaluation utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import KDTree


def compute_nn_metrics(coords: np.ndarray) -> Dict[str, object]:
    """Compute NN distance stats for a set of 2D coordinates.

    Args:
        coords: Array-like of shape (N, 2), interpreted as (x, y).

    Returns:
        {
          "distances": np.ndarray,  # NN distances for each point
          "mean": float,
          "median": float,
          "std": float,
          "n_points": int
        }
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be an array of shape (N, 2)")

    n_points = coords.shape[0]
    if n_points < 2:
        return {
            "distances": np.array([], dtype=float),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "n_points": n_points,
        }

    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn = dists[:, 1]
    return {
        "distances": nn,
        "mean": float(nn.mean()),
        "median": float(np.median(nn)),
        "std": float(nn.std(ddof=0)),
        "n_points": n_points,
    }
