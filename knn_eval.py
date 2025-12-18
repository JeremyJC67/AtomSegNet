#!/usr/bin/env python3
"""kNN distance evaluation utilities for AtomSegNet point sets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import KDTree


def compute_knn_distances(coords: np.ndarray, k_list: Iterable[int] = (4, 6, 8)) -> dict:
    """Compute kNN distance stats for a set of 2D coordinates.

    Args:
        coords: Array-like of shape (N, 2), interpreted as (x, y).
        k_list: Iterable of k values to evaluate.

    Returns:
        Dict keyed by k:
        {
          k: {
            "distances": np.ndarray,  # flattened kNN distances
            "mean": float,
            "std": float,
            "k_used": int,            # min(k, N-1)
            "n_points": int
          }
        }
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be an array of shape (N, 2)")

    n_points = coords.shape[0]
    results: dict[int, dict] = {}

    if n_points < 2:
        for k in k_list:
            k = int(k)
            results[k] = {
                "distances": np.array([], dtype=float),
                "mean": float("nan"),
                "std": float("nan"),
                "k_used": 0,
                "n_points": n_points,
            }
        return results

    tree = KDTree(coords)
    for k in k_list:
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        k_used = min(k, n_points - 1)
        dists, _ = tree.query(coords, k=k_used + 1)
        knn = dists[:, 1:]  # drop self distance
        flat = knn.reshape(-1)
        results[k] = {
            "distances": flat,
            "mean": float(flat.mean()) if flat.size else float("nan"),
            "std": float(flat.std(ddof=0)) if flat.size else float("nan"),
            "k_used": k_used,
            "n_points": n_points,
        }
    return results


def save_knn_results(results: dict, out_path: str | Path) -> None:
    """Save kNN results to a .npz file for later analysis."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray | float | int] = {}
    for k, stats in results.items():
        key = int(k)
        payload[f"k{key}_dists"] = np.asarray(stats["distances"], dtype=float)
        payload[f"k{key}_mean"] = float(stats["mean"])
        payload[f"k{key}_std"] = float(stats["std"])
        payload[f"k{key}_used"] = int(stats.get("k_used", key))
        payload[f"k{key}_n_points"] = int(stats.get("n_points", 0))

    np.savez(out_path, **payload)


def plot_knn_distribution(results: dict, k: int = 6, title: str = "", save_path: str | Path | None = None):
    """Plot a kNN distance distribution histogram (probability density)."""
    import matplotlib.pyplot as plt

    if k not in results:
        raise KeyError(f"k={k} not found in results")

    distances = np.asarray(results[k]["distances"], dtype=float)
    if distances.size == 0:
        raise ValueError("No distances to plot (empty input)")

    fig, ax = plt.subplots()
    ax.hist(distances, bins=50, density=True)
    ax.set_xlabel("distance")
    ax.set_ylabel("probability density")
    if title:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def load_atomsegnet_coords(path: str | Path) -> np.ndarray:
    """Load AtomSegNet TXT coords, returning array of (x, y)."""
    coords = []
    path = Path(path)
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            cy = float(parts[0])
            cx = float(parts[1])
            coords.append((cx, cy))
    return np.asarray(coords, dtype=float)
