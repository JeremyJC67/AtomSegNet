"""Reusable helpers for scoring thresholds and nearest-neighbour deduplication."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from skimage.filters import threshold_otsu  # type: ignore
except ImportError:  # pragma: no cover
    threshold_otsu = None  # type: ignore


def choose_threshold(
    scores: np.ndarray,
    *,
    score_threshold: float | None = None,
    adaptive_method: str = "quantile",
    quantile: float = 0.65,
    min_score: float = 0.4,
) -> float:
    """Return score cutoff value using explicit / adaptive heuristics."""
    if score_threshold is not None:
        return float(score_threshold)
    if scores.size == 0:
        return float(min_score)

    threshold: float | None = None
    unique = np.unique(scores)

    if (
        adaptive_method == "otsu"
        and threshold_otsu is not None
        and unique.size > 1
    ):
        try:
            threshold = float(threshold_otsu(scores))
        except ValueError:
            threshold = None

    if threshold is None:
        q = min(max(quantile, 0.0), 0.99)
        threshold = float(np.quantile(scores, q))

    return max(float(min_score), threshold)


def estimate_nearest_neighbor(coords: np.ndarray) -> float | None:
    """Median nearest-neighbour distance among (N,2) coordinates."""
    n = coords.shape[0]
    if n < 2:
        return None
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def deduplicate_by_pixel(data: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """Collapse points that fall in the same grid cell; keep highest score."""
    if data.size <= 1:
        return data.copy()
    if cell <= 0:
        return data.copy()
    cell = float(cell)
    cy = data["cy"] / cell
    cx = data["cx"] / cell
    keys = np.floor(cy).astype(np.int64) << 32 | np.floor(cx).astype(np.int64)
    order = np.argsort(data["score"])[::-1]
    seen = set()
    keep_flags = np.zeros(data.size, dtype=bool)
    for idx in order:
        key = keys[idx]
        if key in seen:
            continue
        seen.add(key)
        keep_flags[idx] = True
    return data[keep_flags]


def deduplicate(data: np.ndarray, min_dist: float) -> np.ndarray:
    """Greedy keep-highest-score while enforcing `min_dist` separation."""
    if data.size <= 1 or min_dist <= 0:
        return data.copy()
    order = np.argsort(data["score"])[::-1]
    keep_flags = np.zeros(data.size, dtype=bool)
    accepted: list[tuple[float, float]] = []
    for idx in order:
        cy = data["cy"][idx]
        cx = data["cx"][idx]
        if accepted:
            accepted_arr = np.asarray(accepted)
            dist = np.hypot(accepted_arr[:, 0] - cy, accepted_arr[:, 1] - cx)
            if np.any(dist < min_dist):
                continue
        keep_flags[idx] = True
        accepted.append((cy, cx))
    return data[keep_flags]


def clean_positions(
    data: np.ndarray,
    *,
    score_threshold: float | None = None,
    adaptive_method: str = "quantile",
    quantile: float = 0.65,
    min_score: float = 0.4,
    nn_factor: float = 0.5,
    min_dist: float = 0.0,
    cell: float = 0.0,
) -> Tuple[np.ndarray, float, float, int, int]:
    """Apply score filtering then NN dedup; return cleaned data & diagnostics.

    Returns:
        deduped array,
        score threshold used,
        effective min_dist (after NN estimation),
        number removed by score,
        number removed by distance.
    """
    points = data
    if cell and cell > 0:
        points = deduplicate_by_pixel(points, cell=cell)

    scores = points["score"]
    threshold = choose_threshold(
        scores,
        score_threshold=score_threshold,
        adaptive_method=adaptive_method,
        quantile=quantile,
        min_score=min_score,
    )
    keep_mask = scores >= threshold
    filtered = points[keep_mask]
    removed_score = points.size - filtered.size

    effective_dist = float(min_dist)
    if filtered.size >= 2:
        coords = np.column_stack([filtered["cy"], filtered["cx"]])
        nn_est = estimate_nearest_neighbor(coords)
        if nn_est is not None and nn_est > 0:
            effective_dist = max(effective_dist, float(nn_factor) * nn_est)

    deduped = deduplicate(filtered, effective_dist)
    removed_dist = filtered.size - deduped.size
    return deduped, threshold, effective_dist, removed_score, removed_dist
