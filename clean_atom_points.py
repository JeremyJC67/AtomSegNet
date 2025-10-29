#!/usr/bin/env python3
"""Clean AtomSegNet detection files with score thresholding and distance deduplication."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from skimage.filters import threshold_otsu  # type: ignore
except ImportError:  # pragma: no cover
    threshold_otsu = None  # type: ignore


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


def iter_pos_files(root: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            yield path


def load_positions(path: Path) -> np.ndarray:
    records = []
    with path.open("r") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                raise ValueError(f"Unexpected column count in {path} line {line_no}")
            cy = float(parts[0])
            cx = float(parts[1])
            min_row = int(round(float(parts[2])))
            min_col = int(round(float(parts[3])))
            max_row = int(round(float(parts[4])))
            max_col = int(round(float(parts[5])))
            score = float(parts[6])
            records.append((cy, cx, min_row, min_col, max_row, max_col, score))
    if not records:
        return np.zeros(0, dtype=DTYPE)
    return np.array(records, dtype=DTYPE)


def save_positions(path: Path, data: np.ndarray) -> None:
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


def choose_threshold(scores: np.ndarray, args: argparse.Namespace) -> float:
    if args.score_threshold is not None:
        return float(args.score_threshold)
    if scores.size == 0:
        return float(args.min_score)

    threshold = None
    unique_scores = np.unique(scores)

    if args.adaptive_method == "otsu" and threshold_otsu is not None and unique_scores.size > 1:
        try:
            threshold = float(threshold_otsu(scores))
        except ValueError:
            threshold = None

    if threshold is None:
        quantile = min(max(args.quantile, 0.0), 0.99)
        threshold = float(np.quantile(scores, quantile))

    return max(float(args.min_score), threshold)


def estimate_nearest_neighbor(coords: np.ndarray) -> float | None:
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


def deduplicate(data: np.ndarray, min_dist: float) -> np.ndarray:
    if data.size <= 1 or min_dist <= 0:
        return data.copy()
    order = np.argsort(data["score"])[::-1]
    keep_flags = np.zeros(data.size, dtype=bool)
    accepted = []
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


def process_file(path: Path, args: argparse.Namespace) -> tuple[int, int, int, float, float]:
    data = load_positions(path)
    scores = data["score"]
    threshold = choose_threshold(scores, args)
    keep_mask = scores >= threshold
    filtered = data[keep_mask]

    nn_estimate = None
    min_dist = float(args.min_dist)
    if filtered.size >= 2:
        coords = np.column_stack([filtered["cy"], filtered["cx"]])
        nn_estimate = estimate_nearest_neighbor(coords)
        if nn_estimate is not None and nn_estimate > 0:
            min_dist = max(min_dist, float(args.nn_factor) * nn_estimate)

    deduped = deduplicate(filtered, min_dist)
    if args.output_suffix:
        out_path = path.with_name(f"{path.stem}_{args.output_suffix}{path.suffix}")
    else:
        out_path = path
    if not args.dry_run:
        save_positions(out_path, deduped)

    removed_score = data.size - filtered.size
    removed_dist = filtered.size - deduped.size
    return data.size, removed_score, removed_dist, threshold, min_dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean AtomSegNet detection txt files by confidence thresholding and distance deduplication."
    )
    parser.add_argument("input_root", type=Path, help="Root directory containing *_pos_*.txt files.")
    parser.add_argument("--pattern", default="*_pos_*.txt", help="Glob pattern to find detection files.")
    parser.add_argument("--output-suffix", default="clean", help="Suffix for cleaned files (empty for in-place).")
    parser.add_argument("--score-threshold", type=float, default=None, help="Global score threshold.")
    parser.add_argument("--adaptive-method", choices=["quantile", "otsu"], default="quantile",
                        help="Adaptive thresholding method when global threshold is not set.")
    parser.add_argument("--quantile", type=float, default=0.65,
                        help="Quantile used for score thresholding (quantile method).")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum allowed score threshold.")
    parser.add_argument("--nn-factor", type=float, default=0.5,
                        help="Factor applied to the median nearest-neighbor distance.")
    parser.add_argument("--min-dist", type=float, default=0.0,
                        help="Absolute minimum distance used for deduplication (pixels).")
    parser.add_argument("--dry-run", action="store_true", help="Compute statistics without writing output files.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file details.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    files = list(iter_pos_files(args.input_root, args.pattern))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print("No detection files found.")
        return

    total_raw = total_removed_score = total_removed_dist = 0
    thresholds = []
    min_dists = []

    for path in files:
        raw, removed_score, removed_dist, threshold, min_dist = process_file(path, args)
        total_raw += raw
        total_removed_score += removed_score
        total_removed_dist += removed_dist
        thresholds.append(threshold)
        min_dists.append(min_dist)
        if args.verbose:
            print(
                f"{path}: raw={raw} removed_score={removed_score} removed_dist={removed_dist} "
                f"threshold={threshold:.3f} min_dist={min_dist:.2f}"
            )

    cleaned = total_raw - total_removed_score - total_removed_dist
    print(f"Processed {len(files)} files.")
    print(f"Total detections: {total_raw}")
    print(f"Removed by score: {total_removed_score}")
    print(f"Removed by distance: {total_removed_dist}")
    print(f"Remaining detections: {cleaned}")

    thresholds_arr = np.array(thresholds)
    if thresholds_arr.size:
        print(
            "Score thresholds -> "
            f"min {thresholds_arr.min():.3f}, median {np.median(thresholds_arr):.3f}, max {thresholds_arr.max():.3f}"
        )
    min_dists_arr = np.array(min_dists)
    if min_dists_arr.size and np.any(min_dists_arr > 0):
        valid = min_dists_arr[min_dists_arr > 0]
        print(
            "Dedup radii -> "
            f"min {valid.min():.2f}px, median {np.median(valid):.2f}px, max {valid.max():.2f}px"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
