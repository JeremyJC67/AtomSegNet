#!/usr/bin/env python3
"""Cluster AtomSegNet detections using DBSCAN and convex-hull filtering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

try:  # pragma: no cover
    import cupy as cp  # type: ignore
    from cuml.cluster import DBSCAN as cuDBSCAN  # type: ignore

    HAS_CUML = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    cuDBSCAN = None  # type: ignore
    HAS_CUML = False

try:  # pragma: no cover
    import alphashape  # type: ignore
    HAS_ALPHA_SHAPE = True
except Exception:  # pragma: no cover
    alphashape = None  # type: ignore
    HAS_ALPHA_SHAPE = False

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


def trimmed_median_nn(coords: np.ndarray, trim: float = 0.10, max_samples: Optional[int] = None) -> float:
    if coords.shape[0] < 2:
        return 0.0
    if max_samples is not None and max_samples > 0 and coords.shape[0] > max_samples:
        idx = np.random.choice(coords.shape[0], max_samples, replace=False)
        coords = coords[idx]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    nn = np.sort(distances[:, 1])
    if nn.size == 0:
        return 0.0
    trim = max(0.0, min(trim, 0.45))
    lo = int(trim * nn.size)
    hi = int((1.0 - trim) * nn.size)
    trimmed = nn[lo:hi] if hi > lo else nn
    return float(np.median(trimmed))


def convex_hull_metrics(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 3:
        return 0.0, 0.0
    hull = ConvexHull(points)
    area = float(hull.volume)
    perimeter = float(hull.area)
    return area, perimeter


def _disk_structure(radius: int) -> np.ndarray:
    r = max(1, radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def polygon_to_mask(poly_pts_xy: np.ndarray, height: int, width: int, close_radius_px: int) -> np.ndarray:
    if poly_pts_xy.shape[0] < 3 or width <= 0 or height <= 0:
        return np.zeros((height, width), dtype=bool)

    img = Image.new("1", (width, height), 0)
    pts = np.clip(poly_pts_xy.round().astype(int), 0, None)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    ImageDraw.Draw(img).polygon([tuple(p) for p in pts], outline=1, fill=1)
    mask = np.array(img, dtype=bool)

    if close_radius_px > 0 and mask.any():
        structure = _disk_structure(close_radius_px)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=1)
        mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        mask = ndimage.binary_fill_holes(mask)

    return mask


def points_in_mask(points_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(points_xy.shape[0], dtype=bool)
    xs = np.clip(points_xy[:, 0].round().astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(points_xy[:, 1].round().astype(int), 0, mask.shape[0] - 1)
    return mask[ys, xs]


def alpha_polygon(points_xy: np.ndarray, d_nn: float, tight: float, loose: float):
    if not HAS_ALPHA_SHAPE or points_xy.shape[0] < 4:
        return None, None
    geoms = []
    for scale in (tight, loose):
        alpha = 1.0 / max(d_nn * max(scale, 1e-6), 1e-6)
        try:
            geom = alphashape.alphashape(points_xy, alpha)
        except Exception:
            geom = None
        if geom is None or geom.is_empty:
            continue
        if hasattr(geom, "geoms"):
            geom = max(geom.geoms, key=lambda g: g.area)
        geoms.append(geom)
    if not geoms:
        return None, None
    geom = geoms[0]
    for g in geoms[1:]:
        try:
            geom = geom.union(g)
        except Exception:
            pass
    if geom is None or geom.is_empty:
        return None, None
    if hasattr(geom, "geoms"):
        geom = max(list(geom.geoms), key=lambda g: g.area)
    coords = np.array(geom.exterior.coords, dtype=np.float32)
    return geom, coords


def pca_ellipse(points_xy: np.ndarray, k_sigma: float):
    if points_xy.shape[0] < 3:
        return None
    mu = points_xy.mean(axis=0)
    cov = np.cov(points_xy.T)
    if not np.all(np.isfinite(cov)):
        return None
    cov = np.atleast_2d(cov)
    if cov.shape != (2, 2):
        return None
    cov += np.eye(2) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return mu, cov_inv, eigvecs, eigvals, k_sigma


def refine_clusters(data: np.ndarray, labels: np.ndarray, args: argparse.Namespace,
                    median_nn: float) -> np.ndarray:
    if data.size == 0:
        return np.zeros(data.size, dtype=bool)
    coords_yx = np.column_stack([data["cy"], data["cx"]])
    coords_xy = coords_yx[:, ::-1]

    max_y = int(np.ceil(coords_yx[:, 0].max())) + 3 if coords_yx.size else 1
    max_x = int(np.ceil(coords_yx[:, 1].max())) + 3 if coords_yx.size else 1
    height = max(max_y, 1)
    width = max(max_x, 1)

    keep_mask = np.zeros(data.size, dtype=bool)
    use_alpha = (not args.no_alpha_shape) and HAS_ALPHA_SHAPE
    close_radius_px = max(0, int(round(args.mask_close_radius * median_nn)))

    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_mask = labels == label
        core_count = int(cluster_mask.sum())
        if core_count < max(3, args.min_points // 4):
            continue

        cluster_xy = coords_xy[cluster_mask]
        mask = None
        area = perimeter = None

        area0, _ = convex_hull_metrics(cluster_xy[:, ::-1])
        if area0 > 0.0:
            density0 = core_count / max(area0, 1e-6)
            if density0 < args.min_density * 0.5:
                continue

        local_nn = trimmed_median_nn(cluster_xy, trim=args.nn_trim, max_samples=args.nn_sample)
        scale_factor = np.clip(median_nn / max(local_nn, 1e-6), 0.8, 1.5)
        tight_scale = args.alpha_scale_tight * scale_factor
        loose_scale = args.alpha_scale_loose * scale_factor

        if use_alpha:
            geom, poly_pts = alpha_polygon(cluster_xy, median_nn,
                                           tight_scale,
                                           loose_scale)
            if geom is not None and poly_pts is not None and poly_pts.shape[0] >= 3:
                mask = polygon_to_mask(poly_pts, height, width, close_radius_px)
                area = float(geom.area)
                perimeter = float(max(geom.length, 1e-6))

        if mask is None or not mask.any():
            ellipse = pca_ellipse(cluster_xy, args.pca_k_sigma)
            if ellipse is None:
                continue
            mu, cov_inv, eigvecs, eigvals, k_sigma = ellipse
            axes = k_sigma * np.sqrt(eigvals)
            theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
            circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            ellipse_pts = (circle * axes) @ eigvecs.T + mu
            mask = polygon_to_mask(ellipse_pts, height, width, close_radius_px)
            a = float(max(axes))
            b = float(min(axes))
            area = float(np.pi * a * b)
            perimeter = float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))

        if mask is None or not mask.any() or area is None or perimeter is None:
            continue

        in_mask = points_in_mask(coords_xy, mask)
        refined_mask = in_mask | cluster_mask
        refined_count = int(refined_mask.sum())
        if refined_count < args.min_points:
            continue
        if area <= 0.0 or perimeter <= 0.0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        density = refined_count / max(area * args.density_area_scale, 1e-6)
        if circularity < args.min_circularity or density < args.min_density:
            continue
        keep_mask |= refined_mask

    return keep_mask


def filter_clusters(data: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if data.size == 0:
        return data
    keep_mask = np.zeros(data.size, dtype=bool)
    coords = np.column_stack([data["cy"], data["cx"]])
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_mask = labels == label
        cluster_points = coords[cluster_mask]
        count = cluster_points.shape[0]
        if count < args.min_points:
            continue
        area, perimeter = convex_hull_metrics(cluster_points)
        if area <= 0.0 or perimeter <= 0.0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        density = count / max(area * args.density_area_scale, 1e-6)
        if circularity < args.min_circularity:
            continue
        if density < args.min_density:
            continue
        keep_mask |= cluster_mask
    return data[keep_mask]


def run_dbscan(coords: np.ndarray, args: argparse.Namespace, median_nn: float) -> tuple[np.ndarray, float]:
    if coords.shape[0] == 0:
        return np.array([], dtype=int), args.min_eps

    best_labels: Optional[np.ndarray] = None
    best_eps = args.min_eps
    best_max_cluster = -1

    factors = np.arange(args.eps_factor, args.max_eps_factor + 1e-9, args.eps_step)
    use_gpu = args.use_gpu and HAS_CUML
    coords_gpu = cp.asarray(coords.astype(np.float32)) if use_gpu else None

    for factor in factors:
        eps = max(args.min_eps, factor * median_nn)
        if use_gpu:
            clusterer = cuDBSCAN(eps=eps, min_samples=args.min_samples)
            labels = clusterer.fit_predict(coords_gpu)
            labels = cp.asnumpy(labels)
        else:
            clusterer = DBSCAN(eps=eps, min_samples=args.min_samples)
            labels = clusterer.fit_predict(coords)
        core = labels != -1
        max_cluster = 0
        if core.any():
            _, cnt = np.unique(labels[core], return_counts=True)
            max_cluster = int(cnt.max())
        if max_cluster > best_max_cluster:
            best_max_cluster = max_cluster
            best_eps = eps
            best_labels = labels
        if max_cluster > args.eps_max_cluster:
            break

    if best_labels is None:
        best_labels = np.full(coords.shape[0], -1, dtype=int)
    return best_labels, best_eps


def dedup_by_radius(data: np.ndarray, radius: float) -> np.ndarray:
    if data.size == 0 or radius <= 0:
        return data
    coords = np.column_stack([data["cy"], data["cx"]])
    nbrs = NearestNeighbors(radius=radius, algorithm="kd_tree").fit(coords)
    neighborhoods = nbrs.radius_neighbors(coords, return_distance=False)
    keep = np.ones(data.size, dtype=bool)
    for idx, neigh in enumerate(neighborhoods):
        if not keep[idx]:
            continue
        if len(neigh) <= 1:
            continue
        best = idx
        best_score = data[idx]["score"]
        for j in neigh:
            if j == idx:
                continue
            if data[j]["score"] > best_score:
                best = j
                best_score = data[j]["score"]
        for j in neigh:
            if j != best:
                keep[j] = False
    return data[keep]


def process_file(path: Path, args: argparse.Namespace) -> tuple[int, int, int, float, int]:
    data = load_positions(path)
    total = data.size
    if total == 0:
        return 0, 0, 0, 0.0, 0
    coords = np.column_stack([data["cy"], data["cx"]])
    median_nn = trimmed_median_nn(coords, trim=args.nn_trim, max_samples=args.nn_sample)
    dedup_radius = max(args.min_dedup_radius, args.dedup_factor * median_nn)
    if args.enable_dedup and dedup_radius > 0:
        data = dedup_by_radius(data, dedup_radius)
        coords = np.column_stack([data["cy"], data["cx"]])
        median_nn = trimmed_median_nn(coords, trim=args.nn_trim, max_samples=args.nn_sample)
    labels, eps = run_dbscan(coords, args, median_nn)
    clustered = (labels != -1).sum()

    refined_mask = refine_clusters(
        data,
        labels,
        args,
        median_nn,
    )

    if refined_mask.any():
        filtered = data[refined_mask]
    else:
        filtered = filter_clusters(data, labels, args)

    if args.output_suffix:
        out_path = path.with_name(path.name.replace("_pos_", f"_pos_{args.output_suffix}_"))
    else:
        out_path = path
    if not args.dry_run:
        save_positions(out_path, filtered)

    return total, clustered, filtered.size, eps, len(set(labels)) - (1 if -1 in labels else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify atom clusters using DBSCAN and convex-hull filtering."
    )
    parser.add_argument("input_root", type=Path, help="Root directory containing detection txt files.")
    parser.add_argument("--pattern", default="*_pos_*_clean.txt",
                        help="Glob pattern for input files (default: *_pos_*_clean.txt).")
    parser.add_argument("--output-suffix", default="atom_cluster",
                        help="Suffix inserted after _pos_ for output files.")
    parser.add_argument("--eps-factor", type=float, default=0.7,
                        help="Multiplier for median nearest-neighbor distance.")
    parser.add_argument("--min-eps", type=float, default=3.0,
                        help="Absolute minimum eps in pixels.")
    parser.add_argument("--max-eps-factor", type=float, default=1.4,
                        help="Maximum multiplier when searching for eps.")
    parser.add_argument("--eps-step", type=float, default=0.1,
                        help="Increment used when expanding eps search.")
    parser.add_argument("--min-samples", type=int, default=4,
                        help="DBSCAN min_samples parameter.")
    parser.add_argument("--eps-max-cluster", type=int, default=1000,
                        help="Stop increasing eps once largest cluster exceeds this size.")
    parser.add_argument("--nn-trim", type=float, default=0.10,
                        help="Fraction trimmed from both tails when estimating d_nn.")
    parser.add_argument("--nn-sample", type=int, default=20000,
                        help="Maximum samples used when estimating d_nn (0 disables subsampling).")
    parser.add_argument("--min-points", type=int, default=40,
                        help="Minimum points per cluster after DBSCAN.")
    parser.add_argument("--min-circularity", type=float, default=0.35,
                        help="Minimum circularity threshold for convex-hull filtering.")
    parser.add_argument("--min-density", type=float, default=0.0012,
                        help="Minimum density (points per pixel^2) for clusters.")
    parser.add_argument("--density-area-scale", type=float, default=0.8,
                        help="Scale factor applied to convex-hull area before density check.")
    parser.add_argument("--alpha-scale-tight", type=float, default=1.3,
                        help="Tighter alpha scale multiplier relative to d_nn.")
    parser.add_argument("--alpha-scale-loose", type=float, default=1.9,
                        help="Looser alpha scale multiplier relative to d_nn.")
    parser.add_argument("--mask-close-radius", type=float, default=0.7,
                        help="Morphological closing radius (as multiple of d_nn).")
    parser.add_argument("--pca-k-sigma", type=float, default=2.5,
                        help="Mahalanobis radius (sigma) when using PCA ellipse fallback.")
    parser.add_argument("--no-alpha-shape", action="store_true",
                        help="Disable alpha-shape boundary refinement (use PCA ellipse only).")
    parser.add_argument("--enable-dedup", action="store_true",
                        help="Remove near-duplicate points before clustering.")
    parser.add_argument("--dedup-factor", type=float, default=0.5,
                        help="Multiplier applied to d_nn for deduplication radius.")
    parser.add_argument("--min-dedup-radius", type=float, default=2.0,
                        help="Absolute minimum deduplication radius in pixels.")
    parser.add_argument("--head", type=int, default=50,
                        help="Process the first N files (default: 50).")
    parser.add_argument("--tail", type=int, default=50,
                        help="Process the last N files (default: 50).")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Parallel jobs for per-file processing (default: 8). Use 1 to disable parallelism.")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Enable GPU DBSCAN using cuML if available.")
    parser.add_argument("--dry-run", action="store_true", help="Report stats without writing results.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    files = [
        path
        for path in iter_pos_files(args.input_root, args.pattern)
        if f"_pos_{args.output_suffix}_" not in path.name
    ]
    head = max(0, args.head)
    tail = max(0, args.tail)
    if head or tail:
        selection: list[Path] = []
        if head:
            selection.extend(files[:head])
        if tail:
            selection.extend(files[max(len(files) - tail, 0):])
        seen = set()
        unique_selection: list[Path] = []
        for path in selection:
            if path not in seen:
                unique_selection.append(path)
                seen.add(path)
        files = unique_selection if unique_selection else files

    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print("No detection files found.")
        return

    total_points = total_clustered = total_kept = 0
    eps_values = []
    cluster_counts = []

    if args.jobs == 1 or len(files) <= 1:
        results = []
        for path in files:
            res = process_file(path, args)
            results.append(res)
            if args.verbose:
                total, clustered, kept, eps, clusters = res
                print(f"{path}: total={total} clustered={clustered} kept={kept} eps={eps:.2f} clusters={clusters}")
    else:
        results = Parallel(n_jobs=args.jobs)(delayed(process_file)(path, args) for path in files)
        if args.verbose:
            for path, res in zip(files, results):
                total, clustered, kept, eps, clusters = res
                print(f"{path}: total={total} clustered={clustered} kept={kept} eps={eps:.2f} clusters={clusters}")

    for total, clustered, kept, eps, clusters in results:
        total_points += total
        total_clustered += clustered
        total_kept += kept
        eps_values.append(eps)
        cluster_counts.append(clusters)

    print(f"Processed {len(files)} files.")
    print(f"Total detections: {total_points}")
    print(f"DBSCAN clustered: {total_clustered}")
    print(f"Clusters kept after refinement: {total_kept}")
    if eps_values:
        arr = np.array(eps_values)
        print(
            "eps range -> "
            f"min {arr.min():.2f}px, median {np.median(arr):.2f}px, max {arr.max():.2f}px"
        )
    if cluster_counts:
        arr = np.array(cluster_counts)
        print(
            "clusters per image -> "
            f"min {arr.min()}, median {int(np.median(arr))}, max {arr.max()}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
