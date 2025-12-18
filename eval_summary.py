#!/usr/bin/env python3
"""Compute combined NN + kNN + mask coverage + overlap metrics for frames."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from knn_eval import compute_knn_distances, load_atomsegnet_coords, save_knn_results
from mask_coverage_eval import compute_mask_coverage
from nn_eval import compute_nn_metrics
from overlap_eval import compute_overlap


BASE = Path("/home/jicwang/atom-research/CustomAtomSegNet/AtomSegNet")
FRAMES = ["0000", "0007", "0036"]

PIPELINES = {
    "AS": lambda f: BASE / f"atomsegnet_enhanced_results/{f}/{f}_pos_gaussian_blur_Gen1-gaussianMask.txt",
    "UDVD": lambda f: BASE / f"atomsegnet_udvd_results/1742_udvd_incnb_{f}/"
                      f"1742_udvd_incnb_{f}_pos_gaussian_blur_Gen1-gaussianMask.txt",
    "UDVD+AS": lambda f: BASE / f"atomsegnet_udvd_twostage_results/1742_udvd_incnb_{f}_denoise/"
                         f"1742_udvd_incnb_{f}_pos_gaussian_blur_Gen1-gaussianMask_lattice_filled3.txt",
}

MASKS = {
    "AS": lambda f: BASE / f"atom_cluster_out_sigma14/{f}_heavyblur_mask.png",
    "UDVD": lambda f: BASE / f"atom_cluster_out_udvd_sigma14_full/1742_udvd_incnb_{f}_heavyblur_mask.png",
    "UDVD+AS": lambda f: BASE / f"atom_cluster_out_udvd_sigma14_full/1742_udvd_incnb_{f}_heavyblur_mask.png",
}


def format_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def compute_frame_metrics(frame: str, save_knn: bool = True) -> Dict[str, dict]:
    metrics: Dict[str, dict] = {}
    udvd_coords = load_atomsegnet_coords(PIPELINES["UDVD"](frame))

    for name, path_fn in PIPELINES.items():
        path = path_fn(frame)
        if not path.exists() and name == "UDVD+AS":
            alt = BASE / f"atomsegnet_udvd_twostage_results/1742_udvd_incnb_{frame}_denoise/" \
                        f"1742_udvd_incnb_{frame}_pos_gaussian_blur_Gen1-gaussianMask.txt"
            if alt.exists():
                path = alt
        coords = load_atomsegnet_coords(path)

        nn = compute_nn_metrics(coords)
        knn = compute_knn_distances(coords, k_list=(4, 6, 8))
        mask_cov = compute_mask_coverage(coords, MASKS[name](frame))

        if save_knn:
            out_dir = BASE / "knn_eval_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_knn_results(knn, out_dir / f"{frame}_{name}.npz")

        overlap = {"a_to_b": 1.0, "b_to_a": 1.0}
        if name != "UDVD":
            overlap = compute_overlap(udvd_coords, coords, radius=3.0)

        metrics[name] = {
            "n_points": nn["n_points"],
            "nn_mean": nn["mean"],
            "nn_median": nn["median"],
            "nn_std": nn["std"],
            "k4_mean": knn[4]["mean"],
            "k4_std": knn[4]["std"],
            "k6_mean": knn[6]["mean"],
            "k6_std": knn[6]["std"],
            "k8_mean": knn[8]["mean"],
            "k8_std": knn[8]["std"],
            "mask_frac": mask_cov["inside_frac"],
            "udvd_to_col": overlap["a_to_b"],
            "col_to_udvd": overlap["b_to_a"],
        }
    return metrics


def print_frame_table(frame: str, metrics: Dict[str, dict]) -> None:
    header = "| Metric | AS | UDVD | UDVD+AS |"
    sep = "|---|---:|---:|---:|"
    rows = []

    rows.append(("Point count", "n_points", 0))
    rows.append(("NN mean / median / std (px)", "nn", 3))
    rows.append(("k4 mean / std (px)", "k4", 3))
    rows.append(("k6 mean / std (px)", "k6", 3))
    rows.append(("k8 mean / std (px)", "k8", 3))
    rows.append(("Mask coverage (inside frac)", "mask_frac", 3))
    rows.append(("UDVD overlap (UDVD→col, 3px)", "udvd_to_col", 3))
    rows.append(("Col overlap to UDVD (3px)", "col_to_udvd", 3))

    print(f"\nFrame {frame}")
    print(header)
    print(sep)
    for label, key, digits in rows:
        if key == "nn":
            vals = []
            for name in ("AS", "UDVD", "UDVD+AS"):
                m = metrics[name]
                vals.append(
                    f"{format_float(m['nn_mean'], digits)} / {format_float(m['nn_median'], digits)} / {format_float(m['nn_std'], digits)}"
                )
        elif key in ("k4", "k6", "k8"):
            vals = []
            for name in ("AS", "UDVD", "UDVD+AS"):
                m = metrics[name]
                vals.append(
                    f"{format_float(m[f'{key}_mean'], digits)} / {format_float(m[f'{key}_std'], digits)}"
                )
        elif key == "n_points":
            vals = [str(metrics[name][key]) for name in ("AS", "UDVD", "UDVD+AS")]
        else:
            vals = [format_float(metrics[name][key], digits) for name in ("AS", "UDVD", "UDVD+AS")]
        print(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")


def main() -> None:
    for frame in FRAMES:
        metrics = compute_frame_metrics(frame, save_knn=True)
        print_frame_table(frame, metrics)


if __name__ == "__main__":
    main()
