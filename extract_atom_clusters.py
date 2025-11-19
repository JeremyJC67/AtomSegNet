#!/usr/bin/env python3
"""Extract atomic clusters from images with contour detection and coordinate export.

This script demonstrates the enhanced contour detection capabilities:
1. Identifies atomic cluster boundaries automatically
2. Exports per-cluster images with red contours highlighted
3. Saves boundary coordinates for each cluster
4. Generates overlay visualization

Example usage:
    python3 extract_atom_clusters.py blured_1742/ output_clusters/ --sigma 3.0
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import sys

# Import contour functions from blur_filter_points
from blur_filter_points import (
    build_mask,
    extract_topk_contours,
    save_individual_contour_images,
    save_boundary_coords_multi,
)


def process_single_image(image_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Process a single image to extract atomic clusters."""
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return

    print(f"Processing: {image_path.name}")

    # Load image
    image = Image.open(image_path).convert("L")
    image_array = np.array(image, dtype=np.float32)

    # Build mask from image
    mask, blurred = build_mask(
        image_array,
        sigma=args.sigma,
        threshold_scale=args.threshold_scale,
        radius_frac=args.radius_frac,
        open_radius_frac=args.open_radius_frac,
        use_local=not args.use_global_threshold,
        local_block=args.local_block,
        local_offset=args.local_offset,
        min_obj_area=args.min_obj_area,
        split_touching=not args.no_split_touching,
    )

    # Extract contours
    contours_xy = extract_topk_contours(mask, k=args.max_contours, smooth=True)

    if not contours_xy:
        print(f"  No clusters found")
        return

    print(f"  Found {len(contours_xy)} cluster(s)")

    # Normalize blurred image for visualization
    blur_norm = blurred - blurred.min()
    if blur_norm.max() > 0:
        blur_norm = blur_norm / blur_norm.max()
    blur_img = (blur_norm * 255).astype(np.uint8)

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem

    # Save individual cluster images and coordinates
    save_individual_contour_images(base_name, blur_img, contours_xy, type('Args', (), {'save_blur_dir': output_dir})())
    save_boundary_coords_multi(output_dir, base_name, contours_xy)

    # Print cluster statistics
    for idx, contour_xy in enumerate(contours_xy):
        area = 0.5 * np.abs(
            np.dot(contour_xy[:, 0], np.roll(contour_xy[:, 1], -1))
            - np.dot(contour_xy[:, 1], np.roll(contour_xy[:, 0], -1))
        )
        center_x = contour_xy[:, 0].mean()
        center_y = contour_xy[:, 1].mean()
        print(f"  Cluster {idx}: area={area:.1f}, center=({center_x:.1f}, {center_y:.1f})")

    print(f"  Output files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract atomic clusters from images with contour detection."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing images to process")
    parser.add_argument("output_dir", type=Path, help="Directory for output files")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern for input images")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian blur sigma")
    parser.add_argument("--threshold-scale", type=float, default=0.95, help="Otsu threshold scale")
    parser.add_argument("--radius-frac", type=float, default=0.10, help="Morphological radius fraction")
    parser.add_argument("--open-radius-frac", type=float, default=None, help="Opening radius fraction")
    parser.add_argument("--max-contours", type=int, default=2, help="Maximum contours per image")
    parser.add_argument("--local-block", type=int, default=91, help="Local threshold block size")
    parser.add_argument("--local-offset", type=float, default=5.0, help="Local threshold offset")
    parser.add_argument("--min-obj-area", type=int, default=1200, help="Minimum object area")
    parser.add_argument("--use-global-threshold", action="store_true", help="Use global threshold")
    parser.add_argument("--no-split-touching", action="store_true", help="Disable watershed splitting")
    parser.add_argument("--head", type=int, default=5, help="Process only first N images")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Find input files
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        print(f"Error: No images found matching pattern '{args.pattern}'")
        sys.exit(1)

    # Limit to head if specified
    if args.head > 0:
        files = files[: args.head]

    print(f"Found {len(files)} images to process")
    print()

    # Process each image
    for image_path in files:
        process_single_image(image_path, args.output_dir, args)
        print()


if __name__ == "__main__":
    main()
