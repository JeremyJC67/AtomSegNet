#!/usr/bin/env python3
"""Demo script showing contour extraction results with visualization."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import sys

from blur_filter_points import (
    build_mask,
    extract_topk_contours,
    polygon_area,
)


def visualize_contour_process(image_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Create a detailed visualization showing the contour extraction process."""
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

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem

    # 1. Original image
    image.save(output_dir / f"{base_name}_01_original.png")

    # 2. Blurred image
    blur_norm = blurred - blurred.min()
    if blur_norm.max() > 0:
        blur_norm = blur_norm / blur_norm.max()
    blur_img = (blur_norm * 255).astype(np.uint8)
    Image.fromarray(blur_img, mode="L").save(output_dir / f"{base_name}_02_blurred.png")

    # 3. Mask
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(output_dir / f"{base_name}_03_mask.png")

    # 4. Contours on original
    result_img = Image.fromarray(image_array.astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(result_img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for idx, contour_xy in enumerate(contours_xy):
        if contour_xy is not None and contour_xy.size > 0:
            color = colors[idx % len(colors)]
            coords = [(float(x), float(y)) for x, y in contour_xy]
            if len(coords) >= 2:
                draw.line(coords + [coords[0]], fill=color, width=3)
                # Add index label at contour center
                center_x = contour_xy[:, 0].mean()
                center_y = contour_xy[:, 1].mean()
                draw.text((int(center_x), int(center_y)), str(idx), fill=color)

    result_img.save(output_dir / f"{base_name}_04_contours.png")

    # 5. Contours on blurred
    blur_rgb = Image.fromarray(blur_img, mode="L").convert("RGB")
    draw = ImageDraw.Draw(blur_rgb)

    for idx, contour_xy in enumerate(contours_xy):
        if contour_xy is not None and contour_xy.size > 0:
            color = colors[idx % len(colors)]
            coords = [(float(x), float(y)) for x, y in contour_xy]
            if len(coords) >= 2:
                draw.line(coords + [coords[0]], fill=color, width=3)

    blur_rgb.save(output_dir / f"{base_name}_05_contours_on_blur.png")

    # 6. Individual cluster crops with contours
    for idx, contour_xy in enumerate(contours_xy):
        if contour_xy is None or contour_xy.size == 0:
            continue

        min_x = max(0, int(np.floor(contour_xy[:, 0].min())) - 10)
        max_x = min(image_array.shape[1], int(np.ceil(contour_xy[:, 0].max())) + 10)
        min_y = max(0, int(np.floor(contour_xy[:, 1].min())) - 10)
        max_y = min(image_array.shape[0], int(np.ceil(contour_xy[:, 1].max())) + 10)

        region = image_array[min_y:max_y, min_x:max_x].copy()
        region_img = Image.fromarray(region.astype(np.uint8), mode="L").convert("RGB")
        draw = ImageDraw.Draw(region_img)

        contour_local = contour_xy - np.array([min_x, min_y])
        coords = [(float(x), float(y)) for x, y in contour_local]
        if len(coords) >= 2:
            draw.line(coords + [coords[0]], fill=(255, 0, 0), width=2)

        region_img.save(output_dir / f"{base_name}_06_cluster_{idx}_crop.png")

    # Print statistics
    print(f"  Clusters found: {len(contours_xy)}")
    for idx, contour_xy in enumerate(contours_xy):
        area = polygon_area(contour_xy.astype(np.float32))
        center_x = contour_xy[:, 0].mean()
        center_y = contour_xy[:, 1].mean()
        perimeter = np.sum(np.linalg.norm(np.diff(contour_xy, axis=0), axis=1)) + \
                    np.linalg.norm(contour_xy[0] - contour_xy[-1])
        print(f"    Cluster {idx}:")
        print(f"      Area: {area:.1f} px²")
        print(f"      Perimeter: {perimeter:.1f} px")
        print(f"      Center: ({center_x:.1f}, {center_y:.1f})")
        print(f"      Boundary points: {len(contour_xy)}")

    print(f"  Output saved to: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate contour extraction with detailed visualization."
    )
    parser.add_argument("input_image", type=Path, help="Input image to process")
    parser.add_argument("output_dir", type=Path, help="Output directory for results")
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

    args = parser.parse_args()

    if not args.input_image.exists():
        print(f"Error: Input image not found: {args.input_image}")
        sys.exit(1)

    print("=" * 60)
    print("ATOM CLUSTER CONTOUR EXTRACTION - DEMO")
    print("=" * 60)
    print()

    visualize_contour_process(args.input_image, args.output_dir, args)

    print("=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  01_original.png      - Original input image")
    print("  02_blurred.png       - Gaussian blurred image")
    print("  03_mask.png          - Binary segmentation mask")
    print("  04_contours.png      - Contours on original image")
    print("  05_contours_on_blur.png - Contours on blurred image")
    print("  06_cluster_*.png     - Individual cluster crops (RED outlines)")
    print()


if __name__ == "__main__":
    main()
