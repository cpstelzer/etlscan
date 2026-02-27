"""Small diagnostic utility for crop-boundary flag behavior.

This script evaluates a single cropped particle image with
`detect_cropped_particle_on_boundary` and writes human-readable + visual
artifacts under `data/boundary_flag/output`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import skimage.io
import skimage.util

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.core.particle_operations import (
    detect_cropped_particle_on_boundary,
    generate_binary_mask_from_float_image,
)


def _build_threshold_params(args: argparse.Namespace) -> Dict[str, float | int | str]:
    return {
        "thresholding_mode": args.thresholding_mode,
        "otsu_threshold_multiplier": float(args.otsu_threshold_multiplier),
        "fixed_threshold_value": int(args.fixed_threshold_value),
    }


def _compute_binary_mask(
    image_crop_gray: np.ndarray,
    illumination_mode: str,
    threshold_params: Dict[str, float | int | str],
) -> np.ndarray:
    img_float = skimage.util.img_as_float(image_crop_gray)
    if illumination_mode == "collimated":
        img_float = 1.0 - img_float

    binary_mask = generate_binary_mask_from_float_image(img_float, threshold_params)
    if binary_mask is None:
        raise RuntimeError("Binary mask generation failed.")
    return binary_mask.astype(bool)


def _find_boundary_foreground_pixels(binary_mask: np.ndarray) -> List[Tuple[int, int]]:
    h, w = binary_mask.shape
    coords: List[Tuple[int, int]] = []

    for x in range(w):
        if binary_mask[0, x]:
            coords.append((x, 0))
        if h > 1 and binary_mask[h - 1, x]:
            coords.append((x, h - 1))

    for y in range(1, max(1, h - 1)):
        if binary_mask[y, 0]:
            coords.append((0, y))
        if w > 1 and binary_mask[y, w - 1]:
            coords.append((w - 1, y))

    return sorted(set(coords), key=lambda xy: (xy[1], xy[0]))


def _render_overlay(image_crop_gray: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
    overlay = np.stack([image_crop_gray, image_crop_gray, image_crop_gray], axis=-1)

    h, w = image_crop_gray.shape
    overlay[0, :, :] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[h - 1, :, :] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[:, 0, :] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[:, w - 1, :] = np.array([255, 255, 0], dtype=np.uint8)

    for x, y in coords:
        overlay[y, x, :] = np.array([255, 0, 0], dtype=np.uint8)

    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic for detect_cropped_particle_on_boundary")
    parser.add_argument("--input-image", required=True, help="Path to a cropped particle image (grayscale or color).")
    parser.add_argument("--output-dir", default="data/boundary_flag/output", help="Directory for diagnostic outputs.")
    parser.add_argument(
        "--illumination-mode",
        choices=["darkfield", "collimated"],
        default="darkfield",
        help="Illumination mode passed into boundary detection.",
    )
    parser.add_argument(
        "--thresholding-mode",
        choices=["otsu", "fixed"],
        default="otsu",
        help="Thresholding mode used inside boundary detection.",
    )
    parser.add_argument("--otsu-threshold-multiplier", type=float, default=0.45)
    parser.add_argument("--fixed-threshold-value", type=int, default=70)
    args = parser.parse_args()

    input_image_path = Path(args.input_image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = skimage.io.imread(str(input_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_image_path}")

    if image.ndim == 3:
        image = skimage.util.img_as_ubyte(np.mean(image[..., :3], axis=2) / 255.0)
    elif image.ndim == 2:
        image = skimage.util.img_as_ubyte(skimage.util.img_as_float(image))
    else:
        raise ValueError(f"Unsupported image shape {image.shape}; expected 2D or 3D.")

    threshold_params = _build_threshold_params(args)

    is_on_crop_boundary = detect_cropped_particle_on_boundary(
        image,
        str(input_image_path),
        args.illumination_mode,
        threshold_params,
    )

    binary_mask = _compute_binary_mask(image, args.illumination_mode, threshold_params)
    boundary_coords = _find_boundary_foreground_pixels(binary_mask)

    overlay = _render_overlay(image, boundary_coords)
    mask_u8 = (binary_mask.astype(np.uint8) * 255)

    stem = input_image_path.stem
    mask_path = output_dir / f"{stem}_binary_mask.png"
    overlay_path = output_dir / f"{stem}_boundary_overlay.png"
    report_path = output_dir / f"{stem}_boundary_report.txt"

    skimage.io.imsave(str(mask_path), mask_u8, check_contrast=False)
    skimage.io.imsave(str(overlay_path), overlay, check_contrast=False)

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write(f"input_image={input_image_path}\n")
        report_file.write(f"illumination_mode={args.illumination_mode}\n")
        report_file.write(f"threshold_params={threshold_params}\n")
        report_file.write(f"is_on_crop_boundary={bool(is_on_crop_boundary)}\n")
        report_file.write(f"boundary_foreground_pixel_count={len(boundary_coords)}\n")
        report_file.write("boundary_foreground_pixels_xy=\n")
        for x, y in boundary_coords:
            report_file.write(f"  ({x}, {y})\n")

    print(f"is_on_crop_boundary={bool(is_on_crop_boundary)}")
    print(f"boundary_foreground_pixel_count={len(boundary_coords)}")
    print(f"report={report_path}")
    print(f"mask={mask_path}")
    print(f"overlay={overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
