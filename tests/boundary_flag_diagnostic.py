"""Diagnostic utility for crop-boundary flag behavior.

Processes all image files in an input directory, evaluates
`detect_cropped_particle_on_boundary`, and writes outputs under
`data/boundary_flag/output`.

Important: threshold defaults are loaded from the shared pipeline config to
stay consistent with offline/realtime behavior.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import skimage.io
import skimage.util

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils.config_loader import get_unified_segmentation_params, load_config_once

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_boundary_functions() -> tuple[Callable[..., bool], Callable[..., Any]]:
    """Lazy-load heavy imports so --help works in constrained environments."""
    from src.core.particle_operations import (  # pylint: disable=import-outside-toplevel
        detect_cropped_particle_on_boundary,
        generate_binary_mask_from_float_image,
    )

    return detect_cropped_particle_on_boundary, generate_binary_mask_from_float_image


def _resolve_threshold_params(args: argparse.Namespace) -> Dict[str, float | int | str]:
    """Resolve threshold parameters using shared pipeline config as source of truth."""
    config = load_config_once(args.config)
    segmentation_params = get_unified_segmentation_params(config)

    thresholding_mode = (
        args.thresholding_mode
        if args.thresholding_mode is not None
        else str(segmentation_params.get("thresholding_mode", "otsu"))
    )
    otsu_threshold_multiplier = (
        float(args.otsu_threshold_multiplier)
        if args.otsu_threshold_multiplier is not None
        else float(segmentation_params.get("otsu_threshold_multiplier", 0.45))
    )
    fixed_threshold_value = (
        int(args.fixed_threshold_value)
        if args.fixed_threshold_value is not None
        else int(segmentation_params.get("fixed_threshold_value", 70))
    )

    return {
        "thresholding_mode": thresholding_mode,
        "otsu_threshold_multiplier": otsu_threshold_multiplier,
        "fixed_threshold_value": fixed_threshold_value,
    }


def _compute_binary_mask(
    image_crop_gray: np.ndarray,
    illumination_mode: str,
    threshold_params: Dict[str, float | int | str],
    generate_binary_mask_from_float_image: Callable[..., Any],
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


def _load_as_grayscale_u8(image_path: Path) -> np.ndarray:
    image = skimage.io.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    if image.ndim == 3:
        return skimage.util.img_as_ubyte(np.mean(image[..., :3], axis=2) / 255.0)
    if image.ndim == 2:
        return skimage.util.img_as_ubyte(skimage.util.img_as_float(image))

    raise ValueError(f"Unsupported image shape {image.shape}; expected 2D or 3D.")


def _iter_input_images(input_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic for detect_cropped_particle_on_boundary")
    parser.add_argument("--config", default="config/pipeline_config.yml", help="Path to pipeline config YAML.")
    parser.add_argument("--input-dir", default="data/boundary_flag/input", help="Directory with cropped particle images.")
    parser.add_argument("--output-dir", default="data/boundary_flag/output", help="Directory for diagnostic outputs.")
    parser.add_argument(
        "--illumination-mode",
        choices=["darkfield", "collimated"],
        default="collimated",
        help="Illumination mode passed into boundary detection.",
    )
    parser.add_argument(
        "--thresholding-mode",
        choices=["otsu", "fixed"],
        default=None,
        help="Optional override. Default comes from shared pipeline config.",
    )
    parser.add_argument(
        "--otsu-threshold-multiplier",
        type=float,
        default=None,
        help="Optional override. Default comes from shared pipeline config.",
    )
    parser.add_argument(
        "--fixed-threshold-value",
        type=int,
        default=None,
        help="Optional override. Default comes from shared pipeline config.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_images = _iter_input_images(input_dir)
    if not input_images:
        print(f"No supported images found in: {input_dir}")
        return 0

    detect_cropped_particle_on_boundary, generate_binary_mask_from_float_image = _load_boundary_functions()
    threshold_params = _resolve_threshold_params(args)

    summary_lines = [
        f"input_dir={input_dir}",
        f"illumination_mode={args.illumination_mode}",
        f"threshold_params={threshold_params}",
        f"image_count={len(input_images)}",
        "",
    ]

    print(f"Using threshold params: {threshold_params}")

    for image_path in input_images:
        image = _load_as_grayscale_u8(image_path)

        is_on_crop_boundary = detect_cropped_particle_on_boundary(
            image,
            str(image_path),
            args.illumination_mode,
            threshold_params,
        )

        binary_mask = _compute_binary_mask(
            image,
            args.illumination_mode,
            threshold_params,
            generate_binary_mask_from_float_image,
        )
        boundary_coords = _find_boundary_foreground_pixels(binary_mask)

        overlay = _render_overlay(image, boundary_coords)
        mask_u8 = (binary_mask.astype(np.uint8) * 255)

        stem = image_path.stem
        mask_path = output_dir / f"{stem}_binary_mask.png"
        overlay_path = output_dir / f"{stem}_boundary_overlay.png"
        report_path = output_dir / f"{stem}_boundary_report.txt"

        skimage.io.imsave(str(mask_path), mask_u8, check_contrast=False)
        skimage.io.imsave(str(overlay_path), overlay, check_contrast=False)

        with report_path.open("w", encoding="utf-8") as report_file:
            report_file.write(f"input_image={image_path}\n")
            report_file.write(f"illumination_mode={args.illumination_mode}\n")
            report_file.write(f"threshold_params={threshold_params}\n")
            report_file.write(f"is_on_crop_boundary={bool(is_on_crop_boundary)}\n")
            report_file.write(f"boundary_foreground_pixel_count={len(boundary_coords)}\n")
            report_file.write("boundary_foreground_pixels_xy=\n")
            for x, y in boundary_coords:
                report_file.write(f"  ({x}, {y})\n")

        summary_lines.extend(
            [
                f"input_image={image_path}",
                f"is_on_crop_boundary={bool(is_on_crop_boundary)}",
                f"boundary_foreground_pixel_count={len(boundary_coords)}",
                f"report={report_path}",
                f"mask={mask_path}",
                f"overlay={overlay_path}",
                "",
            ]
        )

        print(f"[{image_path.name}] is_on_crop_boundary={bool(is_on_crop_boundary)} | boundary_foreground_pixel_count={len(boundary_coords)}")

    summary_path = output_dir / "boundary_diagnostic_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
