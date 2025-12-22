"""Demo utility for detect_black_mask_artifact and crop-boundary detection.

Given a directory of images, this script loads the configured
``min_area_ratio`` from ``pipeline_config.yml`` and evaluates whether each
image contains a black mask artifact. It also checks whether each image
would be flagged by ``detect_cropped_particle_on_boundary``.
The combined results are written to a CSV file within the same directory.

`scripts/detect_black_mask_demo.py` evaluates every supported image in a directory with `detect_black_mask_artifact` and `detect_cropped_particle_on_boundary` and writes `black_mask_detection_results.csv` beside the inputs. Example:

  ```
  python scripts/detect_black_mask_demo.py data/sanity_checks/masks --config config/pipeline_config.yml
  ```

"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

from skimage import io as skio
from skimage import util as skutil

# Ensure ``src`` can be imported when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.masking_operations import detect_black_mask_artifact
from src.core.particle_operations import detect_cropped_particle_on_boundary
from src.utils.config_loader import load_config

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _configure_logging(verbose: bool) -> None:
    """Configure a basic console logger for CLI usage."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _iter_image_files(directory: Path) -> Iterable[Path]:
    """Yield image files from ``directory`` that match supported extensions."""

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def run_demo(image_dir: Path, config_path: Path) -> Path:
    """Evaluate black mask artifacts and boundary-touching crops for all images."""

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {image_dir}")

    config = load_config(str(config_path))
    black_mask_cfg = (
        config.get("stage_02_characterization", {})
        .get("quality_features", {})
        .get("black_mask_check", {})
    )
    system_cfg = config.get("system", {})
    threshold_cfg = system_cfg.get("thresholding", {})
    min_area_ratio = black_mask_cfg.get("min_area_ratio", 0.05)
    max_black_pixel_value = black_mask_cfg.get("max_black_pixel_value", 15)
    min_edge_thickness_fraction = black_mask_cfg.get("min_edge_thickness_fraction", 0.2)

    LOGGER.info(
        "Using min_area_ratio=%.4f, max_black_pixel_value=%d, and min_edge_thickness_fraction=%.2f from %s",
        min_area_ratio,
        max_black_pixel_value,
        min_edge_thickness_fraction,
        config_path,
    )

    image_files = list(_iter_image_files(image_dir))
    if not image_files:
        raise RuntimeError(
            f"No supported image files were found in {image_dir}. Supported extensions: "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )

    results = []
    for image_path in image_files:
        LOGGER.debug("Analyzing %s", image_path.name)
        has_artifact = detect_black_mask_artifact(
            image_path=str(image_path),
            min_area_ratio=min_area_ratio,
            max_black_pixel_value=max_black_pixel_value,
            min_edge_thickness_fraction=min_edge_thickness_fraction,
        )
        image_gray = skio.imread(str(image_path), as_gray=True)
        if image_gray is None:
            raise RuntimeError(f"Failed to load {image_path} for boundary detection.")
        image_gray_uint8 = skutil.img_as_ubyte(image_gray)

        boundary_result = detect_cropped_particle_on_boundary(
            image_gray_uint8,
            str(image_path),
            str(system_cfg.get("source_image_illumination_type", "darkfield")).strip().lower(),
            {
                "thresholding_mode": threshold_cfg.get("mode", "otsu"),
                "otsu_threshold_multiplier": threshold_cfg.get("otsu_threshold_multiplier", 0.45),
                "fixed_threshold_value": threshold_cfg.get("fixed_threshold_value", 70),
            },
        )
        results.append(
            {
                "image_name": image_path.name,
                "is_black_mask": str(has_artifact).lower(),
                "is_on_crop_boundary": str(bool(boundary_result)).lower(),
            }
        )

    output_path = image_dir / "black_mask_detection_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["image_name", "is_black_mask", "is_on_crop_boundary"],
        )
        writer.writeheader()
        writer.writerows(results)

    LOGGER.info("Saved results to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for the black mask detection demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Run detect_black_mask_artifact on every supported image in a directory "
            "and save the results to CSV."
        )
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory that contains the input images.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "pipeline_config.yml",
        help="Path to pipeline_config.yml (defaults to the repository config).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running the black mask detection demo from the CLI."""
    args = parse_args()
    _configure_logging(args.verbose)

    try:
        output_csv = run_demo(args.image_dir, args.config)
    except Exception as exc:  # noqa: BLE001 - surface full error to CLI user
        raise SystemExit(f"Error while running demo: {exc}") from exc

    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()
