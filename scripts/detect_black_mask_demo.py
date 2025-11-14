"""Demo utility for detect_black_mask_artifact.

Given a directory of images, this script loads the configured
``min_area_ratio`` from ``pipeline_config.yml`` and evaluates whether each
image contains a black mask artifact.  The results are written to a CSV
file within the same directory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

# Ensure ``src`` can be imported when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.masking_operations import detect_black_mask_artifact
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
    """Evaluate the presence of black mask artifacts for all images."""

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {image_dir}")

    config = load_config(str(config_path))
    min_area_ratio = (
        config.get("stage_02_characterization", {})
        .get("quality_features", {})
        .get("black_mask_check", {})
        .get("min_area_ratio", 0.05)
    )

    LOGGER.info(
        "Using min_area_ratio=%.4f from %s", min_area_ratio, config_path
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
        )
        results.append(
            {
                "image_name": image_path.name,
                "is_black_mask": str(has_artifact).lower(),
            }
        )

    output_path = image_dir / "black_mask_detection_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image_name", "is_black_mask"])
        writer.writeheader()
        writer.writerows(results)

    LOGGER.info("Saved results to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    _configure_logging(args.verbose)

    try:
        output_csv = run_demo(args.image_dir, args.config)
    except Exception as exc:  # noqa: BLE001 - surface full error to CLI user
        raise SystemExit(f"Error while running demo: {exc}") from exc

    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()
