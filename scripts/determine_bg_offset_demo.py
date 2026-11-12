"""Utility script to inspect background offset selection.

This script loads image metadata from one or more directories, filters the
frames using the same logic as the stage 01 processing pipeline, and then calls
``_determine_bg_offset_for_fg_bg_pairs`` directly to display the automatically
chosen background offset(s).  It is intended to help diagnose issues with the
background pairing heuristics.

To run this script, navigate to well_plate_analyzer/scripts and execute:
python determine_bg_offset_demo.py --input-batch-directory /path/to/image_directory

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Allow the script to be executed directly from the ``scripts`` directory without
# requiring callers to modify ``PYTHONPATH`` manually.  This mirrors the
# repository root so imports like ``src.*`` resolve correctly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stages.stage_01_initial_detection import (
    _determine_bg_offset_for_fg_bg_pairs,
    _filter_image_df,
)
from src.utils.path_helpers import parse_frame_filenames

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure a simple logging handler for CLI usage."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _load_image_dataframe(image_dir: Path) -> pd.DataFrame:
    """Parse and filter the image metadata for the provided directory."""

    parsed_df = parse_frame_filenames(str(image_dir))
    if parsed_df is None:
        raise RuntimeError("Failed to parse image filenames; see log for details.")

    if parsed_df.empty:
        raise RuntimeError(
            "No matching images were found in the directory after parsing."
        )

    return _filter_image_df(parsed_df, trim_to_last_fd0=True)


def determine_offset(
    image_dir: Path,
    minimum_offset: Optional[int],
    fallback_offset: int,
    tolerance: float,
) -> int:
    """Determine the background offset for the provided dataset."""

    filtered_df = _load_image_dataframe(image_dir)
    LOGGER.info(
        "Loaded %d frames after filtering (input directory: %s)",
        len(filtered_df),
        image_dir,
    )
    return _determine_bg_offset_for_fg_bg_pairs(
        filtered_df,
        minimum_offset=minimum_offset,
        fallback_offset=fallback_offset,
        tolerance=tolerance,
    )


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for the background offset demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Call _determine_bg_offset_for_fg_bg_pairs on a dataset and display the "
            "selected offset."
        )
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        type=Path,
        help="Directory containing the foreground/background image sequence.",
    )
    parser.add_argument(
        "--input-batch-directory",
        dest="batch_dir",
        type=Path,
        help=(
            "Directory containing multiple datasets, each stored in a subdirectory. "
            "The script will determine an offset for every dataset and display the "
            "results in a table."
        ),
    )
    parser.add_argument(
        "--minimum-offset",
        type=int,
        default=None,
        help=(
            "Minimum offset (in frames) to consider when pairing FG/BG frames. "
            "If omitted, the function will fall back immediately."
        ),
    )
    parser.add_argument(
        "--fallback-offset",
        type=int,
        default=1,
        help="Fallback offset to use if no suitable automatic match is found.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for comparing fd_mm values.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI workflow for determining background offsets."""
    args = parse_args()
    _configure_logging(args.verbose)

    image_dir: Optional[Path] = args.image_dir
    batch_dir: Optional[Path] = args.batch_dir

    if (image_dir is None) == (batch_dir is None):
        raise SystemExit(
            "Specify either a single image directory or --input-batch-directory, but not both."
        )

    if image_dir is not None:
        if not image_dir.exists() or not image_dir.is_dir():
            raise SystemExit(f"Image directory not found: {image_dir}")

        try:
            offset = determine_offset(
                image_dir=image_dir,
                minimum_offset=args.minimum_offset,
                fallback_offset=args.fallback_offset,
                tolerance=args.tolerance,
            )
        except Exception as exc:  # noqa: BLE001 - expose full error to CLI user
            raise SystemExit(f"Error determining background offset: {exc}") from exc

        print(f"Determined background offset: {offset}")
        return

    assert batch_dir is not None  # for type-checkers

    if not batch_dir.exists() or not batch_dir.is_dir():
        raise SystemExit(f"Batch directory not found: {batch_dir}")

    dataset_dirs: Iterable[Path] = sorted(
        entry for entry in batch_dir.iterdir() if entry.is_dir()
    )

    if not dataset_dirs:
        raise SystemExit(
            "No dataset subdirectories were found inside the provided batch directory."
        )

    results = []
    for dataset_dir in dataset_dirs:
        dataset_label = dataset_dir.relative_to(batch_dir)
        try:
            offset = determine_offset(
                image_dir=dataset_dir,
                minimum_offset=args.minimum_offset,
                fallback_offset=args.fallback_offset,
                tolerance=args.tolerance,
            )
            results.append({
                "dataset": str(dataset_label),
                "offset": offset,
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001 - expose full error to CLI user
            LOGGER.error("Error determining background offset for %s: %s", dataset_dir, exc)
            results.append({
                "dataset": str(dataset_label),
                "offset": None,
                "status": f"error: {exc}",
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df[["dataset", "offset", "status"]]
        results_df["offset"] = results_df["offset"].fillna("")

    print("\nBackground offsets per dataset:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
