"""Utilities for rebuilding labelled training rows from curated crops.

This helper walks the manually curated PNG directories, identifies the
corresponding Stage-03 focussed CSV for every sample, and extracts the metric
rows tied to each crop.  The recovered rows are combined into a single CSV that
`scripts/train_model.py` can ingest via the `--csv-files` flag.

Usage example::

    python scripts/build_manual_labels_csv.py --config config/pipeline_config.yml \
        --stage3-root C:/sandbox --good-dir C:/sandbox/data/training_data/good \
            --bad-dir C:/sandbox/data/training_data/bad \
                --output-csv C:/sandbox/data/training_data/manual_labels.csv

The script expects manually curated crops in ``data/training_data/good`` and
``data/training_data/bad`` by default.  Each PNG name embeds the Stage-03 sample
identifier up to ``_track`` which is used to locate the CSV inside the Stage-03
output tree.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def parse_sample_id_from_filename(filename: str) -> Optional[str]:
    """Return the Stage-03 sample identifier embedded in a curated filename."""

    if '_track' not in filename:
        return None

    prefix, suffix = filename.split('_track', 1)
    if not suffix:
        return None
    return prefix


def discover_stage3_csvs(stage3_root: Path, csv_name: str) -> Dict[str, Path]:
    """Scan ``stage3_root`` for focussed CSVs and build a sample -> path map."""

    if not stage3_root.exists():
        raise FileNotFoundError(
            f"Stage-03 root directory does not exist: {stage3_root}"
        )

    csv_lookup: Dict[str, Path] = {}
    for csv_path in stage3_root.rglob(csv_name):
        if csv_path.parent.parent is None:
            continue
        sample_id = csv_path.parent.parent.name
        if sample_id in csv_lookup:
            LOGGER.warning(
                "Multiple Stage-03 CSVs detected for sample %s. Keeping %s, ignoring %s",
                sample_id,
                csv_lookup[sample_id],
                csv_path,
            )
            continue
        csv_lookup[sample_id] = csv_path

    if not csv_lookup:
        LOGGER.warning(
            "No focussed CSVs found under %s. Ensure Stage-03 outputs are present.",
            stage3_root,
        )

    return csv_lookup


def _match_crop_row(df: pd.DataFrame, png_name: str) -> Optional[pd.Series]:
    """Return the dataframe row that corresponds to ``png_name`` if present."""

    candidate_columns = [
        'final_crop_filename',
        'crop_filename',
        'meta__final_crop_filename',
        'meta__crop_filename',
    ]

    for column in candidate_columns:
        if column not in df.columns:
            continue
        matches = df.loc[df[column] == png_name]
        if not matches.empty:
            return matches.iloc[0]
    return None


def _load_stage3_dataframe(
    sample_id: str,
    cache: Dict[str, pd.DataFrame],
    csv_lookup: Dict[str, Path],
) -> Optional[pd.DataFrame]:
    """Load and cache the Stage-03 focussed CSV for ``sample_id``."""

    if sample_id in cache:
        return cache[sample_id]

    csv_path = csv_lookup.get(sample_id)
    if csv_path is None:
        LOGGER.warning(
            "No Stage-03 CSV found for sample %s. Skipping associated crops.",
            sample_id,
        )
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - pandas error detail
        LOGGER.error("Failed to load Stage-03 CSV %s: %s", csv_path, exc)
        return None

    cache[sample_id] = df
    return df


def _collect_curated_rows(
    curated_dirs: Iterable[Path],
    label_value: int,
    stage3_cache: Dict[str, pd.DataFrame],
    csv_lookup: Dict[str, Path],
) -> List[dict]:
    """Gather labelled rows for all PNGs inside ``curated_dirs``."""

    rows: List[dict] = []
    for curated_dir in curated_dirs:
        if not curated_dir.exists():
            LOGGER.warning(
                "Curated directory %s does not exist. Skipping.", curated_dir
            )
            continue

        for png_path in sorted(curated_dir.rglob('*.png')):
            png_name = png_path.name
            sample_id = parse_sample_id_from_filename(png_path.stem)
            if not sample_id:
                LOGGER.warning(
                    "Could not parse sample identifier from %s. Skipping.", png_name
                )
                continue

            df = _load_stage3_dataframe(sample_id, stage3_cache, csv_lookup)
            if df is None:
                continue

            matched_row = _match_crop_row(df, png_name)
            if matched_row is None:
                LOGGER.warning(
                    "No matching Stage-03 row for crop %s (sample %s).", png_name, sample_id
                )
                continue

            row_dict = matched_row.to_dict()
            row_dict['label'] = int(label_value)
            rows.append(row_dict)

    return rows


def build_manual_labels_dataframe(
    good_dir: Path,
    bad_dir: Path,
    stage3_root: Path,
    output_csv: Path,
    csv_name: str = 'focussed_and_filtered.csv',
) -> pd.DataFrame:
    """Aggregate Stage-03 rows for curated crops and persist the combined CSV."""

    csv_lookup = discover_stage3_csvs(stage3_root, csv_name)
    stage3_cache: Dict[str, pd.DataFrame] = {}

    labelled_rows = []
    labelled_rows.extend(
        _collect_curated_rows([good_dir], 1, stage3_cache, csv_lookup)
    )
    labelled_rows.extend(
        _collect_curated_rows([bad_dir], 0, stage3_cache, csv_lookup)
    )

    if not labelled_rows:
        LOGGER.warning("No labelled rows were collected. Nothing to write.")
        return pd.DataFrame()

    combined_df = pd.DataFrame(labelled_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    LOGGER.info(
        "Wrote %d labelled rows to %s", len(combined_df), output_csv.as_posix()
    )
    return combined_df


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for rebuilding manual labels."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Stage-03 metrics for curated crops and emit a labelled CSV."
        )
    )
    parser.add_argument(
        '--config',
        default='config/pipeline_config.yml',
        help=(
            'Path to the pipeline config (unused but kept for symmetry with other scripts).'
        ),
    )
    parser.add_argument(
        '--good-dir',
        default='data/training_data/good',
        type=Path,
        help='Directory containing curated "good" crops.',
    )
    parser.add_argument(
        '--bad-dir',
        default='data/training_data/bad',
        type=Path,
        help='Directory containing curated "bad" crops.',
    )
    parser.add_argument(
        '--stage3-root',
        default='data/output',
        type=Path,
        help='Parent directory that contains sample folders with Stage-03 outputs.',
    )
    parser.add_argument(
        '--stage3-csv-name',
        default='focussed_and_filtered.csv',
        help='Name of the Stage-03 CSV to match (default: focussed_and_filtered.csv).',
    )
    parser.add_argument(
        '--output-csv',
        default='data/training_data/manual_labels.csv',
        type=Path,
        help='Destination CSV path for the aggregated rows.',
    )
    return parser


def main() -> None:
    """Parse arguments and run the manual labels aggregation workflow."""
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()

    build_manual_labels_dataframe(
        good_dir=args.good_dir,
        bad_dir=args.bad_dir,
        stage3_root=args.stage3_root,
        output_csv=args.output_csv,
        csv_name=args.stage3_csv_name,
    )


if __name__ == '__main__':  # pragma: no cover - manual invocation entry point
    main()
