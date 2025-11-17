# src/stages/stage_03_filtering_and_classification.py
import logging
import os
import shutil
from typing import Optional

import pandas as pd

from src.utils import get_config, load_dataframe_from_csv, save_dataframe_to_csv
from src.utils.common_exceptions import ConfigError

logger = logging.getLogger(__name__)


def _copy_filtered_crops(
    filtered_df: pd.DataFrame,
    source_dir: str,
    destination_dir: str,
) -> None:
    """Copy the filtered particle crops into the Stage 3 directory."""
    if filtered_df.empty:
        return

    os.makedirs(destination_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for _, row in filtered_df.iterrows():
        crop_filename = row.get('crop_filename')
        if not isinstance(crop_filename, str) or not crop_filename:
            continue

        src_path = os.path.join(source_dir, crop_filename)
        dest_path = os.path.join(destination_dir, crop_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            logger.warning("Missing crop file for Stage 3 copy: %s", src_path)
            missing_count += 1

    logger.info(
        "Copied %d filtered particle crops to %s (%d missing)",
        copied_count,
        destination_dir,
        missing_count,
    )


def run_stage_03_filtering_and_classification(
    input_final_particles_csv_path: str,
    output_dir_stage3: str,
    final_particles_crops_dir: str,
    sample_id: str,
) -> Optional[str]:
    """Apply quality flag filters to Stage 2 final particles."""
    logger.info(
        "--- Stage 3: Filtering & Classification (quality filters only) for Sample ID: %s ---",
        sample_id,
    )

    config = get_config()
    s3_config = config.get('stage_03_filtering_and_classification', {})
    if not s3_config:
        raise ConfigError("Configuration for 'stage_03_filtering_and_classification' not found.")

    final_particles_df = load_dataframe_from_csv(input_final_particles_csv_path)
    if final_particles_df is None or final_particles_df.empty:
        logger.warning(
            "Stage 2 final particles CSV missing or empty: %s. Aborting Stage 3.",
            input_final_particles_csv_path,
        )
        return None

    logger.info("Loaded %d particles from Stage 2 final output.", len(final_particles_df))

    filtered_df = final_particles_df.copy()

    if 'has_black_mask_artifact' in filtered_df.columns:
        mask = filtered_df['has_black_mask_artifact'].fillna(False).astype(bool)
        removed = int(mask.sum())
        filtered_df = filtered_df.loc[~mask]
        logger.info("Removed %d particles with black mask artifacts.", removed)
    else:
        logger.warning("Column 'has_black_mask_artifact' missing; skipping that filter.")

    if 'is_on_crop_boundary' in filtered_df.columns:
        mask = filtered_df['is_on_crop_boundary'].fillna(False).astype(bool)
        removed = int(mask.sum())
        filtered_df = filtered_df.loc[~mask]
        logger.info("Removed %d particles touching crop boundaries.", removed)
    else:
        logger.warning("Column 'is_on_crop_boundary' missing; skipping boundary filter.")

    filtered_df = filtered_df.reset_index(drop=True)

    if filtered_df.empty:
        logger.warning("No particles remain after Stage 3 quality filtering.")

    if 'final_crop_filename' in filtered_df.columns:
        filtered_df['crop_filename'] = filtered_df['final_crop_filename']
    elif 'crop_filename' not in filtered_df.columns:
        filtered_df['crop_filename'] = None
        logger.warning(
            "No crop filename column found in Stage 2 output; downstream stages may fail."
        )

    stage3_crops_dir = os.path.join(output_dir_stage3, 'focussed_and_filtered')
    _copy_filtered_crops(filtered_df, final_particles_crops_dir, stage3_crops_dir)

    output_filename = 'focussed_and_filtered.csv'
    saved_csv_path = save_dataframe_to_csv(filtered_df, output_dir_stage3, output_filename)

    logger.info(
        "--- Stage 3: Filtering & Classification for %s COMPLETED. Saved %d particles. ---",
        sample_id,
        len(filtered_df),
    )
    return saved_csv_path
