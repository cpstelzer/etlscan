# src/stages/stage_03_filtering_and_classification.py
import logging
import os
import shutil
from typing import List, Optional

import joblib
import pandas as pd

from src.utils import (
    get_config,
    get_models_dir,
    load_dataframe_from_csv,
    save_dataframe_to_csv,
)
from src.utils.common_exceptions import ConfigError, MLModelError

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


def _load_ml_artifacts(s3_config: dict) -> tuple:
    """Load the ML model, scaler, and feature list used during training."""
    models_dir = get_models_dir()
    model_filename = s3_config.get('model_filename', 'rf_fem_classifier.joblib')
    scaler_filename = s3_config.get('scaler_filename', 'rf_fem_scaler.joblib')
    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    feature_names_filename = f"{os.path.splitext(model_filename)[0]}_feature_names.txt"
    feature_names_path = os.path.join(models_dir, feature_names_filename)

    if not os.path.exists(model_path):
        raise MLModelError(
            f"ML model file not found at {model_path}. Train the model or update the configuration."
        )
    if not os.path.exists(scaler_path):
        raise MLModelError(
            f"Scaler file not found at {scaler_path}. Ensure training saved the scaler correctly."
        )
    if not os.path.exists(feature_names_path):
        raise MLModelError(
            "Feature names file missing. Expected at "
            f"{feature_names_path}. Re-train or copy the feature list from training output."
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise MLModelError(f"Failed to load ML model from {model_path}: {exc}") from exc

    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise MLModelError(f"Failed to load scaler from {scaler_path}: {exc}") from exc

    with open(feature_names_path, 'r', encoding='utf-8') as handle:
        feature_names: List[str] = [line.strip() for line in handle if line.strip()]

    if not feature_names:
        raise MLModelError(
            f"Feature names list at {feature_names_path} is empty; cannot align ML features."
        )

    return model, scaler, feature_names


def _prepare_ml_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Align dataframe columns with feature names used during training."""
    missing_columns = [col for col in feature_names if col not in df.columns]
    if missing_columns:
        sample_missing = ', '.join(missing_columns[:10])
        logger.warning(
            "Missing %d ML feature columns in Stage 2 output (showing up to 10): %s. Filling with zeros.",
            len(missing_columns),
            sample_missing,
        )

    aligned = df.reindex(columns=feature_names)
    aligned = aligned.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return aligned


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

    initial_particle_count = len(final_particles_df)
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

    quality_removed_count = initial_particle_count - len(filtered_df)

    if filtered_df.empty:
        logger.warning("No particles remain after Stage 3 quality filtering.")

    ml_rejected_count = 0

    if not filtered_df.empty:
        model, scaler, feature_names = _load_ml_artifacts(s3_config)
        feature_matrix = _prepare_ml_features(filtered_df, feature_names)

        try:
            scaled_features = scaler.transform(feature_matrix)
        except Exception as exc:  # pragma: no cover - defensive
            raise MLModelError(f"Scaler failed to transform Stage 3 features: {exc}") from exc

        if not hasattr(model, 'predict_proba'):
            raise MLModelError("Loaded ML model does not support probability predictions.")

        probabilities = model.predict_proba(scaled_features)[:, 1]
        filtered_df['ml_probability'] = probabilities

        ml_filter_config = s3_config.get('ml_filter', {})
        probability_threshold = ml_filter_config.get('min_probability_threshold', 0.5)
        filtered_df['ml_label'] = (filtered_df['ml_probability'] >= probability_threshold).astype(int)

        ml_mask = filtered_df['ml_probability'] >= probability_threshold
        ml_rejected_count = int((~ml_mask).sum())
        filtered_df = filtered_df.loc[ml_mask].reset_index(drop=True)

        if filtered_df.empty:
            logger.warning(
                "All particles (%d) were rejected by the ML filter (threshold=%.2f).",
                ml_rejected_count,
                probability_threshold,
            )
        else:
            logger.info(
                "Applied ML filter with threshold %.2f. %d particles rejected, %d remain.",
                probability_threshold,
                ml_rejected_count,
                len(filtered_df),
            )

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
        "Stage 3 summary for %s -> start: %d, quality removed: %d, ML removed: %d, saved: %d",
        sample_id,
        initial_particle_count,
        quality_removed_count,
        ml_rejected_count,
        len(filtered_df),
    )
    return saved_csv_path
