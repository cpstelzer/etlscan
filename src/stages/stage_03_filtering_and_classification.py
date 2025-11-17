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
from src.utils import get_config, load_dataframe_from_csv, save_dataframe_to_csv
from src.utils.common_exceptions import ConfigError, DataValidationError

logger = logging.getLogger(__name__)


BASE_NUMERIC_FEATURE_COLUMNS: List[str] = [
    # Stage 1 geometric/positional columns
    'frame', 'X', 'Y', 'x_mm', 'y_mm', 'z_mm', 'dioptries',
    'local_threshold_value', 'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h',
    'rot_rect_width', 'rot_rect_height', 'rot_rect_angle', 'rot_rect_area',
    's1_area', 's1_perimeter', 's1_orientation', 's1_aspect_ratio', 's1_solidity',
    's1_eccentricity', 's1_feret_diameter', 's1_circularity', 's1_equivalent_diameter',
    # Stage 2 morphology metrics
    'main_area_raw', 'solidity', 'compactness_rect', 'extent_raw', 'eccentricity_raw',
    'equivalent_diameter_raw', 'major_axis_length_raw', 'minor_axis_length_raw',
    'main_area_filled', 'porosity', 'hu_moment1', 'moment_of_inertia_xx',
    'moment_of_inertia_yy', 'moment_of_inertia_xy', 'type_factor',
    'heywood_circularity_factor', 'elongation_factor', 'compactness_factor_ni',
    # Stage 2 quality metrics
    'variance_laplacian', 'tenengrad_variance', 'energy_laplacian', 'energy_gradient',
]

METADATA_COLUMNS: List[str] = [
    'SAMPLE_ID', 'source_image_filename', 'crop_filename', 'final_crop_filename', 'track_id'
]
METADATA_PREFIX = 'meta__'


def _get_expected_numeric_feature_columns(config: Optional[dict] = None) -> List[str]:
    """Return the numeric schema, expanding LBP columns from the Stage 2 config."""

    if config is None:
        config = get_config()

    stage2_config = config.get('stage_02_characterization', {})
    shape_config = stage2_config.get('shape_descriptors', {})
    lbp_radii = shape_config.get('lbp_radii', [1, 3])
    lbp_points = shape_config.get('lbp_points', 8)

    expected_columns = list(BASE_NUMERIC_FEATURE_COLUMNS)

    if lbp_radii and lbp_points is not None:
        lbp_feature_count = len(lbp_radii) * (lbp_points + 2)
        expected_columns.extend([f'lbp_{i}' for i in range(lbp_feature_count)])

    return expected_columns


def _validate_required_feature_columns(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> None:
    """Ensure Stage 3 receives the expected numeric feature set."""

    expected_columns = _get_expected_numeric_feature_columns(config)
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise DataValidationError(
            "Stage 3 cannot proceed because the following feature columns are missing: "
            + ', '.join(sorted(missing))
        )


def _rename_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix metadata/identifier columns so ML preprocessing can ignore them."""

    rename_map = {}
    for col in METADATA_COLUMNS:
        if col in df.columns and not col.startswith(METADATA_PREFIX):
            rename_map[col] = f"{METADATA_PREFIX}{col}"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


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

    _validate_required_feature_columns(final_particles_df, config=config)
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

    ml_filter_cfg = s3_config.get('ml_filter', {})
    probability_threshold = float(ml_filter_cfg.get('min_probability_threshold', 0.5))

    try:
        model, scaler, feature_names = _load_ml_artifacts(s3_config)
    except MLModelError as exc:
        logger.error("Failed to load ML artifacts for Stage 3: %s", exc)
        raise

    if not filtered_df.empty:
        probabilities = _score_particles_with_model(filtered_df, model, scaler, feature_names)
        filtered_df['ml_probability'] = probabilities
        filtered_df['ml_label'] = np.where(
            filtered_df['ml_probability'] >= probability_threshold, 'accept', 'reject'
        )
        _log_probability_summary(probabilities, sample_id, probability_threshold)
        label_counts = filtered_df['ml_label'].value_counts(dropna=False).to_dict()
        logger.info(
            "Stage 3 ML label distribution before thresholding for %s: %s",
            sample_id,
            label_counts,
        )
    else:
        filtered_df['ml_probability'] = pd.Series(dtype=float)
        filtered_df['ml_label'] = pd.Series(dtype=str)

    accepted_mask = filtered_df['ml_label'] == 'accept'
    accepted_df = filtered_df.loc[accepted_mask].reset_index(drop=True)

    if accepted_df.empty:
        logger.warning(
            "All particles rejected by Stage 3 ML filter for sample %s (threshold %.2f).",
            sample_id,
            probability_threshold,
        )

    stage3_crops_dir = os.path.join(output_dir_stage3, 'focussed_and_filtered')
    _copy_filtered_crops(accepted_df, final_particles_crops_dir, stage3_crops_dir)

    filtered_df = _rename_metadata_columns(filtered_df)

    output_filename = 'focussed_and_filtered.csv'
    saved_csv_path = save_dataframe_to_csv(accepted_df, output_dir_stage3, output_filename)

    logger.info(
        "Stage 3 summary for %s -> start: %d, quality removed: %d, ML removed: %d, saved: %d",
        sample_id,
        initial_particle_count,
        quality_removed_count,
        ml_rejected_count,
        len(filtered_df),
    )
    return saved_csv_path
