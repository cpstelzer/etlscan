# src/stages/stage_03_filtering_and_classification.py
import logging
import os
import shutil
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.utils import (
    get_config,
    get_models_dir,
    load_dataframe_from_csv,
    save_dataframe_to_csv,
)
from src.utils.common_exceptions import ConfigError, DataValidationError, MLModelError

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

# Raw metrics that scale with particle size and should be removed when
# ``size_invariant_features`` is enabled. Each entry lists normalized or
# shape-centric counterparts that remain available as replacements.
SIZE_DEPENDENT_RAW_FEATURE_RULES = {
    # Stage 1 geometry-heavy metrics
    'bounding_w': ['compactness_rect', 'extent_raw'],
    'bounding_h': ['compactness_rect', 'extent_raw'],
    'rot_rect_width': ['elongation_factor', 'compactness_rect'],
    'rot_rect_height': ['elongation_factor', 'compactness_rect'],
    'rot_rect_area': ['compactness_rect'],
    's1_area': ['s1_solidity', 's1_circularity'],
    's1_perimeter': ['s1_circularity', 'heywood_circularity_factor'],
    's1_feret_diameter': ['elongation_factor'],
    's1_equivalent_diameter': ['heywood_circularity_factor'],

    # Stage 2 morphology metrics (raw size driven)
    'main_area_raw': ['solidity', 'heywood_circularity_factor', 'elongation_factor'],
    'main_area_filled': ['solidity', 'heywood_circularity_factor', 'elongation_factor'],
    'equivalent_diameter_raw': ['heywood_circularity_factor', 'elongation_factor'],
    'major_axis_length_raw': ['elongation_factor', 'compactness_factor_ni'],
    'minor_axis_length_raw': ['elongation_factor', 'compactness_factor_ni'],
    'extent_raw': ['compactness_rect', 'solidity'],
    'eccentricity_raw': ['elongation_factor', 's1_eccentricity'],
}


def _get_lbp_feature_count_from_config(config: Optional[dict] = None) -> int:
    """Return how many LBP columns Stage 2 should have produced."""

    if config is None:
        config = get_config()

    stage2_config = config.get('stage_02_characterization', {})
    shape_config = stage2_config.get('shape_descriptors', {})
    lbp_radii = shape_config.get('lbp_radii', [1, 3]) or []
    lbp_points = shape_config.get('lbp_points', 8)

    if not lbp_radii or lbp_points is None:
        return 0

    return len(lbp_radii) * (lbp_points + 2)


def _get_expected_numeric_feature_columns(config: Optional[dict] = None) -> List[str]:
    """Return the numeric schema, expanding LBP columns from the Stage 2 config."""

    lbp_feature_count = _get_lbp_feature_count_from_config(config)

    expected_columns = list(BASE_NUMERIC_FEATURE_COLUMNS)

    if lbp_feature_count:
        expected_columns.extend([f'lbp_{i}' for i in range(lbp_feature_count)])

    return expected_columns


def _infer_lbp_feature_count_from_df(df: pd.DataFrame) -> int:
    """Infer how many LBP columns exist in the Stage 2 output."""

    lbp_columns = [col for col in df.columns if col.startswith('lbp_')]
    if not lbp_columns:
        return 0

    indices: List[int] = []
    for col in lbp_columns:
        try:
            indices.append(int(col.split('_')[1]))
        except (IndexError, ValueError):
            # Unexpected naming; fall back to counting columns instead of crashing.
            continue

    if indices:
        return max(indices) + 1
    return len(lbp_columns)


def apply_size_invariant_feature_strategy(
    df: pd.DataFrame,
    enabled: bool,
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop size-dependent raw metrics when size-invariant mode is enabled.

    All known raw area/length/perimeter metrics are removed to enforce scale
    invariance. When normalized counterparts (e.g., solidity, circularity,
    elongation, compactness) are present, they are preferred; when no
    normalized alternative exists, the raw feature is still removed unless the
    trained model explicitly requires it via ``feature_names``.
    """

    if not enabled:
        return df, []

    working_df = df.copy()
    dropped: List[str] = []

    for raw_feature, normalized_candidates in SIZE_DEPENDENT_RAW_FEATURE_RULES.items():
        if raw_feature not in working_df.columns:
            continue
        if feature_names and raw_feature in feature_names:
            logger.debug(
                "Preserving size-dependent feature '%s' because the trained model artifact expects it.",
                raw_feature,
            )
            continue

        has_normalized_alternative = any(
            candidate in working_df.columns for candidate in normalized_candidates
        )

        working_df = working_df.drop(columns=[raw_feature])
        dropped.append(raw_feature)

        if not has_normalized_alternative:
            logger.debug(
                "Dropping size-dependent feature '%s' despite no normalized alternative;"
                " keeping size invariance strict.",
                raw_feature,
            )

    return working_df, dropped


def _validate_required_feature_columns(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> None:
    """Ensure Stage 3 receives the expected numeric feature set."""

    expected_columns = _get_expected_numeric_feature_columns(config)
    missing_base = [col for col in BASE_NUMERIC_FEATURE_COLUMNS if col not in df.columns]
    if missing_base:
        raise DataValidationError(
            "Stage 3 cannot proceed because the following feature columns are missing: "
            + ', '.join(sorted(missing_base))
        )

    expected_lbp_count = len([col for col in expected_columns if col.startswith('lbp_')])
    observed_lbp_count = _infer_lbp_feature_count_from_df(df)

    if expected_lbp_count and observed_lbp_count < expected_lbp_count:
        missing_lbp = [
            f'lbp_{i}'
            for i in range(expected_lbp_count)
            if f'lbp_{i}' not in df.columns
        ]
        raise DataValidationError(
            "Stage 3 cannot proceed because the following LBP columns are missing: "
            + ', '.join(sorted(missing_lbp))
        )

    # Stage 2 outputs might contain additional lbp_* columns if it was run with a
    # more detailed configuration than the one currently loaded. Those extras do
    # not prevent Stage 3 from running because downstream ML alignment uses the
    # persisted training feature list.


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
    destination_dirs: List[str],
) -> None:
    """Copy the filtered particle crops into one or more Stage 3 directories."""
    if filtered_df.empty or not destination_dirs:
        return

    unique_destinations = [
        dest for dest in dict.fromkeys(destination_dirs) if isinstance(dest, str) and dest
    ]
    if not unique_destinations:
        return

    for destination_dir in unique_destinations:
        os.makedirs(destination_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for _, row in filtered_df.iterrows():
        crop_filename = row.get('crop_filename')
        if not isinstance(crop_filename, str) or not crop_filename:
            continue

        src_path = os.path.join(source_dir, crop_filename)

        if os.path.exists(src_path):
            for destination_dir in unique_destinations:
                dest_path = os.path.join(destination_dir, crop_filename)
                shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            logger.warning("Missing crop file for Stage 3 copy: %s", src_path)
            missing_count += 1

    logger.info(
        "Copied %d filtered particle crops to %s (%d missing)",
        copied_count,
        ', '.join(unique_destinations),
        missing_count,
    )


def _ensure_crop_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that downstream copy/CSV logic has a crop filename column."""

    result_df = df.copy()
    if 'final_crop_filename' in result_df.columns:
        result_df['crop_filename'] = result_df['final_crop_filename']
    elif 'crop_filename' not in result_df.columns:
        result_df['crop_filename'] = None
        logger.warning(
            "No crop filename column found in Stage 2 output; downstream stages may fail."
        )
    return result_df


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


def _prepare_ml_features(
    df: pd.DataFrame,
    feature_names: List[str],
    size_invariant_enabled: bool = False,
) -> pd.DataFrame:
    """Align dataframe columns with feature names used during training."""

    working_df = df
    if size_invariant_enabled:
        working_df, dropped = apply_size_invariant_feature_strategy(
            working_df, enabled=True, feature_names=feature_names
        )
        if dropped:
            logger.debug(
                "Size-invariant mode enabled; dropped raw size features not used by the model: %s",
                ', '.join(sorted(dropped)),
            )
        elif any(raw in feature_names for raw in SIZE_DEPENDENT_RAW_FEATURE_RULES):
            logger.warning(
                "Size-invariant mode requested, but the trained feature list still expects raw size metrics."
            )

    missing_columns = [col for col in feature_names if col not in working_df.columns]
    if missing_columns:
        sample_missing = ', '.join(missing_columns[:10])
        logger.warning(
            "Missing %d ML feature columns in Stage 2 output (showing up to 10): %s. Filling with zeros.",
            len(missing_columns),
            sample_missing,
        )

    aligned = working_df.reindex(columns=feature_names)
    aligned = aligned.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return aligned


def _score_particles_with_model(
    df: pd.DataFrame,
    model,
    scaler,
    feature_names: List[str],
    size_invariant_enabled: bool = False,
) -> np.ndarray:
    """Scale Stage 3 features and compute positive-class probabilities."""

    feature_matrix = _prepare_ml_features(df, feature_names, size_invariant_enabled=size_invariant_enabled)

    try:
        scaled_features = scaler.transform(feature_matrix)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise MLModelError(f"Scaler failed to transform Stage 3 features: {exc}") from exc

    if not hasattr(model, 'predict_proba'):
        raise MLModelError("Loaded ML model does not support probability predictions.")

    probabilities = model.predict_proba(scaled_features)[:, 1]
    return probabilities


def _log_probability_summary(
    probabilities: np.ndarray, sample_id: str, threshold: float
) -> None:
    """Emit a concise summary of the ML probability distribution for diagnostics."""

    if probabilities.size == 0:
        return

    summary = {
        'min': float(np.min(probabilities)),
        'max': float(np.max(probabilities)),
        'mean': float(np.mean(probabilities)),
        'median': float(np.median(probabilities)),
        'threshold': float(threshold),
    }
    logger.info("Stage 3 ML probability summary for %s: %s", sample_id, summary)


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
    size_invariant_enabled = bool(s3_config.get('size_invariant_features', False))

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
    quality_filtered_df = final_particles_df.copy()

    if 'has_black_mask_artifact' in quality_filtered_df.columns:
        mask = quality_filtered_df['has_black_mask_artifact'].fillna(False).astype(bool)
        removed = int(mask.sum())
        quality_filtered_df = quality_filtered_df.loc[~mask]
        logger.info("Removed %d particles with black mask artifacts.", removed)
    else:
        logger.warning("Column 'has_black_mask_artifact' missing; skipping that filter.")

    if 'is_on_crop_boundary' in quality_filtered_df.columns:
        mask = quality_filtered_df['is_on_crop_boundary'].fillna(False).astype(bool)
        removed = int(mask.sum())
        quality_filtered_df = quality_filtered_df.loc[~mask]
        logger.info("Removed %d particles touching crop boundaries.", removed)
    else:
        logger.warning("Column 'is_on_crop_boundary' missing; skipping boundary filter.")

    quality_filtered_df = quality_filtered_df.reset_index(drop=True)

    quality_removed_count = initial_particle_count - len(quality_filtered_df)

    if quality_filtered_df.empty:
        logger.warning("No particles remain after Stage 3 quality filtering.")

    ml_rejected_count = 0
    ml_filter_cfg = s3_config.get('ml_filter', {})
    probability_threshold = float(ml_filter_cfg.get('min_probability_threshold', 0.5))

    ml_classified_df = quality_filtered_df.copy()

    if not ml_classified_df.empty:
        try:
            model, scaler, feature_names = _load_ml_artifacts(s3_config)
        except MLModelError as exc:
            logger.error("Failed to load ML artifacts for Stage 3: %s", exc)
            raise

        probabilities = _score_particles_with_model(
            ml_classified_df,
            model,
            scaler,
            feature_names,
            size_invariant_enabled=size_invariant_enabled,
        )
        ml_classified_df['ml_probability'] = probabilities
        ml_classified_df['ml_label'] = np.where(
            ml_classified_df['ml_probability'] >= probability_threshold, 'accept', 'reject'
        )
        _log_probability_summary(probabilities, sample_id, probability_threshold)
        label_counts = ml_classified_df['ml_label'].value_counts(dropna=False).to_dict()
        logger.info(
            "Stage 3 ML label distribution before thresholding for %s: %s",
            sample_id,
            label_counts,
        )
        accepted_mask = ml_classified_df['ml_label'] == 'accept'
        ml_rejected_count = int((~accepted_mask).sum())
        accepted_df = ml_classified_df.loc[accepted_mask].reset_index(drop=True)
    else:
        ml_classified_df['ml_probability'] = pd.Series(dtype=float)
        ml_classified_df['ml_label'] = pd.Series(dtype=str)
        accepted_df = ml_classified_df

    if accepted_df.empty:
        logger.warning(
            "All particles rejected by Stage 3 ML filter for sample %s (threshold %.2f).",
            sample_id,
            probability_threshold,
        )

    legacy_stage3_crops_dir = os.path.join(output_dir_stage3, 'focussed_and_filtered')
    filtered_good_crops_dir = os.path.join(output_dir_stage3, 'filtered_good')

    focussed_df_for_outputs = _ensure_crop_filename_column(quality_filtered_df)
    _copy_filtered_crops(
        focussed_df_for_outputs,
        final_particles_crops_dir,
        [legacy_stage3_crops_dir],
    )
    focussed_csv_df = _rename_metadata_columns(focussed_df_for_outputs)
    legacy_output_filename = 'focussed_and_filtered.csv'
    legacy_csv_path = save_dataframe_to_csv(
        focussed_csv_df,
        output_dir_stage3,
        legacy_output_filename,
    )

    accepted_df_for_outputs = _ensure_crop_filename_column(accepted_df)
    _copy_filtered_crops(
        accepted_df_for_outputs,
        final_particles_crops_dir,
        [filtered_good_crops_dir],
    )
    filtered_good_csv_df = _rename_metadata_columns(accepted_df_for_outputs)

    filtered_good_filename = 'filtered_good.csv'
    filtered_good_csv_path = save_dataframe_to_csv(
        filtered_good_csv_df,
        output_dir_stage3,
        filtered_good_filename,
    )

    logger.info(
        "Stage 3 summary for %s -> start: %d, quality removed: %d, ML removed: %d, saved: %d",
        sample_id,
        initial_particle_count,
        quality_removed_count,
        ml_rejected_count,
        len(accepted_df),
    )
    if legacy_csv_path:
        logger.debug(
            "Legacy Stage 3 CSV now captures pre-ML filtering results at %s", legacy_csv_path
        )
    return filtered_good_csv_path
