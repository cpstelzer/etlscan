# src/stages/stage_03_filtering_and_classification.py
import logging
import os
import shutil
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np

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


def _read_feature_names(feature_file: str) -> Optional[List[str]]:
    if not os.path.exists(feature_file):
        logger.warning(
            "Feature-name list missing at %s. Falling back to numeric columns order.",
            feature_file,
        )
        return None

    with open(feature_file, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        logger.warning(
            "Feature-name list at %s is empty. Falling back to numeric columns order.",
            feature_file,
        )
        return None
    return names


def _load_ml_artifacts(s3_config: dict) -> Tuple[object, object, Optional[List[str]]]:
    models_dir = get_models_dir()
    model_filename = s3_config.get("model_filename", "rf_fem_classifier.joblib")
    scaler_filename = s3_config.get("scaler_filename", "rf_fem_scaler.joblib")
    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    feature_file = model_path.replace(".joblib", "_feature_names.txt")

    if not os.path.exists(model_path):
        raise MLModelError(f"Stage 3 model artifact missing: {model_path}")
    if not os.path.exists(scaler_path):
        raise MLModelError(f"Stage 3 scaler artifact missing: {scaler_path}")

    logger.info("Loading Stage 3 classifier from %s", model_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = _read_feature_names(feature_file)
    return model, scaler, feature_names


def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_names: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df.fillna(0, inplace=True)

    if feature_names:
        missing = [name for name in feature_names if name not in numeric_df.columns]
        if missing:
            logger.warning(
                "%d configured features missing from Stage 2 CSV; filling with zeros: %s",
                len(missing),
                ", ".join(missing[:10]),
            )
        feature_df = numeric_df.reindex(columns=list(feature_names), fill_value=0)
        ordered_names = list(feature_names)
    else:
        if numeric_df.empty:
            raise MLModelError("No numeric columns available for ML scoring in Stage 3 input.")
        ordered_names = numeric_df.columns.tolist()
        feature_df = numeric_df

    return feature_df, ordered_names


def _score_particles_with_model(
    df: pd.DataFrame,
    model: object,
    scaler: object,
    feature_names: Optional[Sequence[str]],
) -> pd.Series:
    feature_df, ordered_names = _prepare_feature_matrix(df, feature_names)
    logger.debug("Scoring particles with %d ML features.", len(ordered_names))
    scaled_values = scaler.transform(feature_df.values)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_values)[:, -1]
    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(scaled_values)
        probabilities = 1 / (1 + np.exp(-decision_values))
    else:
        raise MLModelError("Loaded model lacks predict_proba/decision_function for probability scoring.")

    return pd.Series(probabilities, index=df.index, name="ml_probability")


def _log_probability_summary(probabilities: pd.Series, sample_id: str, threshold: float) -> None:
    if probabilities.empty:
        return
    summary = {
        "min": float(probabilities.min()),
        "median": float(probabilities.median()),
        "max": float(probabilities.max()),
        "mean": float(probabilities.mean()),
    }
    logger.info(
        "Stage 3 ML probability summary for %s (threshold %.2f): %s",
        sample_id,
        threshold,
        summary,
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

    output_filename = 'focussed_and_filtered.csv'
    saved_csv_path = save_dataframe_to_csv(accepted_df, output_dir_stage3, output_filename)

    logger.info(
        "--- Stage 3: Filtering & Classification for %s COMPLETED. Saved %d particles. ---",
        sample_id,
        len(accepted_df),
    )
    return saved_csv_path
