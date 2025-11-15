# src/stages/stage_03_filtering_and_classification.py
import os
import pandas as pd
import numpy as np
import logging
import joblib
import shutil
from typing import Optional, List

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv, get_models_dir
)
from src.utils.common_exceptions import MLModelError, ConfigError, DataValidationError
from src.stages.stage_02_characterization import (
    filter_focused_frames,
    select_final_particles_by_area,
)

logger = logging.getLogger(__name__)

# --- NEW HELPER FUNCTION ---
def _save_rejected_crops(
    rejected_df: pd.DataFrame,
    s2_crops_dir: str,
    rejection_dir: str,
    reason: str
):
    """
    Copies the cropped images of rejected particles to a specified directory.
    """
    if rejected_df.empty:
        return
    
    logger.info(f"Saving {len(rejected_df)} crops rejected due to: {reason}")
    os.makedirs(rejection_dir, exist_ok=True)
    
    for _, row in rejected_df.iterrows():
        crop_filename = row.get('crop_filename')
        if not crop_filename or not isinstance(crop_filename, str):
            continue
            
        source_path = os.path.join(s2_crops_dir, crop_filename)
        dest_path = os.path.join(rejection_dir, crop_filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
        else:
            logger.warning(f"Could not find crop file to save as rejected: {source_path}")


def run_stage_03_filtering_and_classification(
    input_s2_csv_path: str,
    output_dir_stage3: str,
    s2_crops_dir: str, # Need access to the crops for debug output
    sample_id: str
) -> Optional[str]:
    """
    Uses the rich feature set from Stage 2 to filter out low-quality particles
    and classify the remaining ones, with optional debug output for rejected particles.
    """
    logger.info(f"--- Stage 3: Filtering & Classification for Sample ID: {sample_id} ---")

    config = get_config()
    s3_config = config.get('stage_03_filtering_and_classification', {})
    if not s3_config:
        raise ConfigError("Configuration for 'stage_03_filtering_and_classification' not found.")
    
    # --- MODIFIED: Get debug config ---
    debug_config = s3_config.get('debug_output', {})
    save_rejected = debug_config.get('save_rejected_images', False)

    s2_features_df = load_dataframe_from_csv(input_s2_csv_path)
    if s2_features_df is None or s2_features_df.empty:
        logger.warning(f"No features loaded from Stage 2 CSV: {input_s2_csv_path}. Aborting Stage 3.")
        return None
        
    # --- 1. Load ML Model, Scaler, and Feature List (unchanged) ---
    models_dir = get_models_dir()
    model_filename = s3_config.get('model_filename', 'rf_fem_classifier.joblib')
    # ... (rest of model loading is the same)
    scaler_filename = s3_config.get('scaler_filename', 'rf_fem_scaler.joblib')
    feature_names_filename = model_filename.replace(".joblib", "_feature_names.txt")
    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    feature_names_path = os.path.join(models_dir, feature_names_filename)
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(feature_names_path, "r") as f:
            feature_names_for_model = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise MLModelError(f"Error loading ML components: {e}")

    # --- 2. Predict for ALL particles first ---
    df_with_predictions = s2_features_df.copy()
    missing_features = [f for f in feature_names_for_model if f not in df_with_predictions.columns]
    if missing_features:
        raise DataValidationError(f"DataFrame is missing features required by the model: {missing_features}")
    
    X_predict = df_with_predictions[feature_names_for_model].fillna(0).values
    X_predict_scaled = scaler.transform(X_predict)
    df_with_predictions['ml_predicted_class'] = model.predict(X_predict_scaled)
    df_with_predictions['ml_predicted_proba'] = model.predict_proba(X_predict_scaled)[:, 1]
    
    logger.info("ML prediction complete for all particles.")

    # --- 3. MODIFIED: Reordered and enhanced filtering logic ---
    current_df = df_with_predictions.copy()
    
    # a. Basic Quality Flags Filter
    quality_filter_config = s3_config.get('quality_flags_filter', {})
    if quality_filter_config.get('filter_black_mask', True) and 'has_black_mask_artifact' in current_df.columns:
        rejected_mask = current_df['has_black_mask_artifact']
        if save_rejected:
            _save_rejected_crops(current_df[rejected_mask], s2_crops_dir, os.path.join(output_dir_stage3, "rejected_by_black_mask"), "black mask artifact")
        current_df = current_df[~rejected_mask]
        logger.info(f"Filter (Black Mask): {len(current_df)} particles remain.")
    
    if quality_filter_config.get('filter_on_boundary', True) and 'is_on_crop_boundary' in current_df.columns:
        rejected_mask = current_df['is_on_crop_boundary'].fillna(False).astype(bool)
        if save_rejected:
            _save_rejected_crops(
                current_df[rejected_mask],
                s2_crops_dir,
                os.path.join(output_dir_stage3, "rejected_by_boundary"),
                "crop touches boundary",
            )
        current_df = current_df[~rejected_mask]
        logger.info(f"Filter (Boundary Touch): {len(current_df)} particles remain.")
         
    # b. Focus-based pruning (moved from Stage 2)
    focus_cfg = config.get('stage_02_characterization', {}).get('focussed_particles', {})
    if focus_cfg.get('enabled', False):
        frame_col = focus_cfg.get('frame_column', 'frame')
        track_col = focus_cfg.get('track_column', 'track_id')
        sharp_col = focus_cfg.get('sharpness_column', 'variance_laplacian')
        area_col = focus_cfg.get('area_column', 'main_area_filled')

        required_cols = [frame_col, track_col, sharp_col]
        missing_cols = [col for col in required_cols if col not in current_df.columns]
        if missing_cols:
            logger.warning(
                "Skipping focus-based pruning because required columns are missing: %s",
                ', '.join(missing_cols)
            )
        else:
            focus_dir = os.path.join(output_dir_stage3, 'focus_filter')
            os.makedirs(focus_dir, exist_ok=True)

            filtered_focus_df, diagnostics_df = filter_focused_frames(
                current_df,
                track_col,
                frame_col,
                sharp_col,
                focus_cfg
            )

            save_dataframe_to_csv(filtered_focus_df, focus_dir, 'focus_peaks.csv')
            save_dataframe_to_csv(diagnostics_df, focus_dir, 'focus_diagnostics.csv')

            current_df = select_final_particles_by_area(
                filtered_focus_df,
                track_col,
                area_col
            )

            logger.info(
                "Focus-based pruning complete: %d particles remain after per-track selection.",
                len(current_df)
            )

    # c. ML Filter (temporarily disabled via config)
    ml_filter_config = s3_config.get('ml_filter', {})
    if ml_filter_config.get('enabled', True):
        threshold = ml_filter_config.get('min_probability_threshold', 0.5)
        rejected_mask = current_df['ml_predicted_proba'] < threshold
        if save_rejected:
            _save_rejected_crops(
                current_df[rejected_mask],
                s2_crops_dir,
                os.path.join(output_dir_stage3, "rejected_by_ml"),
                f"ML probability < {threshold}",
            )
        current_df = current_df[~rejected_mask]
        logger.info(f"Filter (ML): {len(current_df)} particles remain.")
    else:
        logger.info("ML filter disabled via configuration; skipping probability thresholding.")
    
    # --- 4. Final Output ---
    final_good_particles_df = current_df.copy()
    logger.info(f"Final count of good, classified particles: {len(final_good_particles_df)}")
    
    output_filename = f"{sample_id}_stage3_classified_good_particles.csv"
    saved_csv_path = save_dataframe_to_csv(final_good_particles_df, output_dir_stage3, output_filename)

    logger.info(f"--- Stage 3: Filtering & Classification for {sample_id} COMPLETED ---")
    return saved_csv_path