# scripts/train_model.py
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import joblib
import cv2
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils import (
    setup_logging, get_config, load_config, get_models_dir, get_training_data_dir
)
from src.utils.common_exceptions import ConfigError, MLModelError, DataValidationError
from src.core.image_io_operations import load_image
import skimage.util

from src.core.particle_operations import crop_rotated
from src.core.segmentation import segment_main_body_from_image_float
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors


logger = logging.getLogger(__name__)

# --- All helper functions (_standardize_training_crop, etc.) remain unchanged ---
def _standardize_training_crop(
    image_crop_gray: np.ndarray,
    system_config: Dict[str, Any]
) -> Optional[np.ndarray]:
    crop_float = skimage.util.img_as_float(image_crop_gray)
    _, filled_mask, _ = segment_main_body_from_image_float(crop_float, system_config)

    if filled_mask is None or filled_mask.sum() == 0:
        logger.warning("Could not segment particle in training crop to standardize orientation.")
        return None

    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    (cx, cy), (w, h), angle = cv2.minAreaRect(largest_contour)

    if w > h:
        w, h = h, w
        angle += 90

    standardized_crop = crop_rotated(image_crop_gray, (cx, cy), (int(w), int(h)), angle, padding=5)
    
    return standardized_crop


def extract_features_for_single_image(
    image_path: str,
    system_config: Dict[str, Any],
    s2_config: Dict[str, Any]
) -> Dict[str, Any]:
    features = {}
    
    image_crop_gray_original = load_image(image_path, mode='gray')
    if image_crop_gray_original is None:
        logger.warning(f"Could not load image {image_path}")
        return None

    image_crop_gray = _standardize_training_crop(image_crop_gray_original, system_config)
    if image_crop_gray is None:
        logger.warning(f"Failed to standardize crop for {image_path}, using original.")
        image_crop_gray = image_crop_gray_original

    crop_float_orig = skimage.util.img_as_float(image_crop_gray)
    raw_mask, filled_mask, _ = segment_main_body_from_image_float(crop_float_orig, system_config)

    if filled_mask is None or raw_mask is None:
        logger.warning(f"Segmentation failed for standardized crop from {image_path}")
        return None

    quality_features = calculate_quality_features(image_crop_gray, image_path, s2_config)
    features.update(quality_features)

    intensity_img_for_props = crop_float_orig.copy()
    if system_config.get('source_image_illumination_type') == 'collimated':
        intensity_img_for_props = 1.0 - intensity_img_for_props
    morphology_features = calculate_detailed_body_morphology(raw_mask, filled_mask, intensity_img_for_props)
    renamed_morphology = {k.replace('_s4', ''): v for k, v in morphology_features.items()}
    features.update(renamed_morphology)

    shape_features = calculate_shape_descriptors(image_crop_gray, filled_mask, s2_config)
    features.update(shape_features)
    
    return features


def load_and_extract_from_directory(
    directory: str,
    label: int,
    system_config: Dict[str, Any],
    s2_config: Dict[str, Any]
) -> pd.DataFrame:
    features_list = []
    logger.info(f"Processing directory for label '{label}': {directory}")
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(directory, filename)
            features = extract_features_for_single_image(image_path, system_config, s2_config)
            if features:
                features_list.append(features)
    
    df = pd.DataFrame(features_list)
    df['label'] = label
    return df


def main():
    # ... (Setup and Data Loading sections are unchanged) ...
    parser = argparse.ArgumentParser(description="Train a new ML model on the rich feature set.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(project_root, "config", "pipeline_config.yml"),
        help="Path to the pipeline configuration YAML file."
    )
    args = parser.parse_args()

    try:
        load_config(args.config)
        setup_logging()
        config = get_config()
        logger.info("Configuration loaded for model training.")
    except Exception as e:
        print(f"FATAL: Error during setup: {e}")
        sys.exit(1)

    s3_config = config.get('stage_03_filtering_and_classification', {})
    s2_config = config.get('stage_02_characterization', {})
    system_config = config.get('system', {})
    training_params = s3_config.get('training_params', {})
    
    training_data_dir = get_training_data_dir()
    good_dir = os.path.join(training_data_dir, training_params.get('good_data_subdir', 'good'))
    bad_dir = os.path.join(training_data_dir, training_params.get('bad_data_subdir', 'bad'))

    good_df = load_and_extract_from_directory(good_dir, 1, system_config, s2_config)
    bad_df = load_and_extract_from_directory(bad_dir, 0, system_config, s2_config)

    if good_df.empty or bad_df.empty:
        raise DataValidationError("One or both training directories are empty or failed processing. Cannot train.")

    full_df = pd.concat([good_df, bad_df], ignore_index=True)
    
    # --- Model Training ---
    y = full_df['label']
    X = full_df.drop(columns=['label'])
    
    X = X.select_dtypes(include=np.number)
    X.fillna(0, inplace=True)
    
    # --- MODIFICATION: Exclude features based on config ---
    prefixes_to_exclude = training_params.get('exclude_feature_prefixes', [])
    if prefixes_to_exclude:
        cols_to_drop = []
        for prefix in prefixes_to_exclude:
            cols_to_drop.extend([col for col in X.columns if col.startswith(prefix)])
        
        # Remove duplicates and drop columns
        cols_to_drop = list(set(cols_to_drop))
        X.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Excluded {len(cols_to_drop)} features based on prefixes: {prefixes_to_exclude}")
    # --- END OF MODIFICATION ---
    
    feature_names = X.columns.tolist()
    logger.info(f"Training model with {len(feature_names)} features.")
    
    # ... (Rest of the script: train/test split, scaling, grid search, evaluation, and saving) ...
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=training_params.get('test_split_size', 0.25),
        random_state=training_params.get('random_state', 42),
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=training_params.get('random_state', 42))
    param_grid_raw = training_params.get('rf_param_grid', {})
    
    param_grid = {}
    for key, value_list in param_grid_raw.items():
        if isinstance(value_list, list):
            param_grid[key] = [None if v == 'null' else v for v in value_list]
        else:
            param_grid[key] = value_list

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f"Best model parameters: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test_scaled)
    logger.info("Classification Report on Test Set:\n" + classification_report(y_test, y_pred, zero_division=0))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    logger.info("--- Top 20 Feature Importances ---")
    logger.info("\n" + importance_df.head(20).to_string())

    models_dir = get_models_dir()
    importance_csv_path = os.path.join(models_dir, "feature_importances.csv")
    importance_df.to_csv(importance_csv_path, index=False)
    logger.info(f"Full feature importance list saved to: {importance_csv_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 20
    top_features = importance_df.head(top_n)
    ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
    ax.invert_yaxis() 
    ax.set_xlabel('Importance (Mean Decrease in Gini Impurity)')
    ax.set_title(f'Top {top_n} Feature Importances for Particle Classification')
    plt.tight_layout()
    
    chart_path = os.path.join(models_dir, "feature_importance_chart.png")
    plt.savefig(chart_path)
    logger.info(f"Feature importance chart saved to: {chart_path}")

    model_filename = s3_config.get('model_filename', 'rf_fem_classifier.joblib')
    scaler_filename = s3_config.get('scaler_filename', 'rf_fem_scaler.joblib')
    feature_names_filename = model_filename.replace(".joblib", "_feature_names.txt")

    joblib.dump(best_model, os.path.join(models_dir, model_filename))
    joblib.dump(scaler, os.path.join(models_dir, scaler_filename))
    with open(os.path.join(models_dir, feature_names_filename), 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
            
    logger.info(f"Model, scaler, and feature names saved successfully to {models_dir}")
    logger.info("--- Model Training Completed ---")

if __name__ == '__main__':
    main()