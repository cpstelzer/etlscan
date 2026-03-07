"""

scripts/train_model.py will merge the labeled rows, normalize the numeric features, 
sweep the RandomForest hyperparameters, and then save the trained model, scaler, 
feature-importance chart, and feature-name list under models/. 

These artifacts are what Stage 03 consumes during normal pipeline runs, so rerunning 
the pipeline afterward will automatically pick up the new classifier. Keep your repo’s 
config/pipeline_config.yml handy, because that file tells the trainer where to store/load 
the artifacts and which Stage 02 feature definitions to use.

If you later curate additional batches, simply append their Stage‑03 CSVs (or a regenerated 
manual_labels.csv) to the --csv-files list so the trainer can see all labeled rows in one go.

Usage example:
python scripts/train_model.py --config config/pipeline_config.yml --csv-files C:/sandbox/data/training_data/manual_labels.csv --label-column label


"""

# scripts/train_model.py
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import logging
import joblib
import cv2
from typing import Dict, Any, Optional, List, Tuple

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
    setup_logging, load_config_once, get_models_dir, get_training_data_dir
)
from src.utils.config_loader import get_unified_segmentation_params
from src.utils.common_exceptions import DataValidationError
from src.core.image_io_operations import load_image
import skimage.util

from src.core.particle_operations import crop_rotated
from src.core.segmentation import segment_main_body_from_image_float
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors
from src.stages.stage_03_filtering_and_classification import apply_size_invariant_feature_strategy


logger = logging.getLogger(__name__)

# --- All helper functions (_standardize_training_crop, etc.) remain unchanged ---
def _standardize_training_crop(
    image_crop_gray: np.ndarray,
    segmentation_params: Dict[str, Any]
) -> Optional[np.ndarray]:
    """Rotate and pad a crop so that the main body is upright for feature extraction."""
    crop_float = skimage.util.img_as_float(image_crop_gray)
    _, filled_mask, _ = segment_main_body_from_image_float(crop_float, segmentation_params)

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
    segmentation_params: Dict[str, Any],
    s2_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute quality, morphology, and shape features for one training crop."""
    features = {}
    
    image_crop_gray_original = load_image(image_path, mode='gray')
    if image_crop_gray_original is None:
        logger.warning(f"Could not load image {image_path}")
        return None

    image_crop_gray = _standardize_training_crop(image_crop_gray_original, segmentation_params)
    if image_crop_gray is None:
        logger.warning(f"Failed to standardize crop for {image_path}, using original.")
        image_crop_gray = image_crop_gray_original

    crop_float_orig = skimage.util.img_as_float(image_crop_gray)
    raw_mask, filled_mask, _ = segment_main_body_from_image_float(crop_float_orig, segmentation_params)

    if filled_mask is None or raw_mask is None:
        logger.warning(f"Segmentation failed for standardized crop from {image_path}")
        return None

    quality_features = calculate_quality_features(image_crop_gray, image_path, s2_config)
    features.update(quality_features)

    intensity_img_for_props = crop_float_orig.copy()
    if segmentation_params.get('source_image_illumination_type') == 'collimated':
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
    segmentation_params: Dict[str, Any],
    s2_config: Dict[str, Any]
) -> pd.DataFrame:
    """Load all images in a directory and return a labelled feature dataframe."""
    features_list = []
    logger.info(f"Processing directory for label '{label}': {directory}")
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(directory, filename)
            features = extract_features_for_single_image(image_path, segmentation_params, s2_config)
            if features:
                features_list.append(features)

    df = pd.DataFrame(features_list)
    df['label'] = label
    return df


def _normalize_sidecar_key(key: str) -> str:
    """Normalize potential sidecar keys for consistent lookup."""
    normalized = key.strip()
    if not normalized:
        return normalized
    normalized = normalized.replace('\\', os.sep)
    try:
        normalized = os.path.normpath(normalized)
    except Exception:
        # If the key cannot be normalized as a path, fall back to the stripped version
        pass
    return normalized


def _load_label_sidecar(sidecar_path: str) -> Dict[str, Any]:
    """Return a mapping from CSV identifiers to labels with normalized keys."""
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Label sidecar not found: {sidecar_path}")

    _, ext = os.path.splitext(sidecar_path)
    ext = ext.lower()
    if ext == '.json':
        with open(sidecar_path, 'r') as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise DataValidationError("Label sidecar JSON must contain an object mapping filename -> label.")
        raw_mapping = mapping
    
    else:
        mapping_df = pd.read_csv(sidecar_path)
        column_candidates = [col for col in mapping_df.columns if col.lower() in {'source', 'filename', 'file', 'csv'}]
        if not column_candidates or 'label' not in mapping_df.columns:
            raise DataValidationError(
                "Label sidecar CSV must contain a 'label' column and one of: source, filename, file, csv"
            )
        key_column = column_candidates[0]
        raw_mapping = {str(row[key_column]): row['label'] for _, row in mapping_df.iterrows()}

    normalized_mapping: Dict[str, Any] = {}
    for key, value in raw_mapping.items():
        if key is None:
            continue
        str_key = str(key).strip()
        if not str_key:
            continue
        normalized_mapping[str_key] = value
        normalized_key = _normalize_sidecar_key(str_key)
        normalized_mapping[normalized_key] = value
    return normalized_mapping


def _generate_label_key_candidates(csv_path: str, sidecar_dir: Optional[str]) -> List[str]:
    """Return possible keys that may match an entry in the sidecar for this CSV."""
    absolute_path = os.path.abspath(csv_path)
    candidates = [absolute_path]

    try:
        rel_to_cwd = os.path.relpath(absolute_path)
        candidates.append(rel_to_cwd)
    except ValueError:
        # Different drives on Windows, ignore
        rel_to_cwd = None

    if sidecar_dir:
        try:
            rel_to_sidecar = os.path.relpath(absolute_path, start=sidecar_dir)
            candidates.append(rel_to_sidecar)
        except ValueError:
            # Different drives on Windows, ignore
            pass

    basename = os.path.basename(csv_path)
    base_no_ext = os.path.splitext(basename)[0]
    parent = os.path.basename(os.path.dirname(csv_path))

    candidates.extend([
        basename,
        base_no_ext,
        os.path.join(parent, basename) if parent else None,
        os.path.join(parent, base_no_ext) if parent else None,
    ])

    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def _load_feature_table_from_csv(
    csv_paths: List[str],
    label_column: str,
    labels_sidecar: Optional[str]
) -> pd.DataFrame:
    """Load training rows from one or more CSVs and ensure a label column is present."""
    if not csv_paths:
        raise DataValidationError("At least one focussed_and_filtered.csv file must be provided.")

    label_lookup = {}
    sidecar_dir: Optional[str] = None
    if labels_sidecar:
        label_lookup = _load_label_sidecar(labels_sidecar)
        sidecar_dir = os.path.dirname(os.path.abspath(labels_sidecar))

    frames = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)

        if label_column in df.columns:
            df['label'] = df[label_column]
            if label_column != 'label':
                df = df.drop(columns=[label_column])
        elif label_lookup:
            candidate_keys = _generate_label_key_candidates(csv_path, sidecar_dir)
            assigned_label = None
            for candidate in candidate_keys:
                normalized_candidate = _normalize_sidecar_key(candidate)
                if candidate in label_lookup:
                    assigned_label = label_lookup[candidate]
                    break
                if normalized_candidate in label_lookup:
                    assigned_label = label_lookup[normalized_candidate]
                    break

            if assigned_label is None:
                raise DataValidationError(
                    "Label sidecar does not contain an entry for any of the keys "
                    f"{candidate_keys} (from {csv_path})."
                )
            df['label'] = assigned_label
        else:
            raise DataValidationError(
                f"No label column '{label_column}' present in {csv_path} and no sidecar provided."
            )

        frames.append(df)

    if not frames:
        raise DataValidationError("No rows were loaded from the provided CSV files.")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def _normalize_feature_table(df: pd.DataFrame, label_column: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and labels, drop identifiers, and impute missing numeric values."""
    if label_column not in df.columns:
        raise DataValidationError(f"Label column '{label_column}' not found in data frame.")

    y_raw = df[label_column]
    y = pd.to_numeric(y_raw, errors='coerce')
    if y.isna().any():
        raise DataValidationError("Label column contains non-numeric values. Ensure labels are 0/1.")

    X = df.drop(columns=[label_column])
    id_like_cols = [
        col for col in X.columns
        if col.lower() in {'id', 'particle_id', 'sample_id', 'image_id', 'filename', 'file', 'source'}
        or col.lower().endswith('_id')
    ]
    if id_like_cols:
        X = X.drop(columns=id_like_cols)

    numeric_X = X.select_dtypes(include=[np.number]).copy()
    numeric_X.fillna(0, inplace=True)

    if numeric_X.empty:
        raise DataValidationError("No numeric feature columns remain after normalization.")

    return numeric_X, y


def main():
    """Entry point for training the RandomForest model or image-based feature extraction."""
    parser = argparse.ArgumentParser(description="Train a new ML model on the rich feature set.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(project_root, "config", "pipeline_config.yml"),
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "--csv-files",
        nargs='+',
        help="One or more focussed_and_filtered.csv files to use for training."
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default='label',
        help="Name of the label column within the focussed_and_filtered.csv files."
    )
    parser.add_argument(
        "--labels-sidecar",
        type=str,
        help="Optional JSON/CSV mapping from csv basename to label when no label column is present.",
    )
    parser.add_argument(
        "--use-image-directories",
        action='store_true',
        help=(
            "Extract features directly from image directories instead of CSVs. "
            "When --csv-files is omitted, this mode is used automatically."
        ),
    )
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=None,
        help=(
            "Directory containing training CSV files (defaults to data/training_data under the project base). "
            "If --csv-files are not provided, all CSVs in this directory will be used."
        ),
    )
    parser.add_argument(
        "--good-image-dir",
        type=str,
        help="Path to directory with GOOD training crops (defaults to <training-data-dir>/<good_data_subdir>).",
    )
    parser.add_argument(
        "--bad-image-dir",
        type=str,
        help="Path to directory with BAD training crops (defaults to <training-data-dir>/<bad_data_subdir>).",
    )
    args = parser.parse_args()

    try:
        config = load_config_once(args.config)
        setup_logging(config)
        logger.info("Configuration loaded for model training.")
    except Exception as e:
        print(f"FATAL: Error during setup: {e}")
        sys.exit(1)

    s3_config = config.get('stage_03_filtering_and_classification', {})
    s2_config = config.get('stage_02_characterization', {})
    segmentation_params = get_unified_segmentation_params(config)
    logger.info(
        "Loaded unified segmentation parameters for training crops; aligns with Stage 02/04 processing."
    )
    training_params = s3_config.get('training_params', {})
    
    csv_files: List[str] = args.csv_files or []
    use_image_mode = args.use_image_directories or not csv_files

    if use_image_mode and csv_files:
        logger.info("--use-image-directories specified; ignoring %d CSV file(s).", len(csv_files))

    if use_image_mode:
        training_data_dir = args.training_data_dir or get_training_data_dir(config)
        if not os.path.isdir(training_data_dir):
            raise DataValidationError(
                f"Training data directory not found for image training: {training_data_dir}"
            )

        def _resolve_image_dir(directory_arg: Optional[str], fallback_subdir: str) -> str:
            """Resolve an image directory from a CLI override or default subdirectory."""
            base_dir = directory_arg or os.path.join(training_data_dir, fallback_subdir)
            return base_dir

        good_dir = _resolve_image_dir(args.good_image_dir, training_params.get('good_data_subdir', 'good'))
        bad_dir = _resolve_image_dir(args.bad_image_dir, training_params.get('bad_data_subdir', 'bad'))

        for label_name, dir_path in [('good', good_dir), ('bad', bad_dir)]:
            if not os.path.isdir(dir_path):
                raise DataValidationError(
                    f"Image directory for label '{label_name}' not found: {dir_path}"
                )
            has_images = any(
                name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
                for name in os.listdir(dir_path)
            )
            if not has_images:
                raise DataValidationError(
                    f"Image directory for label '{label_name}' contains no images: {dir_path}"
                )

        good_df = load_and_extract_from_directory(good_dir, 1, segmentation_params, s2_config)
        bad_df = load_and_extract_from_directory(bad_dir, 0, segmentation_params, s2_config)

        full_df = pd.concat([good_df, bad_df], ignore_index=True)
    else:
        if args.training_data_dir:
            if not os.path.isdir(args.training_data_dir):
                raise DataValidationError(
                    f"Training data directory not found: {args.training_data_dir}"
                )

            csv_files = [
                os.path.join(args.training_data_dir, name)
                for name in os.listdir(args.training_data_dir)
                if name.lower().endswith('.csv')
            ]

            if not csv_files:
                raise DataValidationError(
                    "No CSV files found in training data directory. Provide --csv-files explicitly or "
                    "populate the directory with training tables, or allow image-based training by omitting --csv-files."
                )

            logger.info(
                "Auto-discovered %d CSV file(s) in training data directory %s", len(csv_files), args.training_data_dir
            )

        if not csv_files:
            raise DataValidationError(
                "Provide --csv-files or omit it to trigger image-based training from the default class folders."
            )

        full_df = _load_feature_table_from_csv(csv_files, args.label_column, args.labels_sidecar)

    # --- Model Training ---
    X, y = _normalize_feature_table(full_df, label_column='label')

    size_invariant_enabled = bool(
        training_params.get('size_invariant_features', s3_config.get('size_invariant_features', False))
    )
    if size_invariant_enabled:
        X, dropped_raw_features = apply_size_invariant_feature_strategy(X, enabled=True)
        if dropped_raw_features:
            logger.info(
                "Size-invariant training enabled; dropped raw size-dependent features to favor normalized metrics: %s",
                ', '.join(sorted(dropped_raw_features)),
            )
        else:
            logger.warning(
                "Size-invariant training enabled, but no raw size features were removed."
            )

    
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

    models_dir = get_models_dir(config)
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
