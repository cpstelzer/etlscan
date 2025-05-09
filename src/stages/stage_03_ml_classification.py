# src/stages/stage_03_ml_classification.py
import os
import shutil
import cv2
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import NotFittedError

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    get_models_dir, get_training_data_dir
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError, MLModelError
from src.core.ml_feature_extractor import extract_ml_features, DEFAULT_FEATURE_NAMES
from src.visualization.feature_visualization import visualize_ml_features

logger = logging.getLogger(__name__)

# Global variable to store feature names learned during training
_learned_feature_names_during_training: Optional[List[str]] = None


def _load_and_extract_features_for_training(
    image_directory: str,
    label: int,
    ml_params_config: Dict[str, Any],
    image_extensions: List[str]
) -> Tuple[List[List[Any]], List[int], Optional[List[str]]]:
    """
    Loads images from a directory, extracts features, and assigns a label.
    Determines feature names from the first successful extraction.
    """
    global _learned_feature_names_during_training
    features_list: List[List[Any]] = []
    labels_list: List[int] = []
    
    logger.info(f"Loading and extracting features from '{os.path.basename(image_directory)}' with label {label}.")
    if not os.path.isdir(image_directory):
        logger.warning(f"Training data directory not found: {image_directory}")
        return features_list, labels_list, _learned_feature_names_during_training

    count_processed = 0
    count_extracted = 0
    
    filenames = [f for f in os.listdir(image_directory)
                 if os.path.isfile(os.path.join(image_directory, f)) and
                    f.lower().endswith(tuple(image_extensions))]
    
    total_files = len(filenames)
    logger.info(f"Found {total_files} image files in {os.path.basename(image_directory)}.")

    for i, filename in enumerate(filenames):
        count_processed += 1
        image_path = os.path.join(image_directory, filename)
        features_dict, _, _ = extract_ml_features(image_path, ml_params_config)

        if features_dict:
            if _learned_feature_names_during_training is None:
                _learned_feature_names_during_training = list(features_dict.keys())
                # Ensure DEFAULT_FEATURE_NAMES are present and in order, if possible
                # This can be refined to be more strict about expected features.
                current_keys = set(_learned_feature_names_during_training)
                default_set = set(DEFAULT_FEATURE_NAMES)
                if not default_set.issubset(current_keys):
                    logger.warning(f"Extracted features {current_keys} do not fully cover default expected features {default_set}.")
                # For now, use the extracted keys as the standard.
                logger.info(f"Determined feature names from first successful extraction: {_learned_feature_names_during_training}")

            # Order features according to the learned names
            try:
                ordered_feature_values = [features_dict[name] for name in _learned_feature_names_during_training]
                features_list.append(ordered_feature_values)
                labels_list.append(label)
                count_extracted += 1
            except KeyError as ke:
                logger.error(f"KeyError extracting ordered features for {filename}: {ke}. "
                             f"Expected features: {_learned_feature_names_during_training}, Got from image: {list(features_dict.keys())}")
        
        if (i + 1) % 50 == 0 or (i + 1) == total_files:
            logger.debug(f"  ...{os.path.basename(image_directory)}: {i+1}/{total_files} processed ({count_extracted} extracted).")
            
    logger.info(f"Finished {os.path.basename(image_directory)}. Extracted features for {count_extracted}/{count_processed} images.")
    return features_list, labels_list, _learned_feature_names_during_training


def train_ml_model(
    config_s3: Dict[str, Any],
    force_retrain: bool = False
    ) -> Tuple[Optional[RandomForestClassifier], Optional[StandardScaler], Optional[List[str]]]:
    """
    Trains a RandomForestClassifier model based on configuration.
    Saves the trained model and scaler.

    Args:
        config_s3: Configuration dictionary for Stage 3.
        force_retrain: If True, retrains even if model files exist.

    Returns:
        Tuple containing (trained_model, fitted_scaler, feature_names_used_for_training).
        Returns (None, None, None) on failure.
    """
    global _learned_feature_names_during_training
    _learned_feature_names_during_training = None # Reset for each training call

    logger.info("--- Initiating ML Model Training ---")
    
    training_params = config_s3.get('training_params', {})
    ml_feature_extraction_params = config_s3.get('feature_extraction_params', {})
    image_exts = config_s3.get('image_extensions', ['.png', '.jpg'])
    
    models_output_dir = get_models_dir()
    model_path = os.path.join(models_output_dir, config_s3.get('model_filename', 'rf_fem_classifier.joblib'))
    scaler_path = os.path.join(models_output_dir, config_s3.get('scaler_filename', 'rf_fem_scaler.joblib'))

    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        logger.info(f"Model '{model_path}' and scaler '{scaler_path}' already exist. Loading them.")
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            # We need to know the feature names used for this loaded model.
            # This is a common challenge. Store feature names with model or derive if possible.
            # For now, assume if model loads, we'd need a way to get its expected feature order.
            # One strategy: save feature names as a separate file or in model metadata if supported.
            # Here, we'll try to re-determine if not available.
            # This part needs a robust way to get feature names for PRE-TRAINED models.
            # If joblib doesn't store them, we might need to save a _features.txt alongside model.
            # For this refactor, if not retraining, we return None for feature_names, and caller must handle it.
            logger.warning("Loaded existing model. Feature names used for its training are not explicitly stored with it in this version.")
            logger.warning("For prediction with a pre-loaded model, ensure feature extraction matches its training.")
            # For now, we can't reliably return _learned_feature_names_during_training without re-running extraction.
            # This implies that to know the feature names, training or a part of it must run.
            # Or, the calling function must pass them if known.
            return model, scaler, None # Cannot return feature names without training data pass
        except Exception as e:
            logger.error(f"Error loading existing model/scaler: {e}. Proceeding to retrain.")
            # Fall through to retrain

    base_training_data_dir = get_training_data_dir()
    good_dir = os.path.join(base_training_data_dir, training_params.get('good_data_subdir', 'good'))
    bad_dir = os.path.join(base_training_data_dir, training_params.get('bad_data_subdir', 'bad'))

    features_good, labels_good, f_names_good = _load_and_extract_features_for_training(
        good_dir, 1, ml_feature_extraction_params, image_exts
    )
    # _learned_feature_names_during_training should be set by the first call if successful.
    features_bad, labels_bad, f_names_bad = _load_and_extract_features_for_training(
        bad_dir, 0, ml_feature_extraction_params, image_exts
    )
    
    if _learned_feature_names_during_training is None:
        logger.error("Feature names could not be determined from training data. Training aborted.")
        return None, None, None
    
    all_features_X = np.array(features_good + features_bad)
    all_labels_y = np.array(labels_good + labels_bad)

    if all_features_X.shape[0] == 0 or all_features_X.shape[1] == 0:
        logger.error("No training data or features extracted. Training aborted.")
        return None, None, _learned_feature_names_during_training # Return names for debug

    logger.info(f"Total training samples: {len(all_labels_y)}. Number of features: {all_features_X.shape[1]}")
    logger.info(f"Feature names: {_learned_feature_names_during_training}")
    logger.info(f"Class distribution in training data: {np.bincount(all_labels_y)}")

    # Train/Test Split
    test_size = training_params.get('test_split_size', 0.25)
    random_state = training_params.get('random_state', 42)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            all_features_X, all_labels_y, test_size=test_size, random_state=random_state, stratify=all_labels_y
        )
    except ValueError as e: # Stratify might fail if a class has < 2 samples
        logger.warning(f"Stratified split failed ({e}). Trying non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            all_features_X, all_labels_y, test_size=test_size, random_state=random_state
        )
    logger.info(f"Data split: Train={len(y_train)}, Test={len(y_test)}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Training data scaled. Scaler fitted.")

    # RandomForest Model Training with GridSearchCV
    rf_params = training_params.get('rf_param_grid', {})
    if not rf_params:
        logger.warning("RandomForest param_grid not defined in config. Using default.")
        rf_params = {'n_estimators': [100], 'max_depth': [None], 'class_weight': ['balanced']}
        
    rf_model = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_params,
        cv=training_params.get('grid_search_cv_folds', 3),
        n_jobs=-1, # Use all available cores
        verbose=1,
        scoring=training_params.get('grid_search_scoring', 'f1_weighted')
    )
    logger.info("Starting GridSearchCV for RandomForest model...")
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best RandomForest parameters found: {grid_search.best_params_}")

    # Evaluation on Test Set
    y_pred_test = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_test)
    logger.info(f"Accuracy on test set: {accuracy:.4f}")
    logger.info("Classification Report (Test Set):\n" +
                classification_report(y_test, y_pred_test, target_names=['Bad(0)', 'Good(1)'], zero_division=0))
    logger.info("Confusion Matrix (Test Set):\n" + str(confusion_matrix(y_test, y_pred_test)))

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': _learned_feature_names_during_training,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        logger.info("Feature Importances:\n" + str(feature_importance_df))

    # Save Model and Scaler
    try:
        joblib.dump(best_model, model_path)
        logger.info(f"Trained model saved to: {model_path}")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
        # Save feature names used for training
        feature_names_path = model_path.replace(".joblib", "_feature_names.txt")
        with open(feature_names_path, "w") as f:
            for name in _learned_feature_names_during_training:
                f.write(f"{name}\n")
        logger.info(f"Feature names used for training saved to: {feature_names_path}")

    except Exception as e:
        logger.exception(f"Error saving model or scaler: {e}")
        return None, None, _learned_feature_names_during_training # Return them even if save fails

    logger.info("--- ML Model Training Completed ---")
    return best_model, scaler, _learned_feature_names_during_training


def predict_ideal_fems(
    input_sharp_objects_dir: str, # Directory of images from Stage 2 (e.g., "sharp_object_final_crops")
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_names_for_prediction: List[str], # Crucial: must match training
    config_s3: Dict[str, Any],
    output_classified_dir: str, # Base dir for "ideal_fems_ML_classified"
    sample_id: str
) -> Optional[pd.DataFrame]:
    """
    Predicts ideal FEMs from a directory of sharp object images using a trained model.
    Saves accepted images, visualizations, and a CSV of their features.
    """
    logger.info(f"--- Predicting Ideal FEMs for Sample: {sample_id} ---")
    logger.info(f"Input directory of sharp objects: {input_sharp_objects_dir}")

    if not os.path.isdir(input_sharp_objects_dir):
        logger.error(f"Input directory for prediction '{input_sharp_objects_dir}' not found. Prediction aborted.")
        return None

    ml_feature_params = config_s3.get('feature_extraction_params', {})
    image_exts = config_s3.get('image_extensions', ['.png', '.jpg'])

    # Output subdirectories
    output_accepted_images_dir = os.path.join(output_classified_dir, "accepted_images")
    output_visualizations_dir = os.path.join(output_classified_dir, "visualizations")
    os.makedirs(output_accepted_images_dir, exist_ok=True)
    os.makedirs(output_visualizations_dir, exist_ok=True)

    accepted_features_list = []
    processed_count = 0
    accepted_count = 0
    rejected_count = 0
    error_count = 0

    image_files_to_predict = [
        f for f in os.listdir(input_sharp_objects_dir)
        if os.path.isfile(os.path.join(input_sharp_objects_dir, f)) and
           f.lower().endswith(tuple(image_exts))
    ]
    total_to_predict = len(image_files_to_predict)
    logger.info(f"Found {total_to_predict} images in '{input_sharp_objects_dir}' for prediction.")

    if not image_files_to_predict:
        logger.warning("No images found for prediction.")
        return pd.DataFrame() # Return empty DF

    for i, filename in enumerate(image_files_to_predict):
        processed_count += 1
        image_path = os.path.join(input_sharp_objects_dir, filename)
        
        features_dict, main_props, egg_pair_props = extract_ml_features(image_path, ml_feature_params)

        if features_dict:
            try:
                # Ensure features are in the same order as during training
                current_image_features = [features_dict[name] for name in feature_names_for_prediction]
                features_array = np.array(current_image_features).reshape(1, -1)
                
                if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
                    raise MLModelError("Scaler is not fitted. Cannot transform features for prediction.")
                
                features_scaled = scaler.transform(features_array)
                prediction_label = model.predict(features_scaled)[0]
                # prediction_proba = model.predict_proba(features_scaled)[0] # Optional: get probabilities

                if prediction_label == 1: # Accepted as "ideal FEM"
                    accepted_count += 1
                    # Copy accepted image
                    shutil.copy2(image_path, os.path.join(output_accepted_images_dir, filename))
                    
                    # Create visualization
                    vis_filename = f"{os.path.splitext(filename)[0]}_vis.png"
                    vis_output_path = os.path.join(output_visualizations_dir, vis_filename)
                    visualize_ml_features(image_path, main_props, egg_pair_props, vis_output_path, ml_feature_params=ml_feature_params)
                    
                    # Store features for CSV
                    features_dict['filename'] = filename # Add filename to features
                    features_dict['predicted_label'] = int(prediction_label)
                    # features_dict['prediction_proba_good'] = float(prediction_proba[1]) if len(prediction_proba)>1 else np.nan
                    accepted_features_list.append(features_dict)
                else: # Rejected
                    rejected_count += 1

            except KeyError as ke:
                logger.error(f"Feature mismatch for {filename} during prediction: {ke}. "
                             f"Expected: {feature_names_for_prediction}. Got: {list(features_dict.keys())}")
                error_count += 1
            except NotFittedError as nfe:
                 logger.error(f"Scaler or Model NotFittedError for {filename}: {nfe}")
                 error_count +=1
            except Exception as e_pred:
                logger.exception(f"Error during prediction processing (copy/viz/append) for {filename}: {e_pred}") # Use logger.exception
                error_count += 1
        else:
            logger.warning(f"Feature extraction failed for {filename}. Skipping prediction.")
            error_count += 1
        
        if (i + 1) % 20 == 0 or (i + 1) == total_to_predict:
            logger.info(f"  ...predicted {i+1}/{total_to_predict} images "
                        f"(Accepted: {accepted_count}, Rejected: {rejected_count}, Errors: {error_count})")

    logger.info("--- Prediction Phase Complete ---")
    logger.info(f"Total images processed: {processed_count}")
    logger.info(f"Accepted images: {accepted_count} (saved to: {output_accepted_images_dir})")
    logger.info(f"Visualizations saved to: {output_visualizations_dir}")
    logger.info(f"Rejected images: {rejected_count}")
    logger.info(f"Errors during prediction: {error_count}")

    if accepted_features_list:
        # Ensure DataFrame columns match the feature_names_for_prediction order + filename + label
        cols_for_df = ['filename', 'predicted_label'] + feature_names_for_prediction # + ['prediction_proba_good']
        accepted_df = pd.DataFrame(accepted_features_list, columns=cols_for_df)
        
        # Fill any missing feature columns with NaN (should not happen if KeyErrors are caught)
        for col in feature_names_for_prediction:
            if col not in accepted_df.columns:
                accepted_df[col] = np.nan
        
        return accepted_df
    else:
        return pd.DataFrame() # Return empty if no accepted


def run_stage_03_ml_classification(
    input_dir_or_csv_path: str, # Path to Stage 2 output dir (sharp_object_final_crops)
                                # OR path to a CSV with list of images (not implemented yet)
    output_dir_stage3: str,
    sample_id: str,
    mode: str = "predict", # "train", "predict", "train_and_predict"
    force_model_retrain: bool = False
    ) -> Optional[str]: # Path to the CSV of accepted FEM features
    """
    Main function for Stage 3: ML Classification.
    Can train a model, predict ideal FEMs, or do both.
    """
    logger.info(f"--- Stage 3: ML Classification for Sample ID: {sample_id} (Mode: {mode}) ---")
    
    config = get_config()
    config_s3 = config.get('stage_03_ml_classification', {})
    if not config_s3:
        raise ConfigError("Configuration for 'stage_03_ml_classification' not found.")

    trained_model: Optional[RandomForestClassifier] = None
    fitted_scaler: Optional[StandardScaler] = None
    feature_names: Optional[List[str]] = None # Store feature names learned/used

    # --- Training Phase ---
    if mode in ["train", "train_and_predict"]:
        logger.info(f"Starting model training (force_retrain={force_model_retrain})...")
        trained_model, fitted_scaler, feature_names = train_ml_model(config_s3, force_model_retrain)
        if not trained_model or not fitted_scaler or not feature_names:
            logger.error("ML model training failed or did not complete successfully.")
            if mode == "train": return None # If only training, signal failure
            # If "train_and_predict", we might try to load an existing model as fallback if config allows.
            # For now, if training fails in train_and_predict, prediction is also aborted.
            raise MLModelError("Training failed, cannot proceed to prediction in 'train_and_predict' mode.")
        logger.info("Model training completed successfully.")
        _learned_feature_names_during_training = feature_names # Update global for consistency if needed elsewhere

    # --- Prediction Phase ---
    if mode in ["predict", "train_and_predict"]:
        if not trained_model or not fitted_scaler: # Need to load if only predicting
            logger.info("Loading existing model and scaler for prediction...")
            models_output_dir = get_models_dir()
            model_path = os.path.join(models_output_dir, config_s3.get('model_filename', 'rf_fem_classifier.joblib'))
            scaler_path = os.path.join(models_output_dir, config_s3.get('scaler_filename', 'rf_fem_scaler.joblib'))
            feature_names_path = model_path.replace(".joblib", "_feature_names.txt")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(feature_names_path):
                logger.error(f"Model ('{model_path}'), scaler ('{scaler_path}'), or feature names ('{feature_names_path}') "
                             "not found for prediction. Train the model first or ensure files exist.")
                raise MLModelError("Required model/scaler/feature_names files not found for prediction.")
            try:
                trained_model = joblib.load(model_path)
                fitted_scaler = joblib.load(scaler_path)
                with open(feature_names_path, "r") as f:
                    feature_names = [line.strip() for line in f if line.strip()]
                logger.info("Existing model, scaler, and feature names loaded successfully.")
                if not feature_names:
                     raise MLModelError("Loaded feature names file is empty.")
            except Exception as e:
                logger.exception(f"Error loading model/scaler/feature_names for prediction: {e}")
                raise MLModelError(f"Failed to load model/scaler/feature_names: {e}")

        if not feature_names:
            logger.error("Feature names for prediction are not available. Cannot proceed.")
            raise MLModelError("Feature names required for prediction are missing.")

        # input_dir_or_csv_path for prediction should be the directory containing sharp object images
        # e.g., .../output/{sample_id}/stage2_output/sharp_object_final_crops/
        sharp_objects_image_dir = input_dir_or_csv_path 
        
        # Output for Stage 3 classification results
        stage3_sample_output_dir = os.path.join(output_dir_stage3, "ideal_fems_ML_classified")
        if os.path.exists(stage3_sample_output_dir): # Clean previous results for this sample's S3 output
            logger.warning(f"Removing existing Stage 3 output directory: {stage3_sample_output_dir}")
            shutil.rmtree(stage3_sample_output_dir)
        os.makedirs(stage3_sample_output_dir, exist_ok=True)

        accepted_fems_df = predict_ideal_fems(
            input_sharp_objects_dir=sharp_objects_image_dir,
            model=trained_model,
            scaler=fitted_scaler,
            feature_names_for_prediction=feature_names,
            config_s3=config_s3,
            output_classified_dir=stage3_sample_output_dir, # Base for "accepted_images" and "visualizations"
            sample_id=sample_id
        )

        if accepted_fems_df is not None and not accepted_fems_df.empty:
            output_csv_filename = f"{sample_id}_stage3_accepted_fem_features.csv"
            saved_csv_path = save_dataframe_to_csv(accepted_fems_df, stage3_sample_output_dir, output_csv_filename)
            logger.info(f"--- Stage 3: ML Classification for Sample ID: {sample_id} COMPLETED ---")
            return saved_csv_path
        elif accepted_fems_df is not None and accepted_fems_df.empty:
            logger.info("No ideal FEMs were accepted during prediction. No features CSV saved.")
            logger.info(f"--- Stage 3: ML Classification for Sample ID: {sample_id} COMPLETED (no accepted FEMs) ---")
            return os.path.join(stage3_sample_output_dir, f"{sample_id}_stage3_accepted_fem_features.csv") # Path to (empty) csv
        else: # accepted_fems_df is None
            logger.error("Prediction of ideal FEMs failed.")
            logger.info(f"--- Stage 3: ML Classification for Sample ID: {sample_id} FAILED ---")
            return None
            
    elif mode == "train": # Only training was requested and completed
        logger.info(f"--- Stage 3: ML Model Training for Sample ID: {sample_id} COMPLETED (no prediction run) ---")
        return os.path.join(get_models_dir(), config_s3.get('model_filename', 'rf_fem_classifier.joblib')) # Return path to model
    
    return None # Should not be reached if mode is valid


if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging
    setup_logging()

    try:
        cfg_main = get_config()
        TEST_SAMPLE_ID_S3 = "test_sample_s3_classify" # Assumes this sample exists for input/output
        
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir:
            raise ValueError("project_base_dir not set in config for testing.")

        # --- Test Parameters ---
        # MODE: "train", "predict", "train_and_predict"
        # For a full test:
        # 1. Run with mode="train" (ensure training_data/good, training_data/bad exist with dummy images)
        # 2. Run with mode="predict" (ensure Stage 2 output exists with sharp crops for TEST_SAMPLE_ID_S3)
        # 3. Run with mode="train_and_predict"
        
        TEST_MODE = "train_and_predict" # Change as needed
        FORCE_RETRAIN_TEST = True # Set to True to always retrain during test

        # Path to Stage 2 output (sharp crops) - input for prediction
        # This must point to the output of Stage 2 for the sample being processed.
        stage2_output_dir_for_s3_input = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S3, "stage2_output", "sharp_object_final_crops")
        
        # Output directory for Stage 3 results
        stage3_output_dir_for_sample = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S3, "stage3_output")
        # os.makedirs(stage3_output_dir_for_sample, exist_ok=True) # run_stage_03 will create subdirs

        logger.info(f"Test Mode: {TEST_MODE}")
        logger.info(f"Test S2 Input Dir for S3 predict: {stage2_output_dir_for_s3_input}")
        logger.info(f"Test S3 Output Base Dir: {stage3_output_dir_for_sample}")

        # Ensure dummy training data for "train" or "train_and_predict" modes
        if TEST_MODE in ["train", "train_and_predict"]:
            s3_cfg = cfg_main.get('stage_03_ml_classification', {})
            train_params_cfg = s3_cfg.get('training_params', {})
            good_subdir = os.path.join(get_training_data_dir(), train_params_cfg.get('good_data_subdir', 'good'))
            bad_subdir = os.path.join(get_training_data_dir(), train_params_cfg.get('bad_data_subdir', 'bad'))
            os.makedirs(good_subdir, exist_ok=True)
            os.makedirs(bad_subdir, exist_ok=True)
            if not os.listdir(good_subdir) or not os.listdir(bad_subdir):
                logger.warning(f"Dummy training data dirs '{good_subdir}' or '{bad_subdir}' are empty. "
                               "Please add a few dummy .png images for training test.")
                # Create 2-3 dummy images in each if empty
                for i in range(3):
                    if not os.path.exists(os.path.join(good_subdir,f"good_img_{i}.png")):
                        cv2.imwrite(os.path.join(good_subdir,f"good_img_{i}.png"), np.random.randint(100,250,(50,50), dtype=np.uint8))
                    if not os.path.exists(os.path.join(bad_subdir,f"bad_img_{i}.png")):
                         cv2.imwrite(os.path.join(bad_subdir,f"bad_img_{i}.png"), np.random.randint(0,100,(50,50), dtype=np.uint8))


        # Ensure dummy "sharp_objects" for "predict" or "train_and_predict" modes
        if TEST_MODE in ["predict", "train_and_predict"]:
            os.makedirs(stage2_output_dir_for_s3_input, exist_ok=True)
            if not os.listdir(stage2_output_dir_for_s3_input):
                logger.warning(f"Sharp objects input dir '{stage2_output_dir_for_s3_input}' is empty. "
                               "Prediction test might not find images.")
                # Create a few dummy sharp object images
                for i in range(5):
                    fname = f"dummy_sharp_obj_{i}.png"
                    fpath = os.path.join(stage2_output_dir_for_s3_input, fname)
                    if not os.path.exists(fpath):
                        cv2.imwrite(fpath, np.random.randint(0,255, (150,150,3), dtype=np.uint8))


        # Run Stage 3
        result_s3_path = run_stage_03_ml_classification(
            input_dir_or_csv_path=stage2_output_dir_for_s3_input, # Used for 'predict' mode
            output_dir_stage3=stage3_output_dir_for_sample, # Base for S3 specific outputs
            sample_id=TEST_SAMPLE_ID_S3,
            mode=TEST_MODE,
            force_model_retrain=FORCE_RETRAIN_TEST
        )

        if result_s3_path:
            logger.info(f"Stage 3 test (mode: {TEST_MODE}) completed. Result/Indicator path: {result_s3_path}")
            if TEST_MODE != "train" and os.path.exists(result_s3_path): # If predict mode, check CSV
                df_res_s3 = pd.read_csv(result_s3_path)
                logger.info(f"Accepted FEMs features (head):\n{df_res_s3.head()}")
        else:
            logger.error(f"Stage 3 test (mode: {TEST_MODE}) FAILED or produced no output.")

    except ConfigError as ce:
        logger.error(f"Configuration error during Stage 3 test: {ce}")
    except MLModelError as mle:
        logger.error(f"ML Model error during Stage 3 test: {mle}")
    except FileNotFoundError as fnfe:
        logger.error(f"File not found during Stage 3 test setup: {fnfe}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the Stage 3 test: {e}")