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
from src.utils.config_loader import get_unified_segmentation_params 
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError, MLModelError
from src.core.ml_feature_extractor import extract_ml_features, DEFAULT_FEATURE_NAMES
from src.visualization.feature_visualization import visualize_ml_features
from src.utils.path_helpers import parse_sharp_crop_filename, extract_frame_from_filename
from src.core.image_io_operations import load_image, convert_to_grayscale
from src.core.particle_operations import detect_cropped_particle_on_boundary


logger = logging.getLogger(__name__)

_learned_feature_names_during_training: Optional[List[str]] = None


def _load_and_extract_features_for_training(
    image_directory: str,
    label: int,
    ml_processing_params: Dict[str, Any], 
    image_extensions: List[str]
) -> Tuple[List[List[Any]], List[int], Optional[List[str]]]:
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
        features_dict, _, _ = extract_ml_features(image_path, ml_processing_params)

        if features_dict:
            if _learned_feature_names_during_training is None:
                _learned_feature_names_during_training = list(features_dict.keys())
                logger.info(f"Determined feature names from first successful extraction: {_learned_feature_names_during_training}")

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
    global_system_config: Dict[str, Any], 
    force_retrain: bool = False
    ) -> Tuple[Optional[RandomForestClassifier], Optional[StandardScaler], Optional[List[str]]]:
    global _learned_feature_names_during_training
    _learned_feature_names_during_training = None 

    logger.info("--- Initiating ML Model Training ---")
    
    training_params_cfg = config_s3.get('training_params', {})
    
    main_config_for_helper = get_config() 
    ml_params_for_training = get_unified_segmentation_params(main_config_for_helper)
    
    image_exts = config_s3.get('image_extensions', ['.png', '.jpg'])
    
    models_output_dir = get_models_dir()
    model_filename = config_s3.get('model_filename', 'rf_fem_classifier.joblib')
    scaler_filename = config_s3.get('scaler_filename', 'rf_fem_scaler.joblib')
    model_path = os.path.join(models_output_dir, model_filename)
    scaler_path = os.path.join(models_output_dir, scaler_filename)
    feature_names_path = model_path.replace(".joblib", "_feature_names.txt")

    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_names_path):
        logger.info(f"Model '{model_path}', scaler '{scaler_path}', and feature names file '{feature_names_path}' already exist. Loading them.")
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            with open(feature_names_path, "r") as f:
                loaded_feature_names = [line.strip() for line in f if line.strip()]
            if not loaded_feature_names:
                raise MLModelError("Loaded feature names file is empty.")
            _learned_feature_names_during_training = loaded_feature_names 
            return model, scaler, loaded_feature_names
        except Exception as e:
            logger.error(f"Error loading existing model/scaler/feature_names: {e}. Proceeding to retrain.")

    base_training_data_dir = get_training_data_dir()
    good_dir = os.path.join(base_training_data_dir, training_params_cfg.get('good_data_subdir', 'good'))
    bad_dir = os.path.join(base_training_data_dir, training_params_cfg.get('bad_data_subdir', 'bad'))

    logger.debug(f"Training parameters for feature extraction (unified): {ml_params_for_training}")
    features_good, labels_good, learned_names_good = _load_and_extract_features_for_training(
        good_dir, 1, ml_params_for_training, image_exts
    )
    features_bad, labels_bad, learned_names_bad = _load_and_extract_features_for_training(
        bad_dir, 0, ml_params_for_training, image_exts
    )
    
    if _learned_feature_names_during_training is None: 
        if learned_names_good: _learned_feature_names_during_training = learned_names_good
        elif learned_names_bad: _learned_feature_names_during_training = learned_names_bad
        else:
            logger.error("Feature names could not be determined from training data. Training aborted.")
            return None, None, None
    
    all_features_X = np.array(features_good + features_bad)
    all_labels_y = np.array(labels_good + labels_bad)

    if all_features_X.shape[0] == 0 or (_learned_feature_names_during_training and all_features_X.shape[1] == 0):
        logger.error("No training data or features extracted. Training aborted.")
        return None, None, _learned_feature_names_during_training

    logger.info(f"Total training samples: {len(all_labels_y)}. Number of features: {all_features_X.shape[1]}")
    logger.info(f"Feature names used for training: {_learned_feature_names_during_training}")
    class_counts = np.bincount(all_labels_y)
    if len(class_counts) < 2: logger.warning(f"Training data has only one class: {class_counts}. Stratify and some metrics might fail.")
    logger.info(f"Class distribution in training data: {class_counts}")
    test_size = training_params_cfg.get('test_split_size', 0.25)
    random_state = training_params_cfg.get('random_state', 42)
    try:
        X_train, X_test, y_train, y_test = train_test_split(all_features_X, all_labels_y, test_size=test_size, random_state=random_state, stratify=all_labels_y)
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}). Trying non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(all_features_X, all_labels_y, test_size=test_size, random_state=random_state)
    logger.info(f"Data split: Train={len(y_train)}, Test={len(y_test)}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Training data scaled. Scaler fitted.")
    rf_params_grid_config = training_params_cfg.get('rf_param_grid', {})
    if not rf_params_grid_config:
        logger.warning("RandomForest param_grid not defined in config. Using default.")
        rf_params_parsed = {'n_estimators': [100], 'max_depth': [None], 'class_weight': ['balanced']}
    else: 
        rf_params_parsed = {key: [None if v == 'null' else v for v in value_list] for key, value_list in rf_params_grid_config.items()}
    rf_model = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(estimator=rf_model,param_grid=rf_params_parsed,cv=training_params_cfg.get('grid_search_cv_folds', 3),n_jobs=-1, verbose=1,scoring=training_params_cfg.get('grid_search_scoring', 'f1_weighted'))
    logger.info("Starting GridSearchCV for RandomForest model...")
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    logger.info(f"Best RandomForest parameters found: {grid_search.best_params_}")
    y_pred_test = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_test)
    logger.info(f"Accuracy on test set: {accuracy:.4f}")
    logger.info("Classification Report (Test Set):\n" + classification_report(y_test, y_pred_test, target_names=['Bad(0)', 'Good(1)'], zero_division=0))
    logger.info("Confusion Matrix (Test Set):\n" + str(confusion_matrix(y_test, y_pred_test)))
    if hasattr(best_model, 'feature_importances_') and _learned_feature_names_during_training:
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': _learned_feature_names_during_training,'importance': importances}).sort_values(by='importance', ascending=False)
        logger.info("Feature Importances:\n" + str(feature_importance_df))
    try:
        joblib.dump(best_model, model_path)
        logger.info(f"Trained model saved to: {model_path}")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
        with open(feature_names_path, "w") as f:
            for name in _learned_feature_names_during_training: f.write(f"{name}\n")
        logger.info(f"Feature names used for training saved to: {feature_names_path}")
    except Exception as e:
        logger.exception(f"Error saving model, scaler, or feature names: {e}")
        return None, None, _learned_feature_names_during_training
    logger.info("--- ML Model Training Completed ---")
    return best_model, scaler, _learned_feature_names_during_training


def _predict_ml_candidates(
    input_sharp_objects_dir: str,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_names_for_prediction: List[str],
    config_s3: Dict[str, Any], 
    global_system_config: Dict[str, Any], 
    current_sample_id: str 
) -> pd.DataFrame:
    logger.info(f"--- Predicting ML Candidates for Sample: {current_sample_id} ---")
    
    if not os.path.isdir(input_sharp_objects_dir):
        logger.error(f"Input directory for prediction '{input_sharp_objects_dir}' not found.")
        return pd.DataFrame()

    main_config_for_helper = get_config() 
    ml_params_for_prediction = get_unified_segmentation_params(main_config_for_helper)
    
    logger.debug(f"Using ML params for prediction in Stage 3 (unified) for sample {current_sample_id}: {ml_params_for_prediction}")

    image_exts = config_s3.get('image_extensions', ['.png', '.jpg'])
    candidate_data_list = []
    processed_count = 0; ml_positive_count = 0; error_count = 0
    image_files_to_predict = [f for f in os.listdir(input_sharp_objects_dir) if os.path.isfile(os.path.join(input_sharp_objects_dir, f)) and f.lower().endswith(tuple(image_exts))]
    total_to_predict = len(image_files_to_predict)
    logger.info(f"Found {total_to_predict} images in '{input_sharp_objects_dir}' for ML prediction.")
    if not image_files_to_predict: return pd.DataFrame()

    for i, filename in enumerate(image_files_to_predict):
        processed_count += 1
        image_path = os.path.join(input_sharp_objects_dir, filename)
        features_dict, main_props, egg_area_props = extract_ml_features(image_path, ml_params_for_prediction)
        if features_dict:
            try:
                missing_model_features = [name for name in feature_names_for_prediction if name not in features_dict]
                if missing_model_features:
                    logger.error(f"Missing features {missing_model_features} for {filename}. Expected: {feature_names_for_prediction}. Got: {list(features_dict.keys())}")
                    error_count +=1; continue
                current_image_features = [features_dict[name] for name in feature_names_for_prediction]
                features_array = np.array(current_image_features).reshape(1, -1)
                if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'): 
                    raise MLModelError("Scaler is not fitted.")
                features_scaled = scaler.transform(features_array)
                prediction_label = model.predict(features_scaled)[0]
                if prediction_label == 1:
                    ml_positive_count += 1
                    candidate_data_list.append({'sample_id': current_sample_id, 'filename': filename,'image_path': image_path,'main_props': main_props,'egg_area_props': egg_area_props,'predicted_label': int(prediction_label),**features_dict})
            except KeyError as ke: logger.error(f"Feature mismatch for {filename}: {ke}."); error_count += 1
            except NotFittedError as nfe: logger.error(f"Scaler or Model NotFittedError for {filename}: {nfe}"); error_count +=1
            except Exception as e_pred: logger.exception(f"Error during ML prediction for {filename}: {e_pred}"); error_count += 1
        else: logger.warning(f"Feature extraction failed for {filename}. Skipping prediction."); error_count += 1
        if (i + 1) % 20 == 0 or (i + 1) == total_to_predict: logger.info(f"  ...ML predicted {i+1}/{total_to_predict} images (ML Positives: {ml_positive_count}, Errors: {error_count})")
    logger.info(f"ML Prediction Phase Complete. Total ML Positives: {ml_positive_count}")
    return pd.DataFrame(candidate_data_list) if candidate_data_list else pd.DataFrame()

def _merge_with_stage2_data(ml_candidates_df: pd.DataFrame, stage2_analyzed_csv_path: Optional[str], sample_id: str ) -> pd.DataFrame:
    if ml_candidates_df.empty: return ml_candidates_df
    if not stage2_analyzed_csv_path or not os.path.exists(stage2_analyzed_csv_path):
        logger.warning(f"Stage 2 analyzed CSV path not provided or not found: {stage2_analyzed_csv_path}. 'combined_sharpness' will be missing for sample {sample_id}.")
        ml_candidates_df['combined_sharpness'] = np.nan 
        return ml_candidates_df
    logger.info(f"Merging ML candidates for sample {sample_id} with Stage 2 data from: {stage2_analyzed_csv_path}")
    s2_df = load_dataframe_from_csv(stage2_analyzed_csv_path)
    if s2_df is None or s2_df.empty:
        logger.warning(f"Failed to load or empty Stage 2 data from {stage2_analyzed_csv_path} for sample {sample_id}. 'combined_sharpness' will be missing.")
        ml_candidates_df['combined_sharpness'] = np.nan
        return ml_candidates_df
    parsed_info = ml_candidates_df['filename'].apply(parse_sharp_crop_filename)
    ml_candidates_df['parsed_dpw_str'] = parsed_info.apply(lambda x: x[0] if x else None)
    ml_candidates_df['parsed_frame'] = pd.to_numeric(parsed_info.apply(lambda x: x[1] if x else None), errors='coerce').astype('Int64')
    ml_candidates_df['parsed_X'] = pd.to_numeric(parsed_info.apply(lambda x: x[2] if x else None), errors='coerce').astype('Int64')
    ml_candidates_df['parsed_Y'] = pd.to_numeric(parsed_info.apply(lambda x: x[3] if x else None), errors='coerce').astype('Int64')
    ml_candidates_df.dropna(subset=['parsed_frame', 'parsed_X', 'parsed_Y'], inplace=True)
    if ml_candidates_df.empty:
        logger.warning(f"No ML candidates remaining for sample {sample_id} after parsing filenames for merge keys.")
        ml_candidates_df['combined_sharpness'] = np.nan 
        return ml_candidates_df
    s2_df['s2_parsed_frame'] = s2_df['source_image_filename'].apply(extract_frame_from_filename).astype('Int64')
    s2_df.dropna(subset=['s2_parsed_frame', 'X', 'Y'], inplace=True)
    s2_df['X'] = s2_df['X'].astype('Int64'); s2_df['Y'] = s2_df['Y'].astype('Int64')
    s2_merge_cols = ['X', 'Y', 's2_parsed_frame', 'combined_sharpness']
    missing_s2_cols = [col for col in s2_merge_cols if col not in s2_df.columns]
    if missing_s2_cols:
        logger.warning(f"Missing columns in Stage 2 CSV for merge for sample {sample_id}: {missing_s2_cols}. 'combined_sharpness' will be missing.")
        ml_candidates_df['combined_sharpness'] = np.nan
        return ml_candidates_df
    merged_df = pd.merge(ml_candidates_df,s2_df[s2_merge_cols],left_on=['parsed_X', 'parsed_Y', 'parsed_frame'],right_on=['X', 'Y', 's2_parsed_frame'],how='left',suffixes=('', '_s2'))
    if 'combined_sharpness' not in merged_df.columns:
         logger.error(f"'combined_sharpness' column not found after merge for sample {sample_id}. This indicates a merge logic issue.")
         merged_df['combined_sharpness'] = np.nan
    nan_sharpness_count = merged_df['combined_sharpness'].isnull().sum()
    if nan_sharpness_count > 0: logger.warning(f"{nan_sharpness_count}/{len(merged_df)} ML candidates for sample {sample_id} could not be matched with 'combined_sharpness' from Stage 2 data.")
    logger.info(f"Merge complete for sample {sample_id}. ML candidates DataFrame now has {len(merged_df)} rows.")
    return merged_df

def _apply_temporal_spatial_deduplication(candidates_df: pd.DataFrame, dedup_config: Dict[str, Any]) -> pd.DataFrame:
    if not dedup_config.get('enabled', False) or candidates_df.empty:
        logger.info("Deduplication filter is disabled or no candidates to filter.")
        return candidates_df
    logger.info("Applying temporal-spatial deduplication filter...")
    required_cols = ['parsed_frame', 'parsed_X', 'parsed_Y', 'combined_sharpness']
    if not all(k in candidates_df.columns for k in required_cols):
        missing = [k for k in required_cols if k not in candidates_df.columns]
        logger.error(f"Missing required columns for deduplication: {missing}. Skipping filter.")
        return candidates_df
    candidates_df['combined_sharpness_filled'] = pd.to_numeric(candidates_df['combined_sharpness'], errors='coerce').fillna(float('-inf'))
    prox_thresh_sq = dedup_config.get('proximity_threshold_pixels', 100)**2
    max_frame_diff = dedup_config.get('max_adjacent_frame_difference', 2)
    df = candidates_df.sort_values(by=['parsed_frame', 'parsed_Y', 'parsed_X']).copy()
    df['unique_id'] = range(len(df))
    visited_ids = set(); final_indices_to_keep = [] 
    for i in range(len(df)):
        current_row_unique_id = df.iloc[i]['unique_id']
        current_original_index = df.index[i] 
        if current_row_unique_id in visited_ids: continue
        current_group_original_indices = [current_original_index]; visited_ids.add(current_row_unique_id)
        queue = [current_original_index]; head = 0
        while head < len(queue):
            current_q_original_idx = queue[head]; head += 1
            p_current_frame = df.loc[current_q_original_idx, 'parsed_frame']; p_current_X = df.loc[current_q_original_idx, 'parsed_X']; p_current_Y = df.loc[current_q_original_idx, 'parsed_Y']
            min_search_frame = p_current_frame - max_frame_diff; max_search_frame = p_current_frame + max_frame_diff
            potential_matches_mask = ((df['parsed_frame'] >= min_search_frame) & (df['parsed_frame'] <= max_search_frame) & (~df['unique_id'].isin(visited_ids)))
            for k_original_idx in df.index[potential_matches_mask]:
                p_k_frame = df.loc[k_original_idx, 'parsed_frame']; p_k_X = df.loc[k_original_idx, 'parsed_X']; p_k_Y = df.loc[k_original_idx, 'parsed_Y']
                frame_diff_abs = abs(p_k_frame - p_current_frame)
                if frame_diff_abs <= max_frame_diff:
                    dist_sq = (p_k_X - p_current_X)**2 + (p_k_Y - p_current_Y)**2
                    if dist_sq < prox_thresh_sq:
                        current_group_original_indices.append(k_original_idx)
                        visited_ids.add(df.loc[k_original_idx, 'unique_id']); queue.append(k_original_idx)
        if current_group_original_indices:
            group_sub_df = df.loc[current_group_original_indices]
            best_in_group_original_idx = group_sub_df['combined_sharpness_filled'].idxmax()
            final_indices_to_keep.append(best_in_group_original_idx)
    num_before = len(candidates_df)
    deduplicated_df = candidates_df.loc[final_indices_to_keep].reset_index(drop=True)
    if 'combined_sharpness_filled' in deduplicated_df.columns: deduplicated_df = deduplicated_df.drop(columns=['combined_sharpness_filled'])
    num_after = len(deduplicated_df)
    logger.info(f"Deduplication complete. Kept {num_after} from {num_before} candidates.")
    return deduplicated_df

def _apply_boundary_recheck(
    candidates_df: pd.DataFrame, 
    boundary_config: Dict[str, Any], 
    global_source_illumination: str,
    # MODIFIED: Accept threshold_params dictionary
    threshold_params_for_boundary: Dict[str, Any]
    ) -> pd.DataFrame:
    if not boundary_config.get('enabled', False) or candidates_df.empty:
        logger.info("Boundary re-check filter is disabled or no candidates to filter.")
        return candidates_df

    logger.info(f"Applying boundary re-check filter (Source Illum for check: {global_source_illumination}, ThreshMode: {threshold_params_for_boundary.get('thresholding_mode','N/A')})...")
    if 'image_path' not in candidates_df.columns:
        logger.error("Missing 'image_path' column for boundary re-check. Skipping.")
        return candidates_df

    indices_to_keep = []
    for index, row in candidates_df.iterrows():
        image_path = row['image_path']
        img_gray_uint8 = load_image(image_path, mode='gray') 
        if img_gray_uint8 is None:
            logger.warning(f"Could not load image {image_path} for boundary check. Excluding it."); continue
        
        # MODIFIED: Pass the threshold_params dictionary
        is_on_boundary = detect_cropped_particle_on_boundary(
            img_gray_uint8, 
            image_path, 
            illumination_mode_of_crop=global_source_illumination,
            threshold_params=threshold_params_for_boundary 
        )
        
        if not is_on_boundary: indices_to_keep.append(index)
        else: logger.debug(f"Particle in {os.path.basename(image_path)} detected on boundary during re-check. Removing.")
            
    num_before = len(candidates_df)
    boundary_filtered_df = candidates_df.loc[indices_to_keep].reset_index(drop=True)
    num_after = len(boundary_filtered_df)
    logger.info(f"Boundary re-check complete. Kept {num_after} from {num_before} candidates.")
    return boundary_filtered_df

def _save_final_s3_outputs(
    final_df: pd.DataFrame,
    output_classified_dir: str,
    current_sample_id: str, 
    ml_processing_params_for_vis: Dict[str, Any], 
    feature_names_for_csv: List[str] 
) -> Tuple[Optional[str], pd.DataFrame]: 
    output_csv_filename = f"{current_sample_id}_stage3_accepted_fem_features.csv"
    cols_for_df_schema = ['sample_id', 'filename', 'predicted_label'] + feature_names_for_csv
    if 'combined_sharpness' not in cols_for_df_schema: cols_for_df_schema.append('combined_sharpness')
    cols_for_df_schema = list(dict.fromkeys(cols_for_df_schema))
    final_csv_columns = [col for col in cols_for_df_schema if col in final_df.columns]

    if final_df.empty:
        logger.info(f"No ideal FEMs accepted for {current_sample_id}. No final outputs saved.")
        empty_df_with_schema = pd.DataFrame(columns=final_csv_columns); saved_path = save_dataframe_to_csv(empty_df_with_schema, output_classified_dir, output_csv_filename)
        return saved_path, empty_df_with_schema

    logger.info(f"Saving final {len(final_df)} accepted FEMs for {current_sample_id} and outputs...")
    output_accepted_images_dir = os.path.join(output_classified_dir, "accepted_images"); output_visualizations_dir = os.path.join(output_classified_dir, "visualizations")
    os.makedirs(output_accepted_images_dir, exist_ok=True); os.makedirs(output_visualizations_dir, exist_ok=True)
    for _, row in final_df.iterrows():
        original_image_path = row['image_path']; filename = row['filename']
        try: shutil.copy2(original_image_path, os.path.join(output_accepted_images_dir, filename))
        except Exception as e: logger.error(f"Failed to copy accepted image {filename}: {e}")
        main_props = row.get('main_props'); egg_area_props = row.get('egg_area_props')
        if main_props:
            vis_filename = f"{os.path.splitext(filename)[0]}_vis.png"; vis_output_path = os.path.join(output_visualizations_dir, vis_filename)
            try: visualize_ml_features(original_image_path, main_props, egg_area_props, vis_output_path, ml_feature_params=ml_processing_params_for_vis)
            except Exception as e: logger.error(f"Failed to create visualization for {filename}: {e}")
        else: logger.warning(f"Skipping visualization for {filename} due to missing main_props.")
    final_output_df_for_csv = final_df[final_csv_columns].copy(); saved_csv_path = save_dataframe_to_csv(final_output_df_for_csv, output_classified_dir, output_csv_filename)
    logger.info(f"Final outputs for {current_sample_id} saved. CSV: {saved_csv_path}")
    return saved_csv_path, final_output_df_for_csv


def run_stage_03_ml_classification(
    input_dir_or_csv_path: str, 
    output_dir_stage3: str,
    sample_id: str, 
    stage2_analyzed_csv_path: Optional[str] = None,
    mode: str = "predict", 
    force_model_retrain: bool = False
    ) -> Tuple[Optional[str], Optional[pd.DataFrame]]: 
    
    logger.info(f"--- Stage 3: ML Classification for Sample ID: {sample_id} (Mode: {mode}) ---")
    
    config = get_config() 
    config_s3 = config.get('stage_03_ml_classification', {})
    if not config_s3:
        raise ConfigError("Configuration for 'stage_03_ml_classification' not found.")

    global_system_config = config.get('system', {}) 
    
    unified_seg_and_feature_params = get_unified_segmentation_params(config)
    
    logger.info(f"Stage 3 using unified segmentation parameters derived from global and S3 config.")
    
    # These specific global values are used for _apply_boundary_recheck
    # They are sourced from the unified_params for consistency
    global_illumination_type = unified_seg_and_feature_params.get('source_image_illumination_type')
    
    # Construct the threshold_params dictionary for _apply_boundary_recheck
    boundary_recheck_threshold_params = {
        'thresholding_mode': unified_seg_and_feature_params.get('thresholding_mode'),
        'otsu_threshold_multiplier': unified_seg_and_feature_params.get('otsu_threshold_multiplier'),
        'fixed_threshold_value': unified_seg_and_feature_params.get('fixed_threshold_value')
    }

    trained_model: Optional[RandomForestClassifier] = None
    fitted_scaler: Optional[StandardScaler] = None
    feature_names_for_model: Optional[List[str]] = None
    
    empty_df_cols = ['sample_id', 'filename', 'predicted_label'] + DEFAULT_FEATURE_NAMES + ['combined_sharpness']
    empty_df_cols = list(dict.fromkeys(empty_df_cols)) 
    empty_return_df = pd.DataFrame(columns=empty_df_cols)

    if mode in ["train", "train_and_predict"]:
        logger.info(f"Starting model training (force_retrain={force_model_retrain})...")
        trained_model, fitted_scaler, feature_names_for_model = train_ml_model(
            config_s3, 
            global_system_config, 
            force_model_retrain
        )
        if not trained_model or not fitted_scaler or not feature_names_for_model:
            logger.error("ML model training failed.")
            if mode == "train": return None, empty_return_df
            raise MLModelError("Training failed, cannot proceed in 'train_and_predict' mode.")
        logger.info("Model training completed successfully.")
        global _learned_feature_names_during_training 
        _learned_feature_names_during_training = feature_names_for_model

    if mode in ["predict", "train_and_predict"]:
        if not trained_model or not fitted_scaler or not feature_names_for_model: 
            logger.info("Loading existing model, scaler, and feature names for prediction...")
            models_output_dir = get_models_dir(); model_filename = config_s3.get('model_filename', 'rf_fem_classifier.joblib'); scaler_filename = config_s3.get('scaler_filename', 'rf_fem_scaler.joblib')
            model_path = os.path.join(models_output_dir, model_filename); scaler_path = os.path.join(models_output_dir, scaler_filename)
            feature_names_path = model_path.replace(".joblib", "_feature_names.txt")
            if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
                msg = f"Model ('{model_path}'), scaler ('{scaler_path}'), or feature names ('{feature_names_path}') not found. Train model first."
                logger.error(msg); raise MLModelError(msg)
            try:
                trained_model = joblib.load(model_path); fitted_scaler = joblib.load(scaler_path)
                with open(feature_names_path, "r") as f: feature_names_for_model = [line.strip() for line in f if line.strip()]
                if not feature_names_for_model: raise MLModelError("Loaded feature names file is empty.")
                logger.info("Existing model, scaler, and feature names loaded.")
            except Exception as e: logger.exception(f"Error loading model/scaler/feature_names: {e}"); raise MLModelError(f"Failed to load model/scaler/feature_names: {e}")

        if not feature_names_for_model: 
            logger.warning("feature_names_for_model is None. Using DEFAULT_FEATURE_NAMES.")
            feature_names_for_model = DEFAULT_FEATURE_NAMES 
            if not feature_names_for_model: raise MLModelError("Feature names for model prediction critically unavailable.")

        sharp_objects_image_dir = input_dir_or_csv_path 
        
        ml_candidates_df = _predict_ml_candidates(
            input_sharp_objects_dir=sharp_objects_image_dir,
            model=trained_model, scaler=fitted_scaler,
            feature_names_for_prediction=feature_names_for_model,
            config_s3=config_s3, 
            global_system_config=global_system_config, 
            current_sample_id=sample_id 
        )
        
        stage3_final_output_dir = os.path.join(output_dir_stage3, "ideal_fems_ML_classified")
        os.makedirs(stage3_final_output_dir, exist_ok=True)

        if ml_candidates_df.empty:
            logger.info(f"No ML positive candidates found or prediction failed for sample {sample_id}.")
            fnames_for_empty_save = feature_names_for_model if feature_names_for_model else DEFAULT_FEATURE_NAMES
            # Pass unified_seg_and_feature_params for visualization consistency even if empty
            saved_csv_path, df_to_return = _save_final_s3_outputs(pd.DataFrame(), stage3_final_output_dir, sample_id, unified_seg_and_feature_params, fnames_for_empty_save)
            return saved_csv_path, df_to_return

        post_ml_filter_configs = config_s3.get('post_ml_filters', {})
        df_with_sharpness = _merge_with_stage2_data(ml_candidates_df, stage2_analyzed_csv_path, sample_id)
        if 'combined_sharpness' not in df_with_sharpness.columns or df_with_sharpness['combined_sharpness'].isnull().all():
            if post_ml_filter_configs.get('deduplication',{}).get('enabled',False):
                 logger.warning(f"Deduplication relies on 'combined_sharpness', which is missing or all NaN for sample {sample_id}. Deduplication may not be effective.")
        dedup_config = post_ml_filter_configs.get('deduplication', {'enabled': False})
        df_after_dedup = _apply_temporal_spatial_deduplication(df_with_sharpness, dedup_config)
        
        boundary_config = post_ml_filter_configs.get('boundary_recheck', {'enabled': False})
        # MODIFIED: Pass the assembled threshold_params dictionary
        final_filtered_df = _apply_boundary_recheck(
            df_after_dedup, 
            boundary_config, 
            global_illumination_type,
            boundary_recheck_threshold_params # Pass the dict here
        )
        
        if os.path.exists(stage3_final_output_dir): 
            target_subdir_for_sample_output = os.path.join(output_dir_stage3, "ideal_fems_ML_classified")
            if os.path.exists(target_subdir_for_sample_output) and os.listdir(target_subdir_for_sample_output): 
                logger.info(f"Clearing existing content from Stage 3 output subdir: {target_subdir_for_sample_output} for sample {sample_id}")
                for item in os.listdir(target_subdir_for_sample_output):
                    item_path = os.path.join(target_subdir_for_sample_output, item)
                    if os.path.isdir(item_path): shutil.rmtree(item_path)
                    else: os.remove(item_path)

        fnames_for_final_save = feature_names_for_model if feature_names_for_model else DEFAULT_FEATURE_NAMES
        saved_csv_path, df_to_return = _save_final_s3_outputs(final_filtered_df, stage3_final_output_dir, sample_id, unified_seg_and_feature_params, fnames_for_final_save)
        
        logger.info(f"--- Stage 3: ML Classification for Sample ID: {sample_id} COMPLETED ---")
        return saved_csv_path, df_to_return

    elif mode == "train":
        logger.info(f"--- Stage 3: ML Model Training for Sample ID (context): {sample_id} COMPLETED (no prediction) ---")
        model_filename = config_s3.get('model_filename', 'rf_fem_classifier.joblib')
        if _learned_feature_names_during_training:
            models_output_dir = get_models_dir(); feature_names_path = os.path.join(models_output_dir, model_filename.replace(".joblib", "_feature_names.txt"))
            try:
                with open(feature_names_path, "w") as f:
                    for name in _learned_feature_names_during_training: f.write(f"{name}\n")
                logger.info(f"Feature names used for training saved to: {feature_names_path} (train-only mode)")
            except Exception as e: logger.exception(f"Error saving feature names in train-only mode: {e}")
        return os.path.join(get_models_dir(), model_filename), empty_return_df 
    
    logger.error(f"Invalid mode '{mode}' for Stage 3.")
    return None, empty_return_df