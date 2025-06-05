# src/stages/stage_04_morphological_characterization.py
import os
import pandas as pd
import numpy as np
import logging
import skimage.util 
from typing import Optional, List, Dict, Any

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image
from src.core.detailed_morphology_calculator import (
    reconstruct_main_body_masks,
    calculate_detailed_body_morphology
)
from src.visualization.stage_04_visualization import visualize_detailed_morphology
from src.core.ml_feature_extractor import extract_ml_features, DEFAULT_FEATURE_NAMES 
from src.core.segmentation import segment_main_body_from_image_float # For direct call if needed for props, or reference

logger = logging.getLogger(__name__)

def run_stage_04_morphological_characterization(
    input_s3_csv_path: str,
    s3_accepted_images_dir: str, 
    output_dir_stage4: str,
    sample_id: str
) -> Optional[str]:
    logger.info(f"--- Stage 4: Detailed Morpho Characterization for Sample ID: {sample_id} (Unified Seg) ---")
    config = get_config() 
    
    # Get global system settings (illumination and thresholding)
    global_system_config = config.get('system', {})
    global_threshold_cfg = global_system_config.get('thresholding')
    if not global_threshold_cfg: # Check if the dict itself is missing or empty
        raise ConfigError("System-wide thresholding configuration ('system.thresholding') not found or empty.")
    
    global_source_illumination = global_system_config.get('source_image_illumination_type', 'darkfield')
    
    # S4 uses S3's feature_extraction_params and combines them with global settings
    config_s3_for_s4 = config.get('stage_03_ml_classification', {}) 
    # These are the base params from S3 config (e.g., unsharp, egg params, min_noise, closing_selem)
    s3_feature_extraction_params_base = config_s3_for_s4.get('feature_extraction_params', {})
    
    # Prepare the comprehensive params dict for segmentation and S3 logic in S4
    # This dict will be passed to reconstruct_main_body_masks (which calls segment_main_body)
    # and also to extract_ml_features if used to get S3 main_props for visualization.
    params_for_s4_segmentation_and_s3_logic = s3_feature_extraction_params_base.copy()
    params_for_s4_segmentation_and_s3_logic['source_image_illumination_type'] = global_source_illumination
    params_for_s4_segmentation_and_s3_logic['thresholding_mode'] = global_threshold_cfg.get('mode', 'otsu')
    params_for_s4_segmentation_and_s3_logic['otsu_threshold_multiplier'] = global_threshold_cfg.get('otsu_threshold_multiplier', 0.45)
    params_for_s4_segmentation_and_s3_logic['fixed_threshold_value'] = global_threshold_cfg.get('fixed_threshold_value', 70)
    # Ensure unsharp params are present, defaulting if necessary (though config should provide them now)
    params_for_s4_segmentation_and_s3_logic.setdefault('enable_unsharp_masking', False)
    params_for_s4_segmentation_and_s3_logic.setdefault('unsharp_mask_sigma', 1.0)
    params_for_s4_segmentation_and_s3_logic.setdefault('unsharp_mask_amount', 1.0)

    logger.debug(f"Effective parameters for Stage 4 processing (segmentation & S3 logic): {params_for_s4_segmentation_and_s3_logic}")

    logger.info(f"Stage 4: Attempting to load S3 results from: {input_s3_csv_path}")
    s3_results_df = load_dataframe_from_csv(input_s3_csv_path)

    s4_morph_feature_keys = [
        'main_area_raw_s4', 'solidity_s4', 'compactness_rect_s4', 'extent_raw_s4',
        'eccentricity_raw_s4', 'equivalent_diameter_raw_s4',
        'major_axis_length_raw_s4', 'minor_axis_length_raw_s4',
        'main_area_filled_s4', 'porosity_s4', 'hu_moment1_s4',
        'moment_of_inertia_xx_s4', 'moment_of_inertia_yy_s4', 'moment_of_inertia_xy_s4',
        'type_factor_s4', 'heywood_circularity_factor_s4', 'elongation_factor_s4',
        'compactness_factor_ni_s4'
    ]
    base_cols_for_empty_s4_df = ['sample_id', 'filename', 'predicted_label'] + \
                                DEFAULT_FEATURE_NAMES + ['combined_sharpness'] 
    final_expected_cols_for_empty_s4_df = list(dict.fromkeys(base_cols_for_empty_s4_df + s4_morph_feature_keys))

    if s3_results_df is None or s3_results_df.empty:
        logger.warning(f"Stage 4: S3 results DataFrame from {input_s3_csv_path} is empty or failed to load. No FEMs to process.")
        output_csv_filename_s4_empty = f"{sample_id}_stage4_detailed_morphology.csv"
        empty_df_for_save = pd.DataFrame(columns=final_expected_cols_for_empty_s4_df)
        return save_dataframe_to_csv(empty_df_for_save, output_dir_stage4, output_csv_filename_s4_empty)

    logger.info(f"Stage 4: Loaded {len(s3_results_df)} entries from S3 results for sample {sample_id}.")

    output_visualizations_s4_dir = os.path.join(output_dir_stage4, "visualizations")
    os.makedirs(output_visualizations_s4_dir, exist_ok=True)

    detailed_results_list = []
    processed_image_count = 0

    for index, s3_row in s3_results_df.iterrows():
        processed_image_count +=1
        logger.debug(f"Stage 4: Processing S3 entry {index+1}/{len(s3_results_df)}...")
        image_filename = s3_row.get('filename')
        current_row_data_dict = s3_row.to_dict() 

        for key in s4_morph_feature_keys: 
            current_row_data_dict[key] = np.nan 

        if not image_filename:
            logger.warning(f"Missing 'filename' in S3 CSV row index {index}. Skipping S4.")
            detailed_results_list.append(current_row_data_dict); continue

        full_image_path = os.path.join(s3_accepted_images_dir, image_filename)
        if not os.path.exists(full_image_path):
            logger.warning(f"Image file not found: {full_image_path}. Skipping S4.")
            detailed_results_list.append(current_row_data_dict); continue
        
        img_loaded_for_s4_uint8 = load_image(full_image_path, mode='gray') 
        if img_loaded_for_s4_uint8 is None:
            logger.error(f"Failed to load image {full_image_path} for S4. Skipping S4.")
            detailed_results_list.append(current_row_data_dict); continue
        original_img_gray_float = skimage.util.img_as_float(img_loaded_for_s4_uint8)

        # Re-run S3's extract_ml_features to get main_props_s3 (which uses the unified segmentation)
        # for consistent visualization with S3's perspective.
        # params_for_s4_segmentation_and_s3_logic is used here.
        _, main_props_from_s3_logic_for_vis, _ = extract_ml_features(
            full_image_path, params_for_s4_segmentation_and_s3_logic, image_array_gray=original_img_gray_float.copy()
        )

        # Get masks for S4 detailed morphology using reconstruct_main_body_masks
        # which now internally calls the unified segment_main_body_from_image_float.
        # params_for_s4_segmentation_and_s3_logic is also used here.
        raw_mask_s4, filled_mask_s4, main_props_filled_s4_recalc = reconstruct_main_body_masks(
            original_img_gray_float.copy(), 
            params_for_s4_segmentation_and_s3_logic 
        )
        
        # Prepare intensity image for calculate_detailed_body_morphology:
        # This should be the image *after* inversion (if any) but *before* unsharp masking.
        img_for_s4_intensity_props = original_img_gray_float.copy()
        if params_for_s4_segmentation_and_s3_logic['source_image_illumination_type'] == 'collimated':
            img_for_s4_intensity_props = 1.0 - img_for_s4_intensity_props
        # Unsharp masking (if enabled) happens inside segment_main_body_from_image_float
        # and its result is used for thresholding to get masks, but not for this intensity image.
        
        body_morph_features = calculate_detailed_body_morphology(
            raw_mask_s4, filled_mask_s4, img_for_s4_intensity_props
        )
        current_row_data_dict.update(body_morph_features)
        
        vis_filename_s4 = f"{os.path.splitext(image_filename)[0]}_s4_morph_vis.png"
        vis_output_path_s4 = os.path.join(output_visualizations_s4_dir, vis_filename_s4)
        
        # For visualization, use main_props from S3 logic if available, otherwise the one from S4 recalc.
        # This provides the posterior end marker consistent with S3's view.
        main_props_for_visualization = main_props_from_s3_logic_for_vis if main_props_from_s3_logic_for_vis else main_props_filled_s4_recalc

        try:
            visualize_detailed_morphology(
                original_img_gray_float, # Pass original float image
                main_props_for_visualization, 
                raw_mask_s4, 
                filled_mask_s4, 
                vis_output_path_s4
            )
        except Exception as e_vis:
            logger.error(f"Error during S4 visualization for {image_filename}: {e_vis}")

        detailed_results_list.append(current_row_data_dict)

    logger.info(f"Stage 4: Processed {processed_image_count} images from S3 input.")
    if not detailed_results_list: 
        logger.info("Stage 4: detailed_results_list empty. No CSV saved.") # Changed from error to info
        output_csv_filename_s4_empty = f"{sample_id}_stage4_detailed_morphology.csv"
        empty_df_for_save = pd.DataFrame(columns=final_expected_cols_for_empty_s4_df)
        save_dataframe_to_csv(empty_df_for_save, output_dir_stage4, output_csv_filename_s4_empty)
        return None # Return None if no results

    final_s4_df = pd.DataFrame(detailed_results_list)
    for col in final_expected_cols_for_empty_s4_df: # Ensure all expected columns are present
        if col not in final_s4_df.columns: final_s4_df[col] = np.nan
    final_s4_df = final_s4_df[final_expected_cols_for_empty_s4_df] # Order and select columns

    logger.info(f"Stage 4: Created final DataFrame with {len(final_s4_df)} rows for {sample_id}.")
    
    output_csv_filename_s4 = f"{sample_id}_stage4_detailed_morphology.csv"
    saved_csv_path = save_dataframe_to_csv(final_s4_df, output_dir_stage4, output_csv_filename_s4)

    if saved_csv_path: logger.info(f"--- Stage 4: Detailed Morpho Characterization for {sample_id} COMPLETED ---")
    else: logger.error(f"--- Stage 4: Failed to save results for {sample_id} ---")
    return saved_csv_path

if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging 
    import skimage 
    import cv2 
    setup_logging()

    try:
        cfg_main = get_config()
        system_cfg_test = cfg_main.setdefault('system', {})
        system_cfg_test.setdefault('source_image_illumination_type', 'collimated')
        thresh_cfg_test = system_cfg_test.setdefault('thresholding', {})
        thresh_cfg_test.setdefault('mode', 'otsu')
        thresh_cfg_test.setdefault('otsu_threshold_multiplier', 0.45)
        thresh_cfg_test.setdefault('fixed_threshold_value', 70)
        
        # Ensure S3 feature_extraction_params exist for test
        s3_cfg_test = cfg_main.setdefault('stage_03_ml_classification', {})
        s3_feat_params_test = s3_cfg_test.setdefault('feature_extraction_params', {})
        s3_feat_params_test.setdefault('enable_unsharp_masking', False)
        s3_feat_params_test.setdefault('unsharp_mask_sigma', 1.0)
        s3_feat_params_test.setdefault('unsharp_mask_amount', 1.0)
        s3_feat_params_test.setdefault('min_noise_size', 10)
        s3_feat_params_test.setdefault('closing_selem_size', 1)
        # Add other S3 params if extract_ml_features in S4 needs them (e.g. egg params for main_props_from_s3_logic_for_vis)
        s3_feat_params_test.setdefault('gaussian_sigma', 1.0)
        s3_feat_params_test.setdefault('min_significant_area', 50)
        s3_feat_params_test.setdefault('egg_posterior_pixel_fraction', 0.4)
        s3_feat_params_test.setdefault('egg_threshold_quantile', 0.8)
        s3_feat_params_test.setdefault('egg_min_area', 20)


        TEST_SAMPLE_ID_S4 = "test_sample_s4_unified_seg" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir: raise ValueError("project_base_dir not set in config for S4 testing.")

        s3_output_base_dir_test = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S4, "stage3_ml_classification")
        s3_csv_path_test = os.path.join(s3_output_base_dir_test, "ideal_fems_ML_classified", f"{TEST_SAMPLE_ID_S4}_stage3_accepted_fem_features.csv")
        s3_images_dir_test = os.path.join(s3_output_base_dir_test, "ideal_fems_ML_classified", "accepted_images")
        
        s4_output_dir_test = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S4, "stage4_morph_unified_seg")
        os.makedirs(s4_output_dir_test, exist_ok=True)
        os.makedirs(s3_images_dir_test, exist_ok=True) 

        if not os.path.exists(s3_csv_path_test) or not any(os.path.isfile(os.path.join(s3_images_dir_test, f)) for f in os.listdir(s3_images_dir_test)):
            logger.info(f"Creating dummy S3 CSV and images for S4 unified seg test...")
            dummy_s3_data_list = []
            num_dummy_images = 2 
            
            # Params for dummy S3 feature extraction (to populate the dummy CSV)
            temp_ml_params_for_s3_dummy_creation = s3_feat_params_test.copy()
            temp_ml_params_for_s3_dummy_creation['source_image_illumination_type'] = system_cfg_test['source_image_illumination_type']
            temp_ml_params_for_s3_dummy_creation['thresholding_mode'] = thresh_cfg_test['mode']
            temp_ml_params_for_s3_dummy_creation['otsu_threshold_multiplier'] = thresh_cfg_test['otsu_threshold_multiplier']
            temp_ml_params_for_s3_dummy_creation['fixed_threshold_value'] = thresh_cfg_test['fixed_threshold_value']

            for i in range(num_dummy_images):
                dummy_img_name_s4test = f"dummy_s4_test_img_unified_{i}.png"
                dummy_img_path_s4test = os.path.join(s3_images_dir_test, dummy_img_name_s4test)
                dummy_cv_img_uint8_s4test = np.zeros((200, 200), dtype=np.uint8) 
                if system_cfg_test['source_image_illumination_type'] == 'collimated': 
                    dummy_cv_img_uint8_s4test = np.full((200,200), 220, dtype=np.uint8)
                    cv2.ellipse(dummy_cv_img_uint8_s4test, (100,100), (70,35), angle=20, startAngle=0, endAngle=360, color=50, thickness=-1)
                    cv2.ellipse(dummy_cv_img_uint8_s4test, (90,90), (10,5), angle=20, startAngle=0, endAngle=360, color=220, thickness=-1) 
                else: 
                    cv2.ellipse(dummy_cv_img_uint8_s4test, (100,100), (70,35), angle=20, startAngle=0, endAngle=360, color=150, thickness=-1)
                    cv2.ellipse(dummy_cv_img_uint8_s4test, (90,90), (10,5), angle=20, startAngle=0, endAngle=360, color=0, thickness=-1) 
                cv2.imwrite(dummy_img_path_s4test, dummy_cv_img_uint8_s4test)
                
                row_data_s4test = {'sample_id': TEST_SAMPLE_ID_S4, 'filename': dummy_img_name_s4test, 'predicted_label': 1, 'combined_sharpness': 0.8}
                dummy_img_float_s4test = skimage.util.img_as_float(dummy_cv_img_uint8_s4test)
                feats_dict_s4test, _, _ = extract_ml_features(dummy_img_path_s4test, temp_ml_params_for_s3_dummy_creation, image_array_gray=dummy_img_float_s4test)
                if feats_dict_s4test: row_data_s4test.update({fn: feats_dict_s4test.get(fn, 0.0) for fn in DEFAULT_FEATURE_NAMES})
                else: row_data_s4test.update({fn: np.random.rand() for fn in DEFAULT_FEATURE_NAMES}) # Fallback
                dummy_s3_data_list.append(row_data_s4test)
            pd.DataFrame(dummy_s3_data_list).to_csv(s3_csv_path_test, index=False)

        result_s4_csv_path_test = run_stage_04_morphological_characterization(
            input_s3_csv_path=s3_csv_path_test,
            s3_accepted_images_dir=s3_images_dir_test,
            output_dir_stage4=s4_output_dir_test,
            sample_id=TEST_SAMPLE_ID_S4
        )
        if result_s4_csv_path_test:
            logger.info(f"Stage 4 unified seg test completed. Results CSV: {result_s4_csv_path_test}")
            df_res_s4_test = pd.read_csv(result_s4_csv_path_test)
            print("Stage 4 Test Output DataFrame Head (Unified Seg):")
            print(df_res_s4_test.head())
            # Check if new S4 morph columns are present
            assert 'main_area_raw_s4' in df_res_s4_test.columns, "Missing new S4 morph column"
            assert 'porosity_s4' in df_res_s4_test.columns, "Missing new S4 morph column"
        else: logger.error("Stage 4 unified seg test FAILED or produced no output CSV.")

    except ConfigError as ce: logger.error(f"Configuration error during Stage 4 test: {ce}")
    except FileNotFoundError as fnfe: logger.error(f"File not found during Stage 4 test setup: {fnfe}")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 4 test: {e}")