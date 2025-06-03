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

logger = logging.getLogger(__name__)

def run_stage_04_morphological_characterization(
    input_s3_csv_path: str,
    s3_accepted_images_dir: str, 
    output_dir_stage4: str,
    sample_id: str
) -> Optional[str]:
    logger.info(f"--- Stage 4: Detailed Morpho Characterization for Sample ID: {sample_id} ---")
    config = get_config() 
    
    system_config = config.get('system', {})
    global_source_illumination = system_config.get('source_image_illumination_type', 'darkfield')
    logger.info(f"Stage 4 using global source image illumination: {global_source_illumination}")

    config_s3_for_s4 = config.get('stage_03_ml_classification', {}) # S4 uses S3's feature extraction params
    s3_feature_extraction_params_base = config_s3_for_s4.get('feature_extraction_params', {})
    
    # Add global illumination to S3 feature params for use in S4
    s3_params_for_s4_processing = s3_feature_extraction_params_base.copy()
    s3_params_for_s4_processing['source_image_illumination_type'] = global_source_illumination


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
                                DEFAULT_FEATURE_NAMES + ['combined_sharpness'] # Use updated DEFAULT_FEATURE_NAMES
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
        
        img_loaded_for_s4 = load_image(full_image_path, mode='gray') 
        if img_loaded_for_s4 is None:
            logger.error(f"Failed to load image {full_image_path} for S4. Skipping S4.")
            detailed_results_list.append(current_row_data_dict); continue
        original_img_gray_float = skimage.util.img_as_float(img_loaded_for_s4)

        # Get S3 main_props for visualization (orientation, posterior end)
        # extract_ml_features uses s3_params_for_s4_processing (which includes global_source_illumination)
        _, main_props_from_s3_logic, _ = extract_ml_features(
            full_image_path, s3_params_for_s4_processing, image_array_gray=original_img_gray_float.copy()
        )

        # Reconstruct masks. reconstruct_main_body_masks also uses s3_params_for_s4_processing for illumination.
        # It expects the *original un-inverted* float image as input.
        raw_mask, filled_mask, main_props_filled_s4_recalc = reconstruct_main_body_masks(
            original_img_gray_float.copy(), 
            s3_params_for_s4_processing
        )
        
        # For calculate_detailed_body_morphology, prepare the intensity image based on global illumination
        img_for_s4_intensity_props = original_img_gray_float.copy()
        if global_source_illumination == 'collimated':
            img_for_s4_intensity_props = 1.0 - img_for_s4_intensity_props
        
        body_morph_features = calculate_detailed_body_morphology(
            raw_mask, filled_mask, img_for_s4_intensity_props
        )
        current_row_data_dict.update(body_morph_features)
        
        vis_filename = f"{os.path.splitext(image_filename)[0]}_s4_morph_vis.png"
        vis_output_path = os.path.join(output_visualizations_s4_dir, vis_filename)
        main_props_for_vis = main_props_from_s3_logic if main_props_from_s3_logic else main_props_filled_s4_recalc

        try:
            visualize_detailed_morphology(
                original_img_gray_float, # Pass original non-inverted for display
                main_props_for_vis, 
                raw_mask, 
                filled_mask, 
                vis_output_path
            )
        except Exception as e_vis:
            logger.error(f"Error during S4 visualization for {image_filename}: {e_vis}")

        detailed_results_list.append(current_row_data_dict)

    logger.info(f"Stage 4: Processed {processed_image_count} images from S3 input.")
    if not detailed_results_list: 
        logger.error("Stage 4: detailed_results_list empty. No CSV saved.")
        output_csv_filename_s4_empty = f"{sample_id}_stage4_detailed_morphology.csv"
        empty_df_for_save = pd.DataFrame(columns=final_expected_cols_for_empty_s4_df)
        save_dataframe_to_csv(empty_df_for_save, output_dir_stage4, output_csv_filename_s4_empty)
        return None

    final_s4_df = pd.DataFrame(detailed_results_list)
    for col in final_expected_cols_for_empty_s4_df:
        if col not in final_s4_df.columns: final_s4_df[col] = np.nan
    final_s4_df = final_s4_df[final_expected_cols_for_empty_s4_df] 

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
        # Ensure system config for illumination type is present for testing
        system_cfg_test = cfg_main.setdefault('system', {})
        system_cfg_test.setdefault('source_image_illumination_type', 'collimated') # Example for testing

        TEST_SAMPLE_ID_S4 = "test_sample_s4_global_illum" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir: raise ValueError("project_base_dir not set in config for S4 testing.")

        s3_output_base_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S4, "stage3_ml_classification")
        s3_csv_path = os.path.join(s3_output_base_dir, "ideal_fems_ML_classified", f"{TEST_SAMPLE_ID_S4}_stage3_accepted_fem_features.csv")
        s3_images_dir = os.path.join(s3_output_base_dir, "ideal_fems_ML_classified", "accepted_images")
        
        s4_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S4, "stage4_morph_global_illum")
        os.makedirs(s4_output_dir, exist_ok=True)
        os.makedirs(s3_images_dir, exist_ok=True) 

        # ... (dummy S3 data creation - ensure ml_params used by extract_ml_features includes source_image_illumination_type) ...
        if not os.path.exists(s3_csv_path) or not any(os.path.isfile(os.path.join(s3_images_dir, f)) for f in os.listdir(s3_images_dir)):
            logger.info(f"Creating dummy S3 CSV and images for S4 test (global illumination aware)...")
            dummy_s3_data = []
            num_dummy_images = 2 
            temp_ml_params_for_s3_dummy = cfg_main.get('stage_03_ml_classification', {}).get('feature_extraction_params',{}).copy()
            temp_ml_params_for_s3_dummy['source_image_illumination_type'] = system_cfg_test['source_image_illumination_type']

            for i in range(num_dummy_images):
                dummy_img_name = f"dummy_fem_img_global_illum_s4test_{i}.png"
                dummy_img_path = os.path.join(s3_images_dir, dummy_img_name)
                dummy_cv_img_uint8 = np.zeros((200, 200), dtype=np.uint8) # Dark BG
                if system_cfg_test['source_image_illumination_type'] == 'collimated': # Create dark particles on bright BG
                    dummy_cv_img_uint8 = np.full((200,200), 220, dtype=np.uint8)
                    cv2.ellipse(dummy_cv_img_uint8, (100,100), (70,35), angle=20, startAngle=0, endAngle=360, color=50, thickness=-1)
                    cv2.ellipse(dummy_cv_img_uint8, (90,90), (10,5), angle=20, startAngle=0, endAngle=360, color=220, thickness=-1) 
                else: # Darkfield: bright particles on dark BG
                    cv2.ellipse(dummy_cv_img_uint8, (100,100), (70,35), angle=20, startAngle=0, endAngle=360, color=150, thickness=-1)
                    cv2.ellipse(dummy_cv_img_uint8, (90,90), (10,5), angle=20, startAngle=0, endAngle=360, color=0, thickness=-1) 
                cv2.imwrite(dummy_img_path, dummy_cv_img_uint8)
                
                row_data = {'sample_id': TEST_SAMPLE_ID_S4, 'filename': dummy_img_name, 'predicted_label': 1, 'combined_sharpness': 0.8}
                dummy_img_float = skimage.util.img_as_float(dummy_cv_img_uint8)
                feats_dict, _, _ = extract_ml_features(dummy_img_path, temp_ml_params_for_s3_dummy, image_array_gray=dummy_img_float)
                if feats_dict: row_data.update({fn: feats_dict.get(fn, 0.0) for fn in DEFAULT_FEATURE_NAMES})
                else: row_data.update({fn: np.random.rand() for fn in DEFAULT_FEATURE_NAMES})
                dummy_s3_data.append(row_data)
            pd.DataFrame(dummy_s3_data).to_csv(s3_csv_path, index=False)

        if not os.path.exists(s3_csv_path) or not any(os.path.isfile(os.path.join(s3_images_dir, f)) for f in os.listdir(s3_images_dir)):
            logger.error(f"S3 CSV or images still missing for S4 test.")
        else:
            result_s4_csv_path = run_stage_04_morphological_characterization(
                input_s3_csv_path=s3_csv_path,
                s3_accepted_images_dir=s3_images_dir,
                output_dir_stage4=s4_output_dir,
                sample_id=TEST_SAMPLE_ID_S4
            )
            if result_s4_csv_path:
                logger.info(f"Stage 4 test completed. Results CSV: {result_s4_csv_path}")
                df_res_s4 = pd.read_csv(result_s4_csv_path)
                print("Stage 4 Test Output DataFrame Head:")
                print(df_res_s4.head())
            else: logger.error("Stage 4 test FAILED or produced no output CSV.")

    except ConfigError as ce: logger.error(f"Configuration error during Stage 4 test: {ce}")
    except FileNotFoundError as fnfe: logger.error(f"File not found during Stage 4 test setup: {fnfe}")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 4 test: {e}")