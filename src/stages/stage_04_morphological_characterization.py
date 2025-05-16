# src/stages/stage_04_morphological_characterization.py
import os
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
import cv2

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image
from src.core.egg_fitting import fit_two_eggs_in_area
from src.core.detailed_morphology_calculator import (
    reconstruct_main_body_masks,
    calculate_detailed_body_morphology,
    estimate_ideal_body_mask_and_area 
)
from src.visualization.stage_04_visualization import visualize_detailed_morphology
from src.core.ml_feature_extractor import extract_ml_features 

logger = logging.getLogger(__name__)

def run_stage_04_morphological_characterization(
    input_s3_csv_path: str,
    s3_accepted_images_dir: str, 
    output_dir_stage4: str,
    sample_id: str
) -> Optional[str]:
    logger.info(f"--- Stage 4: Detailed Morpho Characterization for Sample ID: {sample_id} ---")
    logger.info(f"Input Stage 3 CSV: {input_s3_csv_path}")
    logger.info(f"Input Stage 3 Accepted Images Dir: {s3_accepted_images_dir}")
    logger.info(f"Output directory for Stage 4: {output_dir_stage4}")

    config = get_config()
    config_s3 = config.get('stage_03_ml_classification', {}) 
    s3_feature_extraction_params = config_s3.get('feature_extraction_params', {})
    if not s3_feature_extraction_params:
        raise ConfigError("Stage 3 'feature_extraction_params' not found in config. Needed for Stage 4 reconstruction.")
    
    s3_results_df = load_dataframe_from_csv(input_s3_csv_path)
    if s3_results_df is None or s3_results_df.empty:
        logger.warning(f"No accepted FEMs loaded from {input_s3_csv_path}. Stage 4 cannot proceed.")
        output_csv_filename_s4 = f"{sample_id}_stage4_detailed_morphology.csv"
        s4_cols = ['egg1_area_s4', 'egg2_area_s4', 'main_area_raw_s4', 'main_area_filled_s4',
                   'solidity_s4', 'porosity_s4', 'compactness_rect_s4',
                   'extent_raw_s4', 'eccentricity_raw_s4', 'equivalent_diameter_raw_s4',
                   'major_axis_length_raw_s4', 'minor_axis_length_raw_s4',
                   'main_area_ideal_s4'] + \
                  [f'hu_moment{i+1}_s4' for i in range(7)]
        
        base_cols = list(s3_results_df.columns if s3_results_df is not None else [])
        from src.core.ml_feature_extractor import DEFAULT_FEATURE_NAMES # In case s3_results_df is None
        if not base_cols: base_cols = ['sample_id', 'filename', 'predicted_label'] + DEFAULT_FEATURE_NAMES + ['combined_sharpness']
            
        final_expected_cols = list(dict.fromkeys(base_cols + s4_cols)) 
        empty_df = pd.DataFrame(columns=final_expected_cols)
        
        return save_dataframe_to_csv(empty_df, output_dir_stage4, output_csv_filename_s4)

    output_visualizations_s4_dir = os.path.join(output_dir_stage4, "visualizations")
    os.makedirs(output_visualizations_s4_dir, exist_ok=True)

    detailed_results_list = []

    for index, s3_row in s3_results_df.iterrows():
        image_filename = s3_row.get('filename')
        s3_row_dict_for_processing = s3_row.to_dict() 

        if not image_filename:
            logger.warning(f"Missing 'filename' in Stage 3 CSV row {index}. Skipping.")
            detailed_results_list.append(s3_row_dict_for_processing) 
            continue

        full_image_path = os.path.join(s3_accepted_images_dir, image_filename)
        if not os.path.exists(full_image_path):
            logger.warning(f"Image file not found: {full_image_path} (from S3 CSV row {index}). Skipping.")
            detailed_results_list.append(s3_row_dict_for_processing)
            continue

        logger.debug(f"Processing {image_filename} for Stage 4 details...")
        current_s4_features: Dict[str, Any] = {}
        
        original_img_gray = load_image(full_image_path, mode='gray')
        if original_img_gray is None:
            logger.error(f"Failed to load image {full_image_path} for Stage 4. Skipping.")
            detailed_results_list.append(s3_row_dict_for_processing)
            continue
        
        image_shape_hw = original_img_gray.shape[:2]

        _, main_props_from_s3_logic, egg_area_props_from_s3_logic = extract_ml_features(
            full_image_path, s3_feature_extraction_params, image_array_gray=original_img_gray
        )

        egg_coords_for_fitting = None
        if egg_area_props_from_s3_logic and hasattr(egg_area_props_from_s3_logic, 'coords'):
            egg_coords_for_fitting = egg_area_props_from_s3_logic.coords
            egg1_fit, egg2_fit = fit_two_eggs_in_area(egg_coords_for_fitting, image_shape_hw)
            current_s4_features['egg1_area_s4'] = egg1_fit['area'] if egg1_fit and 'area' in egg1_fit else np.nan
            current_s4_features['egg2_area_s4'] = egg2_fit['area'] if egg2_fit and 'area' in egg2_fit else np.nan
            s3_row_dict_for_processing['_egg_fits_s4_obj'] = [egg1_fit, egg2_fit]
        else:
            current_s4_features['egg1_area_s4'] = np.nan
            current_s4_features['egg2_area_s4'] = np.nan
            s3_row_dict_for_processing['_egg_fits_s4_obj'] = None
            logger.debug(f"No egg_area_props or coords found for {image_filename} for S4 egg fitting.")

        raw_mask, filled_mask, main_props_filled_s4_recalc = reconstruct_main_body_masks(
            original_img_gray, s3_feature_extraction_params
        )
        s3_row_dict_for_processing['_raw_body_mask_s4_obj'] = raw_mask
        s3_row_dict_for_processing['_filled_body_mask_s4_obj'] = filled_mask

        ideal_body_mask_s4, main_area_ideal_s4_val = estimate_ideal_body_mask_and_area(
            filled_mask, 
            egg_coords_for_fitting, 
            image_shape_hw
        )
        # main_area_ideal_s4 will be added via calculate_detailed_body_morphology
        s3_row_dict_for_processing['_ideal_body_mask_s4_obj'] = ideal_body_mask_s4

        body_morph_features = calculate_detailed_body_morphology(
            raw_mask, filled_mask, ideal_body_mask_s4, original_img_gray
        )
        current_s4_features.update(body_morph_features)
        
        # Check if main_area_ideal_s4 was calculated, if not, ensure it's in current_s4_features as NaN
        if 'main_area_ideal_s4' not in current_s4_features:
            current_s4_features['main_area_ideal_s4'] = np.nan
            if ideal_body_mask_s4 is not None : # Should have been set by calculate_detailed_body_morphology
                 current_s4_features['main_area_ideal_s4'] = float(np.sum(ideal_body_mask_s4 > 0))


        vis_filename = f"{os.path.splitext(image_filename)[0]}_s4_morph_vis.png"
        vis_output_path = os.path.join(output_visualizations_s4_dir, vis_filename)
        
        visualize_detailed_morphology(
            full_image_path,
            main_props_filled_s4_recalc if main_props_filled_s4_recalc else main_props_from_s3_logic,
            s3_row_dict_for_processing.get('_egg_fits_s4_obj'),
            s3_row_dict_for_processing.get('_raw_body_mask_s4_obj'),
            s3_row_dict_for_processing.get('_filled_body_mask_s4_obj'), 
            s3_row_dict_for_processing.get('_ideal_body_mask_s4_obj'), 
            vis_output_path
        )

        s3_row_dict_for_processing.update(current_s4_features)
        detailed_results_list.append(s3_row_dict_for_processing)

    final_s4_df = pd.DataFrame(detailed_results_list)
    
    cols_to_drop_if_exist = ['_egg_fits_s4_obj', '_raw_body_mask_s4_obj',
                             '_filled_body_mask_s4_obj', '_ideal_body_mask_s4_obj'] 
    for col_drop in cols_to_drop_if_exist:
        if col_drop in final_s4_df.columns:
            final_s4_df = final_s4_df.drop(columns=[col_drop])

    output_csv_filename_s4 = f"{sample_id}_stage4_detailed_morphology.csv"
    saved_csv_path = save_dataframe_to_csv(final_s4_df, output_dir_stage4, output_csv_filename_s4)

    if saved_csv_path:
        logger.info(f"--- Stage 4: Detailed Morpho Characterization for {sample_id} COMPLETED ---")
    else:
        logger.error(f"--- Stage 4: Failed to save results for {sample_id} ---")
    return saved_csv_path

if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging 
    import skimage # for __main__ test image creation
    setup_logging()

    try:
        cfg_main = get_config()
        TEST_SAMPLE_ID_S4 = "test_sample_s4_morpho_ideal" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir:
            raise ValueError("project_base_dir not set in config for S4 testing.")

        s3_output_base_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S4, "stage3_ml_classification")
        s3_csv_path = os.path.join(s3_output_base_dir, "ideal_fems_ML_classified", f"{TEST_SAMPLE_ID_S4}_stage3_accepted_fem_features.csv")
        s3_images_dir = os.path.join(s3_output_base_dir, "ideal_fems_ML_classified", "accepted_images")
        
        s4_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID_S4, "stage4_morphological_characterization")
        os.makedirs(s4_output_dir, exist_ok=True)
        os.makedirs(s3_images_dir, exist_ok=True) 

        logger.info(f"Test S3 Input CSV for S4: {s3_csv_path}")
        logger.info(f"Test S3 Accepted Images Dir for S4: {s3_images_dir}")
        logger.info(f"Test S4 Output Dir: {s4_output_dir}")

        if not os.path.exists(s3_csv_path) or not os.listdir(s3_images_dir): # Check if images dir is also empty
            logger.info(f"Creating dummy S3 CSV and images for S4 test...")
            dummy_s3_data = []
            num_dummy_images = 1 
            for i in range(num_dummy_images):
                dummy_img_name = f"dummy_fem_img_ideal_test_{i}.png"
                dummy_img_path = os.path.join(s3_images_dir, dummy_img_name)
                
                test_h, test_w = 200, 200
                dummy_cv_img = np.zeros((test_h, test_w), dtype=np.uint8)
                cv2.ellipse(dummy_cv_img, (100,100), (80,40), angle=30, startAngle=0, endAngle=360, color=150, thickness=-1)
                cv2.ellipse(dummy_cv_img, (90,90), (15,10), angle=30, startAngle=0, endAngle=360, color=0, thickness=-1) 
                egg_center_y_test, egg_center_x_test = 100 + int(50*np.sin(np.deg2rad(30))), 100 + int(50*np.cos(np.deg2rad(30)))
                rr_egg_test, cc_egg_test = skimage.draw.ellipse(egg_center_y_test, egg_center_x_test, 25, 20, rotation=np.deg2rad(30+70), shape=dummy_cv_img.shape)
                
                # Ensure egg coords are within bounds for direct assignment
                rr_egg_test_clipped = np.clip(rr_egg_test, 0, test_h - 1)
                cc_egg_test_clipped = np.clip(cc_egg_test, 0, test_w - 1)
                dummy_cv_img[rr_egg_test_clipped, cc_egg_test_clipped] = 200 
                cv2.imwrite(dummy_img_path, dummy_cv_img)

                from src.core.ml_feature_extractor import DEFAULT_FEATURE_NAMES
                row_data = {'sample_id': TEST_SAMPLE_ID_S4, 'filename': dummy_img_name, 'predicted_label': 1}
                temp_ml_params = cfg_main.get('stage_03_ml_classification', {}).get('feature_extraction_params',{})
                feats_dict, _, _ = extract_ml_features(dummy_img_path, temp_ml_params, image_array_gray=dummy_cv_img)
                if feats_dict:
                    for feat_name in DEFAULT_FEATURE_NAMES: # Ensure DEFAULT_FEATURE_NAMES is correct
                        row_data[feat_name] = feats_dict.get(feat_name, np.random.rand()*100)
                else: 
                     for feat_name in DEFAULT_FEATURE_NAMES: row_data[feat_name] = np.random.rand()*100
                dummy_s3_data.append(row_data)

            pd.DataFrame(dummy_s3_data).to_csv(s3_csv_path, index=False)
            logger.info(f"Dummy S3 CSV created at: {s3_csv_path}")
            logger.info(f"Dummy S3 images created in: {s3_images_dir}")

        if not os.path.exists(s3_csv_path) or not os.listdir(s3_images_dir):
            logger.error(f"S3 CSV ('{s3_csv_path}') or S3 images dir ('{s3_images_dir}') is still missing or empty. S4 test cannot run.")
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
                assert 'main_area_ideal_s4' in df_res_s4.columns, "'main_area_ideal_s4' not found in S4 output."
                logger.info("Column 'main_area_ideal_s4' check passed.")
            else:
                logger.error("Stage 4 test FAILED or produced no output CSV.")

    except ConfigError as ce: logger.error(f"Configuration error during Stage 4 test: {ce}")
    except FileNotFoundError as fnfe: logger.error(f"File not found during Stage 4 test setup: {fnfe}")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 4 test: {e}")