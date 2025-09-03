# src/stages/stage_02_quality_filter.py
import os
import pandas as pd
import numpy as np
import cv2 
import logging
import shutil 
from typing import Optional, List, Dict, Any, Tuple

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image, save_image, convert_to_grayscale
from src.core.particle_operations import ParticleCropper, detect_cropped_particle_on_boundary
from src.core.masking_operations import detect_black_mask_artifact
from src.core.sharpness_assessment import calculate_sharpness_metrics, normalize_and_combine_sharpness, AVAILABLE_SHARPNESS_METRICS

logger = logging.getLogger(__name__)

class SingleParticleAnalysisResult: 
    def __init__(self, particle_data_in: pd.Series):
        self.input_particle_data = particle_data_in
        self.centroid_x: Optional[int] = particle_data_in.get('X')
        self.centroid_y: Optional[int] = particle_data_in.get('Y')
        self.cropped_image_filename: Optional[str] = None
        self.is_black_mask_artifact: Optional[bool] = None
        self.is_on_crop_boundary: Optional[bool] = None 
        self.sharpness_scores: Dict[str, float] = {key: np.nan for key in AVAILABLE_SHARPNESS_METRICS}
        self.error_message: Optional[str] = None
        self.processed_successfully: bool = False

    def to_dict(self) -> Dict[str, Any]: 
        base_dict = {
            'original_X': self.input_particle_data.get('X'),
            'original_Y': self.input_particle_data.get('Y'),
            'input_particle_index': self.input_particle_data.name,
            'analyzed_centroid_x': self.centroid_x, 
            'analyzed_centroid_y': self.centroid_y,
            'cropped_filename': self.cropped_image_filename,
            'is_black_mask': self.is_black_mask_artifact,
            'is_on_crop_boundary': self.is_on_crop_boundary,
            **self.sharpness_scores 
        }
        return base_dict

def _process_single_particle_for_quality(
    particle_data_series: pd.Series,
    main_full_color_image: np.ndarray,
    main_full_gray_image: np.ndarray, 
    cropper: ParticleCropper,
    temp_intermediate_crop_dir: str, 
    config_s2: Dict[str, Any], 
    source_image_illumination: str, 
    global_threshold_params_for_boundary_check: Dict[str, Any] 
) -> SingleParticleAnalysisResult:
    result = SingleParticleAnalysisResult(particle_data_series)
    particle_id_info = f"(idx={particle_data_series.name}, X={result.centroid_x}, Y={result.centroid_y})"
    logger.debug(f"Processing particle {particle_id_info} for quality (source illum: {source_image_illumination})...")

    try:
        intermediate_crop_prefix = f"intermediate_crop_idx{particle_data_series.name}"
        
        intermediate_crop_gray_uint8, saved_intermediate_color_crop_path = cropper.crop_single_particle_from_image(
            main_full_color_image, main_full_gray_image, particle_data_series,
            output_dir_for_crop=temp_intermediate_crop_dir,
            crop_filename_prefix=intermediate_crop_prefix
        )

        if intermediate_crop_gray_uint8 is None or saved_intermediate_color_crop_path is None:
            logger.warning(f"Cropping failed or intermediate crop not saved for {particle_id_info}.")
            result.error_message = "Cropping or intermediate save failed"
            return result 

        result.cropped_image_filename = os.path.basename(saved_intermediate_color_crop_path)
        result.sharpness_scores = calculate_sharpness_metrics(intermediate_crop_gray_uint8)
        
        black_mask_check_config = config_s2.get('black_mask_check', {})
        result.is_black_mask_artifact = detect_black_mask_artifact(
            image_path=saved_intermediate_color_crop_path, 
            min_area_ratio=black_mask_check_config.get('min_area_ratio', 0.05)
        )
        
        result.is_on_crop_boundary = detect_cropped_particle_on_boundary(
            intermediate_crop_gray_uint8, 
            saved_intermediate_color_crop_path,
            illumination_mode_of_crop=source_image_illumination, 
            threshold_params=global_threshold_params_for_boundary_check 
        )
        result.processed_successfully = True
        logger.debug(f"Successfully processed particle {particle_id_info} for quality.")

    except DataValidationError as dve:
        logger.error(f"Data validation error for particle {particle_id_info}: {dve}")
        result.error_message = str(dve)
    except Exception as e:
        logger.exception(f"Unexpected error processing particle {particle_id_info} for quality: {e}")
        result.error_message = f"Unexpected error: {str(e)}"
    return result

def _filter_and_score_analysis_results( 
    analysis_results_df: pd.DataFrame,
    config_s2: Dict[str, Any] 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    logger.info("Filtering and scoring particle analysis results...")
    s2_filters_enabled_config = config_s2.get('filters_enabled', {})
    enable_black_mask_filter = s2_filters_enabled_config.get('black_mask_artifact', True)
    enable_on_boundary_filter = s2_filters_enabled_config.get('on_crop_boundary', True)
    if analysis_results_df.empty:
        logger.warning("No analysis results to filter or score.")
        expected_cols = list(analysis_results_df.columns)
        for metric in config_s2.get('sharpness_metrics', AVAILABLE_SHARPNESS_METRICS): expected_cols.append(metric + '_norm')
        expected_cols.append('combined_sharpness')
        return pd.DataFrame(columns=expected_cols), pd.DataFrame(columns=analysis_results_df.columns)
    analysis_results_df['is_black_mask'] = analysis_results_df['is_black_mask'].fillna(True).astype(bool)
    analysis_results_df['is_on_crop_boundary'] = analysis_results_df['is_on_crop_boundary'].fillna(True).astype(bool)
    suitable_mask = pd.Series(True, index=analysis_results_df.index)
    if enable_black_mask_filter: suitable_mask &= (~analysis_results_df['is_black_mask']); logger.info(f"Black Mask Artifact Filter: ENABLED. Current suitable: {suitable_mask.sum()}/{len(analysis_results_df)}")
    else: logger.info("Black Mask Artifact Filter: DISABLED.")
    if enable_on_boundary_filter: suitable_mask &= (~analysis_results_df['is_on_crop_boundary']); logger.info(f"On Crop Boundary Filter: ENABLED. Current suitable: {suitable_mask.sum()}/{len(analysis_results_df)}")
    else: logger.info("On Crop Boundary Filter: DISABLED.")
    suitable_df = analysis_results_df[suitable_mask].copy(); count_in = len(analysis_results_df); count_out = len(suitable_df)
    logger.info(f"After artifact filters (black_mask_enabled={enable_black_mask_filter}, on_boundary_enabled={enable_on_boundary_filter}): {count_out} suitable remaining from {count_in} ({count_in - count_out} removed).")
    if suitable_df.empty:
        logger.warning("No suitable objects found after artifact filtering (or all filters disabled and input was empty).")
        all_analyzed_with_scores = analysis_results_df.copy()
        for metric in config_s2.get('sharpness_metrics', AVAILABLE_SHARPNESS_METRICS):
            if metric + '_norm' not in all_analyzed_with_scores.columns: all_analyzed_with_scores[metric + '_norm'] = np.nan
        if 'combined_sharpness' not in all_analyzed_with_scores.columns: all_analyzed_with_scores['combined_sharpness'] = np.nan
        return pd.DataFrame(columns=all_analyzed_with_scores.columns), all_analyzed_with_scores
    metrics_for_norm = config_s2.get('sharpness_metrics', AVAILABLE_SHARPNESS_METRICS)
    suitable_scored_df = normalize_and_combine_sharpness(suitable_df, metrics_for_norm)
    all_analyzed_df_with_scores = analysis_results_df.copy()
    score_cols_to_add = [m + '_norm' for m in metrics_for_norm] + ['combined_sharpness']
    for sc_col in score_cols_to_add:
        if sc_col not in all_analyzed_df_with_scores.columns: all_analyzed_df_with_scores[sc_col] = np.nan
    all_analyzed_df_with_scores.update(suitable_scored_df[score_cols_to_add])
    return suitable_scored_df, all_analyzed_df_with_scores

def _save_final_sharp_object_crops( 
    sharp_objects_df: pd.DataFrame, 
    image_dir_path: str, 
    final_sharp_crops_output_dir: str,
    # MODIFIED: Pass the new flexible settings dictionary
    final_crop_settings: Dict[str, Any],
    # MODIFIED: Pass the crop mode to be used
    crop_mode: str,
    sample_info: Dict[str, str] 
    ):
    logger.info(f"Saving final sharp object crops to: {final_sharp_crops_output_dir} using mode '{crop_mode}'")
    os.makedirs(final_sharp_crops_output_dir, exist_ok=True)
    sharp_crops_saved_count = 0

    # MODIFIED: Construct the configuration for the final cropper dynamically.
    final_cropper_cfg = {'mode': crop_mode}
    if crop_mode == 'centroid_square':
        final_cropper_cfg['square_size'] = final_crop_settings.get('centroid_square_size', 150)
    elif crop_mode == 'rotational_rectangle':
        final_cropper_cfg['rotational_padding'] = final_crop_settings.get('rotational_padding', 10)
    
    try:
        final_cropper = ParticleCropper(final_cropper_cfg)
    except ConfigError as e:
        logger.error(f"Could not initialize final cropper for mode '{crop_mode}': {e}")
        return

    for image_filename, group in sharp_objects_df.groupby('source_image_filename'):
        full_image_path_orig = os.path.join(image_dir_path, image_filename)
        original_color_img = load_image(full_image_path_orig, mode='color')
        if original_color_img is None:
            logger.warning(f"Could not load original image {full_image_path_orig} for final cropping. Skipping {len(group)} sharp objects from this image.")
            continue
        logger.debug(f"Loaded {image_filename} for final sharp object cropping.")
        
        # A gray image is needed by crop_single_particle_from_image signature, even if not always used by the logic.
        dummy_gray_for_final_crop = convert_to_grayscale(original_color_img)
        if dummy_gray_for_final_crop is None:
            logger.error(f"Could not create dummy gray for final crop from {image_filename}. Skipping its sharp objects.")
            continue

        for _, particle_row in group.iterrows():
            # This data is needed for both modes
            try: coord_x = int(particle_row['X']) ; coord_y = int(particle_row['Y'])
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Missing or invalid coordinates for particle in {image_filename} (X={particle_row.get('X')}, Y={particle_row.get('Y')}). Error: {e}. Skipping crop.")
                continue

            frame_num = extract_frame_from_filename(image_filename) 
            if frame_num is None: frame_num_str = "UnkF"
            else: frame_num_str = f"F{frame_num:05d}"
            
            output_crop_filename_prefix = (f"{sample_info['date']}_plate{sample_info['plate']}_well{sample_info['well']}_{frame_num_str}_sharp")
            
            # The particle_row already contains all necessary data (like rot_rect_*)
            # The cropper will pick the columns it needs based on its mode.
            _, saved_final_crop_path = final_cropper.crop_single_particle_from_image(
                original_color_img, 
                dummy_gray_for_final_crop, 
                particle_row, # Pass the entire row
                output_dir_for_crop=final_sharp_crops_output_dir, 
                crop_filename_prefix=output_crop_filename_prefix
            )

            if saved_final_crop_path: 
                sharp_crops_saved_count += 1
            else: 
                logger.warning(f"Failed to save final sharp crop for particle X{coord_x},Y{coord_y} from {image_filename}")
                
    logger.info(f"Saved {sharp_crops_saved_count} final sharp object crops.")


def run_stage_02_quality_filter(
    input_particles_csv_path: str,
    image_dir_path: str,
    output_dir_stage2: str,
    sample_id: str
    ) -> Optional[str]:
    logger.info(f"--- Stage 2: Quality Filter for Sample ID: {sample_id} (Consolidated Thresh for Boundary) ---")
    # MODIFIED: Updated log message for clarity
    logger.info(f"Input particle CSV: {os.path.basename(input_particles_csv_path)}")
    logger.info(f"Original image directory: {image_dir_path}")
    logger.info(f"Output directory for this stage: {output_dir_stage2}")

    config = get_config()
    config_s2 = config.get('stage_02_quality_filter', {})
    if not config_s2:
        raise ConfigError("Configuration for 'stage_02_quality_filter' not found.")
    
    system_config = config.get('system', {})
    source_image_illumination_type = system_config.get('source_image_illumination_type', 'darkfield')
    if source_image_illumination_type not in ['darkfield', 'collimated']:
        logger.warning(f"Invalid system.source_image_illumination_type: '{source_image_illumination_type}'. Defaulting to 'darkfield'.")
        source_image_illumination_type = 'darkfield'
    
    global_threshold_config_for_boundary = system_config.get('thresholding', {})
    if not global_threshold_config_for_boundary:
        raise ConfigError("Stage 2: System-wide thresholding configuration ('system.thresholding') not found or empty for boundary check.")

    s2_filters_enabled_config = config_s2.get('filters_enabled', {})
    enable_final_sharpness_filter = s2_filters_enabled_config.get('final_sharpness', True)

    original_particles_df = load_dataframe_from_csv(input_particles_csv_path)
    if original_particles_df is None or original_particles_df.empty:
        logger.warning(f"No particles loaded from {input_particles_csv_path}. Stage 2 cannot proceed.")
        output_filename = f"{sample_id}_stage2_analyzed_particles.csv"
        expected_cols_list = ['X', 'Y', 'source_image_filename', 'SAMPLE_ID', 'input_particle_index', 'analyzed_centroid_x', 'analyzed_centroid_y', 'cropped_filename', 'is_black_mask', 'is_on_crop_boundary'] + AVAILABLE_SHARPNESS_METRICS + [m + '_norm' for m in AVAILABLE_SHARPNESS_METRICS] + ['combined_sharpness']
        if original_particles_df is not None: 
            for col in original_particles_df.columns:
                if col not in expected_cols_list: expected_cols_list.append(col)
        empty_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols_list))) 
        return save_dataframe_to_csv(empty_df, output_dir_stage2, output_filename)

    intermediate_crop_cfg = config_s2.get('intermediate_crop', {}); 
    try: intermediate_cropper = ParticleCropper(intermediate_crop_cfg)
    except ConfigError as ce: logger.error(f"Error initializing ParticleCropper for intermediate crops: {ce}"); return None
        
    base_temp_crop_dir = os.path.join(output_dir_stage2, "temp_intermediate_particle_crops"); os.makedirs(base_temp_crop_dir, exist_ok=True)
    all_single_particle_results_list: List[Dict] = []; processed_image_count = 0
    required_cols_stage1 = ['source_image_filename', 'X', 'Y']
    if not all(col in original_particles_df.columns for col in required_cols_stage1):
        for col in required_cols_stage1:
            if col not in original_particles_df.columns:
                logger.warning(f"Required column '{col}' missing from Stage 1 CSV. Adding it with default value.")
                if col == 'source_image_filename': original_particles_df[col] = "unknown_source_image.png"
                else: original_particles_df[col] = np.nan 
    
    grouped_particles = original_particles_df.groupby('source_image_filename')
    logger.info(f"Found {len(grouped_particles)} unique source images in Stage 1 particle data.")

    for image_filename, particles_in_image_df in grouped_particles:
        logger.info(f"Processing source image: {image_filename} ({len(particles_in_image_df)} particles)")
        processed_image_count += 1
        full_image_path_color = os.path.join(image_dir_path, image_filename)
        main_img_color = load_image(full_image_path_color, mode='color')
        if main_img_color is None: logger.warning(f"Failed to load main image {full_image_path_color}. Skipping particles for this image."); continue
        main_img_gray = convert_to_grayscale(main_img_color) 
        if main_img_gray is None: logger.warning(f"Failed to convert {full_image_path_color} to grayscale. Skipping."); continue
        image_specific_temp_crop_dir = os.path.join(base_temp_crop_dir, os.path.splitext(image_filename)[0]); os.makedirs(image_specific_temp_crop_dir, exist_ok=True)

        for _, particle_row in particles_in_image_df.iterrows():
            analysis_result_obj = _process_single_particle_for_quality(
                particle_row, main_img_color, main_img_gray, 
                intermediate_cropper, image_specific_temp_crop_dir, 
                config_s2, 
                source_image_illumination_type, 
                global_threshold_config_for_boundary
            )
            all_single_particle_results_list.append(analysis_result_obj.to_dict())
        
    if not all_single_particle_results_list: 
        logger.warning("No particles were processed for quality analysis. Stage 2 results will be empty.")
        output_filename = f"{sample_id}_stage2_analyzed_particles.csv"
        expected_cols_list = ['X', 'Y', 'source_image_filename', 'SAMPLE_ID', 'input_particle_index', 'analyzed_centroid_x', 'analyzed_centroid_y', 'cropped_filename', 'is_black_mask', 'is_on_crop_boundary'] + AVAILABLE_SHARPNESS_METRICS + [m + '_norm' for m in AVAILABLE_SHARPNESS_METRICS] + ['combined_sharpness']
        if original_particles_df is not None: 
            for col in original_particles_df.columns:
                if col not in expected_cols_list: expected_cols_list.append(col)
        empty_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols_list)))
        return save_dataframe_to_csv(empty_df, output_dir_stage2, output_filename)

    detailed_analysis_df = pd.DataFrame(all_single_particle_results_list)
    logger.info(f"Aggregated quality analysis results for {len(detailed_analysis_df)} particles.")
    suitable_scored_df, all_analyzed_with_scores_df = _filter_and_score_analysis_results(detailed_analysis_df.copy(), config_s2)
    left_df_for_merge = original_particles_df.reset_index().rename(columns={'index': 'input_particle_index'})
    final_merged_df = pd.merge(left_df_for_merge, all_analyzed_with_scores_df, on='input_particle_index', how='left', suffixes=('', '_s2_analysis'))
    cols_to_drop = [col for col in final_merged_df.columns if col.endswith('_s2_analysis')]; cols_to_drop.extend(['original_X', 'original_Y', 'analyzed_centroid_x', 'analyzed_centroid_y']) 
    final_merged_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    logger.info(f"Final merged DataFrame created with {len(final_merged_df)} rows and columns: {final_merged_df.columns.tolist()}")
    output_analyzed_filename = f"{sample_id}_stage2_analyzed_particles.csv"
    saved_analyzed_csv_path = save_dataframe_to_csv(final_merged_df, output_dir_stage2, output_analyzed_filename)

    sharp_filter_criteria = config_s2.get('final_sharp_object_filter', {}) 
    if enable_final_sharpness_filter and sharp_filter_criteria:
        df_for_sharp_thresholding = final_merged_df.copy() 
        sharp_mask = pd.Series(True, index=df_for_sharp_thresholding.index) ; missing_metrics_for_filter = False
        for metric_key_config, threshold_value in sharp_filter_criteria.items():
            metric_name = metric_key_config.replace("_min", "") 
            if metric_name not in df_for_sharp_thresholding.columns:
                logger.warning(f"Metric '{metric_name}' for sharp filter not found in DataFrame. Skipping this criterion."); missing_metrics_for_filter = True; continue 
            numeric_col = pd.to_numeric(df_for_sharp_thresholding[metric_name], errors='coerce')
            sharp_mask &= (numeric_col > threshold_value) 
            logger.debug(f"Applied sharp filter: {metric_name} > {threshold_value}. Remaining after this step: {sharp_mask.sum()}")
        if missing_metrics_for_filter and sharp_mask.sum() == len(df_for_sharp_thresholding) and len(sharp_filter_criteria) > 0 :
             logger.warning("Some metrics for sharp filtering were missing. If all were missing, no effective filtering by value occurred.")
        objects_for_final_crop_df = df_for_sharp_thresholding[sharp_mask]
        logger.info(f"Found {len(objects_for_final_crop_df)} objects meeting ALL numerical sharpness criteria for final crop saving.")
        if not objects_for_final_crop_df.empty:
            final_sharp_crops_output_dir = os.path.join(output_dir_stage2, "sharp_object_final_crops")
            
            # MODIFIED: Logic to prepare for final cropping is now more robust.
            final_crop_settings_from_config = config_s2.get('final_sharp_crop_settings', {})
            # Determine which mode to use for final cropping (should match intermediate)
            intermediate_crop_mode = config_s2.get('intermediate_crop', {}).get('mode', 'rotational_rectangle')
            
            parsed_date, parsed_plate, parsed_well = parse_date_plate_well(sample_id)
            sample_meta_info = {'date': parsed_date or "UNKDATE", 'plate': parsed_plate or "UNKP", 'well': parsed_well or "UNKW"}
            
            _save_final_sharp_object_crops(
                objects_for_final_crop_df, 
                image_dir_path, 
                final_sharp_crops_output_dir, 
                final_crop_settings_from_config, 
                intermediate_crop_mode,
                sample_meta_info
            )
            sharp_objects_csv_path = save_dataframe_to_csv(objects_for_final_crop_df, output_dir_stage2, f"{sample_id}_stage2_sharply_filtered_particles.csv") 
            logger.info(f"Details of {len(objects_for_final_crop_df)} sharp objects saved to: {sharp_objects_csv_path}")
        else: logger.info("No objects met the specific final sharpness criteria for saving crops.")
    elif not enable_final_sharpness_filter: logger.info("Final sharpness filter is DISABLED in config. Skipping final sharp crop saving and related CSV.")
    else: logger.info("Final sharpness filter ENABLED but no 'final_sharp_object_filter' criteria defined in config. Skipping final sharp crop saving.")

    try: 
        if os.path.exists(base_temp_crop_dir): shutil.rmtree(base_temp_crop_dir); logger.info(f"Cleaned up base temporary crop directory: {base_temp_crop_dir}")
    except Exception as e: logger.warning(f"Could not clean up temporary directory {base_temp_crop_dir}: {e}")

    logger.info(f"--- Stage 2: Quality Filter for Sample ID: {sample_id} COMPLETED ---")
    return saved_analyzed_csv_path


if __name__ == '__main__': 
    from src.utils.logging_setup import setup_logging
    setup_logging()
    try:
        cfg_main = get_config()
        system_cfg_test = cfg_main.setdefault('system', {})
        system_cfg_test.setdefault('source_image_illumination_type', 'collimated') 
        thresh_cfg_test_s2 = system_cfg_test.setdefault('thresholding', {
            'mode': 'otsu', 
            'otsu_threshold_multiplier': 0.45, 
            'fixed_threshold_value': 70
        })

        s2_cfg_test = cfg_main.setdefault('stage_02_quality_filter', {})
        s2_cfg_test.setdefault('filters_enabled', {'black_mask_artifact': True, 'on_crop_boundary': True, 'final_sharpness': True})
        s2_cfg_test.setdefault('final_sharp_object_filter', {'combined_sharpness_min': 0.1})
        # ADDED: Ensure the new final crop settings are present for testing
        s2_cfg_test.setdefault('final_sharp_crop_settings', {'centroid_square_size': 150, 'rotational_padding': 10})


        TEST_SAMPLE_ID = "test_s2_rotational_crop_default" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir: raise ValueError("project_base_dir not set in config for testing.")

        stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_output")
        test_s1_csv_path = os.path.join(stage1_output_dir, f"{TEST_SAMPLE_ID}_stage1_particles.csv")
        test_image_input_dir_s2 = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_s2_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage2_rotational_crop_test")
        os.makedirs(test_s2_output_dir, exist_ok=True)

        if not os.path.exists(test_s1_csv_path): 
            logger.error(f"Input CSV from Stage 1 ('{test_s1_csv_path}') not found. Creating dummy data.")
            dummy_s1_data = pd.DataFrame({
                'X': [100, 150], 'Y': [50, 75], 'area': [100, 120], 
                'source_image_filename': ['20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png', '20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png'],
                'SAMPLE_ID': [TEST_SAMPLE_ID]*2,
                # ADDED: rot_rect data needed for new default mode
                'rot_rect_width': [20.5, 30.2], 'rot_rect_height': [40.1, 25.8], 'rot_rect_angle': [15.0, 80.0]
                })
            os.makedirs(stage1_output_dir, exist_ok=True); dummy_s1_data.to_csv(test_s1_csv_path, index=False)
            os.makedirs(test_image_input_dir_s2, exist_ok=True)
            fname = '20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png'
            dummy_img_path = os.path.join(test_image_input_dir_s2, fname)
            if not os.path.exists(dummy_img_path): cv2.imwrite(dummy_img_path, np.full((200,200,3), 220, dtype=np.uint8))
            logger.info(f"Created dummy Stage 1 CSV and image.")

        result_s2_csv_path = run_stage_02_quality_filter(
            input_particles_csv_path=test_s1_csv_path,
            image_dir_path=test_image_input_dir_s2,
            output_dir_stage2=test_s2_output_dir,
            sample_id=TEST_SAMPLE_ID
        )
        if result_s2_csv_path: 
            logger.info(f"Stage 2 test (rotational crop) completed. Final analyzed CSV: {result_s2_csv_path}")
            df_out = pd.read_csv(result_s2_csv_path)
            print("Stage 2 Test (Rotational Crop) Output DataFrame Head:")
            print(df_out.head())
        else: 
            logger.error("Stage 2 test (rotational crop) FAILED or produced no output CSV.")

    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 2 test: {e}")