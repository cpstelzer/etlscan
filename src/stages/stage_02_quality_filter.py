# src/stages/stage_02_quality_filter.py
import os
import pandas as pd
import numpy as np
import cv2 # Keep for direct use if ParticleCropper uses it, or for other direct CV ops
import logging
import shutil # For cleaning up temp directories
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

# Helper class for managing results of single particle processing
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
            'original_X': self.input_particle_data.get('X'), # Keep original for merging
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
    config_s2: Dict[str, Any]
) -> SingleParticleAnalysisResult:
    """
    Processes a single particle: crops, analyzes for artifacts and sharpness.
    """
    result = SingleParticleAnalysisResult(particle_data_series)
    
    particle_id_info = f"(idx={particle_data_series.name}, X={result.centroid_x}, Y={result.centroid_y})"
    logger.debug(f"Processing particle {particle_id_info} for quality...")

    try:
        intermediate_crop_gray, saved_intermediate_color_crop_path = cropper.crop_single_particle_from_image(
            main_full_color_image, main_full_gray_image, particle_data_series,
            output_dir_for_crop=temp_intermediate_crop_dir,
            crop_filename_prefix="intermediate_crop"
        )

        if intermediate_crop_gray is None or saved_intermediate_color_crop_path is None:
            logger.warning(f"Cropping failed or intermediate crop not saved for {particle_id_info}.")
            result.error_message = "Cropping or intermediate save failed"
            return result 

        result.cropped_image_filename = os.path.basename(saved_intermediate_color_crop_path)
        result.sharpness_scores = calculate_sharpness_metrics(intermediate_crop_gray)
        black_mask_params = config_s2.get('black_mask_check', {})
        result.is_black_mask_artifact = detect_black_mask_artifact(
            image_path=saved_intermediate_color_crop_path, 
            min_area_ratio=black_mask_params.get('min_area_ratio', 0.05)
        )
        result.is_on_crop_boundary = detect_cropped_particle_on_boundary(
            intermediate_crop_gray, 
            saved_intermediate_color_crop_path 
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
    """
    Filters particles based on artifact flags (no black_mask, not on_crop_boundary).
    Then calculates normalized and combined sharpness for suitable particles.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - suitable_scored_df: DataFrame of suitable particles with sharpness scores.
            - all_analyzed_df_with_scores: Original analysis_results_df with sharpness scores merged back.
    """
    logger.info("Filtering and scoring particle analysis results...")
    if analysis_results_df.empty:
        logger.warning("No analysis results to filter or score.")
        expected_cols = list(analysis_results_df.columns)
        for metric in config_s2.get('sharpness_metrics', AVAILABLE_SHARPNESS_METRICS):
            expected_cols.append(metric + '_norm')
        expected_cols.append('combined_sharpness')
        return pd.DataFrame(columns=expected_cols), pd.DataFrame(columns=analysis_results_df.columns)

    analysis_results_df['is_black_mask'] = analysis_results_df['is_black_mask'].fillna(True).astype(bool)
    analysis_results_df['is_on_crop_boundary'] = analysis_results_df['is_on_crop_boundary'].fillna(True).astype(bool)

    suitable_mask = (~analysis_results_df['is_black_mask']) & (~analysis_results_df['is_on_crop_boundary'])
    suitable_df = analysis_results_df[suitable_mask].copy()
    
    count_in = len(analysis_results_df)
    count_out = len(suitable_df)
    logger.info(f"Filtering for suitable objects (No Black Mask AND Not On Crop Boundary): "
                f"{count_out} suitable remaining from {count_in} ({count_in - count_out} removed).")

    if suitable_df.empty:
        logger.warning("No suitable objects found after artifact filtering.")
        all_analyzed_with_scores = analysis_results_df.copy()
        for metric in config_s2.get('sharpness_metrics', AVAILABLE_SHARPNESS_METRICS):
            all_analyzed_with_scores[metric + '_norm'] = np.nan
        all_analyzed_with_scores['combined_sharpness'] = np.nan
        return pd.DataFrame(columns=all_analyzed_with_scores.columns), all_analyzed_with_scores

    metrics_for_norm = config_s2.get('sharpness_metrics', AVAILABLE_SHARPNESS_METRICS)
    suitable_scored_df = normalize_and_combine_sharpness(suitable_df, metrics_for_norm)

    all_analyzed_df_with_scores = analysis_results_df.copy()
    score_cols_to_add = [m + '_norm' for m in metrics_for_norm] + ['combined_sharpness']
    for sc_col in score_cols_to_add:
        if sc_col not in all_analyzed_df_with_scores.columns:
             all_analyzed_df_with_scores[sc_col] = np.nan
    all_analyzed_df_with_scores.update(suitable_scored_df[score_cols_to_add])

    return suitable_scored_df, all_analyzed_df_with_scores


def _save_final_sharp_object_crops(
    sharp_objects_df: pd.DataFrame, 
    image_dir_path: str, 
    final_sharp_crops_output_dir: str,
    final_crop_size: int,
    sample_info: Dict[str, str] 
    ):
    """
    Loads original images and crops the final sharp objects.
    """
    logger.info(f"Saving final sharp object crops to: {final_sharp_crops_output_dir}")
    os.makedirs(final_sharp_crops_output_dir, exist_ok=True)
    
    sharp_crops_saved_count = 0

    for image_filename, group in sharp_objects_df.groupby('source_image_filename'):
        full_image_path_orig = os.path.join(image_dir_path, image_filename)
        
        original_color_img = load_image(full_image_path_orig, mode='color')
        if original_color_img is None:
            logger.warning(f"Could not load original image {full_image_path_orig} for final cropping. "
                           f"Skipping {len(group)} sharp objects from this image.")
            continue

        logger.debug(f"Loaded {image_filename} for final sharp object cropping.")
        
        # Create dummy gray image once per full image for the cropper interface
        dummy_gray_for_final_crop = convert_to_grayscale(original_color_img)
        if dummy_gray_for_final_crop is None:
            logger.error(f"Could not create dummy gray for final crop from {image_filename}. Skipping its sharp objects.")
            continue

        for _, particle_row in group.iterrows():
            try:
                # Use 'original_X' and 'original_Y' which were renamed from 'X', 'Y' of stage1 df.
                coord_x = int(particle_row['original_X'])
                coord_y = int(particle_row['original_Y'])
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Missing or invalid original coordinates for particle in {image_filename} "
                             f"(X={particle_row.get('original_X')}, Y={particle_row.get('original_Y')}). Error: {e}. Skipping crop.")
                continue

            frame_num = extract_frame_from_filename(image_filename) 
            if frame_num is None: frame_num_str = "UnkF"
            else: frame_num_str = f"F{frame_num:05d}"

            output_crop_filename_base = (
                f"{sample_info['date']}_plate{sample_info['plate']}_well{sample_info['well']}_"
                f"{frame_num_str}_X{coord_x}_Y{coord_y}_sharp" 
            ) # save_image in cropper will add .png
            
            final_cropper_cfg = {'mode': 'centroid_square', 'square_size': final_crop_size}
            final_cropper = ParticleCropper(final_cropper_cfg)
            
            temp_particle_series_for_final_crop = pd.Series({'X': coord_x, 'Y': coord_y}, name=particle_row.name)

            _, saved_final_crop_path = final_cropper.crop_single_particle_from_image(
                original_color_img,
                dummy_gray_for_final_crop, 
                temp_particle_series_for_final_crop, 
                output_dir_for_crop=final_sharp_crops_output_dir,
                crop_filename_prefix=output_crop_filename_base 
            )
            
            if saved_final_crop_path:
                sharp_crops_saved_count += 1
            else:
                logger.warning(f"Failed to save final sharp crop for particle X{coord_x},Y{coord_y} from {image_filename}")

    logger.info(f"Saved {sharp_crops_saved_count} final sharp object crops.")


def run_stage_02_quality_filter(
    input_particles_csv_path: str, # From Stage 1
    image_dir_path: str, # Path to original full-size images for this sample
    output_dir_stage2: str,
    sample_id: str
    ) -> Optional[str]: # Path to the final CSV with all analyses
    """
    Main function for Stage 2: Quality Filtering and Sharp Object Extraction.
    """
    logger.info(f"--- Stage 2: Quality Filter for Sample ID: {sample_id} ---")
    logger.info(f"Input particle CSV: {input_particles_csv_path}")
    logger.info(f"Original image directory: {image_dir_path}")
    logger.info(f"Output directory for this stage: {output_dir_stage2}")

    config = get_config()
    config_s2 = config.get('stage_02_quality_filter', {})
    if not config_s2:
        raise ConfigError("Configuration for 'stage_02_quality_filter' not found.")

    original_particles_df = load_dataframe_from_csv(input_particles_csv_path)
    if original_particles_df is None or original_particles_df.empty:
        logger.warning(f"No particles loaded from {input_particles_csv_path}. Stage 2 cannot proceed.")
        output_filename = f"{sample_id}_stage2_analyzed_particles.csv"
        # Define schema for the empty DataFrame
        expected_cols_list = ['X', 'Y', 'source_image_filename', 'SAMPLE_ID', # from stage 1
                              'input_particle_index', 'analyzed_centroid_x', 'analyzed_centroid_y', 
                              'cropped_filename', 'is_black_mask', 'is_on_crop_boundary'] + \
                             AVAILABLE_SHARPNESS_METRICS + \
                             [m + '_norm' for m in AVAILABLE_SHARPNESS_METRICS] + \
                             ['combined_sharpness']
        # Add other typical stage 1 columns if they exist in a non-empty scenario
        if original_particles_df is not None: # Check if it was loaded but empty
            for col in original_particles_df.columns:
                if col not in expected_cols_list:
                    expected_cols_list.append(col)

        empty_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols_list)))
        return save_dataframe_to_csv(empty_df, output_dir_stage2, output_filename)

    intermediate_crop_cfg = config_s2.get('intermediate_crop', {})
    try:
        intermediate_cropper = ParticleCropper(intermediate_crop_cfg)
    except ConfigError as ce:
        logger.error(f"Error initializing ParticleCropper for intermediate crops: {ce}")
        return None
        
    base_temp_crop_dir = os.path.join(output_dir_stage2, "temp_intermediate_particle_crops")
    os.makedirs(base_temp_crop_dir, exist_ok=True)

    all_single_particle_results_list: List[Dict] = []
    processed_image_count = 0

    required_cols_stage1 = ['source_image_filename', 'X', 'Y']
    if not all(col in original_particles_df.columns for col in required_cols_stage1):
        # Append missing required columns with default values if they don't exist.
        for col in required_cols_stage1:
            if col not in original_particles_df.columns:
                logger.warning(f"Required column '{col}' missing from Stage 1 CSV. Adding it with default (NaN or placeholder).")
                if col == 'source_image_filename':
                    original_particles_df[col] = "unknown_source_image.png"
                else: # X, Y
                    original_particles_df[col] = np.nan 
        # raise DataValidationError(f"Stage 1 CSV missing required columns: {required_cols_stage1}")


    grouped_particles = original_particles_df.groupby('source_image_filename')
    logger.info(f"Found {len(grouped_particles)} unique source images in Stage 1 particle data.")

    for image_filename, particles_in_image_df in grouped_particles:
        logger.info(f"Processing source image: {image_filename} ({len(particles_in_image_df)} particles)")
        processed_image_count += 1
        
        full_image_path_color = os.path.join(image_dir_path, image_filename)
        main_img_color = load_image(full_image_path_color, mode='color')
        if main_img_color is None:
            logger.warning(f"Failed to load main image {full_image_path_color}. Skipping particles for this image.")
            continue
        main_img_gray = convert_to_grayscale(main_img_color)
        if main_img_gray is None:
            logger.warning(f"Failed to convert {full_image_path_color} to grayscale. Skipping.")
            continue

        image_specific_temp_crop_dir = os.path.join(base_temp_crop_dir, os.path.splitext(image_filename)[0])
        os.makedirs(image_specific_temp_crop_dir, exist_ok=True)

        for _, particle_row in particles_in_image_df.iterrows():
            analysis_result_obj = _process_single_particle_for_quality(
                particle_row, main_img_color, main_img_gray,
                intermediate_cropper, image_specific_temp_crop_dir, config_s2
            )
            all_single_particle_results_list.append(analysis_result_obj.to_dict())
        
    if not all_single_particle_results_list:
        logger.warning("No particles were processed for quality analysis. Stage 2 results will be empty.")
        output_filename = f"{sample_id}_stage2_analyzed_particles.csv"
        expected_cols_list = ['X', 'Y', 'source_image_filename', 'SAMPLE_ID',
                              'input_particle_index', 'analyzed_centroid_x', 'analyzed_centroid_y', 
                              'cropped_filename', 'is_black_mask', 'is_on_crop_boundary'] + \
                             AVAILABLE_SHARPNESS_METRICS + \
                             [m + '_norm' for m in AVAILABLE_SHARPNESS_METRICS] + \
                             ['combined_sharpness']
        if original_particles_df is not None:
            for col in original_particles_df.columns:
                if col not in expected_cols_list:
                    expected_cols_list.append(col)
        empty_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols_list)))
        return save_dataframe_to_csv(empty_df, output_dir_stage2, output_filename)

    detailed_analysis_df = pd.DataFrame(all_single_particle_results_list)
    logger.info(f"Aggregated quality analysis results for {len(detailed_analysis_df)} particles.")

    suitable_scored_df, all_analyzed_with_scores_df = _filter_and_score_analysis_results(
        detailed_analysis_df.copy(), 
        config_s2
    )
    
    left_df_for_merge = original_particles_df.reset_index().rename(columns={'index': 'input_particle_index'})
    
    final_merged_df = pd.merge(
        left_df_for_merge,
        all_analyzed_with_scores_df,
        left_on=['input_particle_index', 'X', 'Y'], # X, Y from original Stage 1 df
        right_on=['input_particle_index', 'original_X', 'original_Y'], # original_X, Y from SingleParticleAnalysisResult
        how='left',
        suffixes=('', '_s2_analysis') 
    )
     
    cols_to_drop = [col for col in final_merged_df.columns if col.endswith('_s2_analysis')]
    # Also remove the 'original_X', 'original_Y' columns that came from all_analyzed_with_scores_df,
    # as they are now redundant with 'X' and 'Y' from the left merge side (original_particles_df).
    cols_to_drop.extend(['original_X', 'original_Y'])
    # Remove 'analyzed_centroid_x', 'analyzed_centroid_y' if they are just copies of X,Y for this stage.
    # The original wsutil did not have these separate analyzed_centroid columns in the final merged output.
    cols_to_drop.extend(['analyzed_centroid_x', 'analyzed_centroid_y'])

    final_merged_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    logger.info(f"Final merged DataFrame created with {len(final_merged_df)} rows and columns: {final_merged_df.columns.tolist()}")

    output_analyzed_filename = f"{sample_id}_stage2_analyzed_particles.csv"
    saved_analyzed_csv_path = save_dataframe_to_csv(final_merged_df, output_dir_stage2, output_analyzed_filename)

    sharp_filter_criteria = config_s2.get('final_sharp_object_filter', {})
    if sharp_filter_criteria:
        df_for_sharp_thresholding = final_merged_df.copy()
            
        sharp_mask = pd.Series(True, index=df_for_sharp_thresholding.index)
        missing_metrics_for_filter = False
        for metric_key_config, threshold_value in sharp_filter_criteria.items():
            metric_name = metric_key_config.replace("_min", "") 
            if metric_name not in df_for_sharp_thresholding.columns:
                logger.warning(f"Metric '{metric_name}' for sharp filter not found in DataFrame. Skipping this criterion.")
                missing_metrics_for_filter = True
                continue 
                
            numeric_col = pd.to_numeric(df_for_sharp_thresholding[metric_name], errors='coerce').fillna(float('-inf'))
            sharp_mask &= (numeric_col > threshold_value)
            logger.debug(f"Applied sharp filter: {metric_name} > {threshold_value}. "
                         f"Remaining after this step: {sharp_mask.sum()}")
        
        if missing_metrics_for_filter and sharp_mask.sum() == len(df_for_sharp_thresholding) and len(sharp_filter_criteria) > 0 :
             logger.warning("Some metrics for sharp filtering were missing. If all were missing, no effective filtering by value occurred.")

        objects_for_final_crop_df = df_for_sharp_thresholding[sharp_mask]
        logger.info(f"Found {len(objects_for_final_crop_df)} objects meeting ALL numerical sharpness criteria for final crop saving.")

        if not objects_for_final_crop_df.empty:
            final_sharp_crops_output_dir = os.path.join(output_dir_stage2, "sharp_object_final_crops")
            final_crop_s = config_s2.get('final_sharp_crop_size', 150)
            
            parsed_date, parsed_plate, parsed_well = parse_date_plate_well(sample_id)
            sample_meta_info = {
                'date': parsed_date or "UNKDATE",
                'plate': parsed_plate or "UNKP",
                'well': parsed_well or "UNKW"
            }
            
            # Rename 'X', 'Y' (which are original stage1 coords) to 'original_X', 'original_Y' for the save function
            temp_df_for_saving_sharp = objects_for_final_crop_df.rename(
                columns={'X': 'original_X', 'Y': 'original_Y'}
            )
            # Ensure 'source_image_filename' is present for grouping in _save_final_sharp_object_crops
            if 'source_image_filename' not in temp_df_for_saving_sharp.columns:
                 logger.error("'source_image_filename' is missing from the DataFrame intended for sharp crop saving. Cannot proceed with saving crops.")
            else:
                _save_final_sharp_object_crops(
                    temp_df_for_saving_sharp, 
                    image_dir_path, 
                    final_sharp_crops_output_dir,
                    final_crop_s,
                    sample_meta_info
                )
            sharp_objects_csv_path = save_dataframe_to_csv(
                objects_for_final_crop_df, output_dir_stage2, f"{sample_id}_stage2_sharply_filtered_particles.csv"
            ) 
            logger.info(f"Details of {len(objects_for_final_crop_df)} sharp objects saved to: {sharp_objects_csv_path}")
        else:
            logger.info("No objects met the specific final sharpness criteria for saving crops.")
    else:
        logger.info("No 'final_sharp_object_filter' criteria defined. Skipping final sharp crop saving.")

    try:
        if os.path.exists(base_temp_crop_dir):
            shutil.rmtree(base_temp_crop_dir)
            logger.info(f"Cleaned up base temporary crop directory: {base_temp_crop_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary directory {base_temp_crop_dir}: {e}")

    logger.info(f"--- Stage 2: Quality Filter for Sample ID: {sample_id} COMPLETED ---")
    return saved_analyzed_csv_path

if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging
    setup_logging()

    try:
        cfg_main = get_config()
        TEST_SAMPLE_ID = "test_sample_s2_filter" 
        
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir:
            raise ValueError("project_base_dir not set in config for testing.")

        stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_output")
        test_s1_csv_path = os.path.join(stage1_output_dir, f"{TEST_SAMPLE_ID}_stage1_particles.csv")
        test_image_input_dir_s2 = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_s2_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage2_output")
        os.makedirs(test_s2_output_dir, exist_ok=True)

        logger.info(f"Test S1 Input CSV: {test_s1_csv_path}")
        logger.info(f"Test Image Dir for S2: {test_image_input_dir_s2}")
        logger.info(f"Test S2 Output Dir: {test_s2_output_dir}")

        if not os.path.exists(test_s1_csv_path):
            logger.error(f"Input CSV from Stage 1 ('{test_s1_csv_path}') not found. ")
            dummy_s1_data = pd.DataFrame({
                'X': [100, 150, 20, 200], 'Y': [50, 75, 180, 10],
                'area': [100, 120, 90, 150], 'perimeter': [40,45,35,50], 
                'source_image_filename': ['20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png', 
                                          '20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png', # two particles in first image
                                          '20230101_plate1_well1_f00001_dpt0.1_fd0.0mm.png',
                                          '20230101_plate1_well1_f00001_dpt0.1_fd0.0mm.png'], # two in second
                'SAMPLE_ID': [TEST_SAMPLE_ID]*4,
                'bounding_x': [90,140,10,190], 'bounding_y': [40,65,170,0],
                'bounding_w': [20,20,20,20], 'bounding_h': [20,20,20,20] 
            })
            os.makedirs(stage1_output_dir, exist_ok=True)
            dummy_s1_data.to_csv(test_s1_csv_path, index=False)
            logger.info(f"Created dummy Stage 1 CSV: {test_s1_csv_path}")
            
            os.makedirs(test_image_input_dir_s2, exist_ok=True)
            for fname in dummy_s1_data['source_image_filename'].unique():
                dummy_img_path = os.path.join(test_image_input_dir_s2, fname)
                if not os.path.exists(dummy_img_path):
                     # Create images that might produce varying sharpness
                     img_content = np.random.randint(0,100,(200,200,3), dtype=np.uint8)
                     if 'f00000' in fname : # make one sharper
                         cv2.putText(img_content, "Sharp", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                     else: # make one blurrier
                         img_content = cv2.GaussianBlur(img_content, (15,15), 0)
                     cv2.imwrite(dummy_img_path, img_content)
            logger.info(f"Created dummy images in {test_image_input_dir_s2}")


        result_s2_csv_path = run_stage_02_quality_filter(
            input_particles_csv_path=test_s1_csv_path,
            image_dir_path=test_image_input_dir_s2,
            output_dir_stage2=test_s2_output_dir,
            sample_id=TEST_SAMPLE_ID
        )

        if result_s2_csv_path:
            logger.info(f"Stage 2 test completed. Final analyzed CSV: {result_s2_csv_path}")
        else:
            logger.error("Stage 2 test FAILED or produced no output CSV.")

    except ConfigError as ce:
        logger.error(f"Configuration error during Stage 2 test: {ce}")
    except DataValidationError as dve:
        logger.error(f"Data validation error during Stage 2 test: {dve}")
    except FileNotFoundError as fnfe:
        logger.error(f"File not found during Stage 2 test setup: {fnfe}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the Stage 2 test: {e}")