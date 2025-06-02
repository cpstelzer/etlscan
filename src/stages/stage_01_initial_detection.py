# src/stages/stage_01_initial_detection.py
import os
import re
import cv2
import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, List, Dict, Any, Tuple

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image # Used by generate_stage1_preview_data and __main__
from src.core.image_enhancement import subtract_images, apply_threshold, apply_blur
from src.core.masking_operations import mask_well_area
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area

logger = logging.getLogger(__name__)

def _parse_image_filenames(image_dir_path: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for PNG images matching a specific pattern,
    parses metadata from filenames, and returns a sorted DataFrame.
    """
    if not image_dir_path or not isinstance(image_dir_path, str):
        logger.error(f"Invalid image_dir_path provided to _parse_image_filenames: {image_dir_path} (type: {type(image_dir_path)}).")
        return None 
        
    if not os.path.isdir(image_dir_path):
        logger.error(f"Image directory not found: {image_dir_path}")
        return None

    pattern = re.compile(
        r"^(?P<date>\d{8})_plate(?P<plate>\d+)_well(?P<well>\d+)_f(?P<frame>\d{5})_dpt(?P<dpt>[-+]?[0-9]*\.?[0-9]+)_fd(?P<fd_mm>[-+]?[0-9]*\.?[0-9]+)mm\.png$"
    )
    parsed_file_data = []
    found_files_count = 0
    logger.info(f"Scanning for images in: {image_dir_path}")

    for filename in sorted(os.listdir(image_dir_path)): 
        if filename.lower().endswith(".png"):
            match = pattern.match(filename)
            if match:
                found_files_count += 1
                groups = match.groupdict()
                try:
                    parsed_file_data.append({
                        "filename": filename,
                        "date": groups["date"],
                        "plate": int(groups["plate"]),
                        "well": int(groups["well"]),
                        "frame": int(groups["frame"]),
                        "dpt": float(groups["dpt"]),
                        "fd_mm": float(groups["fd_mm"]),
                        "full_path": os.path.join(image_dir_path, filename) 
                    })
                except ValueError as e:
                    logger.warning(f"Could not parse metadata from filename '{filename}': {e}")
    
    logger.info(f"Found {found_files_count} files matching the pattern in {image_dir_path}.")
    if not parsed_file_data:
        logger.warning("No matching image files found or parsed in the directory.")
        return pd.DataFrame() 

    df = pd.DataFrame(parsed_file_data)
    df = df.sort_values(by="frame").reset_index(drop=True)
    logger.info(f"Created initial DataFrame with {len(df)} image entries, sorted by frame.")
    return df

def _filter_image_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the image DataFrame based on frame number and 'fd_mm' criteria.
    - Removes frame 0.
    - Keeps frames up to the highest frame where 'fd_mm' is 0.
    """
    if df.empty:
        return df

    initial_len = len(df)
    if not df.empty and df.iloc[0]["frame"] == 0:
        df = df.iloc[1:].reset_index(drop=True)
        logger.info(f"Removed frame 0. Rows changed from {initial_len} to {len(df)}.")
    else:
        logger.debug("Frame 0 not found at the first row or DataFrame empty. No frame removed based on this rule.")

    fd0_frames = df[df["fd_mm"] == 0]
    if not fd0_frames.empty:
        max_frame_fd0 = fd0_frames["frame"].max()
        original_len_before_fd_filter = len(df)
        df = df[df["frame"] <= max_frame_fd0].reset_index(drop=True)
        logger.info(f"Filtered frames after max_frame {max_frame_fd0} (where fd_mm=0). "
                    f"Kept {len(df)} out of {original_len_before_fd_filter} entries.")
    else:
        logger.warning("No images with fd_mm == 0 were found. Keeping all remaining frames after potential frame 0 removal.")

    if df.empty:
        logger.warning("DataFrame is empty after filtering steps.")
    
    return df

def _analyze_single_subtracted_image(
    image_to_analyze_gray: np.ndarray, 
    config_stage1_particle_analysis: Dict[str, Any] 
    ) -> Tuple[Optional[pd.DataFrame], int, Optional[np.ndarray]]: 
    """
    Analyzes a single (already background-subtracted and potentially masked)
    grayscale image to find particles. Assumes input is uint8.
    The input image is expected to be in "bright particles on darker background" format.
    Convolution is applied here.
    Returns particle data, count, and the final binary image used for contour detection.
    """
    
    apply_conv = config_stage1_particle_analysis.get('apply_convolution', True)
    kernel_list = config_stage1_particle_analysis.get('convolution_kernel')
    threshold_val = config_stage1_particle_analysis.get('microorganism_threshold', 30)
    min_area = config_stage1_particle_analysis.get('min_microorganism_area', 50)
    blur_kernel_size = config_stage1_particle_analysis.get('microorganism_blur_kernel', 3)

    if image_to_analyze_gray is None:
        return None, 0, None
    
    if image_to_analyze_gray.dtype != np.uint8:
        logger.warning(f"Input image to _analyze_single_subtracted_image is {image_to_analyze_gray.dtype}, expected uint8. Attempting conversion.")
        if np.issubdtype(image_to_analyze_gray.dtype, np.floating):
            image_to_analyze_gray = np.clip(image_to_analyze_gray * 255, 0, 255).astype(np.uint8)
        else:
            image_to_analyze_gray = image_to_analyze_gray.astype(np.uint8)


    processed_img = image_to_analyze_gray.copy()
    
    if apply_conv and kernel_list:
        try:
            conv_kernel = np.array(kernel_list, dtype=np.float32)
            if conv_kernel.ndim == 2 and conv_kernel.shape[0] > 0 and conv_kernel.shape[1] > 0:
                logger.debug(f"Applying convolution with kernel: {conv_kernel.shape}")
                processed_img = cv2.filter2D(processed_img, -1, conv_kernel)
            else:
                logger.warning(f"Invalid convolution_kernel from config: {kernel_list}. Skipping convolution.")
        except Exception as e:
            logger.error(f"Error applying convolution: {e}. Skipping convolution.")
    elif apply_conv and not kernel_list:
        logger.warning("'apply_convolution' is true, but 'convolution_kernel' is missing/empty in config. Skipping convolution.")

    binary_img = apply_threshold(processed_img, threshold_val, "binary")
    if binary_img is None:
        logger.error("Thresholding failed during particle analysis.")
        return None, 0, None 

    if blur_kernel_size > 0 and blur_kernel_size % 2 == 1:
        blurred_binary_img = apply_blur(binary_img, blur_kernel_size, "median")
        if blurred_binary_img is None:
            logger.warning("Blurring failed, proceeding with unblurred binary image.")
            img_for_contours = binary_img
        else:
            img_for_contours = blurred_binary_img
    else:
        if blur_kernel_size > 0: logger.warning(f"Invalid blur_kernel_size {blur_kernel_size}, skipping blur.")
        img_for_contours = binary_img
        
    contours = find_contours(img_for_contours, mode="external", method="simple")
    if not contours:
        logger.debug("No contours found in the processed image.")
        return pd.DataFrame(), 0, img_for_contours 

    valid_contours = filter_contours_by_area(contours, min_area)
    if not valid_contours:
        logger.debug(f"No contours remained after area filter (min_area={min_area}).")
        return pd.DataFrame(), 0, img_for_contours

    particle_measurements_list = []
    for contour in valid_contours:
        measurements = calculate_contour_measurements(contour)
        if measurements:
            particle_measurements_list.append(measurements)
    
    if not particle_measurements_list:
        logger.debug("No valid measurements calculated from contours.")
        return pd.DataFrame(), 0, img_for_contours

    particles_df = pd.DataFrame(particle_measurements_list)
    particles_df = particles_df.sort_values(by=['Y', 'X'], ascending=[True, True]).reset_index(drop=True)
    
    return particles_df, len(particles_df), img_for_contours


def generate_stage1_preview_data(
    image_uint8: np.ndarray, 
    s1_viewer_params: Dict[str, Any] 
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], int]:
    if image_uint8 is None or image_uint8.dtype != np.uint8:
        logger.error("generate_stage1_preview_data requires a uint8 image.")
        return None, None, 0

    image_for_analysis = image_uint8.copy()
    illumination_mode_preview = s1_viewer_params.get('s1_illumination', 'darkfield')
    if illumination_mode_preview == 'collimated':
        logger.debug("S1 Preview: Illumination is collimated. Inverting image.")
        image_for_analysis = cv2.bitwise_not(image_for_analysis)

    particle_analysis_config_for_preview = {
        'apply_convolution': s1_viewer_params.get('s1_apply_convolution', True),
        'convolution_kernel': s1_viewer_params.get('s1_convolution_kernel_list'), 
        'microorganism_threshold': s1_viewer_params.get('s1_microorganism_threshold', 25),
        'min_microorganism_area': s1_viewer_params.get('s1_min_microorganism_area', 200),
        'microorganism_blur_kernel': s1_viewer_params.get('s1_microorganism_blur_kernel', 5)
    }
    
    particles_df, particle_count, processed_binary_img = _analyze_single_subtracted_image(
        image_for_analysis, 
        particle_analysis_config_for_preview
    )

    if particles_df is None: 
        return processed_binary_img, None, 0

    final_contours_for_preview = []
    if processed_binary_img is not None and not particles_df.empty: 
        temp_contours = find_contours(processed_binary_img, mode="external", method="simple")
        if temp_contours:
            min_area_preview = particle_analysis_config_for_preview.get('min_microorganism_area', 200)
            final_contours_for_preview = filter_contours_by_area(temp_contours, min_area_preview)
    
    return processed_binary_img, final_contours_for_preview, particle_count


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str 
    ) -> Optional[str]:
    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} ---")
    logger.info(f"Input image directory: {image_dir_path}")
    logger.info(f"Output directory for this stage: {output_dir_stage1}")

    config = get_config()
    config_s1 = config.get('stage_01_initial_detection', {})
    if not config_s1:
        raise ConfigError("Configuration for 'stage_01_initial_detection' not found.")

    particle_analysis_config = config_s1.get('particle_analysis', {})
    illumination_mode_pipeline = particle_analysis_config.get('illumination', 'darkfield')

    bg_offset = config_s1.get('bg_offset_val', 110)
    mask_cfg = config_s1.get('well_masking', {})
    system_cfg = config.get('system', {}) 

    images_df = _parse_image_filenames(image_dir_path)
    if images_df is None or images_df.empty:
        logger.error("Failed to parse or find any images. Stage 1 cannot proceed.")
        return None

    images_df_filtered = _filter_image_df(images_df.copy())
    if images_df_filtered.empty:
        logger.error("Image DataFrame became empty after filtering. Stage 1 cannot proceed.")
        return None
    
    total_frames_to_process = len(images_df_filtered)
    logger.info(f"Processing {total_frames_to_process} frames after filtering.")
    logger.info(f"Pipeline Illumination mode: {illumination_mode_pipeline}")

    all_particle_data_dfs: List[pd.DataFrame] = []

    for idx in range(total_frames_to_process):
        fg_row = images_df_filtered.iloc[idx]
        bg_index = (idx + bg_offset) % total_frames_to_process 
        if bg_index < 0: bg_index += total_frames_to_process 
        bg_row = images_df_filtered.iloc[bg_index]

        fg_filepath = fg_row['full_path']
        bg_filepath = bg_row['full_path']

        logger.debug(f"Processing FG: '{fg_row['filename']}' (idx {idx}), BG: '{bg_row['filename']}' (idx {bg_index})")

        if not os.path.exists(fg_filepath) or not os.path.exists(bg_filepath):
            logger.warning(f"FG or BG file not found. FG: {fg_filepath}, BG: {bg_filepath}. Skipping pair.")
            continue
        
        if fg_row['fd_mm'] != bg_row['fd_mm']:
            logger.warning(f"Mismatch in fd_mm for FG ('{fg_row['filename']}', fd={fg_row['fd_mm']}) "
                           f"and BG ('{bg_row['filename']}', fd={bg_row['fd_mm']}). Skipping pair.")
            continue

        try:
            img_a_gray_orig = load_image(fg_filepath, mode='gray') 
            img_b_gray_orig = load_image(bg_filepath, mode='gray') 

            if img_a_gray_orig is None or img_b_gray_orig is None:
                logger.error(f"Failed to load FG or BG image for subtraction. Skipping.")
                continue
            
            img_a_to_subtract = img_a_gray_orig.copy()
            img_b_to_subtract = img_b_gray_orig.copy()

            if illumination_mode_pipeline == 'collimated':
                logger.debug(f"Pipeline (collimated): Inverting FG and BG images before subtraction for {fg_row['filename']}.")
                img_a_to_subtract = cv2.bitwise_not(img_a_to_subtract)
                img_b_to_subtract = cv2.bitwise_not(img_b_to_subtract)
            
            subtracted_img = subtract_images(img_a_to_subtract, img_b_to_subtract)
            if subtracted_img is None:
                logger.error(f"Image subtraction failed for FG '{fg_row['filename']}'.")
                continue

            final_img_for_analysis = subtracted_img
            if mask_cfg.get('create_mask', False):
                masked_a, binary_well_mask = mask_well_area(img_a_gray_orig, mask_cfg, system_cfg)
                if masked_a is not None and binary_well_mask is not None:
                    final_img_for_analysis = cv2.bitwise_and(subtracted_img, subtracted_img, mask=binary_well_mask)
                else:
                    logger.warning(f"Well masking failed for FG '{fg_row['filename']}'. Proceeding with unmasked subtracted image.")
            
            particles_in_frame_df, particle_count, _ = _analyze_single_subtracted_image(
                final_img_for_analysis, particle_analysis_config 
            )

            if particles_in_frame_df is not None and not particles_in_frame_df.empty:
                particles_in_frame_df['SAMPLE_ID'] = sample_id
                particles_in_frame_df['source_image_filename'] = fg_row['filename']
                all_particle_data_dfs.append(particles_in_frame_df)
                logger.info(f"  FG '{fg_row['filename']}': Found {particle_count} particles.")
            elif particle_count == 0:
                logger.debug(f"  FG '{fg_row['filename']}': No particles found.")
            else:
                logger.warning(f"  FG '{fg_row['filename']}': Particle analysis returned an error or no valid DataFrame.")

        except Exception as e:
            logger.exception(f"Critical error during processing of FG image '{fg_row['filename']}': {e}")
            continue 

    if not all_particle_data_dfs:
        logger.warning("No particle data collected from any frame. Stage 1 produced no particle results.")
        expected_cols = ['X', 'Y', 'area', 'perimeter', 'orientation', 'aspect_ratio', 'solidity', 
                         'eccentricity', 'feret_diameter', 'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h',
                         'SAMPLE_ID', 'source_image_filename']
        empty_df = pd.DataFrame(columns=expected_cols)
        output_filename = f"{sample_id}_stage1_particles.csv"
        saved_csv_path = save_dataframe_to_csv(empty_df, output_dir_stage1, output_filename)
        return saved_csv_path 

    logger.info(f"Combining particle data from {len(all_particle_data_dfs)} frames...")
    try:
        combined_particles_df = pd.concat(all_particle_data_dfs, ignore_index=True)
        logger.info(f"Stage 1 combined particle table has {len(combined_particles_df)} rows.")
        
        output_filename = f"{sample_id}_stage1_particles.csv"
        saved_csv_path = save_dataframe_to_csv(combined_particles_df, output_dir_stage1, output_filename)

        if saved_csv_path:
            logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} COMPLETED ---")
        else:
            logger.error(f"--- Stage 1: Failed to save results for Sample ID: {sample_id} ---")
        return saved_csv_path

    except Exception as e:
        logger.exception(f"Error concatenating or saving particle DataFrames for Stage 1: {e}")
        return None


if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging
    # CORRECTED: Import load_image from the correct module if it's used directly in __main__
    # from src.core.image_io_operations import load_image # This was the missing part for the Pylance error in __main__
                                                       # if core_load_image was used. Now __main__ uses load_image directly.
    setup_logging() 

    try:
        cfg_main = get_config() 
        
        TEST_SAMPLE_ID = "test_sample_s1_detect_illum_fix" 
        
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir:
            raise ValueError("project_base_dir not set in config for testing.")

        test_image_input_dir = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_output_illum_fix")
        os.makedirs(test_stage1_output_dir, exist_ok=True)
        os.makedirs(test_image_input_dir, exist_ok=True) 

        logger.info(f"Test Input Dir: {test_image_input_dir}")
        logger.info(f"Test Output Dir: {test_stage1_output_dir}")

        if not os.listdir(test_image_input_dir):
            logger.info(f"Test image directory '{test_image_input_dir}' is empty. Creating dummy PNG files...")
            date_str, plate_str, well_str = "20230101", "1", "1" 
            num_frames_total = 120 
            for i in range(num_frames_total):
                dummy_fname = f"{date_str}_plate{plate_str}_well{well_str}_f{i:05d}_dpt0.0_fd0.0mm.png"
                dummy_fpath = os.path.join(test_image_input_dir, dummy_fname)
                dummy_img_content = np.random.randint(0, 50, (100,100), dtype=np.uint8)
                if i % 10 == 0: 
                    cv2.circle(dummy_img_content, (30,30), 10, 200, -1)
                    cv2.circle(dummy_img_content, (70,70), 8, 180, -1)
                if i % 20 == 5: 
                    dummy_img_content = cv2.bitwise_not(dummy_img_content)
                    logger.debug(f"Created dummy image {dummy_fname} with inverted content (simulating collimated source).")
                cv2.imwrite(dummy_fpath, dummy_img_content)
            logger.info(f"Created {num_frames_total} dummy images in {test_image_input_dir}")

        logger.info("\n--- Running S1 Test with 'darkfield' (default from config) ---")
        result_csv_path_darkfield = run_stage_01_initial_detection(
            image_dir_path=test_image_input_dir,
            output_dir_stage1=test_stage1_output_dir,
            sample_id=TEST_SAMPLE_ID + "_darkfield"
        )
        if result_csv_path_darkfield: logger.info(f"S1 test (darkfield) completed. CSV: {result_csv_path_darkfield}")
        else: logger.error("S1 test (darkfield) FAILED.")

        logger.info("\n--- Running S1 Test with 'collimated' (manually overriding config for test) ---")
        original_s1_config = cfg_main['stage_01_initial_detection'].copy()
        cfg_main['stage_01_initial_detection']['particle_analysis']['illumination'] = 'collimated'
        result_csv_path_collimated = run_stage_01_initial_detection(
            image_dir_path=test_image_input_dir,
            output_dir_stage1=test_stage1_output_dir,
            sample_id=TEST_SAMPLE_ID + "_collimated"
        )
        if result_csv_path_collimated: logger.info(f"S1 test (collimated) completed. CSV: {result_csv_path_collimated}")
        else: logger.error("S1 test (collimated) FAILED.")
        cfg_main['stage_01_initial_detection'] = original_s1_config
        logger.info("Restored S1 config after collimated test.")

        logger.info("\n--- Testing generate_stage1_preview_data (viewer function) ---")
        if os.listdir(test_image_input_dir):
            first_image_file = sorted(os.listdir(test_image_input_dir))[0] 
            first_image_path = os.path.join(test_image_input_dir, first_image_file)
            # Correctly use load_image here as it's imported at the top of the file
            img_for_preview_uint8 = load_image(first_image_path, mode='gray') # Changed from core_load_image
            
            if img_for_preview_uint8 is not None:
                s1_particle_cfg_from_config = cfg_main.get('stage_01_initial_detection',{}).get('particle_analysis',{})
                viewer_params_base = {
                    's1_apply_convolution': s1_particle_cfg_from_config.get('apply_convolution', True),
                    's1_convolution_kernel_list': s1_particle_cfg_from_config.get('convolution_kernel'),
                    's1_microorganism_threshold': s1_particle_cfg_from_config.get('microorganism_threshold', 25),
                    's1_min_microorganism_area': s1_particle_cfg_from_config.get('min_microorganism_area', 200),
                    's1_microorganism_blur_kernel': s1_particle_cfg_from_config.get('microorganism_blur_kernel', 5)
                }
                logger.info("  Previewing with DARKFIELD settings:")
                viewer_params_dark = {**viewer_params_base, 's1_illumination': 'darkfield'}
                prev_bin_dark, _, prev_count_dark = generate_stage1_preview_data(img_for_preview_uint8.copy(), viewer_params_dark)
                logger.info(f"  Darkfield preview: {prev_count_dark} particles found.")
                if prev_bin_dark is not None: cv2.imwrite(os.path.join(test_stage1_output_dir, "s1_preview_test_darkfield.png"), prev_bin_dark)

                logger.info("  Previewing with COLLIMATED settings:")
                viewer_params_collim = {**viewer_params_base, 's1_illumination': 'collimated'}
                prev_bin_col, _, prev_count_col = generate_stage1_preview_data(img_for_preview_uint8.copy(), viewer_params_collim)
                logger.info(f"  Collimated preview: {prev_count_col} particles found.")
                if prev_bin_col is not None: cv2.imwrite(os.path.join(test_stage1_output_dir, "s1_preview_test_collimated.png"), prev_bin_col)
            else: logger.error(f"Could not load image for preview test: {first_image_path}")
        else: logger.warning("No images in test input dir to test generate_stage1_preview_data.")

    except ConfigError as ce: logger.error(f"Configuration error during test: {ce}")
    except FileNotFoundError as fnfe: logger.error(f"File not found during test setup (check config paths and test data): {fnfe}")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 1 test: {e}")