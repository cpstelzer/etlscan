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
from src.core.image_io_operations import load_image
from src.core.image_enhancement import subtract_images, apply_threshold, apply_blur
from src.core.masking_operations import mask_well_area
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area

logger = logging.getLogger(__name__)

def _parse_image_filenames(image_dir_path: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for PNG images matching a specific pattern,
    parses metadata from filenames, and returns a sorted DataFrame.
    """
    if not os.path.isdir(image_dir_path):
        logger.error(f"Image directory not found: {image_dir_path}")
        return None

    # Example filename: 20250402_plate1_well1_f00000_dpt-0.260_fd0.00mm.png
    # This pattern is from the original ws_background_subtraction.py
    pattern = re.compile(
        r"^(?P<date>\d{8})_plate(?P<plate>\d+)_well(?P<well>\d+)_f(?P<frame>\d{5})_dpt(?P<dpt>[-+]?[0-9]*\.?[0-9]+)_fd(?P<fd_mm>[-+]?[0-9]*\.?[0-9]+)mm\.png$"
    )
    parsed_file_data = []
    found_files_count = 0
    logger.info(f"Scanning for images in: {image_dir_path}")

    for filename in sorted(os.listdir(image_dir_path)): # Sorted for consistent ordering
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
                        "full_path": os.path.join(image_dir_path, filename) # Add full path
                    })
                except ValueError as e:
                    logger.warning(f"Could not parse metadata from filename '{filename}': {e}")
            # else:
            #     logger.debug(f"Filename '{filename}' did not match expected pattern.")
    
    logger.info(f"Found {found_files_count} files matching the pattern in {image_dir_path}.")
    if not parsed_file_data:
        logger.warning("No matching image files found or parsed in the directory.")
        return pd.DataFrame() # Return empty DF

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

    # Remove frame 0
    initial_len = len(df)
    if not df.empty and df.iloc[0]["frame"] == 0:
        df = df.iloc[1:].reset_index(drop=True)
        logger.info(f"Removed frame 0. Rows changed from {initial_len} to {len(df)}.")
    else:
        logger.debug("Frame 0 not found at the first row or DataFrame empty. No frame removed based on this rule.")

    # Filter by fd_mm == 0 criterion
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
    subtracted_image_gray: np.ndarray,
    config_stage1: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], int]: # Returns (DataFrame_of_particles, particle_count)
    """
    Analyzes a single subtracted grayscale image to find particles.
    Corresponds to the core logic of wellcounter_imaging_module.analyze_microorganisms.
    """
    particle_config = config_stage1.get('particle_analysis', {})
    threshold_val = particle_config.get('microorganism_threshold', 30)
    min_area = particle_config.get('min_microorganism_area', 50)
    blur_kernel = particle_config.get('microorganism_blur_kernel', 3)
    # filter_by_shape = particle_config.get('filter_by_shape', False) # Shape filtering currently hardcoded in original

    if subtracted_image_gray is None:
        return None, 0

    # 1. Thresholding
    binary_img = apply_threshold(subtracted_image_gray, threshold_val, "binary")
    if binary_img is None:
        logger.error("Thresholding failed during particle analysis.")
        return None, 0

    # 2. Blurring (Median Blur)
    if blur_kernel > 0 and blur_kernel % 2 == 1:
        blurred_binary_img = apply_blur(binary_img, blur_kernel, "median")
        if blurred_binary_img is None:
            logger.warning("Blurring failed, proceeding with unblurred binary image.")
            img_for_contours = binary_img
        else:
            img_for_contours = blurred_binary_img
    else:
        if blur_kernel > 0: logger.warning(f"Invalid blur_kernel {blur_kernel}, skipping blur.")
        img_for_contours = binary_img
        
    # 3. Find Contours
    contours = find_contours(img_for_contours, mode="external", method="simple")
    if not contours:
        logger.debug("No contours found in the processed image.")
        return pd.DataFrame(), 0 # Return empty DF and 0 count

    # 4. Filter Contours by Area
    valid_contours = filter_contours_by_area(contours, min_area)
    if not valid_contours:
        logger.debug(f"No contours remained after area filter (min_area={min_area}).")
        return pd.DataFrame(), 0

    # 5. Calculate Measurements for Valid Contours
    particle_measurements_list = []
    for contour in valid_contours:
        measurements = calculate_contour_measurements(contour)
        if measurements:
            particle_measurements_list.append(measurements)
    
    if not particle_measurements_list:
        logger.debug("No valid measurements calculated from contours.")
        return pd.DataFrame(), 0

    particles_df = pd.DataFrame(particle_measurements_list)
    
    # Optional: filter_by_shape (original wim had hardcoded quantiles)
    # if filter_by_shape:
    #     df = df[(df['solidity'] >= 0.655) & (df['solidity'] <= 0.987) & ...] # Example
    #     logger.info(f"Applied shape filter, {len(df)} particles remaining.")

    # Sort by Y then X for consistent output
    particles_df = particles_df.sort_values(by=['Y', 'X'], ascending=[True, True]).reset_index(drop=True)
    
    return particles_df, len(particles_df)


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str # e.g., "20250410_plate4_well4"
    ) -> Optional[str]:
    """
    Main function for Stage 1: Initial Particle Detection.
    Orchestrates background subtraction and particle analysis on a series of images.

    Args:
        image_dir_path: Path to the directory containing raw images for a single sample.
        output_dir_stage1: Directory where the output CSV for this stage will be saved.
        sample_id: Identifier for the current sample.

    Returns:
        Optional[str]: Path to the generated CSV file of detected particles, or None on failure.
    """
    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} ---")
    logger.info(f"Input image directory: {image_dir_path}")
    logger.info(f"Output directory for this stage: {output_dir_stage1}")

    config = get_config()
    config_s1 = config.get('stage_01_initial_detection', {})
    if not config_s1:
        raise ConfigError("Configuration for 'stage_01_initial_detection' not found.")

    bg_offset = config_s1.get('bg_offset_val', 110)
    mask_cfg = config_s1.get('well_masking', {})
    system_cfg = config.get('system', {}) # For fixed mask params

    # 1. Parse image filenames and create initial DataFrame
    images_df = _parse_image_filenames(image_dir_path)
    if images_df is None or images_df.empty:
        logger.error("Failed to parse or find any images. Stage 1 cannot proceed.")
        return None

    # 2. Filter DataFrame
    images_df_filtered = _filter_image_df(images_df.copy())
    if images_df_filtered.empty:
        logger.error("Image DataFrame became empty after filtering. Stage 1 cannot proceed.")
        return None
    
    total_frames_to_process = len(images_df_filtered)
    logger.info(f"Processing {total_frames_to_process} frames after filtering.")

    all_particle_data_dfs: List[pd.DataFrame] = []

    # 3. Image Analysis Loop (Foreground-Background processing)
    for idx in range(total_frames_to_process):
        fg_row = images_df_filtered.iloc[idx]
        
        # Compute background index with wrap-around
        bg_index = (idx + bg_offset) % total_frames_to_process # Simpler wrap-around

        # Original ws_background_subtraction had a more complex wrap-around.
        # This simpler modulo might differ if bg_offset is very large relative to total_frames_to_process
        # but should be fine for typical offsets. Let's stick to the original's logic if critical.
        # Re-implementing original wrap-around logic:
        temp_bg_index = idx + bg_offset
        if temp_bg_index >= total_frames_to_process:
            offset_within_set = bg_offset - (total_frames_to_process - idx) # How much it overshot
            # If offset_within_set is negative (e.g. bg_offset smaller than remaining list), then this logic has issues.
            # The original logic for wrap-around:
            # bg_index = offset_within_set  # This was potentially problematic if offset_within_set < 0
            # Corrected wrap-around:
            bg_index = temp_bg_index % total_frames_to_process
            if bg_index < 0: bg_index += total_frames_to_process # Ensure positive index

            # Defensive check against the original's calculation leading to invalid index
            if not (0 <= bg_index < total_frames_to_process):
                 logger.error(f"Invalid wrap-around BG index {bg_index} for FG index {idx}. Total: {total_frames_to_process}. Using simple modulo.")
                 bg_index = (idx + bg_offset) % total_frames_to_process


        bg_row = images_df_filtered.iloc[bg_index]

        fg_filepath = fg_row['full_path']
        bg_filepath = bg_row['full_path']

        logger.debug(f"Processing FG: '{fg_row['filename']}' (idx {idx}), BG: '{bg_row['filename']}' (idx {bg_index})")

        if not os.path.exists(fg_filepath) or not os.path.exists(bg_filepath):
            logger.warning(f"FG or BG file not found. FG: {fg_filepath}, BG: {bg_filepath}. Skipping pair.")
            continue
        
        # Check for fd_mm consistency (focal depth)
        if fg_row['fd_mm'] != bg_row['fd_mm']:
            logger.warning(f"Mismatch in fd_mm for FG ('{fg_row['filename']}', fd={fg_row['fd_mm']}) "
                           f"and BG ('{bg_row['filename']}', fd={bg_row['fd_mm']}). Skipping pair.")
            continue

        try:
            # Load images (grayscale)
            img_a_gray = load_image(fg_filepath, mode='gray')
            img_b_gray = load_image(bg_filepath, mode='gray')

            if img_a_gray is None or img_b_gray is None:
                logger.error(f"Failed to load FG or BG image for subtraction. Skipping.")
                continue
            
            # Image Subtraction
            subtracted_img = subtract_images(img_a_gray, img_b_gray)
            if subtracted_img is None:
                logger.error(f"Image subtraction failed for FG '{fg_row['filename']}'.")
                continue

            # Optional Well Masking (on subtracted image or img_a_gray)
            # Original wim applied mask to img_a, then to subtracted.
            # Applying to subtracted makes more sense if mask is purely geometric.
            final_img_for_analysis = subtracted_img
            if mask_cfg.get('create_mask', False):
                # Masking can be applied to image_a (as in original wim) or directly to subtracted_img
                # Let's apply to image_a to get the binary mask, then apply that mask to subtracted_img
                masked_a, binary_well_mask = mask_well_area(img_a_gray, mask_cfg, system_cfg)
                if masked_a is not None and binary_well_mask is not None:
                    final_img_for_analysis = cv2.bitwise_and(subtracted_img, subtracted_img, mask=binary_well_mask)
                else:
                    logger.warning(f"Well masking failed for FG '{fg_row['filename']}'. Proceeding with unmasked subtracted image.")
            
            # Analyze particles in the (potentially masked) subtracted image
            particles_in_frame_df, particle_count = _analyze_single_subtracted_image(final_img_for_analysis, config_s1)

            if particles_in_frame_df is not None and not particles_in_frame_df.empty:
                particles_in_frame_df['SAMPLE_ID'] = sample_id
                particles_in_frame_df['source_image_filename'] = fg_row['filename'] # FG image is the source
                all_particle_data_dfs.append(particles_in_frame_df)
                logger.info(f"  FG '{fg_row['filename']}': Found {particle_count} particles.")
            elif particle_count == 0:
                logger.debug(f"  FG '{fg_row['filename']}': No particles found.")
            else: # particles_in_frame_df is None
                logger.warning(f"  FG '{fg_row['filename']}': Particle analysis returned an error or no valid DataFrame.")

        except Exception as e:
            logger.exception(f"Critical error during processing of FG image '{fg_row['filename']}': {e}")
            continue # Robust: continue to next frame

    # 4. Combine and Save Results
    if not all_particle_data_dfs:
        logger.warning("No particle data collected from any frame. Stage 1 produced no particle results.")
        # Save an empty dataframe with expected columns if necessary for downstream consistency
        # For now, just return None for the path.
        # Consider creating an empty DataFrame with expected schema.
        expected_cols = ['X', 'Y', 'area', 'perimeter', 'orientation', 'aspect_ratio', 'solidity', 
                         'eccentricity', 'feret_diameter', 'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h',
                         'SAMPLE_ID', 'source_image_filename']
        empty_df = pd.DataFrame(columns=expected_cols)
        output_filename = f"{sample_id}_stage1_particles.csv"
        saved_csv_path = save_dataframe_to_csv(empty_df, output_dir_stage1, output_filename)
        return saved_csv_path # Return path to empty CSV


    logger.info(f"Combining particle data from {len(all_particle_data_dfs)} frames...")
    try:
        combined_particles_df = pd.concat(all_particle_data_dfs, ignore_index=True)
        logger.info(f"Stage 1 combined particle table has {len(combined_particles_df)} rows.")
        
        # Ensure consistent column order (optional, but good for clarity)
        # Define expected columns based on _analyze_single_subtracted_image and added ones
        # This can be made more robust by defining schema earlier.
        
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
    # This block is for testing stage_01_initial_detection.py directly.
    # It requires:
    # 1. A valid 'config/pipeline_config.yml' (or mocked config).
    # 2. A sample image directory structured as expected.
    # 3. The 'src' directory to be in Python's path or running from project root.

    from src.utils.logging_setup import setup_logging
    setup_logging() # Initialize logging based on config

    try:
        cfg_main = get_config() # Load config to ensure it's available

        # --- Setup for a Test Run ---
        # You'll need to create dummy images in this structure or point to real data.
        # Example: well_plate_analyzer/data/input/test_sample_s1/
        #          Contains: 20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png, ..._f00001_..., ..._f00110_... etc.
        
        TEST_SAMPLE_ID = "test_sample_s1_detect" # Name of your test sample subdirectory
        
        # Construct paths using path_helpers (which use the config for base_dir)
        # Assume project_base_dir in config points to 'well_plate_analyzer'
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir:
            raise ValueError("project_base_dir not set in config for testing.")

        test_image_input_dir = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_output")
        os.makedirs(test_stage1_output_dir, exist_ok=True)

        logger.info(f"Test Input Dir: {test_image_input_dir}")
        logger.info(f"Test Output Dir: {test_stage1_output_dir}")

        if not os.path.isdir(test_image_input_dir) or not os.listdir(test_image_input_dir):
            logger.error(f"Test image directory '{test_image_input_dir}' is missing or empty. Cannot run test.")
            logger.error("Please create dummy PNG files matching the expected naming pattern for testing.")
            # Example filenames:
            # 20230101_plate1_well1_f00000_dpt0.0_fd0.0mm.png
            # 20230101_plate1_well1_f00001_dpt0.1_fd0.0mm.png
            # ...
            # 20230101_plate1_well1_f00110_dpt1.0_fd1.0mm.png (bg_offset=110)
            # Create simple black/white images for these.
        else:
            # Run the stage
            result_csv_path = run_stage_01_initial_detection(
                image_dir_path=test_image_input_dir,
                output_dir_stage1=test_stage1_output_dir,
                sample_id=TEST_SAMPLE_ID
            )

            if result_csv_path:
                logger.info(f"Stage 1 test completed. Results CSV: {result_csv_path}")
                # Optionally, load and print some info from the CSV
                # df_res = pd.read_csv(result_csv_path)
                # print(df_res.head())
            else:
                logger.error("Stage 1 test FAILED or produced no output.")

    except ConfigError as ce:
        logger.error(f"Configuration error during test: {ce}")
    except FileNotFoundError as fnfe:
        logger.error(f"File not found during test setup (check config paths and test data): {fnfe}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the Stage 1 test: {e}")