# src/stages/stage_01_initial_detection.py
import os
import re
import cv2
import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, List, Dict, Any, Tuple
from sklearn.neighbors import BallTree # Added for compare_detected_particles

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image 
from src.core.image_enhancement import subtract_images, apply_threshold, apply_blur
from src.core.masking_operations import mask_well_area, apply_mask # apply_mask is used
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
    blur_kernel_size = config_stage1_particle_analysis.get('microorganism_blur_kernel', 3) # For subtracted

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
        if blur_kernel_size > 0: logger.warning(f"Invalid blur_kernel_size {blur_kernel_size} for subtracted analysis, skipping blur.")
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


def _analyze_unsubtracted_fg_image(
    original_fg_image_gray: np.ndarray, # uint8, potentially already masked
    config_s1_unsubtracted_analysis: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], int, Optional[np.ndarray]]:
    """
    Analyzes a single original (unsubtracted) grayscale foreground image to find particles.
    The input image might have already been masked by the calling function.
    Returns particle data, count, and the binary image used for contour detection.
    """
    illumination_mode = config_s1_unsubtracted_analysis.get('illumination', 'darkfield')
    blur_ksize = config_s1_unsubtracted_analysis.get('blur_kernel_size', 0)
    threshold_mode = config_s1_unsubtracted_analysis.get('thresholding_mode', 'otsu')
    fixed_thresh_val = config_s1_unsubtracted_analysis.get('fixed_threshold_value', 128)
    min_area = config_s1_unsubtracted_analysis.get('min_particle_area', 50)

    if original_fg_image_gray is None or original_fg_image_gray.dtype != np.uint8:
        logger.error("Unsubtracted FG image is None or not uint8.")
        return None, 0, None

    img_proc = original_fg_image_gray.copy() # This image is now what was prepared by the caller

    if illumination_mode == 'collimated':
        logger.debug("Unsubtracted analysis: Illumination is 'collimated'. Inverting (potentially pre-masked) image.")
        img_proc = cv2.bitwise_not(img_proc)

    if blur_ksize > 0 and blur_ksize % 2 == 1:
        img_proc = apply_blur(img_proc, blur_ksize, "gaussian") 
        if img_proc is None:
            logger.warning("Blurring failed in unsubtracted analysis. Using unblurred (but potentially inverted) image.")
            img_proc = cv2.bitwise_not(original_fg_image_gray) if illumination_mode == 'collimated' else original_fg_image_gray.copy()
    elif blur_ksize > 0:
        logger.warning(f"Invalid blur_kernel_size {blur_ksize} for unsubtracted analysis, skipping blur.")

    binary_img: Optional[np.ndarray] = None
    if threshold_mode == 'otsu':
        if img_proc.min() == img_proc.max(): 
            logger.warning("Image for unsubtracted Otsu is uniform. Thresholding may fail or be meaningless.")
        _, binary_img = cv2.threshold(img_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == 'fixed':
        binary_img = apply_threshold(img_proc, fixed_thresh_val, "binary")
    else:
        logger.error(f"Invalid thresholding_mode '{threshold_mode}' for unsubtracted analysis.")
        return None, 0, None
    
    if binary_img is None:
        logger.error("Thresholding failed for unsubtracted image.")
        return None, 0, None

    contours = find_contours(binary_img, mode="external", method="simple")
    if not contours:
        logger.debug("No contours found in unsubtracted FG image.")
        return pd.DataFrame(), 0, binary_img

    valid_contours = filter_contours_by_area(contours, min_area)
    if not valid_contours:
        logger.debug(f"No contours in unsubtracted FG image remained after area filter (min_area={min_area}).")
        return pd.DataFrame(), 0, binary_img

    particle_measurements_list = []
    for contour in valid_contours:
        measurements = calculate_contour_measurements(contour)
        if measurements:
            particle_measurements_list.append(measurements)

    if not particle_measurements_list:
        logger.debug("No valid measurements from unsubtracted FG image contours.")
        return pd.DataFrame(), 0, binary_img
        
    particles_df = pd.DataFrame(particle_measurements_list)
    particles_df = particles_df.sort_values(by=['Y', 'X'], ascending=[True, True]).reset_index(drop=True)
    
    return particles_df, len(particles_df), binary_img


def _compare_and_fuse_particle_dfs(df_ref: pd.DataFrame, df_query: pd.DataFrame, config_s1_comparison: Dict[str, Any]) -> pd.DataFrame:
    """
    Compares particles from df_ref (e.g., subtracted) with df_query (e.g., unsubtracted).
    Returns a DataFrame of matched particles, using attributes from df_query.
    """
    logger.info(f"Comparing {len(df_ref)} ref particles with {len(df_query)} query particles.")
    if df_ref.empty or df_query.empty:
        logger.info("One or both DataFrames for comparison are empty. Returning empty DataFrame.")
        return pd.DataFrame()

    if 'X' not in df_ref.columns or 'Y' not in df_ref.columns or \
       'X' not in df_query.columns or 'Y' not in df_query.columns:
        raise DataValidationError("Both reference and query DataFrames must contain 'X' and 'Y' columns for comparison.")

    sr_factor = config_s1_comparison.get('search_radius_factor', 2.0)
    dp_area = config_s1_comparison.get('default_particle_area', 500.0)
    search_radius = sr_factor * np.sqrt(dp_area / np.pi)
    logger.debug(f"Particle comparison search radius: {search_radius:.2f} pixels.")

    df_ref = df_ref.reset_index(drop=True)
    df_query = df_query.reset_index(drop=True)

    if df_query[['X', 'Y']].isnull().any().any():
        logger.warning("NaN coordinates found in query DataFrame. Dropping these rows before BallTree.")
        df_query_cleaned = df_query.dropna(subset=['X', 'Y'])
    else:
        df_query_cleaned = df_query
    
    if df_query_cleaned.empty:
        logger.warning("Query DataFrame is empty after cleaning NaN coordinates. No matches possible.")
        return pd.DataFrame()

    tree = BallTree(df_query_cleaned[['X', 'Y']].astype(float).values) 

    matched_particles_list = []
    matched_query_original_indices = set()

    for ref_idx, ref_row in df_ref.iterrows():
        ref_x, ref_y = ref_row['X'], ref_row['Y']
        if pd.isna(ref_x) or pd.isna(ref_y):
            logger.debug(f"Skipping ref particle at index {ref_idx} due to NaN coordinates.")
            continue

        indices_in_radius = tree.query_radius([[float(ref_x), float(ref_y)]], r=search_radius)[0]
        
        candidate_query_original_indices = [df_query_cleaned.index[i] for i in indices_in_radius if df_query_cleaned.index[i] not in matched_query_original_indices]

        if candidate_query_original_indices:
            min_dist = float('inf')
            best_query_original_idx = -1

            for query_original_idx in candidate_query_original_indices:
                query_row = df_query_cleaned.loc[query_original_idx]
                dist_sq = (query_row['X'] - ref_x)**2 + (query_row['Y'] - ref_y)**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    best_query_original_idx = query_original_idx
            
            if best_query_original_idx != -1:
                query_particle_data = df_query_cleaned.loc[best_query_original_idx].to_dict()
                query_particle_data['source_image_filename'] = ref_row.get('source_image_filename')
                query_particle_data['SAMPLE_ID'] = ref_row.get('SAMPLE_ID')
                
                matched_particles_list.append(query_particle_data)
                matched_query_original_indices.add(best_query_original_idx)
    
    if not matched_particles_list:
        logger.info("No particles were matched between reference and query sets.")
        return pd.DataFrame()

    final_matched_df = pd.DataFrame(matched_particles_list)
    logger.info(f"Successfully matched {len(final_matched_df)} particles. Using their unsubtracted image coordinates.")
    return final_matched_df


def generate_stage1_preview_data(
    image_uint8: np.ndarray, 
    s1_viewer_params: Dict[str, Any] 
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], int]:
    # This function in the viewer directly calls _analyze_single_subtracted_image
    # or _analyze_unsubtracted_fg_image.
    # The viewer's `process_and_display` method now handles the specific input preparation
    # (like specialized masking for collimated unsubtracted) before calling these core functions.
    # So, this function can remain relatively simple, assuming its input `image_uint8` is
    # already prepared by the viewer according to the specific analysis path being previewed.

    if image_uint8 is None or image_uint8.dtype != np.uint8:
        logger.error("generate_stage1_preview_data requires a uint8 image.")
        return None, None, 0

    # The viewer will decide if `image_uint8` needs inversion *before* calling this.
    # This function now just runs the core analysis with the provided params.
    # The `s1_viewer_params` should contain the *specific* set of parameters
    # for either subtracted or unsubtracted analysis path.

    # Determine if we're previewing subtracted or unsubtracted path based on param keys
    is_subtracted_preview = 's1_sub_illumination' in s1_viewer_params
    is_unsubtracted_preview = 's1_unsub_illumination' in s1_viewer_params

    particles_df = None
    particle_count = 0
    processed_binary_img = None
    
    if is_subtracted_preview:
        # Parameters for this preview should align with 'particle_analysis' (for subtracted images)
        # The viewer already prepared `image_uint8` (e.g., inverted it if s1_sub_illumination was collimated)
        particle_analysis_config_for_preview = {
            'apply_convolution': s1_viewer_params.get('s1_sub_apply_convolution', True),
            'convolution_kernel': s1_viewer_params.get('s1_convolution_kernel_list'), # Viewer should pass this if needed
            'microorganism_threshold': s1_viewer_params.get('s1_sub_microorganism_threshold', 25),
            'min_microorganism_area': s1_viewer_params.get('s1_sub_min_microorganism_area', 200),
            'microorganism_blur_kernel': s1_viewer_params.get('s1_sub_microorganism_blur_kernel', 5)
        }
        particles_df, particle_count, processed_binary_img = _analyze_single_subtracted_image(
            image_uint8, particle_analysis_config_for_preview
        )
    elif is_unsubtracted_preview:
         # Parameters for this preview should align with 'unsubtracted_particle_analysis'
         # The viewer already prepared `image_uint8` (e.g., applied special masking if s1_unsub_illumination was collimated)
        unsub_analysis_config_for_preview = {
            'illumination': s1_viewer_params.get('s1_unsub_illumination', 'collimated'),
            'blur_kernel_size': s1_viewer_params.get('s1_unsub_blur_kernel_size', 3),
            'thresholding_mode': s1_viewer_params.get('s1_unsub_thresholding_mode', 'otsu'),
            'fixed_threshold_value': s1_viewer_params.get('s1_unsub_fixed_threshold_value', 70),
            'min_particle_area': s1_viewer_params.get('s1_unsub_min_particle_area', 300),
        }
        particles_df, particle_count, processed_binary_img = _analyze_unsubtracted_fg_image(
            image_uint8, unsub_analysis_config_for_preview
        )
    else:
        logger.error("Could not determine S1 preview type (subtracted/unsubtracted) from viewer params.")
        return None, None, 0


    if particles_df is None: 
        return processed_binary_img, None, 0

    final_contours_for_preview = []
    if processed_binary_img is not None and not particles_df.empty: 
        temp_contours = find_contours(processed_binary_img, mode="external", method="simple")
        if temp_contours:
            # Use the correct min_area based on which path was simulated
            min_area_preview = 0
            if is_subtracted_preview:
                min_area_preview = s1_viewer_params.get('s1_sub_min_microorganism_area', 200)
            elif is_unsubtracted_preview:
                min_area_preview = s1_viewer_params.get('s1_unsub_min_particle_area', 300)
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

    particle_analysis_subtracted_config = config_s1.get('particle_analysis', {})
    illumination_mode_subtracted = particle_analysis_subtracted_config.get('illumination', 'darkfield')

    unsubtracted_analysis_config = config_s1.get('unsubtracted_particle_analysis', {})
    if not unsubtracted_analysis_config:
        raise ConfigError("Missing 'unsubtracted_particle_analysis' config in Stage 1.")
    illumination_mode_unsubtracted = unsubtracted_analysis_config.get('illumination', 'collimated')


    particle_comparison_config = config_s1.get('particle_comparison', {})
    if not particle_comparison_config:
        raise ConfigError("Missing 'particle_comparison' config in Stage 1.")


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
    logger.info(f"Subtracted analysis illumination mode: {illumination_mode_subtracted}")
    logger.info(f"Unsubtracted FG analysis illumination mode: {illumination_mode_unsubtracted}")


    all_subtracted_particle_dfs: List[pd.DataFrame] = []
    all_unsubtracted_particle_dfs: List[pd.DataFrame] = []


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
                logger.error(f"Failed to load FG or BG image. Skipping pair for {fg_row['filename']}.")
                continue
            
            # --- Prepare Well Mask (once per FG image) ---
            well_mask_uint8 = None
            if mask_cfg.get('create_mask', False):
                _, well_mask_uint8 = mask_well_area(img_a_gray_orig.copy(), mask_cfg, system_cfg)
                if well_mask_uint8 is None:
                    logger.warning(f"Well masking failed for FG '{fg_row['filename']}'. Analyses will use unmasked images.")
            
            # 1. Analysis of SUBTRACTED image (df_ref source)
            img_a_for_subtraction = img_a_gray_orig.copy()
            img_b_for_subtraction = img_b_gray_orig.copy()

            if illumination_mode_subtracted == 'collimated': 
                logger.debug(f"Subtracted Path (collimated): Inverting FG and BG before subtraction for {fg_row['filename']}.")
                img_a_for_subtraction = cv2.bitwise_not(img_a_for_subtraction)
                img_b_for_subtraction = cv2.bitwise_not(img_b_for_subtraction)
            
            subtracted_img = subtract_images(img_a_for_subtraction, img_b_for_subtraction)
            if subtracted_img is None:
                logger.error(f"Image subtraction failed for FG '{fg_row['filename']}'.")
                continue

            img_for_sub_analysis = subtracted_img
            if well_mask_uint8 is not None: # Apply mask to the SUBTRACTED image
                img_for_sub_analysis = apply_mask(subtracted_img, well_mask_uint8)
            
            subtracted_particles_df, p_count_sub, _ = _analyze_single_subtracted_image(
                img_for_sub_analysis, particle_analysis_subtracted_config
            )
            if subtracted_particles_df is not None and not subtracted_particles_df.empty:
                subtracted_particles_df['SAMPLE_ID'] = sample_id
                subtracted_particles_df['source_image_filename'] = fg_row['filename']
                all_subtracted_particle_dfs.append(subtracted_particles_df)
                logger.debug(f"  FG '{fg_row['filename']}' (Subtracted): Found {p_count_sub} particles.")

            # 2. Analysis of UNSUBTRACTED original FG image (df_query source)
            img_for_unsub_analysis_prepared = img_a_gray_orig.copy()
            if well_mask_uint8 is not None:
                if illumination_mode_unsubtracted == 'collimated':
                    # For 'collimated' unsubtracted analysis, make area outside mask white (255)
                    # so after internal inversion in _analyze_unsubtracted_fg_image, it becomes black (0).
                    img_filled_outside_white = np.full_like(img_a_gray_orig, 255)
                    img_for_unsub_analysis_prepared = cv2.bitwise_and(img_a_gray_orig, img_a_gray_orig, mask=well_mask_uint8) + \
                                                      cv2.bitwise_and(img_filled_outside_white, img_filled_outside_white, mask=cv2.bitwise_not(well_mask_uint8))
                    logger.debug(f"Prepared unsubtracted FG for collimated analysis by setting area outside mask to 255 for {fg_row['filename']}.")
                else: # 'darkfield' unsubtracted analysis, or if mask is None
                    img_for_unsub_analysis_prepared = apply_mask(img_a_gray_orig.copy(), well_mask_uint8)
            # If well_mask_uint8 is None, img_for_unsub_analysis_prepared remains the original unmasked FG image.
            
            unsubtracted_particles_df, p_count_unsub, _ = _analyze_unsubtracted_fg_image(
                img_for_unsub_analysis_prepared, unsubtracted_analysis_config
            )
            if unsubtracted_particles_df is not None and not unsubtracted_particles_df.empty:
                unsubtracted_particles_df['SAMPLE_ID'] = sample_id
                unsubtracted_particles_df['source_image_filename'] = fg_row['filename']
                all_unsubtracted_particle_dfs.append(unsubtracted_particles_df)
                logger.debug(f"  FG '{fg_row['filename']}' (Unsubtracted): Found {p_count_unsub} particles.")

        except Exception as e:
            logger.exception(f"Critical error during processing of FG image '{fg_row['filename']}': {e}")
            continue 

    final_df_ref = pd.concat(all_subtracted_particle_dfs, ignore_index=True) if all_subtracted_particle_dfs else pd.DataFrame()
    final_df_query = pd.concat(all_unsubtracted_particle_dfs, ignore_index=True) if all_unsubtracted_particle_dfs else pd.DataFrame()
    
    logger.info(f"Total subtracted particles found: {len(final_df_ref)}")
    logger.info(f"Total unsubtracted particles found: {len(final_df_query)}")

    combined_particles_df = _compare_and_fuse_particle_dfs(final_df_ref, final_df_query, particle_comparison_config)
    
    if combined_particles_df.empty:
        logger.warning("No particle data matched and fused. Stage 1 results will be empty.")
        expected_cols = ['SAMPLE_ID', 'source_image_filename', 'X', 'Y', 'area', 'perimeter', 'orientation', 
                         'aspect_ratio', 'solidity', 'eccentricity', 'feret_diameter', 
                         'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h']
        empty_df = pd.DataFrame(columns=expected_cols)
        output_filename = f"{sample_id}_stage1_particles.csv"
        saved_csv_path = save_dataframe_to_csv(empty_df, output_dir_stage1, output_filename)
        return saved_csv_path

    logger.info(f"Stage 1 final combined particle table has {len(combined_particles_df)} rows.")
    
    output_filename = f"{sample_id}_stage1_particles.csv"
    cols_order = ['SAMPLE_ID', 'source_image_filename', 'X', 'Y', 'area', 'perimeter', 
                  'orientation', 'aspect_ratio', 'solidity', 'eccentricity', 
                  'feret_diameter', 'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h']
    existing_cols_in_df = [col for col in cols_order if col in combined_particles_df.columns]
    other_cols = [col for col in combined_particles_df.columns if col not in existing_cols_in_df]
    final_cols_order = existing_cols_in_df + other_cols
    
    saved_csv_path = save_dataframe_to_csv(combined_particles_df[final_cols_order], output_dir_stage1, output_filename)

    if saved_csv_path:
        logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} COMPLETED ---")
    else:
        logger.error(f"--- Stage 1: Failed to save results for Sample ID: {sample_id} ---")
    return saved_csv_path


if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging
    setup_logging() 

    try:
        cfg_main = get_config() 
        
        TEST_SAMPLE_ID = "test_sample_s1_core_fix" 
        
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir:
            raise ValueError("project_base_dir not set in config for testing.")

        test_image_input_dir = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_output_core_fix")
        os.makedirs(test_stage1_output_dir, exist_ok=True)
        os.makedirs(test_image_input_dir, exist_ok=True) 

        logger.info(f"Test Input Dir: {test_image_input_dir}")
        logger.info(f"Test Output Dir: {test_stage1_output_dir}")

        if not os.listdir(test_image_input_dir):
            logger.info(f"Test image directory '{test_image_input_dir}' is empty. Creating dummy PNG files...")
            date_str, plate_str, well_str = "20230101", "1", "1" 
            num_frames_total = 5 
            for i in range(num_frames_total):
                dummy_fname = f"{date_str}_plate{plate_str}_well{well_str}_f{i:05d}_dpt0.0_fd0.0mm.png"
                dummy_fpath = os.path.join(test_image_input_dir, dummy_fname)
                
                dummy_img_content = np.full((200, 200), 200, dtype=np.uint8) # Bright BG
                # Particle that moves
                cv2.circle(dummy_img_content, (50 + i * 10, 50 + i * 10), 15, 30, -1) # Dark particle
                # Static particle (should be filtered out by subtraction, then not found in ref)
                if i % 2 == 0:
                    cv2.circle(dummy_img_content, (150,150), 10, 40, -1)
                cv2.imwrite(dummy_fpath, dummy_img_content)
            logger.info(f"Created {num_frames_total} dummy images in {test_image_input_dir}")

        logger.info("\n--- Running S1 Test with core logic fix ---")
        
        # Ensure necessary S1 config subsections exist for the test
        s1_cfg = cfg_main.setdefault('stage_01_initial_detection', {})
        s1_cfg.setdefault('particle_analysis', {'illumination': 'collimated', 'microorganism_threshold': 25, 'min_microorganism_area': 50})
        s1_cfg.setdefault('unsubtracted_particle_analysis', {'illumination': 'collimated', 'thresholding_mode': 'otsu', 'min_particle_area': 30})
        s1_cfg.setdefault('particle_comparison', {'search_radius_factor': 2.0, 'default_particle_area': 100})
        s1_cfg.setdefault('well_masking', {'create_mask': True, 'auto_center_mask': True, 'radius_px': 180, 'radius_fraction': 0.9}) # Smaller radius for 200x200 image

        result_csv_path = run_stage_01_initial_detection(
            image_dir_path=test_image_input_dir,
            output_dir_stage1=test_stage1_output_dir,
            sample_id=TEST_SAMPLE_ID
        )
        if result_csv_path: 
            logger.info(f"S1 test with core logic fix completed. CSV: {result_csv_path}")
            df_out = pd.read_csv(result_csv_path)
            logger.info(f"Output DataFrame head:\n{df_out.head()}")
            logger.info(f"Number of particles in output: {len(df_out)}")
            assert len(df_out) > 0, "Expected some fused particles for this test case."

        else: logger.error("S1 test with core logic fix FAILED.")

    except ConfigError as ce: logger.error(f"Configuration error during test: {ce}")
    except FileNotFoundError as fnfe: logger.error(f"File not found during test setup (check config paths and test data): {fnfe}")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 1 test: {e}")