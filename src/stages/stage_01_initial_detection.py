# src/stages/stage_01_initial_detection.py
import os
import re
import cv2
import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, List, Dict, Any, Tuple
from sklearn.neighbors import BallTree 
import shutil
import skimage.util # Added for img_as_float
import skimage.filters # Added for skimage.filters.threshold_otsu

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image 
from src.core.image_enhancement import subtract_images, apply_threshold, apply_blur # apply_blur might still be used for subtracted path
from src.core.masking_operations import mask_well_area, apply_mask 
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area

logger = logging.getLogger(__name__)

# Removed global _s1_unsub_debug_saved

def _parse_image_filenames(image_dir_path: str) -> Optional[pd.DataFrame]:
    # ... (no changes from previous correct version)
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
                        "filename": filename, "date": groups["date"], "plate": int(groups["plate"]),
                        "well": int(groups["well"]), "frame": int(groups["frame"]),
                        "dpt": float(groups["dpt"]), "fd_mm": float(groups["fd_mm"]),
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
    # ... (no changes from previous correct version)
    if df.empty: return df
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
        logger.info(f"Filtered frames after max_frame {max_frame_fd0} (where fd_mm=0). Kept {len(df)} out of {original_len_before_fd_filter} entries.")
    else:
        logger.warning("No images with fd_mm == 0 were found. Keeping all remaining frames after potential frame 0 removal.")
    if df.empty: logger.warning("DataFrame is empty after filtering steps.")
    return df

def _analyze_single_subtracted_image(
    image_to_analyze_gray: np.ndarray, 
    config_stage1_particle_analysis: Dict[str, Any] 
    ) -> Tuple[Optional[pd.DataFrame], int, Optional[np.ndarray]]: 
    # ... (no changes from previous correct version, this uses its own thresholding for subtracted images)
    apply_conv = config_stage1_particle_analysis.get('apply_convolution', True)
    kernel_list = config_stage1_particle_analysis.get('convolution_kernel')
    threshold_val = config_stage1_particle_analysis.get('microorganism_threshold', 30)
    min_area = config_stage1_particle_analysis.get('min_microorganism_area', 50)
    blur_kernel_size_sub = config_stage1_particle_analysis.get('microorganism_blur_kernel', 3) # Renamed to avoid clash
    if image_to_analyze_gray is None: return None, 0, None
    if image_to_analyze_gray.dtype != np.uint8:
        logger.warning(f"Input image to _analyze_single_subtracted_image is {image_to_analyze_gray.dtype}, expected uint8. Attempting conversion.")
        if np.issubdtype(image_to_analyze_gray.dtype, np.floating): image_to_analyze_gray = np.clip(image_to_analyze_gray * 255, 0, 255).astype(np.uint8)
        else: image_to_analyze_gray = image_to_analyze_gray.astype(np.uint8)
    processed_img = image_to_analyze_gray.copy()
    if apply_conv and kernel_list:
        try:
            conv_kernel = np.array(kernel_list, dtype=np.float32)
            if conv_kernel.ndim == 2 and conv_kernel.shape[0] > 0 and conv_kernel.shape[1] > 0: processed_img = cv2.filter2D(processed_img, -1, conv_kernel)
            else: logger.warning(f"Invalid convolution_kernel. Skipping.")
        except Exception as e: logger.error(f"Error applying convolution: {e}. Skipping.")
    elif apply_conv and not kernel_list: logger.warning("Apply_convolution true, but kernel missing. Skipping.")
    binary_img = apply_threshold(processed_img, threshold_val, "binary")
    if binary_img is None: logger.error("Thresholding failed in _analyze_single_subtracted_image."); return None, 0, None 
    img_for_contours = binary_img
    if blur_kernel_size_sub > 0 and blur_kernel_size_sub % 2 == 1:
        blurred_binary_img = apply_blur(binary_img, blur_kernel_size_sub, "median")
        if blurred_binary_img is not None: img_for_contours = blurred_binary_img
        else: logger.warning("Blurring failed, using unblurred binary for subtracted analysis.")
    elif blur_kernel_size_sub > 0 : logger.warning(f"Invalid blur_kernel_size {blur_kernel_size_sub}, skipping blur.")
    contours = find_contours(img_for_contours, mode="external", method="simple")
    if not contours: return pd.DataFrame(), 0, img_for_contours 
    valid_contours = filter_contours_by_area(contours, min_area)
    if not valid_contours: return pd.DataFrame(), 0, img_for_contours
    particle_measurements_list = [m for cnt in valid_contours if (m := calculate_contour_measurements(cnt))]
    if not particle_measurements_list: return pd.DataFrame(), 0, img_for_contours
    particles_df = pd.DataFrame(particle_measurements_list)
    particles_df = particles_df.sort_values(by=['Y', 'X'], ascending=[True, True]).reset_index(drop=True)
    return particles_df, len(particles_df), img_for_contours

def _analyze_unsubtracted_fg_image(
    original_fg_image_gray_uint8: np.ndarray, # Expects uint8
    config_s1_unsubtracted_analysis: Dict[str, Any], # Contains min_particle_area (blur_kernel_size is now ignored here)
    global_threshold_config: Dict[str, Any], # Contains mode, otsu_multiplier, fixed_value
    source_image_illumination: str 
) -> Tuple[Optional[pd.DataFrame], int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Analyzes a single original (unsubtracted) grayscale foreground image.
    Uses global thresholding configuration. No pre-blurring for main segmentation.
    Returns: particles_df, count, final_binary_mask (uint8), image_after_potential_inversion (float, for debug)
    """
    # blur_ksize = config_s1_unsubtracted_analysis.get('blur_kernel_size', 0) # This is now ignored for segmentation
    min_area = config_s1_unsubtracted_analysis.get('min_particle_area', 50)

    threshold_mode = global_threshold_config.get('mode', 'otsu')
    fixed_thresh_val_int = global_threshold_config.get('fixed_threshold_value', 70) # Integer 0-255
    otsu_multiplier = global_threshold_config.get('otsu_threshold_multiplier', 0.45) # Float

    if original_fg_image_gray_uint8 is None:
        logger.error("Unsubtracted FG image is None.")
        return None, 0, None, None
    if original_fg_image_gray_uint8.dtype != np.uint8:
        logger.error(f"Unsubtracted FG image expected uint8, got {original_fg_image_gray_uint8.dtype}")
        return None, 0, None, None

    # 1. Convert to float [0,1] for consistent processing
    img_float_for_segment = skimage.util.img_as_float(original_fg_image_gray_uint8)
    img_float_after_inversion_for_debug = None 

    # 2. Invert if collimated (operates on float image)
    if source_image_illumination == 'collimated':
        logger.debug("Unsubtracted analysis: Source illumination is 'collimated'. Inverting image.")
        img_float_for_segment = 1.0 - img_float_for_segment
        img_float_after_inversion_for_debug = img_float_for_segment.copy()
    
    # --- MODIFICATION: Pre-thresholding blur REMOVED ---
    # if blur_ksize > 0 and blur_ksize % 2 == 1:
    #     # apply_blur works on uint8, so convert back if needed, or adapt apply_blur
    #     # For now, let's assume if blur was intended, it'd be on the uint8 version before float conversion.
    #     # Since we removed it, this section is commented.
    #     pass
    # --- END MODIFICATION ---
    
    image_to_threshold_float = img_float_for_segment # Use unblurred float image

    final_binary_mask_bool: Optional[np.ndarray] = None
    if threshold_mode == 'otsu':
        if image_to_threshold_float.min() == image_to_threshold_float.max(): 
            logger.warning("Image for unsubtracted Otsu (float) is uniform. Thresholding may fail or be meaningless.")
            # Create an all-black mask if uniform, as Otsu might give unexpected results
            final_binary_mask_bool = np.zeros_like(image_to_threshold_float, dtype=bool)
        else:
            try:
                otsu_thresh_val_raw_sk = skimage.filters.threshold_otsu(image_to_threshold_float) # float 0-1
            except ValueError:
                logger.warning("Skimage Otsu failed for unsubtracted image. Assuming no particles.")
                final_binary_mask_bool = np.zeros_like(image_to_threshold_float, dtype=bool)
            else:
                final_threshold_for_binary_float = otsu_thresh_val_raw_sk * otsu_multiplier
                logger.debug(f"Unsub S1 Otsu: Raw Skimage Otsu val={otsu_thresh_val_raw_sk:.4f}, mult={otsu_multiplier}, final_thresh_float={final_threshold_for_binary_float:.4f}")
                final_binary_mask_bool = image_to_threshold_float > final_threshold_for_binary_float

    elif threshold_mode == 'fixed':
        final_threshold_for_binary_float = fixed_thresh_val_int / 255.0
        logger.debug(f"Unsub S1 Fixed: Config value (0-255)={fixed_thresh_val_int}, Converted threshold_float={final_threshold_for_binary_float:.4f}")
        final_binary_mask_bool = image_to_threshold_float > final_threshold_for_binary_float
    else:
        logger.error(f"Invalid thresholding_mode '{threshold_mode}' for unsubtracted analysis.")
        debug_img_to_return = skimage.util.img_as_ubyte(img_float_after_inversion_for_debug) if img_float_after_inversion_for_debug is not None else None
        return None, 0, None, debug_img_to_return
    
    if final_binary_mask_bool is None: # Should not happen if logic is correct
        logger.error("Binary mask was not generated in unsubtracted analysis.")
        debug_img_to_return = skimage.util.img_as_ubyte(img_float_after_inversion_for_debug) if img_float_after_inversion_for_debug is not None else None
        return None, 0, None, debug_img_to_return

    final_binary_mask_uint8 = final_binary_mask_bool.astype(np.uint8) * 255
    
    # Note: Stage 1 unsubtracted does not have the full morphological chain of S3 (remove small, close, fill holes)
    # It directly finds contours on the (now unblurred) thresholded image.
    contours = find_contours(final_binary_mask_uint8, mode="external", method="simple")
    
    debug_img_float_return = img_float_after_inversion_for_debug if img_float_after_inversion_for_debug is not None else img_float_for_segment
    debug_img_uint8_return = skimage.util.img_as_ubyte(debug_img_float_return)


    if not contours: return pd.DataFrame(), 0, final_binary_mask_uint8, debug_img_uint8_return

    valid_contours = filter_contours_by_area(contours, min_area)
    if not valid_contours: return pd.DataFrame(), 0, final_binary_mask_uint8, debug_img_uint8_return

    particle_measurements_list = [m for cnt in valid_contours if (m := calculate_contour_measurements(cnt))]
    if not particle_measurements_list: return pd.DataFrame(), 0, final_binary_mask_uint8, debug_img_uint8_return
        
    particles_df = pd.DataFrame(particle_measurements_list)
    particles_df = particles_df.sort_values(by=['Y', 'X'], ascending=[True, True]).reset_index(drop=True)
    
    return particles_df, len(particles_df), final_binary_mask_uint8, debug_img_uint8_return


def _compare_and_fuse_particle_dfs(df_ref: pd.DataFrame, df_query: pd.DataFrame, config_s1_comparison: Dict[str, Any]) -> pd.DataFrame:
    # ... (no changes from previous correct version)
    logger.info(f"Comparing {len(df_ref)} ref particles with {len(df_query)} query particles.")
    if df_ref.empty or df_query.empty:
        logger.info("One or both DataFrames for comparison are empty. Returning empty DataFrame.")
        return pd.DataFrame()
    if 'X' not in df_ref.columns or 'Y' not in df_ref.columns or 'X' not in df_query.columns or 'Y' not in df_query.columns:
        raise DataValidationError("Both reference and query DataFrames must contain 'X' and 'Y' columns for comparison.")
    sr_factor = config_s1_comparison.get('search_radius_factor', 2.0)
    dp_area = config_s1_comparison.get('default_particle_area', 500.0)
    search_radius = sr_factor * np.sqrt(dp_area / np.pi)
    logger.debug(f"Particle comparison search radius: {search_radius:.2f} pixels.")
    df_ref = df_ref.reset_index(drop=True); df_query = df_query.reset_index(drop=True)
    if df_query[['X', 'Y']].isnull().any().any():
        logger.warning("NaN coordinates found in query DataFrame. Dropping these rows before BallTree.")
        df_query_cleaned = df_query.dropna(subset=['X', 'Y'])
    else: df_query_cleaned = df_query
    if df_query_cleaned.empty:
        logger.warning("Query DataFrame is empty after cleaning NaN coordinates. No matches possible.")
        return pd.DataFrame()
    tree = BallTree(df_query_cleaned[['X', 'Y']].astype(float).values) 
    matched_particles_list = []; matched_query_original_indices = set()
    for ref_idx, ref_row in df_ref.iterrows():
        ref_x, ref_y = ref_row['X'], ref_row['Y']
        if pd.isna(ref_x) or pd.isna(ref_y):
            logger.debug(f"Skipping ref particle at index {ref_idx} due to NaN coordinates.")
            continue
        indices_in_radius = tree.query_radius([[float(ref_x), float(ref_y)]], r=search_radius)[0]
        candidate_query_original_indices = [df_query_cleaned.index[i] for i in indices_in_radius if df_query_cleaned.index[i] not in matched_query_original_indices]
        if candidate_query_original_indices:
            min_dist = float('inf'); best_query_original_idx = -1
            for query_original_idx in candidate_query_original_indices:
                query_row = df_query_cleaned.loc[query_original_idx]
                dist_sq = (query_row['X'] - ref_x)**2 + (query_row['Y'] - ref_y)**2
                if dist_sq < min_dist: min_dist = dist_sq; best_query_original_idx = query_original_idx
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


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str 
    ) -> Optional[str]:
    
    s1_unsub_debug_images_saved_this_run = False # Reset for each run call

    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} ---")
    config = get_config()
    config_s1 = config.get('stage_01_initial_detection', {})
    if not config_s1: raise ConfigError("Configuration for 'stage_01_initial_detection' not found.")
    
    system_cfg = config.get('system', {})
    source_image_illumination = system_cfg.get('source_image_illumination_type', 'darkfield')
    global_threshold_config = system_cfg.get('thresholding', {})
    if not global_threshold_config:
        raise ConfigError("System-wide thresholding configuration ('system.thresholding') not found.")
    
    logger.info(f"Stage 1 processing with source_image_illumination_type: {source_image_illumination}")
    logger.info(f"Stage 1 using global thresholding config for unsubtracted analysis: {global_threshold_config}")
    
    particle_analysis_subtracted_config = config_s1.get('particle_analysis', {})
    # Unsubtracted analysis config now mainly for min_particle_area; blur_kernel_size is ignored for segmentation.
    unsubtracted_analysis_config = config_s1.get('unsubtracted_particle_analysis', {}) 
    if not unsubtracted_analysis_config: raise ConfigError("Missing 'unsubtracted_particle_analysis' config in Stage 1.")
    
    particle_comparison_config = config_s1.get('particle_comparison', {})
    if not particle_comparison_config: raise ConfigError("Missing 'particle_comparison' config in Stage 1.")
    bg_offset = config_s1.get('bg_offset_val', 110)
    mask_cfg = config_s1.get('well_masking', {})
    
    images_df = _parse_image_filenames(image_dir_path)
    if images_df is None or images_df.empty: logger.error("Failed to parse or find any images."); return None
    images_df_filtered = _filter_image_df(images_df.copy())
    if images_df_filtered.empty: logger.error("Image DataFrame empty after filtering."); return None
    total_frames_to_process = len(images_df_filtered)
    logger.info(f"Processing {total_frames_to_process} frames after filtering.")
    all_subtracted_particle_dfs: List[pd.DataFrame] = []
    all_unsubtracted_particle_dfs: List[pd.DataFrame] = []

    for idx in range(total_frames_to_process):
        fg_row = images_df_filtered.iloc[idx]
        bg_index = (idx + bg_offset) % total_frames_to_process 
        if bg_index < 0: bg_index += total_frames_to_process 
        bg_row = images_df_filtered.iloc[bg_index]
        fg_filepath = fg_row['full_path']; bg_filepath = bg_row['full_path']
        logger.debug(f"Processing FG: '{fg_row['filename']}' (idx {idx}), BG: '{bg_row['filename']}' (idx {bg_index})")
        if not os.path.exists(fg_filepath) or not os.path.exists(bg_filepath): logger.warning(f"FG or BG file not found. Skipping pair."); continue
        if fg_row['fd_mm'] != bg_row['fd_mm']: logger.warning(f"Mismatch fd_mm for FG/BG. Skipping pair."); continue
        try:
            img_a_gray_orig_uint8 = load_image(fg_filepath, mode='gray') # uint8
            img_b_gray_orig_uint8 = load_image(bg_filepath, mode='gray') # uint8
            if img_a_gray_orig_uint8 is None or img_b_gray_orig_uint8 is None: 
                logger.error(f"Failed to load FG/BG. Skipping {fg_row['filename']}.")
                continue
            
            well_mask_uint8 = None
            if mask_cfg.get('create_mask', False):
                _, well_mask_uint8 = mask_well_area(img_a_gray_orig_uint8.copy(), mask_cfg, system_cfg)
                if well_mask_uint8 is None: logger.warning(f"Well masking failed for {fg_row['filename']}.")
            
            img_a_for_sub = img_a_gray_orig_uint8.copy(); img_b_for_sub = img_b_gray_orig_uint8.copy()
            # Subtracted path inversion and thresholding remains specific to its config
            if source_image_illumination == 'collimated': 
                 img_a_for_sub = cv2.bitwise_not(img_a_for_sub); img_b_for_sub = cv2.bitwise_not(img_b_for_sub)
            subtracted_img = subtract_images(img_a_for_sub, img_b_for_sub)
            if subtracted_img is None: logger.error(f"Subtraction failed for {fg_row['filename']}."); continue
            img_for_sub_analysis = apply_mask(subtracted_img, well_mask_uint8) if well_mask_uint8 is not None else subtracted_img
            sub_particles_df, p_count_sub, _ = _analyze_single_subtracted_image(img_for_sub_analysis, particle_analysis_subtracted_config)
            if sub_particles_df is not None and not sub_particles_df.empty:
                sub_particles_df['SAMPLE_ID'] = sample_id; sub_particles_df['source_image_filename'] = fg_row['filename']
                all_subtracted_particle_dfs.append(sub_particles_df); logger.debug(f"  FG '{fg_row['filename']}' (Subtracted): Found {p_count_sub} particles.")

            # Prepare input for unsubtracted analysis (still uint8)
            img_for_unsub_analysis_input_uint8 = img_a_gray_orig_uint8.copy()
            if well_mask_uint8 is not None:
                if source_image_illumination == 'collimated': # masking after potential inversion of main analysis
                    img_filled_outside_white = np.full_like(img_a_gray_orig_uint8, 255, dtype=np.uint8)
                    img_for_unsub_analysis_input_uint8 = cv2.bitwise_and(img_a_gray_orig_uint8, img_a_gray_orig_uint8, mask=well_mask_uint8) + \
                                                         cv2.bitwise_and(img_filled_outside_white, img_filled_outside_white, mask=cv2.bitwise_not(well_mask_uint8))
                else: 
                    img_for_unsub_analysis_input_uint8 = apply_mask(img_a_gray_orig_uint8.copy(), well_mask_uint8)
            
            unsub_particles_df, p_count_unsub, unsub_binary_debug_uint8, unsub_inverted_debug_uint8 = _analyze_unsubtracted_fg_image(
                img_for_unsub_analysis_input_uint8, # Pass uint8
                unsubtracted_analysis_config, 
                global_threshold_config,      
                source_image_illumination
            )
            
            if not s1_unsub_debug_images_saved_this_run and idx == 0: 
                if source_image_illumination == 'collimated' and unsub_inverted_debug_uint8 is not None:
                    cv2.imwrite(os.path.join(output_dir_stage1, f"{sample_id}_DEBUG_s1_unsub_inverted_input.png"), unsub_inverted_debug_uint8)
                    logger.info(f"DEBUG: Saved {sample_id}_DEBUG_s1_unsub_inverted_input.png")
                if unsub_binary_debug_uint8 is not None:
                    cv2.imwrite(os.path.join(output_dir_stage1, f"{sample_id}_DEBUG_s1_unsub_binary_output.png"), unsub_binary_debug_uint8)
                    logger.info(f"DEBUG: Saved {sample_id}_DEBUG_s1_unsub_binary_output.png")
                if well_mask_uint8 is not None and source_image_illumination == 'collimated':
                     cv2.imwrite(os.path.join(output_dir_stage1, f"{sample_id}_DEBUG_s1_unsub_prepped_masked_input.png"), img_for_unsub_analysis_input_uint8)
                     logger.info(f"DEBUG: Saved {sample_id}_DEBUG_s1_unsub_prepped_masked_input.png")
                s1_unsub_debug_images_saved_this_run = True

            if unsub_particles_df is not None and not unsub_particles_df.empty:
                unsub_particles_df['SAMPLE_ID'] = sample_id; unsub_particles_df['source_image_filename'] = fg_row['filename']
                all_unsubtracted_particle_dfs.append(unsub_particles_df); logger.debug(f"  FG '{fg_row['filename']}' (Unsubtracted): Found {p_count_unsub} particles.")
        except Exception as e: logger.exception(f"Critical error processing FG '{fg_row['filename']}': {e}"); continue 
    
    final_df_ref = pd.concat(all_subtracted_particle_dfs, ignore_index=True) if all_subtracted_particle_dfs else pd.DataFrame()
    final_df_query = pd.concat(all_unsubtracted_particle_dfs, ignore_index=True) if all_unsubtracted_particle_dfs else pd.DataFrame()
    logger.info(f"Total subtracted particles found: {len(final_df_ref)}")
    logger.info(f"Total unsubtracted particles found (after no-blur thresholding): {len(final_df_query)}")
    combined_particles_df = _compare_and_fuse_particle_dfs(final_df_ref, final_df_query, particle_comparison_config)
    if combined_particles_df.empty:
        logger.warning("No particle data matched and fused. Stage 1 results will be empty.")
        expected_cols = ['SAMPLE_ID', 'source_image_filename', 'X', 'Y', 'area', 'perimeter', 'orientation', 'aspect_ratio', 'solidity', 'eccentricity', 'feret_diameter', 'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h']
        empty_df = pd.DataFrame(columns=expected_cols)
        output_filename = f"{sample_id}_stage1_particles.csv"
        saved_csv_path = save_dataframe_to_csv(empty_df, output_dir_stage1, output_filename)
        return saved_csv_path
    logger.info(f"Stage 1 final combined particle table has {len(combined_particles_df)} rows.")
    output_filename = f"{sample_id}_stage1_particles.csv"
    cols_order = ['SAMPLE_ID', 'source_image_filename', 'X', 'Y', 'area', 'perimeter', 'orientation', 'aspect_ratio', 'solidity', 'eccentricity', 'feret_diameter', 'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h']
    existing_cols_in_df = [col for col in cols_order if col in combined_particles_df.columns]; other_cols = [col for col in combined_particles_df.columns if col not in existing_cols_in_df]
    final_cols_order = existing_cols_in_df + other_cols
    saved_csv_path = save_dataframe_to_csv(combined_particles_df[final_cols_order], output_dir_stage1, output_filename)
    if saved_csv_path: logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} COMPLETED ---")
    else: logger.error(f"--- Stage 1: Failed to save results for Sample ID: {sample_id} ---")
    return saved_csv_path


if __name__ == '__main__':
    from src.utils.logging_setup import setup_logging
    setup_logging() 
    try:
        cfg_main = get_config() 
        system_cfg_test = cfg_main.setdefault('system', {})
        system_cfg_test.setdefault('source_image_illumination_type', 'collimated') 
        thresh_cfg_test = system_cfg_test.setdefault('thresholding', {})
        thresh_cfg_test.setdefault('mode', 'otsu')
        thresh_cfg_test.setdefault('otsu_threshold_multiplier', 0.45)
        thresh_cfg_test.setdefault('fixed_threshold_value', 70)
        
        TEST_SAMPLE_ID = "test_sample_s1_no_blur_unsub" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir: raise ValueError("project_base_dir not set in config for testing.")
        test_image_input_dir = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_output_no_blur_unsub")
        os.makedirs(test_stage1_output_dir, exist_ok=True); os.makedirs(test_image_input_dir, exist_ok=True) 
        logger.info(f"Test Input Dir: {test_image_input_dir}"); logger.info(f"Test Output Dir: {test_stage1_output_dir}")
        if not os.listdir(test_image_input_dir) or len(os.listdir(test_image_input_dir)) < 5 : 
            if os.listdir(test_image_input_dir): shutil.rmtree(test_image_input_dir); os.makedirs(test_image_input_dir) 
            logger.info(f"Test image directory '{test_image_input_dir}' is empty/insufficient. Creating dummy PNG files...")
            date_str, plate_str, well_str = "20230101", "1", "1" ; num_frames_total = 5 
            for i in range(num_frames_total):
                dummy_fname = f"{date_str}_plate{plate_str}_well{well_str}_f{i:05d}_dpt0.0_fd0.0mm.png"
                dummy_fpath = os.path.join(test_image_input_dir, dummy_fname)
                dummy_img_content_uint8 = np.full((200, 200), 200, dtype=np.uint8) 
                cv2.circle(dummy_img_content_uint8, (50 + i * 20, 50 + i * 5), 15, 30, -1) 
                if i % 2 == 0: cv2.circle(dummy_img_content_uint8, (150,150), 10, 40, -1) 
                cv2.imwrite(dummy_fpath, dummy_img_content_uint8)
            logger.info(f"Created {num_frames_total} dummy 'collimated-like' images in {test_image_input_dir}")
        
        logger.info("\n--- Running S1 Test (NO BLUR for unsubtracted thresholding) ---")
        s1_cfg = cfg_main.setdefault('stage_01_initial_detection', {})
        s1_unsub_cfg = s1_cfg.setdefault('unsubtracted_particle_analysis', {})
        s1_unsub_cfg.setdefault('min_particle_area', 30)
        # s1_unsub_cfg.setdefault('blur_kernel_size',3) # This will be ignored by the updated _analyze_unsubtracted_fg_image

        s1_cfg.setdefault('particle_analysis', {'microorganism_threshold': 25, 'min_microorganism_area': 50, 'apply_convolution':False}) 
        s1_cfg.setdefault('particle_comparison', {'search_radius_factor': 2.0, 'default_particle_area': 100})
        s1_cfg.setdefault('well_masking', {'create_mask': True, 'radius_px':80, 'radius_fraction':0.9}) 
        
        result_csv_path = run_stage_01_initial_detection(image_dir_path=test_image_input_dir, output_dir_stage1=test_stage1_output_dir, sample_id=TEST_SAMPLE_ID)
        if result_csv_path: 
            logger.info(f"S1 test (no unsub blur) completed. CSV: {result_csv_path}")
            df_out = pd.read_csv(result_csv_path)
            logger.info(f"Output DataFrame head:\n{df_out.head()}")
            logger.info(f"Number of particles in output: {len(df_out)}")
            assert len(df_out) > 0, "Expected some fused particles for this test case."
        else: logger.error("S1 test (no unsub blur) FAILED.")
    except ConfigError as ce: logger.error(f"Configuration error during test: {ce}")
    except FileNotFoundError as fnfe: logger.error(f"File not found during test setup (check config paths and test data): {fnfe}")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 1 test: {e}")