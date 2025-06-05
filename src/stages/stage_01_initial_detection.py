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
import skimage.util 
import skimage.filters 

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image 
from src.core.image_enhancement import subtract_images, apply_blur 
# apply_threshold is no longer directly used here as logic is embedded
from src.core.masking_operations import mask_well_area, apply_mask 
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area

logger = logging.getLogger(__name__)


def _parse_image_filenames(image_dir_path: str) -> Optional[pd.DataFrame]:
    if not image_dir_path or not isinstance(image_dir_path, str):
        logger.error(f"Invalid image_dir_path: {image_dir_path}"); return None 
    if not os.path.isdir(image_dir_path):
        logger.error(f"Image directory not found: {image_dir_path}"); return None
    pattern = re.compile(
        r"^(?P<date>\d{8})_plate(?P<plate>\d+)_well(?P<well>\d+)_f(?P<frame>\d{5})_dpt(?P<dpt>[-+]?[0-9]*\.?[0-9]+)_fd(?P<fd_mm>[-+]?[0-9]*\.?[0-9]+)mm\.png$")
    parsed_file_data = []
    for filename in sorted(os.listdir(image_dir_path)): 
        if filename.lower().endswith(".png"):
            match = pattern.match(filename)
            if match:
                try:
                    groups = match.groupdict()
                    parsed_file_data.append({
                        "filename": filename, "date": groups["date"], "plate": int(groups["plate"]),
                        "well": int(groups["well"]), "frame": int(groups["frame"]),
                        "dpt": float(groups["dpt"]), "fd_mm": float(groups["fd_mm"]),
                        "full_path": os.path.join(image_dir_path, filename)})
                except ValueError as e: logger.warning(f"Parse error for '{filename}': {e}")
    if not parsed_file_data: logger.warning(f"No matching images in {image_dir_path}"); return pd.DataFrame() 
    df = pd.DataFrame(parsed_file_data).sort_values(by="frame").reset_index(drop=True)
    logger.info(f"Parsed {len(df)} images from {image_dir_path}.")
    return df

def _filter_image_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    initial_len = len(df)
    if df.iloc[0]["frame"] == 0:
        df = df.iloc[1:].reset_index(drop=True)
        logger.info(f"Removed frame 0. Rows: {initial_len} -> {len(df)}.")
    fd0_frames = df[df["fd_mm"] == 0]
    if not fd0_frames.empty:
        max_frame_fd0 = fd0_frames["frame"].max()
        df = df[df["frame"] <= max_frame_fd0].reset_index(drop=True)
        logger.info(f"Filtered after max_frame {max_frame_fd0} (fd_mm=0). Kept {len(df)}.")
    else: logger.warning("No images with fd_mm == 0 found for filtering rule.")
    return df

def _analyze_single_subtracted_image_global_thresh( 
    image_to_analyze_gray_uint8: np.ndarray, 
    config_stage1_particle_analysis: Dict[str, Any], 
    global_threshold_config: Dict[str, Any] 
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]: 
    
    particles_data = []
    binary_mask_display = None
    if image_to_analyze_gray_uint8 is None: return particles_data, binary_mask_display
    
    processed_img_uint8 = image_to_analyze_gray_uint8.copy()
    apply_conv = config_stage1_particle_analysis.get('apply_convolution', False)
    kernel_list = config_stage1_particle_analysis.get('convolution_kernel')
    if apply_conv and kernel_list:
        try:
            conv_kernel = np.array(kernel_list, dtype=np.float32)
            if conv_kernel.ndim == 2 and conv_kernel.shape[0] > 0:
                processed_img_uint8 = cv2.filter2D(processed_img_uint8, -1, conv_kernel)
        except Exception as e: logger.error(f"S1 Sub Conv Error: {e}")

    img_float_for_segment = skimage.util.img_as_float(processed_img_uint8)
    binary_mask_bool: Optional[np.ndarray] = None
    threshold_mode = global_threshold_config.get('mode', 'otsu')

    if threshold_mode == 'otsu':
        otsu_multiplier = global_threshold_config.get('otsu_threshold_multiplier', 0.45)
        if img_float_for_segment.min() == img_float_for_segment.max():
            binary_mask_bool = np.zeros_like(img_float_for_segment, dtype=bool)
        else:
            try:
                otsu_thresh_raw_sk = skimage.filters.threshold_otsu(img_float_for_segment)
                final_thresh = otsu_thresh_raw_sk * otsu_multiplier
                binary_mask_bool = img_float_for_segment > final_thresh
            except ValueError: binary_mask_bool = np.zeros_like(img_float_for_segment, dtype=bool)
    elif threshold_mode == 'fixed':
        fixed_thresh_val_int = global_threshold_config.get('fixed_threshold_value', 70)
        final_thresh = fixed_thresh_val_int / 255.0
        binary_mask_bool = img_float_for_segment > final_thresh
    else: return particles_data, binary_mask_display
    
    if binary_mask_bool is None: return particles_data, binary_mask_display
    binary_mask_uint8 = binary_mask_bool.astype(np.uint8) * 255
    binary_mask_display = binary_mask_uint8.copy() 

    img_for_contours = binary_mask_uint8
    blur_kernel_size_sub_binary = config_stage1_particle_analysis.get('microorganism_blur_kernel', 0)
    if blur_kernel_size_sub_binary > 0 and blur_kernel_size_sub_binary % 2 == 1:
        blurred_binary = apply_blur(binary_mask_uint8, blur_kernel_size_sub_binary, "median")
        if blurred_binary is not None: img_for_contours = blurred_binary
    
    contours = find_contours(img_for_contours, mode="external", method="simple")
    if not contours: return particles_data, binary_mask_display
    
    min_area = config_stage1_particle_analysis.get('min_microorganism_area', 50)
    valid_contours = filter_contours_by_area(contours, min_area)
    particles_data = [m for cnt in valid_contours if (m := calculate_contour_measurements(cnt))]
    return particles_data, binary_mask_display


def _analyze_unsubtracted_locally(
    full_unsub_prepped_uint8: np.ndarray, 
    candidate_particles_subtracted: List[Dict[str, Any]], 
    roi_crop_size: int,
    analysis_config: Dict[str, Any], 
    threshold_config: Dict[str, Any], 
    source_illumination: str
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]]:
    # Returns: list of particle_data dicts, composite_binary_mask_uint8, 
    #          (example_roi_img_2d_for_display, example_roi_hist_data_ravel, example_roi_otsu_thresh_raw)
    
    if full_unsub_prepped_uint8 is None: 
        return [], None, (None, None, None)
    
    all_local_particle_data = []
    composite_binary_mask_full_image = np.zeros_like(full_unsub_prepped_uint8, dtype=np.uint8)
    
    first_roi_img_2d_for_display = None 
    first_roi_hist_data_ravel = None 
    first_roi_otsu_thresh_raw = None 

    min_area_unsub = analysis_config.get('min_particle_area', 50)
    h_full, w_full = full_unsub_prepped_uint8.shape
    half_crop = roi_crop_size // 2

    logger.debug(f"Local Unsub: Processing {len(candidate_particles_subtracted)} candidate ROIs with crop_size {roi_crop_size}")

    for i, cand_p_sub in enumerate(candidate_particles_subtracted):
        if 'X' not in cand_p_sub or 'Y' not in cand_p_sub: continue
        
        cx, cy = int(cand_p_sub['X']), int(cand_p_sub['Y'])
        x1, y1 = max(0, cx - half_crop), max(0, cy - half_crop)
        x2, y2 = min(w_full, cx + half_crop), min(h_full, cy + half_crop)
        
        roi_img_uint8 = full_unsub_prepped_uint8[y1:y2, x1:x2]

        if roi_img_uint8.size == 0 or roi_img_uint8.shape[0] < 3 or roi_img_uint8.shape[1] < 3:
            continue

        roi_float_for_segment_2d = skimage.util.img_as_float(roi_img_uint8)
        if source_illumination == 'collimated':
            roi_float_for_segment_2d = 1.0 - roi_float_for_segment_2d
        
        current_roi_hist_data_ravel = roi_float_for_segment_2d.ravel()

        if i == 0: 
            first_roi_img_2d_for_display = roi_float_for_segment_2d.copy() 
            first_roi_hist_data_ravel = current_roi_hist_data_ravel

        local_binary_mask_roi_bool = None
        local_otsu_raw = None
        threshold_mode = threshold_config.get('mode', 'otsu')

        if threshold_mode == 'otsu':
            otsu_multiplier = threshold_config.get('otsu_threshold_multiplier', 0.45)
            if roi_float_for_segment_2d.min() == roi_float_for_segment_2d.max():
                local_binary_mask_roi_bool = np.zeros_like(roi_float_for_segment_2d, dtype=bool)
                local_otsu_raw = roi_float_for_segment_2d.min() if roi_float_for_segment_2d.size > 0 else 0
            else:
                try:
                    local_otsu_raw = skimage.filters.threshold_otsu(roi_float_for_segment_2d) 
                    local_final_thresh = local_otsu_raw * otsu_multiplier
                    local_binary_mask_roi_bool = roi_float_for_segment_2d > local_final_thresh
                except ValueError:
                    local_binary_mask_roi_bool = np.zeros_like(roi_float_for_segment_2d, dtype=bool)
            if i == 0: first_roi_otsu_thresh_raw = local_otsu_raw

        elif threshold_mode == 'fixed':
            fixed_val = threshold_config.get('fixed_threshold_value', 70)
            local_final_thresh = fixed_val / 255.0
            local_binary_mask_roi_bool = roi_float_for_segment_2d > local_final_thresh
            local_otsu_raw = local_final_thresh 
            if i == 0: first_roi_otsu_thresh_raw = local_otsu_raw
        else:
            logger.error(f"Error: Invalid local threshold mode '{threshold_mode}' for ROI {i+1}")
            continue
        
        if local_binary_mask_roi_bool is None: continue # Should not happen if logic above is correct
        local_binary_mask_roi_uint8 = local_binary_mask_roi_bool.astype(np.uint8) * 255
        
        composite_binary_mask_full_image[y1:y2, x1:x2] = np.maximum(
            composite_binary_mask_full_image[y1:y2, x1:x2], 
            local_binary_mask_roi_uint8 
        )

        contours_in_roi = find_contours(local_binary_mask_roi_uint8, mode="external", method="simple") # Use full find_contours
        if not contours_in_roi: continue
        
        valid_contours_in_roi = filter_contours_by_area(contours_in_roi, min_area_unsub)
        if not valid_contours_in_roi: continue

        roi_center_x, roi_center_y = roi_img_uint8.shape[1] // 2, roi_img_uint8.shape[0] // 2
        best_contour_in_roi = None; min_dist_to_roi_center_sq = float('inf')
        for contour_roi in valid_contours_in_roi:
            m_roi = cv2.moments(contour_roi)
            if m_roi["m00"] == 0: continue 
            contour_cx_roi = int(m_roi["m10"] / m_roi["m00"]); contour_cy_roi = int(m_roi["m01"] / m_roi["m00"])
            dist_sq = (contour_cx_roi - roi_center_x)**2 + (contour_cy_roi - roi_center_y)**2
            if dist_sq < min_dist_to_roi_center_sq:
                min_dist_to_roi_center_sq = dist_sq; best_contour_in_roi = contour_roi
        
        if best_contour_in_roi is not None:
            particle_measure_roi = calculate_contour_measurements(best_contour_in_roi) 
            if particle_measure_roi:
                particle_measure_roi['X'] += x1; particle_measure_roi['Y'] += y1
                particle_measure_roi['bounding_x'] += x1; particle_measure_roi['bounding_y'] += y1
                particle_measure_roi['local_threshold_value'] = local_otsu_raw if local_otsu_raw is not None else np.nan
                all_local_particle_data.append(particle_measure_roi)
    
    return all_local_particle_data, composite_binary_mask_full_image, \
           (first_roi_img_2d_for_display, first_roi_hist_data_ravel, first_roi_otsu_thresh_raw)


def _compare_and_fuse_particle_dfs(df_ref: pd.DataFrame, df_query: pd.DataFrame, config_s1_comparison: Dict[str, Any]) -> pd.DataFrame:
    if df_ref.empty or df_query.empty: return pd.DataFrame()
    if not all(k in df_ref.columns and k in df_query.columns for k in ['X', 'Y']):
        raise DataValidationError("X, Y columns missing for fusion.")
    
    sr_factor = config_s1_comparison.get('search_radius_factor', 2.0)
    dp_area = config_s1_comparison.get('default_particle_area', 500.0)
    search_radius = sr_factor * np.sqrt(dp_area / np.pi)
    
    df_query_cleaned = df_query.dropna(subset=['X', 'Y']).copy()
    if df_query_cleaned.empty: return pd.DataFrame()
    
    if 'original_query_idx' not in df_query_cleaned.columns:
        df_query_cleaned.loc[:, 'original_query_idx'] = range(len(df_query_cleaned))

    tree = BallTree(df_query_cleaned[['X', 'Y']].astype(float).values)
    matched_query_original_indices = set()

    for _, ref_row in df_ref.iterrows():
        ref_x, ref_y = ref_row['X'], ref_row['Y']
        if pd.isna(ref_x) or pd.isna(ref_y): continue
        
        indices_in_radius_tree = tree.query_radius([[float(ref_x), float(ref_y)]], r=search_radius)[0]
        
        best_query_tree_idx = -1
        min_dist_sq = float('inf')

        for tree_idx in indices_in_radius_tree:
            current_query_original_idx = df_query_cleaned.iloc[tree_idx]['original_query_idx']
            if current_query_original_idx not in matched_query_original_indices:
                query_row_for_dist = df_query_cleaned.iloc[tree_idx]
                dist_sq = (query_row_for_dist['X'] - ref_x)**2 + (query_row_for_dist['Y'] - ref_y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_query_tree_idx = tree_idx
        
        if best_query_tree_idx != -1:
            best_query_original_idx = df_query_cleaned.iloc[best_query_tree_idx]['original_query_idx']
            matched_query_original_indices.add(best_query_original_idx)

    if not matched_query_original_indices: return pd.DataFrame()
    
    final_matched_df = df_query[df_query.index.isin(matched_query_original_indices)].copy()

    if not df_ref.empty:
        final_matched_df.loc[:, 'SAMPLE_ID'] = df_ref['SAMPLE_ID'].iloc[0] if 'SAMPLE_ID' in df_ref.columns else "UNKNOWN_SAMPLE"
        final_matched_df.loc[:, 'source_image_filename'] = df_ref['source_image_filename'].iloc[0] if 'source_image_filename' in df_ref.columns else "UNKNOWN_FILE"
    
    logger.info(f"Fused {len(final_matched_df)} particles.")
    return final_matched_df


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str 
    ) -> Optional[str]:
    
    s1_unsub_debug_images_saved_this_run = False

    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} (Local Unsub Thresh) ---")
    config = get_config()
    config_s1 = config.get('stage_01_initial_detection', {})
    system_cfg = config.get('system', {})
    source_image_illumination = system_cfg.get('source_image_illumination_type', 'darkfield')
    global_threshold_config = system_cfg.get('thresholding', {})
    
    particle_analysis_subtracted_config = config_s1.get('particle_analysis', {})
    unsubtracted_analysis_config = config_s1.get('unsubtracted_particle_analysis', {}) 
    local_otsu_roi_size = unsubtracted_analysis_config.get('local_otsu_roi_size', 150)

    particle_comparison_config = config_s1.get('particle_comparison', {})
    bg_offset = config_s1.get('bg_offset_val', 110)
    mask_cfg = config_s1.get('well_masking', {})
    
    images_df = _parse_image_filenames(image_dir_path)
    if images_df is None or images_df.empty: return None
    images_df_filtered = _filter_image_df(images_df.copy())
    if images_df_filtered.empty: return None
    
    all_fused_particles_for_sample_list: List[pd.DataFrame] = []

    for idx_frame_list, fg_row in images_df_filtered.iterrows():
        bg_index_in_list = (idx_frame_list + bg_offset) % len(images_df_filtered)
        bg_row = images_df_filtered.iloc[bg_index_in_list]
        
        fg_filepath = fg_row['full_path']; bg_filepath = bg_row['full_path']
        logger.debug(f"Processing S1 Frame: {fg_row['filename']} (BG: {bg_row['filename']})")

        img_a_gray_orig_uint8 = load_image(fg_filepath, mode='gray')
        img_b_gray_orig_uint8 = load_image(bg_filepath, mode='gray')
        if img_a_gray_orig_uint8 is None or img_b_gray_orig_uint8 is None: continue
            
        well_mask_uint8 = None
        if mask_cfg.get('create_mask', False):
            _, well_mask_uint8 = mask_well_area(img_a_gray_orig_uint8.copy(), mask_cfg, system_cfg)

        img_a_for_sub = img_a_gray_orig_uint8.copy(); img_b_for_sub = img_b_gray_orig_uint8.copy()
        if source_image_illumination == 'collimated': 
             img_a_for_sub = cv2.bitwise_not(img_a_for_sub); img_b_for_sub = cv2.bitwise_not(img_b_for_sub)
        subtracted_img_uint8 = subtract_images(img_a_for_sub, img_b_for_sub)
        if subtracted_img_uint8 is None: continue
        img_for_sub_analysis_uint8 = apply_mask(subtracted_img_uint8, well_mask_uint8) if well_mask_uint8 is not None else subtracted_img_uint8
        
        particles_subtracted_list_dicts, _ = _analyze_single_subtracted_image_global_thresh(
            img_for_sub_analysis_uint8, 
            particle_analysis_subtracted_config, 
            global_threshold_config
        )
        df_subtracted_particles = pd.DataFrame(particles_subtracted_list_dicts)
        if not df_subtracted_particles.empty:
            df_subtracted_particles['SAMPLE_ID'] = sample_id
            df_subtracted_particles['source_image_filename'] = fg_row['filename']

        img_for_unsub_analysis_prepared_uint8 = img_a_gray_orig_uint8.copy()
        if well_mask_uint8 is not None:
            if source_image_illumination == 'collimated':
                pixels_inside_well = img_a_gray_orig_uint8[well_mask_uint8 == 255]
                if pixels_inside_well.size > 0:
                    hist = cv2.calcHist([img_a_gray_orig_uint8], [0], well_mask_uint8, [256], [0,256])
                    dominant_bg_val = np.argmax(hist)
                    filled_outside = np.full_like(img_a_gray_orig_uint8, dominant_bg_val)
                    filled_outside[well_mask_uint8 == 255] = img_a_gray_orig_uint8[well_mask_uint8 == 255]
                    img_for_unsub_analysis_prepared_uint8 = filled_outside
                else: 
                    img_for_unsub_analysis_prepared_uint8 = apply_mask(img_a_gray_orig_uint8.copy(), well_mask_uint8) if well_mask_uint8 is not None else img_a_gray_orig_uint8.copy()
            else: 
                img_for_unsub_analysis_prepared_uint8 = apply_mask(img_a_gray_orig_uint8.copy(), well_mask_uint8) if well_mask_uint8 is not None else img_a_gray_orig_uint8.copy()
        
        if particles_subtracted_list_dicts: 
            particles_unsubtracted_list_dicts, _, _ = _analyze_unsubtracted_locally(
                img_for_unsub_analysis_prepared_uint8,
                particles_subtracted_list_dicts, 
                local_otsu_roi_size,
                unsubtracted_analysis_config,
                global_threshold_config,
                source_image_illumination
            )
            df_unsubtracted_particles = pd.DataFrame(particles_unsubtracted_list_dicts)
        else:
            df_unsubtracted_particles = pd.DataFrame() 
            logger.debug(f"No subtracted particles for {fg_row['filename']}, skipping local unsubtracted analysis.")

        fused_df_frame = _compare_and_fuse_particle_dfs(df_subtracted_particles, df_unsubtracted_particles, particle_comparison_config)
        
        if not fused_df_frame.empty:
            all_fused_particles_for_sample_list.append(fused_df_frame)
    
    if not all_fused_particles_for_sample_list:
        logger.warning(f"No particles detected or fused for the entire sample {sample_id}.")
        expected_cols = ['SAMPLE_ID', 'source_image_filename', 'X', 'Y', 'area', 
                         'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h', 
                         'local_threshold_value'] 
        empty_df = pd.DataFrame(columns=expected_cols)
        output_filename = f"{sample_id}_stage1_particles.csv"
        return save_dataframe_to_csv(empty_df, output_dir_stage1, output_filename)

    final_sample_df = pd.concat(all_fused_particles_for_sample_list, ignore_index=True)
    logger.info(f"Total fused particles for sample {sample_id}: {len(final_sample_df)}")

    cols_order = ['SAMPLE_ID', 'source_image_filename', 'X', 'Y', 'area', 
                  'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h',
                  'local_threshold_value'] 
    other_measurement_cols = [col for col in final_sample_df.columns if col not in cols_order and col not in ['original_query_idx']]
    final_cols_ordered = cols_order + other_measurement_cols
    
    for col in cols_order: # Ensure all primary columns exist
        if col not in final_sample_df.columns:
            final_sample_df[col] = np.nan
            
    final_sample_df = final_sample_df[final_cols_ordered]

    output_filename = f"{sample_id}_stage1_particles.csv"
    saved_csv_path = save_dataframe_to_csv(final_sample_df, output_dir_stage1, output_filename)
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
        thresh_cfg_test.setdefault('otsu_threshold_multiplier', 1.0) 
        thresh_cfg_test.setdefault('fixed_threshold_value', 70)
        
        TEST_SAMPLE_ID = "test_s1_local_otsu_final" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir: raise ValueError("project_base_dir not set in config for testing.")
        test_image_input_dir = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_local_otsu_final_test")
        os.makedirs(test_stage1_output_dir, exist_ok=True); os.makedirs(test_image_input_dir, exist_ok=True) 
        
        if not os.listdir(test_image_input_dir) or len(os.listdir(test_image_input_dir)) < 200 : 
            if os.listdir(test_image_input_dir): shutil.rmtree(test_image_input_dir); os.makedirs(test_image_input_dir) 
            logger.info(f"Creating dummy images for {TEST_SAMPLE_ID}...")
            date_str, plate_str, well_str = "20230101", "1", "1" ; num_frames_total = 200 
            for i in range(num_frames_total):
                dummy_fname = f"{date_str}_plate{plate_str}_well{well_str}_f{i:05d}_dpt0.0_fd0.0mm.png"
                dummy_fpath = os.path.join(test_image_input_dir, dummy_fname)
                dummy_img_content_uint8 = np.full((300, 300), 200, dtype=np.uint8) 
                cv2.circle(dummy_img_content_uint8, (70 + i//10, 70 + i//10), 10, 50, -1) 
                if i % 10 == 0: cv2.ellipse(dummy_img_content_uint8, (200,150), (15,8), angle=i*2, startAngle=0, endAngle=360, color=60, thickness=-1)
                cv2.imwrite(dummy_fpath, dummy_img_content_uint8)
            logger.info(f"Created dummy images.")
        
        logger.info("\n--- Running S1 Test (Local Otsu for Unsubtracted) ---")
        s1_cfg = cfg_main.setdefault('stage_01_initial_detection', {})
        s1_unsub_cfg = s1_cfg.setdefault('unsubtracted_particle_analysis', {})
        s1_unsub_cfg.setdefault('min_particle_area', 30)
        s1_unsub_cfg.setdefault('local_otsu_roi_size', 100) 

        s1_particle_cfg = s1_cfg.setdefault('particle_analysis', {'min_microorganism_area': 50, 'apply_convolution':False}) 
        s1_cfg.setdefault('particle_comparison', {'search_radius_factor': 2.0, 'default_particle_area': 100})
        s1_cfg.setdefault('well_masking', {'create_mask': True, 'radius_px':120, 'radius_fraction':0.9}) 
        
        result_csv_path = run_stage_01_initial_detection(image_dir_path=test_image_input_dir, output_dir_stage1=test_stage1_output_dir, sample_id=TEST_SAMPLE_ID)
        if result_csv_path: 
            logger.info(f"S1 local unsub test completed. CSV: {result_csv_path}")
            df_out = pd.read_csv(result_csv_path)
            logger.info(f"Output DataFrame head:\n{df_out.head()}")
            logger.info(f"Number of particles in output: {len(df_out)}")
            if not df_out.empty:
                assert 'local_threshold_value' in df_out.columns, "local_threshold_value column missing!"
                logger.info(f"Example local thresholds: {df_out['local_threshold_value'].unique()[:5]}")
        else: logger.error("S1 local unsub test FAILED.")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 1 test: {e}")