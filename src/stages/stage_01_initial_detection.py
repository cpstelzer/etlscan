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
import trackpy as tp
import matplotlib.pyplot as plt

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image 
from src.core.image_enhancement import subtract_images, apply_blur 
from src.core.masking_operations import mask_well_area, apply_mask # mask_well_area is the general circular one
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area
from src.core.segmentation import generate_binary_mask_from_float_image 
# MODIFIED: Import new well analysis functions
from src.core.well_analysis import (
    create_well_mask_black_plate, 
    calculate_well_brightness_peak, # Not directly used by S1 runner, but by calc_avg and match
    calc_avg_well_brightness_from_dir,
    match_well_brightness
)

logger = logging.getLogger(__name__)


def _parse_image_filenames(image_dir_path: str) -> Optional[pd.DataFrame]: # Unchanged
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

def _filter_image_df(df: pd.DataFrame) -> pd.DataFrame: # Unchanged
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

def _analyze_single_subtracted_image_global_thresh( # Unchanged
    image_to_analyze_gray_uint8: np.ndarray, 
    config_stage1_particle_analysis: Dict[str, Any], 
    global_threshold_config: Dict[str, Any] 
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]: 
    particles_data = []; binary_mask_display_uint8 = None 
    if image_to_analyze_gray_uint8 is None: return particles_data, binary_mask_display_uint8
    processed_img_uint8 = image_to_analyze_gray_uint8.copy()
    apply_conv = config_stage1_particle_analysis.get('apply_convolution', False)
    kernel_list = config_stage1_particle_analysis.get('convolution_kernel')
    if apply_conv and kernel_list:
        try:
            conv_kernel = np.array(kernel_list, dtype=np.float32)
            if conv_kernel.ndim == 2 and conv_kernel.shape[0] > 0: processed_img_uint8 = cv2.filter2D(processed_img_uint8, -1, conv_kernel); logger.debug("S1 Subtracted: Applied convolution.")
        except Exception as e: logger.error(f"S1 Sub Conv Error: {e}")
    img_float_for_segment = skimage.util.img_as_float(processed_img_uint8)
    binary_mask_bool = generate_binary_mask_from_float_image(img_float_for_segment, global_threshold_config)
    if binary_mask_bool is None: logger.error("S1 Subtracted: Failed to generate binary mask via helper."); return particles_data, binary_mask_display_uint8
    binary_mask_uint8 = binary_mask_bool.astype(np.uint8) * 255; binary_mask_display_uint8 = binary_mask_uint8.copy() 
    img_for_contours = binary_mask_uint8
    blur_kernel_size_sub_binary = config_stage1_particle_analysis.get('microorganism_blur_kernel', 0)
    if blur_kernel_size_sub_binary > 0 and blur_kernel_size_sub_binary % 2 == 1:
        blurred_binary = apply_blur(binary_mask_uint8, blur_kernel_size_sub_binary, "median")
        if blurred_binary is not None: img_for_contours = blurred_binary; logger.debug(f"S1 Subtracted: Applied blur (kernel {blur_kernel_size_sub_binary}) to binary mask.")
    contours = find_contours(img_for_contours, mode="external", method="simple")
    if not contours: logger.debug("S1 Subtracted: No contours found after thresholding/blur."); return particles_data, binary_mask_display_uint8
    min_area = config_stage1_particle_analysis.get('min_microorganism_area', 50)
    valid_contours = filter_contours_by_area(contours, min_area)
    particles_data = [m for cnt in valid_contours if (m := calculate_contour_measurements(cnt))]
    logger.debug(f"S1 Subtracted: Found {len(particles_data)} particles after area filter.")
    return particles_data, binary_mask_display_uint8

def _analyze_unsubtracted_locally( # Unchanged
    full_unsub_prepped_uint8: np.ndarray, 
    candidate_particles_subtracted: List[Dict[str, Any]], 
    roi_crop_size: int,
    analysis_config: Dict[str, Any], 
    global_threshold_config: Dict[str, Any], 
    source_illumination: str
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]]:
    if full_unsub_prepped_uint8 is None: return [], None, (None, None, None)
    all_local_particle_data = []; composite_binary_mask_full_image = np.zeros_like(full_unsub_prepped_uint8, dtype=np.uint8)
    first_roi_img_2d_for_display, first_roi_hist_data_ravel, first_roi_effective_thresh_val_float = None, None, None
    min_area_unsub = analysis_config.get('min_particle_area', 50); h_full, w_full = full_unsub_prepped_uint8.shape; half_crop = roi_crop_size // 2
    logger.debug(f"Local Unsub: Processing {len(candidate_particles_subtracted)} candidate ROIs with crop_size {roi_crop_size}")
    for i, cand_p_sub in enumerate(candidate_particles_subtracted):
        if 'X' not in cand_p_sub or 'Y' not in cand_p_sub: continue
        cx, cy = int(cand_p_sub['X']), int(cand_p_sub['Y']); x1, y1 = max(0, cx - half_crop), max(0, cy - half_crop); x2, y2 = min(w_full, cx + half_crop), min(h_full, cy + half_crop)
        roi_img_uint8 = full_unsub_prepped_uint8[y1:y2, x1:x2]
        if roi_img_uint8.size == 0 or roi_img_uint8.shape[0] < 3 or roi_img_uint8.shape[1] < 3: continue
        roi_float_original_2d = skimage.util.img_as_float(roi_img_uint8); roi_float_for_segment_2d = roi_float_original_2d.copy()
        if source_illumination == 'collimated': roi_float_for_segment_2d = 1.0 - roi_float_for_segment_2d
        current_roi_hist_data_ravel = roi_float_for_segment_2d.ravel()
        local_binary_mask_roi_bool = generate_binary_mask_from_float_image(roi_float_for_segment_2d, global_threshold_config)
        if local_binary_mask_roi_bool is None: logger.error(f"S1 Local Unsub: Failed to generate binary mask for ROI {i+1}."); continue
        local_threshold_value_for_data = np.nan
        if i == 0:
            first_roi_img_2d_for_display, first_roi_hist_data_ravel = roi_float_for_segment_2d.copy(), current_roi_hist_data_ravel
            if global_threshold_config.get('mode') == 'otsu':
                if roi_float_for_segment_2d.min() != roi_float_for_segment_2d.max():
                    try: otsu_val_disp, otsu_mult_disp = skimage.filters.threshold_otsu(roi_float_for_segment_2d), global_threshold_config.get('otsu_threshold_multiplier', 0.45); first_roi_effective_thresh_val_float, local_threshold_value_for_data = otsu_val_disp * otsu_mult_disp, otsu_val_disp * otsu_mult_disp
                    except ValueError: pass
            elif global_threshold_config.get('mode') == 'fixed': first_roi_effective_thresh_val_float, local_threshold_value_for_data = global_threshold_config.get('fixed_threshold_value', 70) / 255.0, global_threshold_config.get('fixed_threshold_value', 70) / 255.0
        else:
            if global_threshold_config.get('mode') == 'otsu':
                 if roi_float_for_segment_2d.min() != roi_float_for_segment_2d.max():
                    try: otsu_val_data, otsu_mult_data = skimage.filters.threshold_otsu(roi_float_for_segment_2d), global_threshold_config.get('otsu_threshold_multiplier', 0.45); local_threshold_value_for_data = otsu_val_data * otsu_mult_data
                    except ValueError: pass
            elif global_threshold_config.get('mode') == 'fixed': local_threshold_value_for_data = global_threshold_config.get('fixed_threshold_value', 70) / 255.0
        local_binary_mask_roi_uint8 = local_binary_mask_roi_bool.astype(np.uint8) * 255
        composite_binary_mask_full_image[y1:y2, x1:x2] = np.maximum(composite_binary_mask_full_image[y1:y2, x1:x2], local_binary_mask_roi_uint8)
        contours_in_roi = find_contours(local_binary_mask_roi_uint8, mode="external", method="simple")
        if not contours_in_roi: continue
        valid_contours_in_roi = filter_contours_by_area(contours_in_roi, min_area_unsub)
        if not valid_contours_in_roi: continue
        roi_center_x, roi_center_y = roi_img_uint8.shape[1] // 2, roi_img_uint8.shape[0] // 2
        best_contour_in_roi, min_dist_to_roi_center_sq = None, float('inf')
        for contour_roi in valid_contours_in_roi:
            m_roi = cv2.moments(contour_roi)
            if m_roi["m00"] == 0: continue 
            contour_cx_roi, contour_cy_roi = int(m_roi["m10"] / m_roi["m00"]), int(m_roi["m01"] / m_roi["m00"]); dist_sq = (contour_cx_roi - roi_center_x)**2 + (contour_cy_roi - roi_center_y)**2
            if dist_sq < min_dist_to_roi_center_sq: min_dist_to_roi_center_sq, best_contour_in_roi = dist_sq, contour_roi
        if best_contour_in_roi is not None:
            particle_measure_roi = calculate_contour_measurements(best_contour_in_roi) 
            if particle_measure_roi: particle_measure_roi['X'] += x1; particle_measure_roi['Y'] += y1; particle_measure_roi['bounding_x'] += x1; particle_measure_roi['bounding_y'] += y1; particle_measure_roi['local_threshold_value'] = local_threshold_value_for_data; all_local_particle_data.append(particle_measure_roi)
    return all_local_particle_data, composite_binary_mask_full_image, (first_roi_img_2d_for_display, first_roi_hist_data_ravel, first_roi_effective_thresh_val_float)

def _compare_and_fuse_particle_dfs(df_ref: pd.DataFrame, df_query: pd.DataFrame, config_s1_comparison: Dict[str, Any]) -> pd.DataFrame: # Unchanged
    if df_ref.empty or df_query.empty: return pd.DataFrame()
    if not all(k in df_ref.columns and k in df_query.columns for k in ['X', 'Y']): raise DataValidationError("X, Y columns missing for fusion.")
    sr_factor, dp_area = config_s1_comparison.get('search_radius_factor', 2.0), config_s1_comparison.get('default_particle_area', 500.0); search_radius = sr_factor * np.sqrt(dp_area / np.pi)
    df_query_cleaned = df_query.dropna(subset=['X', 'Y']).copy()
    if df_query_cleaned.empty: return pd.DataFrame()
    if 'original_query_idx' not in df_query_cleaned.columns: df_query_cleaned.loc[:, 'original_query_idx'] = range(len(df_query_cleaned))
    tree = BallTree(df_query_cleaned[['X', 'Y']].astype(float).values); matched_query_original_indices = set()
    for _, ref_row in df_ref.iterrows():
        ref_x, ref_y = ref_row['X'], ref_row['Y']
        if pd.isna(ref_x) or pd.isna(ref_y): continue
        indices_in_radius_tree = tree.query_radius([[float(ref_x), float(ref_y)]], r=search_radius)[0]
        best_query_tree_idx, min_dist_sq = -1, float('inf')
        for tree_idx in indices_in_radius_tree:
            current_query_original_idx = df_query_cleaned.iloc[tree_idx]['original_query_idx']
            if current_query_original_idx not in matched_query_original_indices:
                query_row_for_dist = df_query_cleaned.iloc[tree_idx]; dist_sq = (query_row_for_dist['X'] - ref_x)**2 + (query_row_for_dist['Y'] - ref_y)**2
                if dist_sq < min_dist_sq: min_dist_sq, best_query_tree_idx = dist_sq, tree_idx
        if best_query_tree_idx != -1: matched_query_original_indices.add(df_query_cleaned.iloc[best_query_tree_idx]['original_query_idx'])
    if not matched_query_original_indices: return pd.DataFrame()
    final_matched_df = df_query[df_query.index.isin(matched_query_original_indices)].copy()
    if not df_ref.empty:
        final_matched_df.loc[:, 'SAMPLE_ID'] = df_ref['SAMPLE_ID'].iloc[0] if 'SAMPLE_ID' in df_ref.columns else "UNKNOWN_SAMPLE"
        final_matched_df.loc[:, 'source_image_filename'] = df_ref['source_image_filename'].iloc[0] if 'source_image_filename' in df_ref.columns else "UNKNOWN_FILE"
    logger.info(f"Fused {len(final_matched_df)} particles.")
    return final_matched_df


def _process_particle_coordinates_for_tracking(input_csv_path, pixel_size_um, magnification):
    """
    Loads raw particle data, converts coordinates, and extracts metadata from filenames.
    """
    if not os.path.exists(input_csv_path):
        logger.error(f"Error: Input file not found at '{input_csv_path}'")
        return pd.DataFrame()

    logger.info(f"Tracking Step 1: Loading and processing data from: {os.path.basename(input_csv_path)}")
    df = pd.read_csv(input_csv_path)

    if df.empty:
        logger.warning("Input CSV for tracking is empty. No particles to process.")
        return pd.DataFrame()

    pixel_to_mm_factor = pixel_size_um / (magnification * 1000)
    logger.info(f"   - Using Pixel-to-MM factor: {pixel_to_mm_factor:.6f}")
    df['x_mm'] = df['X'] * pixel_to_mm_factor
    df['y_mm'] = df['Y'] * pixel_to_mm_factor

    def extract_metadata(filename):
        frame_match = re.search(r'_f(\d+)', filename)
        z_match = re.search(r'_fd([-\d.]+?)mm', filename)
        dpt_match = re.search(r'_dpt([-\d.]+?)_', filename)
        frame = int(frame_match.group(1)) if frame_match else -1
        z_mm = float(z_match.group(1)) if z_match else np.nan
        dioptries = float(dpt_match.group(1)) if dpt_match else np.nan
        return frame, z_mm, dioptries

    logger.info("   - Extracting frame, Z coordinate (mm), and dioptries from filenames...")
    df[['frame', 'z_mm', 'dioptries']] = df['source_image_filename'].apply(
        lambda fname: pd.Series(extract_metadata(fname))
    )

    initial_rows = len(df)
    df.dropna(subset=['z_mm', 'dioptries'], inplace=True)
    df = df[df['frame'] != -1].copy()
    final_rows = len(df)

    if final_rows < initial_rows:
        logger.warning(f"   - Dropped {initial_rows - final_rows} rows due to failed metadata extraction.")

    key_cols = ['frame', 'x_mm', 'y_mm', 'z_mm', 'dioptries']
    other_cols = [col for col in df.columns if col not in key_cols]
    df = df[key_cols + other_cols]
    return df

def _track_particles_3d(processed_df, search_range_mm, memory_frames, min_track_length):
    """
    Performs 3D particle tracking.
    """
    if processed_df.empty:
        logger.warning("Input DataFrame for tracking is empty. Skipping tracking.")
        return pd.DataFrame()

    logger.info("Tracking Step 2: Performing 3D tracking with trackpy...")
    tp_df = processed_df.rename(columns={'x_mm': 'x', 'y_mm': 'y', 'z_mm': 'z'})
    
    tp_df['frame'] = tp_df['frame'].astype(int)
    pos_cols = ['x', 'y', 'z']
    for col in pos_cols:
        tp_df[col] = pd.to_numeric(tp_df[col], errors='coerce')
    tp_df.dropna(subset=pos_cols, inplace=True)

    trajectories = tp.link(
        tp_df,
        search_range=search_range_mm,
        memory=memory_frames,
        pos_columns=pos_cols,
        t_column='frame'
    )
    num_tracks_raw = trajectories['particle'].nunique()
    logger.info(f"   - trackpy identified {num_tracks_raw} raw tracks.")

    logger.info(f"   - Filtering out tracks shorter than {min_track_length} frames...")
    filtered_trajectories = tp.filter_stubs(trajectories, min_track_length)
    num_tracks_final = filtered_trajectories['particle'].nunique()
    logger.info(f"   - Kept {num_tracks_final} tracks after filtering.")

    filtered_trajectories.rename(columns={'particle': 'tracked_particle'}, inplace=True)
    filtered_trajectories = filtered_trajectories.reset_index(drop=True)
    return filtered_trajectories

def _run_particle_tracking(
    stage1_csv_path: str,
    output_dir: str,
    tracking_config: Dict[str, Any]
    ):
    """
    Orchestrates the particle tracking process based on the output of Stage 1.
    """
    logger.info(f"--- Post-S1 Tracking initiated for: {os.path.basename(stage1_csv_path)} ---")
    
    # --- Configuration ---
    pixel_size_um = tracking_config.get('pixel_size_um')
    magnification = tracking_config.get('magnification')
    search_range_mm = tracking_config.get('search_range_mm')
    memory_frames = tracking_config.get('memory_frames')
    min_track_length = tracking_config.get('min_track_length')
    
    if not all([pixel_size_um, magnification, search_range_mm, memory_frames, min_track_length]):
        raise ConfigError("One or more required tracking parameters are missing from the configuration.")

    base_filename = os.path.splitext(os.path.basename(stage1_csv_path))[0]
    output_csv_all = os.path.join(output_dir, f"{base_filename}_with_tracking_all.csv")
    output_csv_tracked = os.path.join(output_dir, f"{base_filename}_with_tracking_only.csv")
    figure_3d_path = os.path.join(output_dir, f"{base_filename}_tracks_3d.png")
    figure_2d_path = os.path.join(output_dir, f"{base_filename}_tracks_2d.png")
    
    # Process coordinates and track particles
    processed_data = _process_particle_coordinates_for_tracking(stage1_csv_path, pixel_size_um, magnification)
    if processed_data.empty:
        logger.warning("Tracking workflow stopped: No data after initial processing.")
        return

    tracked_data = _track_particles_3d(processed_data, search_range_mm, memory_frames, min_track_length)
    if tracked_data.empty:
        logger.warning("Tracking workflow stopped: No tracks survived the filtering process.")
        return

    # Merge to create full output with tracked_particle ID
    logger.info("Tracking Step 3: Merging tracking results with original data...")
    merge_cols = ['frame','x_mm','y_mm','z_mm']
    tracked_cols = ['frame','x','y','z','tracked_particle']
    
    processed_rounded = processed_data.copy()
    processed_rounded[merge_cols[1:]] = processed_rounded[merge_cols[1:]].round(6)
    tracked_rounded = tracked_data.copy()
    tracked_rounded[['x','y','z']] = tracked_rounded[['x','y','z']].round(6)

    merged_df = pd.merge(
        processed_rounded,
        tracked_rounded[tracked_cols],
        how='left',
        left_on=merge_cols,
        right_on=['frame','x','y','z']
    )
    merged_df.drop(columns=['x','y','z'], inplace=True, errors='ignore')
    merged_df['tracked_particle'] = merged_df['tracked_particle'].fillna(0).astype(int)

    save_dataframe_to_csv(merged_df, output_dir, os.path.basename(output_csv_all))
    save_dataframe_to_csv(merged_df[merged_df['tracked_particle'] != 0], output_dir, os.path.basename(output_csv_tracked))

    logger.info("Tracking Step 4: Visualizing and saving tracks...")
    base_cmap = plt.cm.get_cmap('tab20')
    unique_particles = sorted(tracked_data['tracked_particle'].unique())
    colors = {pid: base_cmap(i % 20) for i, pid in enumerate(unique_particles)}
    
    # Visualize 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Particle Tracks")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    for particle_no, track in tracked_data.groupby('tracked_particle'):
        ax.plot(track['x'], track['y'], track['z'], color=colors[particle_no], linewidth=1)
    plt.tight_layout()
    plt.savefig(figure_3d_path, dpi=150)
    plt.close(fig)
    logger.info(f"   - Saved 3D figure to: {figure_3d_path}")

    # Visualize 2D
    fig2d, ax2d = plt.subplots(figsize=(10, 8))
    ax2d.set_title("2D Particle Tracks (X vs Y)")
    ax2d.set_xlabel("X (mm)"); ax2d.set_ylabel("Y (mm)")
    for particle_no, track in tracked_data.groupby('tracked_particle'):
        ax2d.plot(track['x'], track['y'], color=colors[particle_no], linewidth=1)
    plt.tight_layout()
    plt.savefig(figure_2d_path, dpi=150)
    plt.close(fig2d)
    logger.info(f"   - Saved 2D figure to: {figure_2d_path}")
    logger.info(f"--- Post-S1 Tracking for {os.path.basename(stage1_csv_path)} COMPLETED ---")


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str 
    ) -> Optional[str]:
    
    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} (Brightness Adjustment Enabled) ---")
    config = get_config()
    config_s1 = config.get('stage_01_initial_detection', {})
    system_cfg = config.get('system', {})
    source_image_illumination = system_cfg.get('source_image_illumination_type', 'darkfield')
    global_threshold_config = system_cfg.get('thresholding', {}) 
    if not global_threshold_config:
        raise ConfigError("Stage 1: System-level thresholding configuration not found or empty.")
    
    # MODIFIED: Get black plate well analysis config
    config_black_plate_well = config_s1.get('black_plate_well_analysis', {})
    enable_brightness_adjustment = config_black_plate_well.get('enable_brightness_adjustment', False)
    logger.info(f"Stage 1: Brightness adjustment for BG subtraction is {'ENABLED' if enable_brightness_adjustment else 'DISABLED'}.")

    particle_analysis_subtracted_config = config_s1.get('particle_analysis', {})
    unsubtracted_analysis_config = config_s1.get('unsubtracted_particle_analysis', {}) 
    local_otsu_roi_size = unsubtracted_analysis_config.get('local_otsu_roi_size', 150)
    particle_comparison_config = config_s1.get('particle_comparison', {})
    bg_offset = config_s1.get('bg_offset_val', 110)
    
    general_well_masking_cfg = config_s1.get('well_masking', {})
    
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
        if img_a_gray_orig_uint8 is None or img_b_gray_orig_uint8 is None: 
            logger.warning(f"Could not load FG ({fg_filepath}) or BG ({bg_filepath}). Skipping pair.")
            continue
        
        # MODIFIED: Apply brightness matching if enabled
        img_a_for_sub = img_a_gray_orig_uint8.copy() 
        img_b_for_sub_final = img_b_gray_orig_uint8.copy()

        if enable_brightness_adjustment:
            logger.debug(f"Applying brightness matching for FG: {fg_row['filename']}, BG: {bg_row['filename']}")
            _, img_b_for_sub_final = match_well_brightness(
                img_a_gray_orig_uint8,
                img_b_gray_orig_uint8,
                config_black_plate_well
            )
        
        general_well_mask_for_roi_uint8 = None
        if general_well_masking_cfg.get('create_mask', False):
            _, general_well_mask_for_roi_uint8 = mask_well_area(
                img_a_gray_orig_uint8.copy(), 
                general_well_masking_cfg, 
                system_cfg
            )
            
        img_a_inv_ready = img_a_for_sub.copy()
        img_b_inv_ready = img_b_for_sub_final.copy()

        if source_image_illumination == 'collimated': 
             img_a_inv_ready = cv2.bitwise_not(img_a_inv_ready)
             img_b_inv_ready = cv2.bitwise_not(img_b_inv_ready)
        
        subtracted_img_uint8 = subtract_images(img_a_inv_ready, img_b_inv_ready)
        if subtracted_img_uint8 is None: 
            logger.warning(f"Subtraction resulted in None for FG: {fg_row['filename']}. Skipping pair.")
            continue
        
        img_for_sub_analysis_uint8 = apply_mask(subtracted_img_uint8, general_well_mask_for_roi_uint8) \
                                     if general_well_mask_for_roi_uint8 is not None else subtracted_img_uint8
        
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
        if general_well_mask_for_roi_uint8 is not None:
            if source_image_illumination == 'collimated':
                pixels_inside_well = img_a_gray_orig_uint8[general_well_mask_for_roi_uint8 == 255]
                if pixels_inside_well.size > 0:
                    hist_unsub = cv2.calcHist([img_a_gray_orig_uint8], [0], general_well_mask_for_roi_uint8, [256], [0,256])
                    dominant_bg_val = np.argmax(hist_unsub)
                    
                    filled_outside = np.full_like(img_a_gray_orig_uint8, dominant_bg_val)
                    filled_outside[general_well_mask_for_roi_uint8 == 255] = img_a_gray_orig_uint8[general_well_mask_for_roi_uint8 == 255]
                    img_for_unsub_analysis_prepared_uint8 = filled_outside
                else:
                    img_for_unsub_analysis_prepared_uint8 = apply_mask(img_a_gray_orig_uint8.copy(), general_well_mask_for_roi_uint8)
            else:
                img_for_unsub_analysis_prepared_uint8 = apply_mask(img_a_gray_orig_uint8.copy(), general_well_mask_for_roi_uint8)
        
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
    
    for col in cols_order: 
        if col not in final_sample_df.columns:
            final_sample_df[col] = np.nan
            
    final_sample_df = final_sample_df[final_cols_ordered]

    output_filename = f"{sample_id}_stage1_particles.csv"
    saved_csv_path = save_dataframe_to_csv(final_sample_df, output_dir_stage1, output_filename)
    
    if saved_csv_path: 
        logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} COMPLETED ---")
        
        # --- NEW: Run tracking if enabled ---
        tracking_config = config_s1.get('tracking', {})
        if tracking_config.get('enabled', False):
            try:
                _run_particle_tracking(saved_csv_path, output_dir_stage1, tracking_config)
            except Exception as e:
                logger.exception(f"Particle tracking failed for sample {sample_id}: {e}")
        else:
            logger.info("Particle tracking is disabled in the configuration. Skipping.")
            
    else: 
        logger.error(f"--- Stage 1: Failed to save results for Sample ID: {sample_id} ---")
        
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
        
        s1_cfg = cfg_main.setdefault('stage_01_initial_detection', {})
        # Ensure black_plate_well_analysis config exists for testing
        bpwa_cfg = s1_cfg.setdefault('black_plate_well_analysis', {
            'enable_brightness_adjustment': True,
            'well_identification': {'low_threshold_value': 10, 'min_expected_diameter_px': 150, 'max_expected_diameter_px': 4000, 'min_circularity': 0.80},
            'brightness_peak_calculation': {'min_intensity_for_peak_search': 75, 'percentile_for_peak_proxy': 75},
            'brightness_matching': {'max_peak_difference_allowed': 15, 'adjustment_method': "global_shift", 'target_peak_offset_after_shift': 3},
            'avg_brightness_max_samples': 10 # For faster testing if many images
        })
        
        TEST_SAMPLE_ID = "test_s1_brightness_adjust" 
        base_project_dir = cfg_main.get('project_base_dir')
        if not base_project_dir: raise ValueError("project_base_dir not set in config for testing.")
        test_image_input_dir = os.path.join(base_project_dir, "data", "input", TEST_SAMPLE_ID)
        test_stage1_output_dir = os.path.join(base_project_dir, "data", "output", TEST_SAMPLE_ID, "stage1_brightness_adjust_test")
        os.makedirs(test_stage1_output_dir, exist_ok=True); os.makedirs(test_image_input_dir, exist_ok=True) 
        
        # Create dummy images with varying brightness for testing
        if not os.listdir(test_image_input_dir) or len(os.listdir(test_image_input_dir)) < 20 : 
            if os.listdir(test_image_input_dir): shutil.rmtree(test_image_input_dir); os.makedirs(test_image_input_dir) 
            logger.info(f"Creating dummy images for {TEST_SAMPLE_ID} with brightness variation...")
            date_str, plate_str, well_str = "20240101", "1", "1" ; num_frames_total = 20
            base_bgs = [170, 180, 190, 200, 175] # Different base brightness levels
            for i in range(num_frames_total):
                dummy_fname = f"{date_str}_plate{plate_str}_well{well_str}_f{i:05d}_dpt0.0_fd0.0mm.png"
                dummy_fpath = os.path.join(test_image_input_dir, dummy_fname)
                dummy_img_content_uint8 = np.zeros((300, 300), dtype=np.uint8) 
                well_bg_val = base_bgs[i % len(base_bgs)] + np.random.randint(-5, 5)
                cv2.circle(dummy_img_content_uint8, (150,150), 120, well_bg_val, -1) 
                cv2.circle(dummy_img_content_uint8, (120,120), 15, max(0, well_bg_val - 100), -1) # particle
                cv2.imwrite(dummy_fpath, dummy_img_content_uint8)
            logger.info(f"Created {num_frames_total} dummy images.")
        
        logger.info("\n--- Running S1 Test (with Brightness Adjustment Logic) ---")
        # Ensure other S1 configs are present for the test
        s1_unsub_cfg = s1_cfg.setdefault('unsubtracted_particle_analysis', {})
        s1_unsub_cfg.setdefault('min_particle_area', 30)
        s1_unsub_cfg.setdefault('local_otsu_roi_size', 100) 
        s1_particle_cfg = s1_cfg.setdefault('particle_analysis', {'min_microorganism_area': 50, 'apply_convolution':False}) 
        s1_cfg.setdefault('particle_comparison', {'search_radius_factor': 2.0, 'default_particle_area': 100})
        s1_cfg.setdefault('well_masking', {'create_mask': True, 'radius_px':120, 'radius_fraction':0.9}) # General mask
        
        result_csv_path = run_stage_01_initial_detection(image_dir_path=test_image_input_dir, output_dir_stage1=test_stage1_output_dir, sample_id=TEST_SAMPLE_ID)
        if result_csv_path: 
            logger.info(f"S1 brightness adjust test completed. CSV: {result_csv_path}")
            df_out = pd.read_csv(result_csv_path)
            logger.info(f"Output DataFrame head:\n{df_out.head()}")
        else: logger.error("S1 brightness adjust test FAILED.")
    except Exception as e: logger.exception(f"An unexpected error occurred during the Stage 1 test: {e}")