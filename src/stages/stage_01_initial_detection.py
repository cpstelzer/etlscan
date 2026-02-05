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
from src.core.image_io_operations import (
    load_image,
    load_flat_field_assets,
    apply_flat_field_correction,
)
from src.core.image_enhancement import subtract_images, apply_blur
from src.core.masking_operations import mask_well_area, apply_mask
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area
from src.core.segmentation import generate_binary_mask_from_float_image
from src.core.well_analysis import (
    create_well_mask_black_plate,
    match_well_brightness
)

logger = logging.getLogger(__name__)


def select_background_frame(images_df: pd.DataFrame, fg_index: int, bg_offset: int) -> Dict[str, Any]:
    """Select FG/BG pairing metadata without performing any I/O."""
    if images_df.empty:
        return {"fg_row": None, "bg_row": None, "fg_index": fg_index, "bg_index": None}
    bg_index = (fg_index + bg_offset) % len(images_df)
    return {
        "fg_row": images_df.iloc[fg_index],
        "bg_row": images_df.iloc[bg_index],
        "fg_index": fg_index,
        "bg_index": bg_index,
    }


def prepare_masked_foreground(
    fg_img_uint8: np.ndarray,
    bg_img_uint8: np.ndarray,
    *,
    enable_brightness_adjustment: bool,
    brightness_config: Dict[str, Any],
    well_masking_cfg: Dict[str, Any],
    system_cfg: Dict[str, Any],
    source_illumination: str
) -> Dict[str, Any]:
    """Prepare FG/BG images and reusable masks for subtraction and local analysis."""
    fg_for_sub = fg_img_uint8.copy()
    bg_for_sub = bg_img_uint8.copy()

    if enable_brightness_adjustment:
        _, bg_for_sub = match_well_brightness(
            fg_img_uint8,
            bg_img_uint8,
            brightness_config
        )

    well_mask_uint8 = None
    if well_masking_cfg.get('create_mask', False):
        _, well_mask_uint8 = mask_well_area(
            fg_img_uint8.copy(),
            well_masking_cfg,
            system_cfg
        )

    unsub_prepared = fg_img_uint8.copy()
    if well_mask_uint8 is not None:
        if source_illumination == 'collimated':
            hist_unsub = cv2.calcHist([fg_img_uint8], [0], well_mask_uint8, [256], [0, 256])
            dominant_bg_val = int(np.argmax(hist_unsub))
            filled_outside = np.full_like(fg_img_uint8, dominant_bg_val)
            filled_outside[well_mask_uint8 == 255] = fg_img_uint8[well_mask_uint8 == 255]
            unsub_prepared = filled_outside
        else:
            unsub_prepared = apply_mask(fg_img_uint8.copy(), well_mask_uint8)

    if source_illumination == 'collimated':
        fg_for_sub = cv2.bitwise_not(fg_for_sub)
        bg_for_sub = cv2.bitwise_not(bg_for_sub)

    return {
        "fg_for_sub": fg_for_sub,
        "bg_for_sub": bg_for_sub,
        "well_mask": well_mask_uint8,
        "unsub_prepared": unsub_prepared
    }


def analyze_subtracted_image(
    subtracted_img_uint8: Optional[np.ndarray],
    well_mask_uint8: Optional[np.ndarray],
    particle_analysis_config: Dict[str, Any],
    global_threshold_config: Dict[str, Any],
    sample_id: str,
    source_filename: str
) -> Dict[str, Any]:
    """Find particles on the subtracted image and return structured results."""
    if subtracted_img_uint8 is None:
        return {
            "particles": [],
            "particles_df": pd.DataFrame(),
            "binary_mask": None,
            "masked_subtracted_image": None,
        }

    masked_subtracted_img_uint8 = (
        apply_mask(subtracted_img_uint8, well_mask_uint8)
        if well_mask_uint8 is not None
        else subtracted_img_uint8
    )

    particles_subtracted_list_dicts, binary_mask_display_uint8 = _analyze_single_subtracted_image_global_thresh(
        masked_subtracted_img_uint8,
        particle_analysis_config,
        global_threshold_config
    )
    df_subtracted_particles = pd.DataFrame(particles_subtracted_list_dicts)
    if not df_subtracted_particles.empty:
        df_subtracted_particles['SAMPLE_ID'] = sample_id
        df_subtracted_particles['source_image_filename'] = source_filename

    return {
        "particles": particles_subtracted_list_dicts,
        "particles_df": df_subtracted_particles,
        "binary_mask": binary_mask_display_uint8,
        "masked_subtracted_image": masked_subtracted_img_uint8,
    }


def refine_with_local_otsu(
    unsub_prepared_uint8: np.ndarray,
    candidate_particles: Union[List[Dict[str, Any]], pd.DataFrame],
    roi_crop_size: int,
    analysis_config: Dict[str, Any],
    global_threshold_config: Dict[str, Any],
    source_illumination: str
) -> Dict[str, Any]:
    """Refine detections using local Otsu analysis on the unsubtracted frame."""
    candidate_records = (
        candidate_particles.to_dict('records') if isinstance(candidate_particles, pd.DataFrame) else candidate_particles
    )

    particles_unsubtracted_list_dicts, composite_binary_mask_full_image, roi_debug = _analyze_unsubtracted_locally(
        unsub_prepared_uint8,
        candidate_records,
        roi_crop_size,
        analysis_config,
        global_threshold_config,
        source_illumination
    )
    df_unsubtracted_particles = pd.DataFrame(particles_unsubtracted_list_dicts)

    return {
        "particles": particles_unsubtracted_list_dicts,
        "particles_df": df_unsubtracted_particles,
        "composite_mask": composite_binary_mask_full_image,
        "roi_debug": roi_debug,
    }


def process_frame_pair(
    fg_row: pd.Series,
    bg_row: pd.Series,
    config: Dict[str, Any],
    ffc_assets,
    masks: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Process a FG/BG pair and return fused particle detections."""
    masks = masks or {}
    fg_filepath = fg_row.get('full_path')
    bg_filepath = bg_row.get('full_path')
    fg_img = load_image(fg_filepath, mode='gray')
    bg_img = load_image(bg_filepath, mode='gray')
    if fg_img is None or bg_img is None:
        logger.warning(f"Could not load FG ({fg_filepath}) or BG ({bg_filepath}). Skipping pair.")
        return pd.DataFrame()

    # Correct both frames for uneven lighting before any masking, subtraction, or thresholding.
    # This makes intensity comparisons more reliable across the full field of view.
    fg_img_float = apply_flat_field_correction(fg_img, ffc_assets)
    bg_img_float = apply_flat_field_correction(bg_img, ffc_assets)

    # Convert corrected [0, 1] floats back to uint8 because Stage 1 helpers below
    # (masking, subtraction, and local Otsu routines) currently expect uint8 images.
    fg_img_ffc = np.round(fg_img_float * 255.0).astype(np.uint8)
    bg_img_ffc = np.round(bg_img_float * 255.0).astype(np.uint8)

    prep_data = prepare_masked_foreground(
        fg_img_ffc,
        bg_img_ffc,
        enable_brightness_adjustment=config.get('enable_brightness_adjustment', False),
        brightness_config=config.get('config_black_plate_well', {}),
        well_masking_cfg=config.get('well_masking', {}),
        system_cfg=config.get('system_cfg', {}),
        source_illumination=config.get('source_image_illumination', 'darkfield')
    )
    general_well_mask_for_roi_uint8 = prep_data.get("well_mask")
    if general_well_mask_for_roi_uint8 is None:
        general_well_mask_for_roi_uint8 = masks.get("well_mask")
    elif masks.get("well_mask") is None:
        masks["well_mask"] = general_well_mask_for_roi_uint8
    # Subtract the prepared background from the prepared foreground to isolate moving particles.
    subtracted_img_uint8 = subtract_images(prep_data["fg_for_sub"], prep_data["bg_for_sub"])
    sub_analysis = analyze_subtracted_image(
        subtracted_img_uint8,
        general_well_mask_for_roi_uint8,
        config.get('particle_analysis_subtracted_config', {}),
        config.get('global_threshold_config', {}),
        config.get('sample_id', 'UNKNOWN_SAMPLE'),
        fg_row.get('filename', 'UNKNOWN_FILE')
    )
    unsub_analysis = refine_with_local_otsu(
        prep_data["unsub_prepared"],
        sub_analysis["particles_df"],
        config.get('local_otsu_roi_size', 150),
        config.get('unsubtracted_analysis_config', {}),
        config.get('global_threshold_config', {}),
        config.get('source_image_illumination', 'darkfield')
    )
    # Fuse subtracted and unsubtracted detections to keep confident matches.
    fused_df_frame = _compare_and_fuse_particle_dfs(
        sub_analysis["particles_df"],
        unsub_analysis["particles_df"],
        config.get('particle_comparison_config', {})
    )
    return fused_df_frame


# --- _parse_image_filenames, _filter_image_df, _analyze_single_subtracted_image_global_thresh, _analyze_unsubtracted_locally, _compare_and_fuse_particle_dfs ---
# These helper functions remain unchanged.
def _parse_image_filenames(image_dir_path: str) -> Optional[pd.DataFrame]: # Unchanged
    """Parse image filenames into a dataframe of metadata needed for Stage 1 processing."""
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
    """Trim the parsed image table to frames usable for pairing foreground/background."""
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

def _determine_bg_offset_for_fg_bg_pairs(
    images_df_filtered: pd.DataFrame,
    minimum_offset: Optional[int],
    fallback_offset: int,
    *,
    tolerance: float = 1e-6
) -> int:
    """Return an offset that pairs FG and BG frames with matching fd_mm.

    Parameters
    ----------
    images_df_filtered:
        DataFrame of parsed image metadata, already filtered/sorted.
    minimum_offset:
        Minimum offset (in frames) allowed when searching for a match.
    fallback_offset:
        Offset from the configuration to use if no automatic match is found.
    tolerance:
        Absolute tolerance for comparing floating-point fd_mm values.

    Returns
    -------
    int
        The offset to use when selecting background frames.
    """

    if images_df_filtered.empty:
        logger.warning("BG offset auto-mode: no frames available; using fallback offset.")
        return fallback_offset

    if "fd_mm" not in images_df_filtered:
        logger.warning("BG offset auto-mode: fd_mm column missing; using fallback offset.")
        return fallback_offset

    num_frames = len(images_df_filtered)
    if minimum_offset is None:
        logger.warning("BG offset auto-mode: minimum offset not provided; using fallback offset.")
        return fallback_offset

    if num_frames <= max(minimum_offset, 0):
        logger.warning(
            "BG offset auto-mode: insufficient frames (%d) for minimum offset %d; using fallback offset.",
            num_frames,
            minimum_offset,
        )
        return fallback_offset

    fd_values = images_df_filtered["fd_mm"].to_numpy(dtype=float)
    min_offset = max(minimum_offset, 1)
    best_offset = None
    best_match_ratio = -1.0

    for offset in range(min_offset, num_frames):
        fg_fd_values = fd_values[:-offset]
        bg_fd_values = fd_values[offset:]
        if fg_fd_values.size == 0 or bg_fd_values.size == 0:
            continue

        matches = np.isclose(fg_fd_values, bg_fd_values, atol=tolerance)
        match_ratio = matches.mean()

        if matches.size and np.all(matches):
            logger.info(
                "BG offset auto-mode: selected offset %d (perfect fd_mm match across %d frames).",
                offset,
                matches.size,
            )
            return offset

        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_offset = offset

    if best_offset is not None and best_match_ratio > 0.0:
        logger.info(
            "BG offset auto-mode: using offset %d with %.1f%% fd_mm match rate (no perfect match found).",
            best_offset,
            best_match_ratio * 100,
        )
        return best_offset

    logger.warning(
        "BG offset auto-mode: could not find suitable offset; reverting to fallback (%d).",
        fallback_offset,
    )
    return fallback_offset

def _analyze_single_subtracted_image_global_thresh( # Unchanged
    image_to_analyze_gray_uint8: np.ndarray,
    config_stage1_particle_analysis: Dict[str, Any],
    global_threshold_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """Detect particles on a subtracted frame using global thresholding settings."""
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
    """Refine subtracted detections by re-segmenting local ROIs on the unsubtracted image."""
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
    """Fuse subtracted and unsubtracted detections by spatial proximity to keep confident particles."""
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
    rename_dict = {}
    metrics_to_prefix = [
        'area', 'perimeter', 'orientation', 'aspect_ratio', 'solidity',
        'eccentricity', 'feret_diameter', 'circularity', 'equivalent_diameter'
    ]
    for col in final_matched_df.columns:
        if col in metrics_to_prefix:
            rename_dict[col] = f"s1_{col}"
    if rename_dict:
        final_matched_df.rename(columns=rename_dict, inplace=True)
        logger.debug(f"Prefixed {len(rename_dict)} columns with 's1_'.")
    logger.info(f"Fused {len(final_matched_df)} particles.")
    return final_matched_df

def _process_particle_coordinates_for_tracking_df(input_df: pd.DataFrame, pixel_size_um, magnification):
    """Convert fused particle coordinates to millimeters and extract tracking metadata."""
    logger.info("Tracking Step 1: Loading and processing fused particle data for tracking.")
    df = input_df.copy()
    if df.empty:
        logger.warning("Input DataFrame for tracking is empty. No particles to process.")
        return pd.DataFrame()
    pixel_to_mm_factor = pixel_size_um / (magnification * 1000)
    logger.info(f"   - Using Pixel-to-MM factor: {pixel_to_mm_factor:.6f}")
    df['x_mm'] = df['X'] * pixel_to_mm_factor
    df['y_mm'] = df['Y'] * pixel_to_mm_factor
    def extract_metadata(filename):
        """Parse frame, z position, and dioptries from a source image filename."""
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

def _track_particles_3d(processed_df, search_range_mm, memory_frames, min_track_length): # Unchanged
    """Run trackpy to build 3D trajectories from processed particle coordinates."""
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
    filtered_trajectories.rename(columns={'particle': 'track_id'}, inplace=True)
    filtered_trajectories = filtered_trajectories.reset_index(drop=True)
    return filtered_trajectories

def _build_tracking_figure(tracked_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Create a 3D matplotlib figure summarizing all tracked particle paths."""
    if tracked_df.empty:
        return None
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Particle Tracks")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    for _, track in tracked_df.groupby('track_id'):
        ax.plot(track['x_mm'], track['y_mm'], track['z_mm'], linewidth=1)
    plt.tight_layout()
    return fig


def run_particle_tracking(
    fused_df: pd.DataFrame,
    sample_id: str,
    tracking_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """Perform particle tracking on fused detections and return data instead of writing files."""
    logger.info(f"--- Post-S1 Tracking initiated for: {sample_id} ---")
    pixel_size_um = tracking_config.get('pixel_size_um')
    magnification = tracking_config.get('magnification')
    search_range_mm = tracking_config.get('search_range_mm')
    memory_frames = tracking_config.get('memory_frames')
    min_track_length = tracking_config.get('min_track_length')
    if not all([pixel_size_um, magnification, search_range_mm, memory_frames, min_track_length]):
        raise ConfigError("One or more required tracking parameters are missing from the configuration.")

    processed_data = _process_particle_coordinates_for_tracking_df(fused_df, pixel_size_um, magnification)
    if processed_data.empty:
        logger.warning("Tracking workflow stopped: No data after initial processing.")
        return processed_data, None

    tracked_data = _track_particles_3d(processed_data, search_range_mm, memory_frames, min_track_length)
    if tracked_data.empty:
        logger.warning("Tracking workflow stopped: No tracks survived the filtering process.")
        processed_data['track_id'] = np.nan
        return processed_data, None

    logger.info("Tracking Step 3: Merging tracking results with original data...")
    merge_cols = ['frame', 'x_mm', 'y_mm', 'z_mm']
    tracked_cols = ['frame', 'x', 'y', 'z', 'track_id']

    processed_rounded = processed_data.copy()
    processed_rounded[merge_cols[1:]] = processed_rounded[merge_cols[1:]].round(6)
    tracked_rounded = tracked_data.copy()
    tracked_rounded[['x', 'y', 'z']] = tracked_rounded[['x', 'y', 'z']].round(6)

    merged_df = pd.merge(
        processed_rounded,
        tracked_rounded[tracked_cols],
        how='left',
        left_on=merge_cols,
        right_on=['frame', 'x', 'y', 'z']
    )
    merged_df.drop(columns=['x', 'y', 'z'], inplace=True, errors='ignore')

    final_tracked_particles_df = merged_df.dropna(subset=['track_id']).copy()
    if not final_tracked_particles_df.empty:
        final_tracked_particles_df['track_id'] = final_tracked_particles_df['track_id'].astype(int)

    tracking_figure = _build_tracking_figure(final_tracked_particles_df)
    logger.info(f"--- Post-S1 Tracking for {sample_id} COMPLETED ---")
    return final_tracked_particles_df, tracking_figure


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Input: image dir, output dir, sample id, config. Output: Stage 1 CSV path or None."""
    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} ---")
    config = config or get_config()
    config_s1 = config.get('stage_01_initial_detection', {})
    system_cfg = config.get('system', {})
    source_image_illumination = system_cfg.get('source_image_illumination_type', 'darkfield')
    global_threshold_config = system_cfg.get('thresholding', {})
    if not global_threshold_config:
        raise ConfigError("Stage 1: System-level thresholding configuration not found or empty.")
    
    config_black_plate_well = config_s1.get('black_plate_well_analysis', {})
    enable_brightness_adjustment = config_black_plate_well.get('enable_brightness_adjustment', False)
    
    particle_analysis_subtracted_config = config_s1.get('particle_analysis', {})
    unsubtracted_analysis_config = config_s1.get('unsubtracted_particle_analysis', {})
    local_otsu_roi_size = unsubtracted_analysis_config.get('local_otsu_roi_size', 150)
    particle_comparison_config = config_s1.get('particle_comparison', {})
    bg_offset_mode = config_s1.get('BG_frame_offset_mode', 'fixed')
    bg_offset_default = config_s1.get('bg_offset_val', 110)

    recording_fps_raw = system_cfg.get('recording_fps')
    try:
        recording_fps = float(recording_fps_raw)
    except (TypeError, ValueError):
        recording_fps = 0.0

    recording_fps = max(recording_fps, 0.0)
    min_offset_for_auto: Optional[int]
    if recording_fps > 0:
        min_offset_for_auto = max(int(round(recording_fps * 5)), 1)
    else:
        min_offset_for_auto = None

    general_well_masking_cfg = config_s1.get('well_masking', {})

    images_df = _parse_image_filenames(image_dir_path)
    if images_df is None or images_df.empty: return None
    images_df_filtered = _filter_image_df(images_df.copy())
    if images_df_filtered.empty: return None

    # Load flat/dark calibration once per sample directory and reuse it for all frames.
    # Reuse keeps the correction consistent and avoids repeated disk reads.
    ffc_assets = load_flat_field_assets(image_dir_path, mode='gray')
    
    frame_pair_config = {
        "enable_brightness_adjustment": enable_brightness_adjustment,
        "config_black_plate_well": config_black_plate_well,
        "well_masking": general_well_masking_cfg,
        "system_cfg": system_cfg,
        "source_image_illumination": source_image_illumination,
        "particle_analysis_subtracted_config": particle_analysis_subtracted_config,
        "global_threshold_config": global_threshold_config,
        "local_otsu_roi_size": local_otsu_roi_size,
        "unsubtracted_analysis_config": unsubtracted_analysis_config,
        "particle_comparison_config": particle_comparison_config,
        "sample_id": sample_id,
    }

    all_fused_particles_for_sample_list: List[pd.DataFrame] = []
    reusable_masks: Dict[str, Any] = {}

    effective_mode = bg_offset_mode
    if bg_offset_mode == 'automatic':
        if min_offset_for_auto is None:
            logger.warning(
                "Stage 1: Automatic background offset requested but system.recording_fps is missing or invalid; "
                "falling back to configured offset %d.",
                bg_offset_default,
            )
            bg_offset = bg_offset_default
            effective_mode = 'fixed-fallback'
        else:
            bg_offset = _determine_bg_offset_for_fg_bg_pairs(
                images_df_filtered,
                min_offset_for_auto,
                bg_offset_default
            )
    else:
        bg_offset = bg_offset_default

    min_offset_display = min_offset_for_auto if min_offset_for_auto is not None else 'N/A'
    logger.info(
        "Stage 1: Using background frame offset %s (mode=%s, minimum_auto_offset=%s, recording_fps=%.3f)",
        bg_offset,
        effective_mode,
        min_offset_display,
        recording_fps,
    )

    for idx_frame_list in range(len(images_df_filtered)):
        # Pair foreground frame with its background counterpart using configured offset.
        pairing_data = select_background_frame(images_df_filtered, idx_frame_list, bg_offset)
        fg_row = pairing_data.get("fg_row")
        bg_row = pairing_data.get("bg_row")
        if fg_row is None or bg_row is None:
            logger.warning("Failed to select FG/BG pairing; skipping frame index %s.", idx_frame_list)
            continue

        logger.debug(f"Processing S1 Frame: {fg_row['filename']} (BG: {bg_row['filename']})")
        fused_df_frame = process_frame_pair(
            fg_row,
            bg_row,
            frame_pair_config,
            ffc_assets,
            reusable_masks
        )

        # Collect fused detections for the sample.
        if not fused_df_frame.empty:
            all_fused_particles_for_sample_list.append(fused_df_frame)

    if not all_fused_particles_for_sample_list:
        logger.warning(f"No particles detected or fused for the entire sample {sample_id}.")
        final_sample_df = pd.DataFrame(columns=['SAMPLE_ID', 'source_image_filename', 'X', 'Y'])
    else:
        final_sample_df = pd.concat(all_fused_particles_for_sample_list, ignore_index=True)
    
    logger.info(f"Total fused particles for sample {sample_id}: {len(final_sample_df)}")

    final_output_filename = f"{sample_id}_stage1_tracked_particles.csv"
    final_output_path = os.path.join(output_dir_stage1, final_output_filename)
    
    tracking_config = config_s1.get('tracking', {})
    if tracking_config.get('enabled', False):
        # Track fused particles and defer file I/O to the stage wrapper.
        try:
            tracked_df, tracking_fig = run_particle_tracking(final_sample_df, sample_id, tracking_config)
        except Exception as e:
            logger.exception(f"Particle tracking failed for sample {sample_id}: {e}")
            tracked_df, tracking_fig = pd.DataFrame(), None

        if tracked_df.empty and not final_sample_df.empty:
            logger.warning("Particle tracking returned no tracks; saving fused detections with empty track IDs.")
            final_sample_df['track_id'] = np.nan
            save_dataframe_to_csv(final_sample_df, output_dir_stage1, final_output_filename)
        else:
            save_dataframe_to_csv(tracked_df, output_dir_stage1, final_output_filename)
        if tracking_fig is not None:
            figure_3d_path = os.path.join(output_dir_stage1, f"{sample_id}_stage1_tracks_3d.png")
            tracking_fig.savefig(figure_3d_path, dpi=150)
            plt.close(tracking_fig)
            # Tracking figure documents the trajectories visually.
    else:
        logger.info("Particle tracking is disabled. Saving fused data as final output.")
        final_sample_df['track_id'] = np.nan
        save_dataframe_to_csv(final_sample_df, output_dir_stage1, final_output_filename)
    
    if os.path.exists(final_output_path):
        logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} COMPLETED ---")
        return final_output_path
    else:
        logger.error(f"Stage 1 final output file '{final_output_path}' was not created.")
        return None
