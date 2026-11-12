# src/stages/stage_01_initial_detection.py
import logging
import os
import re
from typing import Union, Optional, List, Dict, Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.filters
import skimage.util
import trackpy as tp
from sklearn.neighbors import BallTree

from src.utils import (
    load_dataframe_from_csv, save_dataframe_to_csv,
    parse_date_plate_well, parse_frame_filenames, extract_frame_from_filename
)
from src.utils.common_exceptions import ProcessingError, ImageProcessingError, ConfigError, DataValidationError
from src.utils.acquisition_metadata import get_scan_extent_mm, read_frames_lost_at_acquisition
from src.utils.config_loader import get_pixel_to_mm_factor
from src.core.image_io_operations import (
    load_image,
    load_flat_field_assets,
    apply_flat_field_correction,
)
from src.core.image_enhancement import subtract_images, apply_blur
from src.core.masking_operations import mask_well_area, apply_mask
from src.core.region_props_calculator import find_contours, calculate_contour_measurements, filter_contours_by_area
from src.core.segmentation import generate_binary_mask_from_float_image, assess_subtracted_foreground
from src.visualization.figure_style import draw_floor_shadow, plot_track_line, track_figure_style
from src.core.well_analysis import (
    create_well_mask_black_plate,
    create_rim_excluded_well_mask,
    create_well_mask_with_fallback,
    match_low_frequency_illumination,
    match_well_brightness
)

logger = logging.getLogger(__name__)

# Two frames show the same focal plane when their dpt / fd_mm (parsed from the same
# filename tokens) agree within this tolerance. Shared by both FG/BG pairing strategies:
# the time-shift offset search and the external-background frame matching.
_FOCAL_PLANE_TOLERANCE = 1e-6


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
    well_analysis_config: Dict[str, Any],
    well_masking_cfg: Dict[str, Any],
    system_cfg: Dict[str, Any],
    source_illumination: str
) -> Dict[str, Any]:
    """Prepare FG/BG images and reusable masks for subtraction and local analysis.

    Shared by the batch (both background sources) and the realtime Stage 1.
    ``well_analysis_config`` is ``stage_01_initial_detection.black_plate_well_analysis``.
    Each frame's well mask (:func:`create_well_mask_with_fallback`) serves the brightness
    matching and the well-rim exclusion. ``subtraction_roi_mask`` is where the subtracted
    analysis searches: inside both frames' wells, ``rim_exclusion_px`` away from either
    rim (:func:`create_rim_excluded_well_mask`), and inside ``well_mask``. ``well_mask``
    is the optional circular mask of ``well_masking`` (None unless ``create_mask``),
    which also shapes the unsubtracted frame.

    Raises:
        ImageProcessingError: a frame has no well mask (well detection failed and the
            full-frame fallback is disabled).
    """
    fg_for_sub = fg_img_uint8.copy()
    bg_for_sub = bg_img_uint8.copy()

    well_identification_config = well_analysis_config['well_identification']
    fg_well_mask_uint8 = create_well_mask_with_fallback(fg_img_uint8, well_identification_config)
    bg_well_mask_uint8 = create_well_mask_with_fallback(bg_img_uint8, well_identification_config)
    if fg_well_mask_uint8 is None or bg_well_mask_uint8 is None:
        raise ImageProcessingError(
            "No well mask could be built for the "
            f"{'FG' if fg_well_mask_uint8 is None else 'BG'} frame (well detection failed and "
            "well_identification.fallback_full_frame_on_failure is disabled), so the well rim "
            "cannot be excluded."
        )
    rim_excluded_well_mask_uint8 = create_rim_excluded_well_mask(
        [fg_well_mask_uint8, bg_well_mask_uint8],
        float(well_analysis_config['rim_exclusion_px']),
    )

    if enable_brightness_adjustment:
        _, bg_for_sub = match_well_brightness(
            fg_img_uint8,
            bg_img_uint8,
            well_analysis_config,
            fg_well_mask_uint8=fg_well_mask_uint8,
            bg_well_mask_uint8=bg_well_mask_uint8,
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
        "subtraction_roi_mask": (
            rim_excluded_well_mask_uint8 if well_mask_uint8 is None
            else cv2.bitwise_and(rim_excluded_well_mask_uint8, well_mask_uint8)
        ),
        "unsub_prepared": unsub_prepared
    }


def analyze_subtracted_image(
    subtracted_img_uint8: Optional[np.ndarray],
    well_mask_uint8: Optional[np.ndarray],
    particle_analysis_config: Dict[str, Any],
    global_threshold_config: Dict[str, Any],
    sample_id: str,
    source_filename: str,
    subtracted_foreground_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find particles on the subtracted image and return structured results.

    ``subtracted_foreground_config`` (shared FG/BG gate, from system.subtracted_foreground)
    applies a fixed foreground floor and a max-foreground-fraction guard via
    :func:`assess_subtracted_foreground`. This is the single place where both the batch and
    realtime pipelines evaluate that gate. On a failed subtraction the result carries
    ``subtraction_failed=True`` and no candidates. ``foreground_fraction`` reports the gate's
    measured fraction (None when the gate is not configured).
    """
    empty_result = {
        "particles": [],
        "particles_df": pd.DataFrame(),
        "binary_mask": None,
        "masked_subtracted_image": None,
        "foreground_fraction": None,
        "subtraction_failed": False,
    }
    if subtracted_img_uint8 is None:
        return empty_result

    masked_subtracted_img_uint8 = (
        apply_mask(subtracted_img_uint8, well_mask_uint8)
        if well_mask_uint8 is not None
        else subtracted_img_uint8
    )

    # Shared FG/BG foreground gate: a fixed floor Otsu may tighten but not fall below, plus a
    # guard that rejects the whole frame when the subtraction failed (flood of false positives).
    foreground_floor_mask = None
    foreground_fraction = None
    if subtracted_foreground_config:
        well_area_px = (
            int(np.count_nonzero(well_mask_uint8)) if well_mask_uint8 is not None
            else int(masked_subtracted_img_uint8.size)
        )
        foreground_floor_mask, foreground_fraction, subtraction_failed = assess_subtracted_foreground(
            masked_subtracted_img_uint8,
            int(subtracted_foreground_config['foreground_diff_threshold']),
            float(subtracted_foreground_config['max_foreground_fraction']),
            area_px=well_area_px,
        )
        if subtraction_failed:
            logger.warning(
                "S1 Subtracted: foreground fraction %.3f exceeds %.3f of the well area; treating the "
                "FG/BG subtraction as failed and emitting no candidates for this frame.",
                foreground_fraction, float(subtracted_foreground_config['max_foreground_fraction']),
            )
            return {
                **empty_result,
                "binary_mask": foreground_floor_mask.astype(np.uint8) * 255,
                "masked_subtracted_image": masked_subtracted_img_uint8,
                "foreground_fraction": foreground_fraction,
                "subtraction_failed": True,
            }

    particles_subtracted_list_dicts, binary_mask_display_uint8 = _analyze_single_subtracted_image_global_thresh(
        masked_subtracted_img_uint8,
        particle_analysis_config,
        global_threshold_config,
        foreground_floor_mask=foreground_floor_mask,
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
        "foreground_fraction": foreground_fraction,
        "subtraction_failed": False,
    }


def refine_with_local_otsu(
    unsub_prepared_uint8: np.ndarray,
    candidate_particles: Union[List[Dict[str, Any]], pd.DataFrame],
    adaptive_padding_factor: float,
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
        adaptive_padding_factor,
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


def _prepare_external_background(
    fg_img_uint8: np.ndarray,
    bg_img_uint8: np.ndarray,
    external_background_config: Dict[str, Any],
) -> np.ndarray:
    """Give an external (separately recorded, specimen-free) BG frame its FG frame's illumination.

    External BG runs are recorded minutes to an hour apart from the FG run, so the
    illumination drifts as a smooth spatial gradient that would flood the shared
    subtracted-foreground gate. Returns the BG with the FG's low-frequency illumination.
    The slight shift of the well rim / meniscus edge between the runs is covered by the
    well-rim exclusion shared with the time-shift mode (:func:`prepare_masked_foreground`).
    Time-shift pairs come from the same recording and never take this step.
    """
    return match_low_frequency_illumination(
        fg_img_uint8,
        bg_img_uint8,
        int(external_background_config['illumination_match_downsample']),
        int(external_background_config['illumination_match_median_kernel_px']),
    )


def process_frame_pair(
    fg_row: pd.Series,
    bg_row: pd.Series,
    config: Dict[str, Any],
    ffc_assets: Optional[Any],
    masks: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Process a FG/BG pair and return fused particle detections plus pair diagnostics.

    The diagnostics (None when an image could not be loaded) record the FG/BG files and
    the shared foreground gate's verdict for the per-sample pairing audit. With
    ``config['external_background_config']`` set, the BG is first adapted by
    :func:`_prepare_external_background`. In both modes the subtracted analysis is
    restricted to the rim-excluded well area of :func:`prepare_masked_foreground`.
    """
    masks = masks or {}
    fg_filepath = fg_row['full_path']
    bg_filepath = bg_row['full_path']
    fg_img = load_image(fg_filepath, mode='gray')
    bg_img = load_image(bg_filepath, mode='gray')
    if fg_img is None or bg_img is None:
        logger.warning(f"Could not load FG ({fg_filepath}) or BG ({bg_filepath}). Skipping pair.")
        return pd.DataFrame(), None

    if ffc_assets is not None:
        fg_img_ffc = np.round(apply_flat_field_correction(fg_img, ffc_assets) * 255.0).astype(np.uint8)
        bg_img_ffc = np.round(apply_flat_field_correction(bg_img, ffc_assets) * 255.0).astype(np.uint8)
    else:
        fg_img_ffc = fg_img
        bg_img_ffc = bg_img

    external_background_config = config['external_background_config']
    if external_background_config is not None:
        bg_img_ffc = _prepare_external_background(fg_img_ffc, bg_img_ffc, external_background_config)

    prep_data = prepare_masked_foreground(
        fg_img_ffc,
        bg_img_ffc,
        enable_brightness_adjustment=config['enable_brightness_adjustment'],
        well_analysis_config=config['config_black_plate_well'],
        well_masking_cfg=config['well_masking'],
        system_cfg=config['system_cfg'],
        source_illumination=config['source_image_illumination']
    )
    subtraction_roi_mask_uint8 = prep_data["subtraction_roi_mask"]
    if prep_data["well_mask"] is not None:
        masks.setdefault("well_mask", prep_data["well_mask"])
    elif masks.get("well_mask") is not None:
        # This frame's circular well mask failed: keep the sample's earlier one.
        subtraction_roi_mask_uint8 = cv2.bitwise_and(subtraction_roi_mask_uint8, masks["well_mask"])
    # Subtract the prepared background from the prepared foreground to isolate particles
    # absent from the background (moving ones for time-shift pairs, all for external BG).
    subtracted_img_uint8 = subtract_images(prep_data["fg_for_sub"], prep_data["bg_for_sub"])
    sub_analysis = analyze_subtracted_image(
        subtracted_img_uint8,
        subtraction_roi_mask_uint8,
        config['particle_analysis_subtracted_config'],
        config['global_threshold_config'],
        config['sample_id'],
        fg_row['filename'],
        subtracted_foreground_config=config['subtracted_foreground_config'],
    )
    unsubtracted_cfg = config['unsubtracted_analysis_config']
    unsub_analysis = refine_with_local_otsu(
        prep_data["unsub_prepared"],
        sub_analysis["particles_df"],
        float(unsubtracted_cfg['adaptive_padding_factor']),
        unsubtracted_cfg,
        config['global_threshold_config'],
        config['source_image_illumination']
    )
    # Fuse subtracted and unsubtracted detections to keep confident matches.
    fused_df_frame = _compare_and_fuse_particle_dfs(
        sub_analysis["particles_df"],
        unsub_analysis["particles_df"],
        config['particle_comparison_config']
    )
    pair_diagnostics = {
        "fg_filename": fg_row['filename'],
        "bg_filename": bg_row['filename'],
        "frame": int(fg_row['frame']),
        "dpt": float(fg_row['dpt']),
        "fd_mm": float(fg_row['fd_mm']),
        "foreground_fraction": sub_analysis["foreground_fraction"],
        "subtraction_failed": sub_analysis["subtraction_failed"],
        "fused_detections": len(fused_df_frame),
    }
    return fused_df_frame, pair_diagnostics


# --- _filter_image_df, _analyze_single_subtracted_image_global_thresh, _analyze_unsubtracted_locally, _compare_and_fuse_particle_dfs ---
# These helper functions remain unchanged.
def _filter_image_df(df: pd.DataFrame, *, trim_to_last_fd0: bool) -> pd.DataFrame:
    """Trim the parsed image table to frames usable for pairing foreground/background.

    Frame 0 is always dropped. ``trim_to_last_fd0`` additionally cuts every frame after
    the last fd_mm == 0 frame, keeping whole focus-sweep cycles so the modulo-wrapped
    time-shift pairing still pairs equal focal planes. External-background pairing gives
    each frame its own BG partner, so it keeps the tail (``trim_to_last_fd0=False``).
    """
    if df.empty: return df
    initial_len = len(df)
    if df.iloc[0]["frame"] == 0:
        df = df.iloc[1:].reset_index(drop=True)
        logger.info(f"Removed frame 0. Rows: {initial_len} -> {len(df)}.")
    if not trim_to_last_fd0:
        return df
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
    tolerance: float = _FOCAL_PLANE_TOLERANCE
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


def _build_time_shift_pairs(
    images_df_filtered: pd.DataFrame,
    config_s1: Dict[str, Any],
    recording_fps: float,
) -> List[Tuple[pd.Series, pd.Series]]:
    """Pair every frame with a later frame of the same recording (time-shifted BG).

    The offset is ``bg_offset_val``, or in ``automatic`` mode the smallest offset of at
    least ~5 s whose frames match in fd_mm; pairs wrap modulo the frame count.
    """
    bg_offset_mode = config_s1['BG_frame_offset_mode']
    bg_offset_default = config_s1['bg_offset_val']
    min_offset_for_auto = max(int(round(recording_fps * 5)), 1) if recording_fps > 0 else None

    effective_mode = bg_offset_mode
    if bg_offset_mode == 'automatic':
        if min_offset_for_auto is None:
            logger.warning(
                "Stage 1: Automatic background offset requested but acquisition.camera.fps is non-positive; "
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

    frame_pairs: List[Tuple[pd.Series, pd.Series]] = []
    for idx_frame_list in range(len(images_df_filtered)):
        # Pair foreground frame with its background counterpart using configured offset.
        pairing_data = select_background_frame(images_df_filtered, idx_frame_list, bg_offset)
        fg_row = pairing_data.get("fg_row")
        bg_row = pairing_data.get("bg_row")
        if fg_row is None or bg_row is None:
            logger.warning(f"Failed to select FG/BG pairing; skipping frame index {idx_frame_list}.")
            continue
        frame_pairs.append((fg_row, bg_row))
    return frame_pairs


def _build_external_background_pairs(
    fg_images_df: pd.DataFrame,
    bg_images_df: pd.DataFrame,
    bg_metadata_log_path: str,
) -> Tuple[List[Tuple[pd.Series, pd.Series]], List[pd.Series]]:
    """Pair every FG frame with the same-index frame of the external (specimen-free) BG run.

    Pairs are keyed on the frame index only: sample folders may have been renamed, so the
    date/plate/well tokens inside the filenames are not trusted. Each pair must then share
    its focal plane (dpt and fd_mm). Strict: subtracting a BG from another focal plane
    yields plausible but wrong detections, so any mismatched FG frame fails.

    An FG frame without a BG frame is accepted only when the BG run's metadata log (read
    only then) records that frame as lost at acquisition: the camera's exposure never
    reached the PC, so the BG run has a known gap and the FG frame cannot be subtracted.
    Any other missing BG frame (e.g. a file deleted after the run) fails.

    Returns:
        The FG/BG pairs, and the FG rows whose BG frame was lost at acquisition.

    Raises:
        DataValidationError: duplicate BG frame indices, FG frames without a BG frame that
            the BG log does not record as lost, or focal-plane mismatches.
        FileNotFoundError: a BG frame is missing and the BG run has no metadata log.
    """
    duplicated_bg_frames = bg_images_df.loc[bg_images_df['frame'].duplicated(), 'frame']
    if not duplicated_bg_frames.empty:
        raise DataValidationError(
            f"External BG has duplicate frame indices: {sorted(duplicated_bg_frames.unique())[:10]}"
        )
    bg_rows_by_frame = {int(bg_row['frame']): bg_row for _, bg_row in bg_images_df.iterrows()}

    frame_pairs: List[Tuple[pd.Series, pd.Series]] = []
    fg_rows_without_bg: List[pd.Series] = []
    mismatched_frames: List[str] = []
    for _, fg_row in fg_images_df.iterrows():
        frame = int(fg_row['frame'])
        bg_row = bg_rows_by_frame.get(frame)
        if bg_row is None:
            fg_rows_without_bg.append(fg_row)
            continue
        same_focal_plane = (
            np.isclose(fg_row['dpt'], bg_row['dpt'], atol=_FOCAL_PLANE_TOLERANCE)
            and np.isclose(fg_row['fd_mm'], bg_row['fd_mm'], atol=_FOCAL_PLANE_TOLERANCE)
        )
        if not same_focal_plane:
            mismatched_frames.append(
                f"f{frame:05d} FG dpt={fg_row['dpt']} fd={fg_row['fd_mm']} / BG dpt={bg_row['dpt']} fd={bg_row['fd_mm']}"
            )
            continue
        frame_pairs.append((fg_row, bg_row))

    fg_rows_with_bg_lost: List[pd.Series] = []
    unexplained_missing_frames: List[int] = []
    if fg_rows_without_bg:
        bg_frames_lost_at_acquisition = read_frames_lost_at_acquisition(bg_metadata_log_path)
        for fg_row in fg_rows_without_bg:
            if int(fg_row['frame']) in bg_frames_lost_at_acquisition:
                fg_rows_with_bg_lost.append(fg_row)
            else:
                unexplained_missing_frames.append(int(fg_row['frame']))

    problems = []
    if unexplained_missing_frames:
        problems.append(
            f"{len(unexplained_missing_frames)} FG frame(s) have no BG frame, and {bg_metadata_log_path!r} "
            f"does not record them as lost at acquisition (first: {unexplained_missing_frames[:5]})"
        )
    if mismatched_frames:
        problems.append(f"{len(mismatched_frames)} focal-plane mismatch(es) (first: {mismatched_frames[:3]})")
    if problems:
        raise DataValidationError(f"External background pairing failed: {'; '.join(problems)}")
    return frame_pairs, fg_rows_with_bg_lost


def _audit_external_background_pairs(
    pair_diagnostics_records: List[Dict[str, Any]],
    fg_rows_with_bg_lost: List[pd.Series],
    max_failed_frame_fraction: float,
    output_dir_stage1: str,
    sample_id: str,
) -> None:
    """Write the FG/BG pairing audit CSV and fail the sample on a mostly failed subtraction.

    An external BG is an independent recording (a time-shifted BG comes from the same
    one): when too many frames trip the shared subtracted-foreground guard it does not
    show the same scene (e.g. a moved dish), and the sample's detections would be
    silently incomplete. FG frames whose BG frame was lost at acquisition could not be
    subtracted either: they are listed with ``bg_lost_at_acquisition`` and
    ``subtraction_failed`` set and count against the same limit.
    """
    pairs_df = pd.DataFrame(pair_diagnostics_records)
    if not pairs_df.empty:
        pairs_df['bg_lost_at_acquisition'] = False
    lost_df = pd.DataFrame([
        {
            "fg_filename": fg_row['filename'],
            "bg_filename": "",
            "frame": int(fg_row['frame']),
            "dpt": float(fg_row['dpt']),
            "fd_mm": float(fg_row['fd_mm']),
            "foreground_fraction": np.nan,
            "subtraction_failed": True,
            "fused_detections": 0,
            "bg_lost_at_acquisition": True,
        }
        for fg_row in fg_rows_with_bg_lost
    ])
    audit_parts = [part_df for part_df in (pairs_df, lost_df) if not part_df.empty]
    audit_df = (
        pd.concat(audit_parts, ignore_index=True).sort_values('frame', ignore_index=True)
        if audit_parts else pd.DataFrame()
    )
    save_dataframe_to_csv(audit_df, output_dir_stage1, f"{sample_id}_stage1_bg_pairing.csv")
    if pairs_df.empty:
        raise ProcessingError(f"External background: no FG/BG pair of sample {sample_id} could be processed.")

    lost_frame_count = len(lost_df)
    guard_failed_frame_count = int(pairs_df['subtraction_failed'].sum())
    failed_frame_fraction = (guard_failed_frame_count + lost_frame_count) / len(audit_df)
    logger.info(
        f"External background: {guard_failed_frame_count + lost_frame_count}/{len(audit_df)} frames of {sample_id} "
        f"could not be subtracted ({failed_frame_fraction:.1%}; limit {max_failed_frame_fraction:.1%}): "
        f"{guard_failed_frame_count} tripped the subtracted-foreground guard, "
        f"{lost_frame_count} had their BG frame lost at acquisition."
    )
    if lost_frame_count:
        logger.warning(
            f"External background for {sample_id}: skipped FG frame(s) "
            f"{[int(fg_row['frame']) for fg_row in fg_rows_with_bg_lost]} - the BG run's metadata log records "
            f"their BG frame as lost at acquisition."
        )
    if failed_frame_fraction > max_failed_frame_fraction:
        raise ProcessingError(
            f"External background for {sample_id}: {failed_frame_fraction:.1%} of frames could not be "
            f"subtracted (limit {max_failed_frame_fraction:.1%}): {guard_failed_frame_count} tripped the "
            f"subtracted-foreground guard (the BG run does not match the FG scene) and {lost_frame_count} "
            f"had their BG frame lost at acquisition. See {sample_id}_stage1_bg_pairing.csv."
        )

def _analyze_single_subtracted_image_global_thresh(
    image_to_analyze_gray_uint8: np.ndarray,
    config_stage1_particle_analysis: Dict[str, Any],
    global_threshold_config: Dict[str, Any],
    foreground_floor_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """Detect particles on a subtracted frame using global thresholding settings.

    ``foreground_floor_mask`` (from :func:`assess_subtracted_foreground`) is ANDed into the
    Otsu mask so the detection cannot fall below the shared fixed foreground floor. The
    subtraction-failed guard is applied upstream in :func:`analyze_subtracted_image`.
    """
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
    binary_mask = generate_binary_mask_from_float_image(img_float_for_segment, global_threshold_config)
    if binary_mask is None: logger.error("S1 Subtracted: Failed to generate binary mask via helper."); return particles_data, binary_mask_display_uint8
    if foreground_floor_mask is not None:
        binary_mask = binary_mask & foreground_floor_mask
    binary_mask_uint8 = binary_mask.astype(np.uint8) * 255; binary_mask_display_uint8 = binary_mask_uint8.copy()
    img_for_contours = binary_mask_uint8
    blur_kernel_size_sub_binary = config_stage1_particle_analysis.get('microorganism_blur_kernel', 0)
    if blur_kernel_size_sub_binary > 0 and blur_kernel_size_sub_binary % 2 == 1:
        blurred_binary = apply_blur(binary_mask_uint8, blur_kernel_size_sub_binary, "median")
        if blurred_binary is not None: img_for_contours = blurred_binary; logger.debug(f"S1 Subtracted: Applied blur (kernel {blur_kernel_size_sub_binary}) to binary mask.")
    contours = find_contours(img_for_contours, mode="external", method="simple")
    if not contours: logger.debug("S1 Subtracted: No contours found after thresholding/blur."); return particles_data, binary_mask_display_uint8
    min_area = config_stage1_particle_analysis['min_microorganism_area']
    valid_contours = filter_contours_by_area(contours, min_area)
    particles_data = [m for cnt in valid_contours if (m := calculate_contour_measurements(cnt))]
    logger.debug(f"S1 Subtracted: Found {len(particles_data)} particles after area filter.")
    return particles_data, binary_mask_display_uint8

def _suppress_roi_background(
    roi_signal_float: np.ndarray,
    background_floor_percentile: float,
) -> np.ndarray:
    """Shift a bright-particle-on-background ROI so its background floor sits at ~0.

    The local refinement thresholds at ``Otsu * otsu_threshold_multiplier`` with a
    multiplier < 1. That only behaves when the background floor is already near zero
    -- true on the background-subtracted frame, but NOT on the unsubtracted frame
    when the raw background is mid-gray rather than near-white. There the (inverted)
    background sits well above zero, the multiplied threshold lands below it, and the
    whole square ROI floods to foreground (perfect-square false particles).
    Subtracting a robust low-percentile floor restores the near-zero-background
    assumption independently of absolute exposure/illumination level. It is a near
    no-op when the background is already near zero (bright collimated or darkfield).
    """
    floor = float(np.percentile(roi_signal_float, background_floor_percentile))
    shifted = roi_signal_float - floor
    np.clip(shifted, 0.0, 1.0, out=shifted)
    return shifted


def _analyze_unsubtracted_locally(
    full_unsub_prepped_uint8: np.ndarray,
    candidate_particles_subtracted: List[Dict[str, Any]],
    adaptive_padding_factor: float,
    analysis_config: Dict[str, Any],
    global_threshold_config: Dict[str, Any],
    source_illumination: str
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]]:
    """Refine subtracted detections by re-segmenting local ROIs on the unsubtracted image."""
    if full_unsub_prepped_uint8 is None: return [], None, (None, None, None)
    all_local_particle_data = []; composite_binary_mask_full_image = np.zeros_like(full_unsub_prepped_uint8, dtype=np.uint8)
    first_roi_img_2d_for_display, first_roi_hist_data_ravel, first_roi_effective_thresh_val_float = None, None, None
    min_area_unsub = analysis_config.get('min_particle_area', 50); h_full, w_full = full_unsub_prepped_uint8.shape
    background_floor_percentile = float(analysis_config.get('background_floor_percentile', 25))
    max_roi_foreground_fraction = float(analysis_config.get('max_roi_foreground_fraction', 0.85))
    logger.debug(f"Local Unsub: Processing {len(candidate_particles_subtracted)} candidate ROIs with adaptive padding factor {adaptive_padding_factor}")
    for i, cand_p_sub in enumerate(candidate_particles_subtracted):
        if 'X' not in cand_p_sub or 'Y' not in cand_p_sub: continue
        cx, cy = int(cand_p_sub['X']), int(cand_p_sub['Y'])
        candidate_w = max(float(cand_p_sub.get('bounding_w', 1.0)), 1.0)
        candidate_h = max(float(cand_p_sub.get('bounding_h', 1.0)), 1.0)
        pad_x = int(np.ceil(candidate_w * adaptive_padding_factor))
        pad_y = int(np.ceil(candidate_h * adaptive_padding_factor))
        adaptive_half_crop = int(np.ceil(max(candidate_w + 2 * pad_x, candidate_h + 2 * pad_y) / 2.0))
        x1, y1 = max(0, cx - adaptive_half_crop), max(0, cy - adaptive_half_crop); x2, y2 = min(w_full, cx + adaptive_half_crop), min(h_full, cy + adaptive_half_crop)
        roi_img_uint8 = full_unsub_prepped_uint8[y1:y2, x1:x2]
        if roi_img_uint8.size == 0 or roi_img_uint8.shape[0] < 3 or roi_img_uint8.shape[1] < 3: continue
        roi_float_original_2d = skimage.util.img_as_float(roi_img_uint8); roi_float_for_segment_2d = roi_float_original_2d.copy()
        if source_illumination == 'collimated': roi_float_for_segment_2d = 1.0 - roi_float_for_segment_2d
        # Pull the local background floor down to ~0 so the sub-1.0 Otsu multiplier
        # cannot drop below an elevated (mid-gray) background and flood the ROI.
        roi_float_for_segment_2d = _suppress_roi_background(roi_float_for_segment_2d, background_floor_percentile)
        current_roi_hist_data_ravel = roi_float_for_segment_2d.ravel()
        local_binary_mask_roi = generate_binary_mask_from_float_image(roi_float_for_segment_2d, global_threshold_config)
        if local_binary_mask_roi is None: logger.error(f"S1 Local Unsub: Failed to generate binary mask for ROI {i+1}."); continue
        # Safety net: a mask that still fills most of the square ROI is a degenerate
        # flood, not a particle. Drop it so it cannot surface as a perfect-square false
        # detection (circularity == pi/4). Real particles stay well under this fraction.
        roi_foreground_fraction = float(np.count_nonzero(local_binary_mask_roi)) / float(local_binary_mask_roi.size)
        if roi_foreground_fraction > max_roi_foreground_fraction:
            logger.debug(
                f"S1 Local Unsub: ROI {i+1} foreground fraction {roi_foreground_fraction:.2f} exceeds "
                f"{max_roi_foreground_fraction:.2f}; skipping degenerate (flooded) segmentation."
            )
            continue
        local_threshold_value_for_data = np.nan
        if i == 0:
            first_roi_img_2d_for_display, first_roi_hist_data_ravel = roi_float_for_segment_2d.copy(), current_roi_hist_data_ravel
            if global_threshold_config.get('thresholding_mode') == 'otsu':
                if roi_float_for_segment_2d.min() != roi_float_for_segment_2d.max():
                    try: otsu_val_disp, otsu_mult_disp = skimage.filters.threshold_otsu(roi_float_for_segment_2d), global_threshold_config.get('otsu_threshold_multiplier', 0.45); first_roi_effective_thresh_val_float, local_threshold_value_for_data = otsu_val_disp * otsu_mult_disp, otsu_val_disp * otsu_mult_disp
                    except ValueError: pass
            elif global_threshold_config.get('thresholding_mode') == 'fixed': first_roi_effective_thresh_val_float, local_threshold_value_for_data = global_threshold_config.get('fixed_threshold_value', 70) / 255.0, global_threshold_config.get('fixed_threshold_value', 70) / 255.0
        else:
            if global_threshold_config.get('thresholding_mode') == 'otsu':
                 if roi_float_for_segment_2d.min() != roi_float_for_segment_2d.max():
                    try: otsu_val_data, otsu_mult_data = skimage.filters.threshold_otsu(roi_float_for_segment_2d), global_threshold_config.get('otsu_threshold_multiplier', 0.45); local_threshold_value_for_data = otsu_val_data * otsu_mult_data
                    except ValueError: pass
            elif global_threshold_config.get('thresholding_mode') == 'fixed': local_threshold_value_for_data = global_threshold_config.get('fixed_threshold_value', 70) / 255.0
        local_binary_mask_roi_uint8 = local_binary_mask_roi.astype(np.uint8) * 255
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

def _process_particle_coordinates_for_tracking_df(input_df: pd.DataFrame, pixel_to_mm_factor: float):
    """Convert fused particle coordinates to millimeters and extract tracking metadata."""
    logger.info("Tracking Step 1: Loading and processing fused particle data for tracking.")
    df = input_df.copy()
    if df.empty:
        logger.warning("Input DataFrame for tracking is empty. No particles to process.")
        return pd.DataFrame()
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

def _track_particles_3d(processed_df, search_range_mm, memory_frames, min_track_length, track_in_2d=False):
    """Run trackpy to build particle trajectories from processed coordinates.

    Links in 3D ``(x, y, z)`` by default. When ``track_in_2d`` is True, links in
    the ``(x, y)`` plane only. On focus-sweep acquisitions ``z`` is the ETL focal
    depth: it is identical for every particle within a frame and steps ~0.2 mm per
    frame, so it carries no particle-discriminating information yet consumes the
    ``search_range`` budget and throttles ``memory`` (a gap of *k* frames drifts z
    by ~0.2*k mm, breaking the link once that reaches ``search_range``). Dropping z
    from the link metric fixes both. ``z`` is always retained in the returned frame
    for the downstream depth merge and reconstruction, regardless of this flag.
    """
    if processed_df.empty:
        logger.warning("Input DataFrame for tracking is empty. Skipping tracking.")
        return pd.DataFrame()
    pos_cols = ['x', 'y'] if track_in_2d else ['x', 'y', 'z']
    logger.info(f"Tracking Step 2: Performing {'2D' if track_in_2d else '3D'} ({', '.join(pos_cols)}) tracking with trackpy...")
    tp_df = processed_df.rename(columns={'x_mm': 'x', 'y_mm': 'y', 'z_mm': 'z'})
    tp_df['frame'] = tp_df['frame'].astype(int)
    # z is always coerced and retained (the downstream merge reconstructs depth from
    # it) even when it is excluded from the link metric in 2D mode.
    for col in ('x', 'y', 'z'):
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

def _build_tracking_figure(
    tracked_df: pd.DataFrame,
    extent_mm: Tuple[float, float, float]
) -> Optional[plt.Figure]:
    """Create a 3D matplotlib figure summarizing all tracked particle paths.

    Axes are pinned to ``extent_mm`` (the full imaged volume) rather than
    autoscaled to the tracks, so the plot always shows the whole field of view
    and Z sweep and stays comparable across samples. Each track's grey copy on
    the floor shows its x/y position. Build and save inside
    :func:`track_figure_style`, which sets the line sizes.
    """
    if tracked_df.empty:
        return None
    x_max_mm, y_max_mm, z_max_mm = extent_mm
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D particle tracks")
    ax.set_xlabel("x (mm)", labelpad=14)
    ax.set_ylabel("y (mm)", labelpad=14)
    ax.set_zlabel("z (mm)", labelpad=14)
    tracks = [track for _, track in tracked_df.groupby('track_id')]
    for track in tracks:
        draw_floor_shadow(ax, track['x_mm'].values, track['y_mm'].values)
    for rank, track in enumerate(tracks):
        plot_track_line(ax, (track['x_mm'], track['y_mm'], track['z_mm']), rank)
    ax.set_xlim(0, x_max_mm)
    ax.set_ylim(0, y_max_mm)
    ax.set_zlim(0, z_max_mm)
    plt.tight_layout()
    return fig


def _build_tracking_xy_figure(
    tracked_df: pd.DataFrame,
    extent_mm: Tuple[float, float, float]
) -> Optional[plt.Figure]:
    """Create a 2D XY-projection figure of all tracked particle paths.

    Companion to :func:`_build_tracking_figure`. The 3D view is subject to
    perspective distortion (a point's screen position depends on its Z), which
    can make the XY layout look different from how it really is. This flat
    projection plots the same trajectories on the true XY plane so the spatial
    layout can be read and compared against Stage 2 without that distortion.

    Axes are pinned to the full field of view from ``extent_mm``; the aspect is
    adjusted via the axes box because the ``datalim`` alternative would stretch
    those fixed limits back out to fill the box. Tracks keep the colours they
    have in the 3D view.
    """
    if tracked_df.empty:
        return None
    x_max_mm, y_max_mm, _ = extent_mm
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title("XY projection of particle tracks")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    for rank, (_, track) in enumerate(tracked_df.groupby('track_id')):
        plot_track_line(ax, (track['x_mm'], track['y_mm']), rank)
    ax.set_xlim(0, x_max_mm)
    ax.set_ylim(0, y_max_mm)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig


def _sweep_period_frames(images_df: pd.DataFrame) -> int:
    """Frames per focus-sweep cycle of a recording, from the spacing of its fd_mm == 0 frames.

    Zigzag and sawtooth sweeps both return to fd_mm == 0 once per cycle. A run shorter
    than one cycle has a single fd_mm == 0 frame; every gap in it is then shorter than
    the run, so the run length is returned.

    Raises:
        DataValidationError: no fd_mm == 0 frame, so the sweep cycle is unknown.
    """
    fd0_frames = np.sort(images_df.loc[images_df['fd_mm'] == 0, 'frame'].to_numpy())
    if fd0_frames.size == 0:
        raise DataValidationError("No fd_mm == 0 frame found: the focus-sweep period cannot be determined.")
    if fd0_frames.size == 1:
        return int(images_df['frame'].max() - images_df['frame'].min() + 1)
    return int(np.median(np.diff(fd0_frames)))


def run_particle_tracking(
    fused_df: pd.DataFrame,
    sample_id: str,
    tracking_config: Dict[str, Any],
    memory_frames: int,
    search_range_mm: float,
) -> pd.DataFrame:
    """Perform particle tracking on fused detections and return data instead of writing files.

    Returns the detections that belong to a track (with ``track_id``); when no track
    survives, the processed detections with ``track_id`` NaN. ``memory_frames`` (how long
    a particle may go undetected) and ``search_range_mm`` (linking radius) come from
    ``tracking`` for time-shift runs and are the static-object values for
    external-background runs (see :func:`run_stage_01_initial_detection`). The track
    figures are built by the caller from the final (object-filtered) tracks.
    """
    logger.info(f"--- Post-S1 Tracking initiated for: {sample_id} ---")
    pixel_size_um = tracking_config.get('pixel_size_um')
    magnification = tracking_config.get('magnification')
    min_track_length = tracking_config.get('min_track_length')
    if not all([pixel_size_um, magnification, search_range_mm, memory_frames, min_track_length]):
        raise ConfigError("One or more required tracking parameters are missing from the configuration.")
    # Link in the XY plane only when requested. A plain .get default (not part of the
    # all() check above) because False is a valid, intended value; absent key -> 3D.
    track_in_2d = bool(tracking_config.get('track_in_2d', False))

    processed_data = _process_particle_coordinates_for_tracking_df(
        fused_df, get_pixel_to_mm_factor(tracking_config)
    )
    if processed_data.empty:
        logger.warning("Tracking workflow stopped: No data after initial processing.")
        return processed_data

    tracked_data = _track_particles_3d(
        processed_data, search_range_mm, memory_frames, min_track_length, track_in_2d=track_in_2d
    )
    if tracked_data.empty:
        logger.warning("Tracking workflow stopped: No tracks survived the filtering process.")
        processed_data['track_id'] = np.nan
        return processed_data

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

    logger.info(f"--- Post-S1 Tracking for {sample_id} COMPLETED ---")
    return final_tracked_particles_df


def _drop_lint_detections(
    fused_df: pd.DataFrame,
    lint_max_solidity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop lint-shaped detections before tracking; return (kept detections, dropped detections).

    A detection is lint-shaped when ``s1_solidity <= lint_max_solidity``: thin, curled or
    branched lint fills little of its convex hull, but only while it is in focus. Dropping
    exactly those detections removes the sharp frames of an isolated lint object, so its
    remaining (blurred) track has no sharpness peak in Stage 2. Animals keep their compact
    shape, so at most a stray frame of theirs goes, which the sweep-period tracking memory
    bridges. Detections without a solidity value are kept.
    """
    lint_mask = fused_df['s1_solidity'] <= lint_max_solidity
    return fused_df.loc[~lint_mask].reset_index(drop=True), fused_df.loc[lint_mask].reset_index(drop=True)


def run_stage_01_initial_detection(
    image_dir_path: str,
    output_dir_stage1: str,
    sample_id: str,
    config: Dict[str, Any],
    bg_image_dir_path: Optional[str] = None,
) -> Optional[str]:
    """Input: image dir, output dir, sample id, config. Output: Stage 1 CSV path or None.

    Background source: without ``bg_image_dir_path`` each frame is paired with a
    time-shifted frame of the same recording, so only particles that moved are found.
    With it, each frame is paired with the same frame of that separate specimen-free
    recording (external background), so non-moving specimens are found as well. Every
    step after pairing (subtraction gate, refinement, fusion, tracking) is shared.
    """
    logger.info(f"--- Stage 1: Initial Detection for Sample ID: {sample_id} ---")
    config_s1 = config['stage_01_initial_detection']
    system_cfg = config['system']
    source_image_illumination = system_cfg['source_image_illumination_type']
    global_threshold_config = system_cfg['thresholding']
    subtracted_foreground_config = system_cfg['subtracted_foreground']

    config_black_plate_well = config_s1['black_plate_well_analysis']
    enable_brightness_adjustment = config_black_plate_well['enable_brightness_adjustment']

    particle_analysis_subtracted_config = config_s1['particle_analysis']
    unsubtracted_analysis_config = config_s1['unsubtracted_particle_analysis']
    particle_comparison_config = config_s1['particle_comparison']

    general_well_masking_cfg = config_s1['well_masking']

    images_df = parse_frame_filenames(image_dir_path)
    if images_df is None or images_df.empty: return None

    if bg_image_dir_path is None:
        external_background_config = None
        fg_rows_with_bg_lost: List[pd.Series] = []
        sweep_period_frames = None
        static_search_range_mm = None
        lint_max_solidity = None
        images_df_filtered = _filter_image_df(images_df.copy(), trim_to_last_fd0=True)
        if images_df_filtered.empty: return None
        frame_pairs = _build_time_shift_pairs(
            images_df_filtered, config_s1, float(config['acquisition']['camera']['fps'])
        )
    else:
        external_background_config = config_s1['external_background']
        images_df_filtered = _filter_image_df(images_df.copy(), trim_to_last_fd0=False)
        if images_df_filtered.empty: return None
        bg_images_df = parse_frame_filenames(bg_image_dir_path)
        if bg_images_df is None or bg_images_df.empty:
            raise DataValidationError(f"No background images found in {bg_image_dir_path!r} for sample {sample_id}.")
        frame_pairs, fg_rows_with_bg_lost = _build_external_background_pairs(
            images_df_filtered,
            bg_images_df,
            os.path.join(bg_image_dir_path, config['acquisition']['paths']['log_filename']),
        )
        logger.info(
            f"Stage 1: External background from {bg_image_dir_path!r}: {len(frame_pairs)} FG/BG frame pairs "
            f"matched by frame index and focal plane; {len(fg_rows_with_bg_lost)} FG frame(s) without BG "
            f"(BG frame lost at acquisition)."
        )
        # Specimens seen against an external BG are static, and many are only detectable
        # while the sweep passes their focal plane: they vanish for up to one sweep cycle and
        # reappear in place. Tracking therefore bridges one full cycle (instead of
        # tracking.memory_frames, tuned for moving specimens), so each object keeps one track
        # rather than one per pass, and links within the static-object radius
        # (external_background.static_search_range_mm instead of tracking.search_range_mm):
        # with a one-cycle memory, the movers' radius lets a waiting track absorb a
        # neighbouring object. Bridging passes requires XY linking: between passes z
        # differs by up to the whole scan height.
        if config_s1['tracking']['enabled'] and not config_s1['tracking']['track_in_2d']:
            raise ConfigError(
                "External background requires stage_01_initial_detection.tracking.track_in_2d: true "
                "(tracks are bridged across focus-sweep passes, where z differs by up to the scan height)."
            )
        sweep_period_frames = _sweep_period_frames(images_df)
        logger.info(
            f"Stage 1: External background: tracking memory = one focus-sweep period = {sweep_period_frames} frames."
        )
        static_search_range_mm = float(external_background_config['static_search_range_mm'])
        lint_max_solidity = float(external_background_config['lint_max_solidity'])

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
        "subtracted_foreground_config": subtracted_foreground_config,
        "unsubtracted_analysis_config": unsubtracted_analysis_config,
        "particle_comparison_config": particle_comparison_config,
        "external_background_config": external_background_config,
        "sample_id": sample_id,
    }

    all_fused_particles_for_sample_list: List[pd.DataFrame] = []
    pair_diagnostics_records: List[Dict[str, Any]] = []
    reusable_masks: Dict[str, Any] = {}

    for fg_row, bg_row in frame_pairs:
        logger.debug(f"Processing S1 Frame: {fg_row['filename']} (BG: {bg_row['filename']})")
        fused_df_frame, pair_diagnostics = process_frame_pair(
            fg_row,
            bg_row,
            frame_pair_config,
            ffc_assets,
            reusable_masks
        )
        if pair_diagnostics is not None:
            pair_diagnostics_records.append(pair_diagnostics)

        # Collect fused detections for the sample.
        if not fused_df_frame.empty:
            all_fused_particles_for_sample_list.append(fused_df_frame)

    if external_background_config is not None:
        _audit_external_background_pairs(
            pair_diagnostics_records,
            fg_rows_with_bg_lost,
            float(external_background_config['max_failed_frame_fraction']),
            output_dir_stage1,
            sample_id,
        )

    if not all_fused_particles_for_sample_list:
        logger.warning(f"No particles detected or fused for the entire sample {sample_id}.")
        final_sample_df = pd.DataFrame(columns=['SAMPLE_ID', 'source_image_filename', 'X', 'Y'])
    else:
        final_sample_df = pd.concat(all_fused_particles_for_sample_list, ignore_index=True)
    
    logger.info(f"Total fused particles for sample {sample_id}: {len(final_sample_df)}")

    if lint_max_solidity is not None and not final_sample_df.empty:
        final_sample_df, lint_detections_df = _drop_lint_detections(final_sample_df, lint_max_solidity)
        save_dataframe_to_csv(lint_detections_df, output_dir_stage1, f"{sample_id}_stage1_rejected_lint_detections.csv")
        logger.info(
            f"Stage 1: Dropped {len(lint_detections_df)} lint-shaped detections (solidity <= {lint_max_solidity}) "
            f"before tracking for {sample_id}; {len(final_sample_df)} remain."
        )

    final_output_filename = f"{sample_id}_stage1_tracked_particles.csv"
    final_output_path = os.path.join(output_dir_stage1, final_output_filename)
    
    tracking_config = config_s1['tracking']
    if tracking_config['enabled']:
        # Track fused particles and defer file I/O to the stage wrapper.
        try:
            if external_background_config is not None:
                memory_frames, search_range_mm = sweep_period_frames, static_search_range_mm
            else:
                memory_frames = tracking_config.get('memory_frames')
                search_range_mm = tracking_config.get('search_range_mm')
            tracked_df = run_particle_tracking(
                final_sample_df, sample_id, tracking_config, memory_frames, search_range_mm
            )
        except Exception as e:
            logger.exception(f"Particle tracking failed for sample {sample_id}: {e}")
            tracked_df = pd.DataFrame()

        if tracked_df.empty and not final_sample_df.empty:
            logger.warning("Particle tracking returned no tracks; saving fused detections with empty track IDs.")
            final_sample_df['track_id'] = np.nan
            save_dataframe_to_csv(final_sample_df, output_dir_stage1, final_output_filename)
        else:
            save_dataframe_to_csv(tracked_df, output_dir_stage1, final_output_filename)

        tracks_for_figures_df = (
            tracked_df.dropna(subset=['track_id']) if 'track_id' in tracked_df.columns else pd.DataFrame()
        )
        # Outside the tracking try-block: a frame that cannot be read must not pass as a tracking failure.
        extent_mm = get_scan_extent_mm(config, image_dir_path)
        with track_figure_style(config):
            tracking_fig = _build_tracking_figure(tracks_for_figures_df, extent_mm) if not tracks_for_figures_df.empty else None
            tracking_xy_fig = _build_tracking_xy_figure(tracks_for_figures_df, extent_mm) if not tracks_for_figures_df.empty else None
            if tracking_fig is not None:
                figure_3d_path = os.path.join(output_dir_stage1, f"{sample_id}_stage1_tracks_3d.png")
                # The tight bbox of a 3D axes omits its z label; include it explicitly.
                tracking_fig.savefig(figure_3d_path, bbox_extra_artists=[tracking_fig.axes[0].zaxis.label])
                plt.close(tracking_fig)
                # Tracking figure documents the trajectories visually.
            if tracking_xy_fig is not None:
                figure_xy_path = os.path.join(output_dir_stage1, f"{sample_id}_stage1_tracks_xy.png")
                tracking_xy_fig.savefig(figure_xy_path)
                plt.close(tracking_xy_fig)
                # Flat XY projection makes the spatial layout directly comparable to Stage 2.
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
