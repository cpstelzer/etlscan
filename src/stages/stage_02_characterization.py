# src/stages/stage_02_characterization.py
"""Stage 2: crop, segment and measure every Stage-1 particle, select each track's in-focus
frames, reconstruct its depth and write the feature tables, crops and diagnostics.

The stage flow lives here; the focus-peak detection (``src/core/focus_peak_detection.py``),
the depth reconstruction (``src/core/depth_reconstruction.py``) and the plots
(``src/visualization/focus_diagnostics.py``, ``track_maps.py``) are separate modules.
"""
import logging
import numbers
import os
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import skimage.util

from src.utils import (
    load_dataframe_from_csv, save_dataframe_to_csv, build_crop_filename
)
from src.utils.acquisition_metadata import get_scan_extent_mm, read_recording_parameters
from src.utils.config_loader import get_unified_segmentation_params, require_config_keys
from src.utils.path_helpers import extract_frame_from_filename
from src.utils.common_exceptions import ConfigError
from src.core.depth_reconstruction import (
    add_reconstructed_depth,
    attach_reconstructed_depth_to_subset,
    flag_depth_knot_speed_outliers,
)
from src.core.focus_peak_detection import FOCUS_CONFIG_PATH, filter_focused_frames
from src.core.image_io_operations import load_image, convert_to_grayscale
from src.core.particle_operations import (
    crop_particle_from_arrays,
    detect_cropped_particle_on_boundary,
    normalized_crop_config,
)
from src.core.segmentation import segment_main_body_from_image_float
from src.core.track_columns import FOCUS_Z_COLUMN
from src.visualization.figure_style import figure_style, track_figure_style
from src.visualization.focus_diagnostics import plot_diagnostics_summary, plot_example_tracks_with_collages
from src.visualization.track_maps import (
    SIDE_VIEW_HORIZONTAL_AXES,
    plot_3d_reconstructed,
    plot_side_view_reconstructed,
    plot_xy_reconstructed,
)
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors

logger = logging.getLogger(__name__)


# Keys of that section the batch stage reads (sharpness_min_threshold is the realtime gate).
FOCUS_CONFIG_KEYS = (
    'enabled', 'frame_column', 'track_column', 'sharpness_column', 'peak_detection',
    'max_tracks_to_plot', 'trajectory_smoothing_factor', 'min_points_for_spline',
    'depth_knot_speed_gate', 'save_plots', 'crop_filename_prefix',
)


def run_stage_02_characterization(
    input_s1_csv_path: str,
    image_dir_path: str,
    output_dir_stage2: str,
    sample_id: str,
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Performs detailed characterization for every particle candidate from Stage 1
    using a modular feature calculator architecture.
    """
    logger.info(f"--- Stage 2: Universal Characterization for Sample ID: {sample_id} ---")
    recording = read_recording_parameters(config, image_dir_path)
    logger.info(
        "Stage 2: recording at %.2f fps, %.6f mm per pixel (optics from %s).",
        recording.frame_rate_fps, recording.pixel_to_mm_factor, recording.optics_source,
    )
    s2_config = config['stage_02_characterization']
    segmentation_params = get_unified_segmentation_params(config)

    s1_particles_df = load_dataframe_from_csv(input_s1_csv_path)
    if s1_particles_df is None or s1_particles_df.empty:
        logger.warning(f"No particles loaded from Stage 1 CSV: {input_s1_csv_path}. Aborting Stage 2.")
        return None

    cropping_config = s2_config['cropping']

    try:
        normalized_cropping = normalized_crop_config(cropping_config)
        logger.info(f"Initialized ParticleCropper with mode: '{normalized_cropping['mode']}'")
    except ConfigError as e:
        logger.error(f"Failed to initialize ParticleCropper: {e}")
        return None

    boundary_threshold_params = {
        'thresholding_mode': segmentation_params['thresholding_mode'],
        'otsu_threshold_multiplier': segmentation_params['otsu_threshold_multiplier'],
        'fixed_threshold_value': segmentation_params['fixed_threshold_value'],
    }
    illumination_mode = str(segmentation_params['source_image_illumination_type']).strip().lower()

    s1_particles_df['s1_original_index'] = s1_particles_df.index

    # crop
    crop_records: List[dict] = []
    for image_filename, group in s1_particles_df.groupby('source_image_filename'):
        logger.info(f"Characterizing {len(group)} particles from image: {image_filename}")
        full_image_path = os.path.join(image_dir_path, image_filename)
        
        main_image_color = load_image(full_image_path, mode='color')
        main_image_gray = convert_to_grayscale(main_image_color)

        if main_image_color is None or main_image_gray is None:
            logger.warning(f"Failed to load image {full_image_path}. Skipping {len(group)} particles.")
            continue

        frame_num = extract_frame_from_filename(image_filename)
        crop_records.extend(
            crop_particles(
                main_image_color,
                main_image_gray,
                group,
                normalized_cropping,
                sample_id,
                image_filename,
                frame_num,
                illumination_mode,
                boundary_threshold_params,
            )
        )

    if not crop_records:
        logger.warning(f"No particles were successfully characterized for sample {sample_id}.")
        return None

    crops_df = pd.DataFrame(crop_records)

    # segment
    segmented_df = segment_particles(crops_df, segmentation_params)

    # measure
    features_df = compute_features(segmented_df, segmentation_params, s2_config)

    if features_df.empty:
        logger.warning(f"No particles were successfully characterized for sample {sample_id}.")
        return None

    s1_particles_df_indexed = s1_particles_df.copy()
    overlapping_cols = set(features_df.columns) & set(s1_particles_df_indexed.columns)
    overlapping_cols.discard('s1_original_index')
    features_for_merge = features_df.drop(columns=list(overlapping_cols)) if overlapping_cols else features_df

    comprehensive_df = pd.merge(
        s1_particles_df_indexed, features_for_merge, on='s1_original_index', how='right'
    )
    comprehensive_df.drop(columns=['s1_original_index'], inplace=True, errors='ignore')

    focus_cfg = require_config_keys(s2_config['focussed_particles'], FOCUS_CONFIG_KEYS, FOCUS_CONFIG_PATH)
    track_col = focus_cfg['track_column']

    # focus filter
    filtered_df, diagnostics_df = select_focused_tracks(comprehensive_df, focus_cfg)

    # flag in-focus depths that no neighbouring knot of their track supports (time between
    # knots from the recording's frame rate, as for the Stage-1 background offset)
    filtered_df = flag_depth_knot_speed_outliers(
        filtered_df, comprehensive_df, focus_cfg, recording.frame_rate_fps
    )
    # reconstruct continuous per-frame depth (z) from the sparse in-focus points
    comprehensive_df = add_reconstructed_depth(comprehensive_df, filtered_df, focus_cfg)
    # carry that per-frame depth onto the in-focus selections so it also reaches
    # filtered_in_focus.csv and final_particles.csv (a pass-through, not an ML feature)
    filtered_df = attach_reconstructed_depth_to_subset(
        filtered_df, comprehensive_df, track_col, focus_cfg['frame_column']
    )

    final_particles_df = pd.DataFrame()
    if filtered_df is not None and not filtered_df.empty:
        final_particles_df = select_final_particles_by_area(
            filtered_df,
            track_col,
            area_col='main_area_filled'
        )

    # save (figure_style governs the focus-diagnostic plots written here; the
    # reconstructed-trajectory maps inside switch to track_figure_style)
    with figure_style(config):
        saved_csv_path = emit_stage2_artifacts(
            comprehensive_df,
            crops_df,
            filtered_df,
            diagnostics_df,
            final_particles_df,
            output_dir_stage2,
            sample_id,
            normalized_cropping,
            image_dir_path,
            focus_cfg,
            get_scan_extent_mm(image_dir_path, recording.pixel_to_mm_factor),
            bool(cropping_config['save_all_particle_crops']),
            config,
        )

    logger.info(f"--- Stage 2: Characterization for {sample_id} COMPLETED ---")
    return saved_csv_path


def crop_particles(
    main_image_color: np.ndarray,
    main_image_gray: np.ndarray,
    particles_df: pd.DataFrame,
    crop_config: Dict[str, Any],
    sample_id: str,
    image_filename: str,
    frame_num: Optional[int],
    illumination_mode: str,
    boundary_threshold_params: Dict[str, Any]
) -> List[dict]:
    """Crop all particles from an image and annotate metadata for downstream features."""
    if frame_num is None:
        logger.warning(
            "Could not parse frame from %s. Skipping %d particles.",
            image_filename,
            len(particles_df)
        )
        return []

    crop_records: List[dict] = []
    for _, s1_particle_row in particles_df.iterrows():
        original_index = s1_particle_row['s1_original_index']
        track_label = _format_track_label(s1_particle_row, fallback_suffix=original_index)

        x_raw, y_raw = s1_particle_row.get('X'), s1_particle_row.get('Y')
        x_int = int(x_raw) if pd.notna(x_raw) else 0
        y_int = int(y_raw) if pd.notna(y_raw) else 0
        crop_filename = build_crop_filename(sample_id, track_label, frame_num, x_int, y_int)

        crop_gray, crop_metadata = crop_particle_from_arrays(
            crop_config,
            main_image_color,
            main_image_gray,
            s1_particle_row
        )

        if crop_gray is None:
            logger.warning(f"Failed to crop particle at index {original_index}. Skipping.")
            continue

        crop_identifier_for_logs = crop_filename or "unsaved_crop"
        is_on_crop_boundary = detect_cropped_particle_on_boundary(
            crop_gray,
            crop_identifier_for_logs,
            illumination_mode,
            boundary_threshold_params,
        )

        crop_records.append({
            's1_original_index': original_index,
            'track_id': s1_particle_row.get('track_id'),
            'X': s1_particle_row.get('X'),
            'Y': s1_particle_row.get('Y'),
            'frame': s1_particle_row.get('frame', frame_num),
            'source_image_filename': image_filename,
            'crop_gray': crop_gray,
            'crop_metadata': crop_metadata,
            'crop_filename': crop_filename,
            'is_on_crop_boundary': bool(is_on_crop_boundary),
        })

    return crop_records


def segment_particles(
    crops_df: pd.DataFrame,
    segmentation_params: Dict[str, Any]
) -> pd.DataFrame:
    """Generate raw and filled masks for every crop to prepare for feature extraction."""
    segmented_records: List[Dict[str, Any]] = []

    for _, crop_row in crops_df.iterrows():
        crop_gray = crop_row.get('crop_gray')
        if crop_gray is None:
            continue

        crop_float_orig = skimage.util.img_as_float(crop_gray)
        raw_mask, filled_mask, _ = segment_main_body_from_image_float(crop_float_orig, segmentation_params)

        if filled_mask is None or raw_mask is None:
            logger.warning(
                "High-precision segmentation failed for crop %s. Skipping feature calculation.",
                crop_row.get('crop_filename')
            )
            continue

        record = crop_row.to_dict()
        record['raw_mask'] = raw_mask
        record['filled_mask'] = filled_mask
        segmented_records.append(record)

    return pd.DataFrame(segmented_records)


def compute_features(
    segmented_df: pd.DataFrame,
    segmentation_params: Dict[str, Any],
    s2_config: Dict[str, Any]
) -> pd.DataFrame:
    """Calculate quality, morphology, and shape descriptors for segmented particles."""
    feature_rows: List[Dict[str, Any]] = []

    for _, seg_row in segmented_df.iterrows():
        crop_gray = seg_row.get('crop_gray')
        raw_mask = seg_row.get('raw_mask')
        filled_mask = seg_row.get('filled_mask')

        if crop_gray is None or raw_mask is None or filled_mask is None:
            continue

        particle_features = {
            's1_original_index': seg_row['s1_original_index'],
            'crop_filename': seg_row.get('crop_filename'),
            'is_on_crop_boundary': seg_row.get('is_on_crop_boundary', False),
            'frame': seg_row.get('frame'),
            'track_id': seg_row.get('track_id'),
            'X': seg_row.get('X'),
            'Y': seg_row.get('Y'),
            'source_image_filename': seg_row.get('source_image_filename'),
        }

        quality_features = calculate_quality_features(crop_gray, None, s2_config)
        particle_features.update(quality_features)

        crop_float_orig = skimage.util.img_as_float(crop_gray)
        intensity_img_for_props = crop_float_orig.copy()
        if segmentation_params['source_image_illumination_type'] == 'collimated':
            intensity_img_for_props = 1.0 - intensity_img_for_props
        morphology_features = calculate_detailed_body_morphology(raw_mask, filled_mask, intensity_img_for_props)
        renamed_morphology = {k.replace('_s4', ''): v for k, v in morphology_features.items()}
        particle_features.update(renamed_morphology)

        shape_features = calculate_shape_descriptors(crop_gray, filled_mask, s2_config)
        particle_features.update(shape_features)

        feature_rows.append(particle_features)

    return pd.DataFrame(feature_rows)


def select_focused_tracks(
    comprehensive_df: pd.DataFrame,
    focus_cfg: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Limit each track to frames near sharpness peaks and emit diagnostics."""
    if not focus_cfg['enabled']:
        return None, None

    frame_col = focus_cfg['frame_column']
    track_col = focus_cfg['track_column']
    sharp_col = focus_cfg['sharpness_column']
    z_col = FOCUS_Z_COLUMN

    required_cols = {frame_col, track_col, sharp_col, z_col}
    missing_cols = [col for col in required_cols if col not in comprehensive_df.columns]
    if missing_cols:
        logger.warning(
            "Skipping focussed particle analysis because required columns are missing: %s",
            ', '.join(missing_cols)
        )
        return None, None

    filtered_df, diagnostics_df = filter_focused_frames(
        comprehensive_df,
        track_col,
        frame_col,
        sharp_col,
        z_col,
        focus_cfg
    )

    return filtered_df, diagnostics_df


def save_all_particle_crops(
    crops_df: pd.DataFrame,
    crops_dir: str
) -> None:
    """Persist every intermediate particle crop to disk."""
    if crops_df is None or crops_df.empty:
        return

    os.makedirs(crops_dir, exist_ok=True)
    for _, row in crops_df.iterrows():
        crop_gray = row.get('crop_gray')
        crop_filename = row.get('crop_filename')
        if crop_gray is None or not crop_filename:
            continue
        crop_path = os.path.join(crops_dir, crop_filename)
        try:
            cv2.imwrite(crop_path, crop_gray)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to save crop to {crop_path}: {exc}")


def emit_focus_artifacts(
    filtered_df: Optional[pd.DataFrame],
    diagnostics_df: Optional[pd.DataFrame],
    comprehensive_df: pd.DataFrame,
    focus_cfg: Dict[str, Any],
    extent_mm: Tuple[float, float, float],
    crops_dir: Optional[str],
    sample_id: str,
    output_dir_stage2: str,
    config: Dict[str, Any]
) -> None:
    """Save focus-filtered CSVs and optional visual diagnostics.

    The reconstructed-trajectory maps are drawn in ``config``'s track-map style
    (:func:`track_figure_style`), shared with the Stage-1 track maps.
    """
    if filtered_df is None or diagnostics_df is None:
        return

    save_dataframe_to_csv(filtered_df, output_dir_stage2, 'filtered_in_focus.csv')
    save_dataframe_to_csv(diagnostics_df, output_dir_stage2, 'diagnostics_selected_peaks.csv')

    track_col = focus_cfg['track_column']
    frame_col = focus_cfg['frame_column']
    smoothing_factor = focus_cfg['trajectory_smoothing_factor']
    min_points_for_spline = focus_cfg['min_points_for_spline']

    if focus_cfg['save_plots']:
        focus_plots_dir = os.path.join(output_dir_stage2, 'focussed_particles')
        os.makedirs(focus_plots_dir, exist_ok=True)

        prefix = focus_cfg['crop_filename_prefix'] or sample_id
        max_tracks = focus_cfg['max_tracks_to_plot']
        plot_example_tracks_with_collages(
            comprehensive_df,
            track_col,
            frame_col,
            focus_cfg['sharpness_column'],
            FOCUS_Z_COLUMN,
            focus_cfg,
            crops_dir,
            prefix,
            focus_plots_dir,
            max_tracks
        )

        with track_figure_style(config):
            plot_3d_reconstructed(
                comprehensive_df,
                filtered_df,
                track_col,
                frame_col,
                FOCUS_Z_COLUMN,
                os.path.join(focus_plots_dir, 'reconstructed_trajectories_3d.png'),
                max_tracks,
                smoothing_factor,
                min_points_for_spline,
                extent_mm
            )

            plot_xy_reconstructed(
                comprehensive_df,
                filtered_df,
                track_col,
                frame_col,
                os.path.join(focus_plots_dir, 'reconstructed_trajectories_xy.png'),
                max_tracks,
                extent_mm
            )

            for horizontal_axis in SIDE_VIEW_HORIZONTAL_AXES:
                plot_side_view_reconstructed(
                    comprehensive_df,
                    filtered_df,
                    track_col,
                    frame_col,
                    FOCUS_Z_COLUMN,
                    os.path.join(focus_plots_dir, f'reconstructed_trajectories_{horizontal_axis}z.png'),
                    max_tracks,
                    smoothing_factor,
                    min_points_for_spline,
                    extent_mm,
                    horizontal_axis
                )

        plot_diagnostics_summary(
            diagnostics_df,
            os.path.join(focus_plots_dir, 'diagnostics_summary.png')
        )

    logger.info(
        'Focussed particle analysis complete: %d peaks across %d tracks.',
        len(filtered_df),
        diagnostics_df.shape[0]
    )


def emit_final_particle_crops(
    final_particles_df: pd.DataFrame,
    image_dir_path: str,
    crop_config: Dict[str, Any],
    sample_id: str,
    output_dir_stage2: str
) -> None:
    """Create and save final crops for area-selected particles."""
    if final_particles_df.empty:
        logger.info('No final particles selected after area-based filtering.')
        return

    final_particles_df = crop_and_save_final_particle_images(
        final_particles_df,
        image_dir_path,
        crop_config,
        sample_id,
        output_dir_stage2
    )

    if final_particles_df is not None and not final_particles_df.empty:
        save_dataframe_to_csv(
            final_particles_df,
            output_dir_stage2,
            'final_particles.csv'
        )


def emit_stage2_artifacts(
    comprehensive_df: pd.DataFrame,
    crops_df: pd.DataFrame,
    filtered_df: Optional[pd.DataFrame],
    diagnostics_df: Optional[pd.DataFrame],
    final_particles_df: pd.DataFrame,
    output_dir_stage2: str,
    sample_id: str,
    crop_config: Dict[str, Any],
    image_dir_path: str,
    focus_cfg: Dict[str, Any],
    extent_mm: Tuple[float, float, float],
    save_all_particle_crops_flag: bool,
    config: Dict[str, Any]
) -> Optional[str]:
    """Persist Stage 2 outputs, including crops, focus artifacts, and summary CSVs."""
    os.makedirs(output_dir_stage2, exist_ok=True)
    output_filename = f"{sample_id}_stage2_all_particle_features.csv"
    saved_csv_path = save_dataframe_to_csv(comprehensive_df, output_dir_stage2, output_filename)

    crops_dir = None
    if save_all_particle_crops_flag:
        crops_dir = os.path.join(output_dir_stage2, "stage2_all_particle_crops")
        save_all_particle_crops(crops_df, crops_dir)

    emit_focus_artifacts(
        filtered_df, diagnostics_df, comprehensive_df, focus_cfg, extent_mm,
        crops_dir, sample_id, output_dir_stage2, config
    )
    emit_final_particle_crops(final_particles_df, image_dir_path, crop_config, sample_id, output_dir_stage2)

    return saved_csv_path


def select_final_particles_by_area(
    filtered_df: pd.DataFrame,
    track_col: str,
    area_col: str
) -> pd.DataFrame:
    """Pick the largest-area particle per track from the focus-filtered dataframe."""
    if filtered_df is None or filtered_df.empty:
        columns = list(filtered_df.columns) if filtered_df is not None else None
        return pd.DataFrame(columns=columns)

    if area_col not in filtered_df.columns:
        logger.warning(
            "Cannot select final particles because '%s' column is missing in filtered dataframe.",
            area_col
        )
        return pd.DataFrame(columns=filtered_df.columns)

    if track_col not in filtered_df.columns:
        logger.warning(
            "Cannot select final particles because track column '%s' is missing in filtered dataframe.",
            track_col
        )
        return pd.DataFrame(columns=filtered_df.columns)

    working_df = filtered_df.copy()
    working_df['_area_numeric'] = pd.to_numeric(working_df[area_col], errors='coerce')
    valid_df = working_df.dropna(subset=[track_col, '_area_numeric'])

    if valid_df.empty:
        logger.info(f"No valid rows with both track id and {area_col!r} for final particle selection.")
        return pd.DataFrame(columns=filtered_df.columns)

    idx = valid_df.groupby(track_col)['_area_numeric'].idxmax()
    selected_df = working_df.loc[idx].drop(columns=['_area_numeric'])
    return selected_df.reset_index(drop=True)


def _format_track_label(row: pd.Series, fallback_suffix: int) -> str:
    """Create a stable label for crops, preferring track IDs and falling back gracefully."""
    raw_track_id = row.get('track_id')
    if pd.notna(raw_track_id):
        try:
            return f"track{int(raw_track_id)}"
        except (ValueError, TypeError):
            return f"track{raw_track_id}"

    fallback_value = row.get('particle_id')
    if pd.isna(fallback_value):
        fallback_value = row.get('s1_original_index')
    if pd.isna(fallback_value):
        fallback_value = fallback_suffix

    if isinstance(fallback_value, numbers.Number):
        try:
            fallback_value = int(fallback_value)
        except (ValueError, TypeError, OverflowError):
            fallback_value = str(fallback_value)

    return f"untracked{fallback_value}"


def crop_and_save_final_particle_images(
    final_particles_df: pd.DataFrame,
    image_dir_path: str,
    crop_config: Dict[str, Any],
    sample_id: str,
    output_dir_stage2: str
) -> pd.DataFrame:
    """Recreate final particle crops and record their filenames."""
    if final_particles_df.empty:
        return final_particles_df

    final_dir = os.path.join(output_dir_stage2, 'final_particles')
    os.makedirs(final_dir, exist_ok=True)

    updated_df = final_particles_df.copy()
    updated_df['final_crop_filename'] = None

    groupby_obj = updated_df.groupby('source_image_filename', dropna=False)
    for image_filename, group in groupby_obj:
        if not isinstance(image_filename, str) or not image_filename:
            logger.warning('Skipping final particle crops for rows with invalid source image filename.')
            continue

        full_image_path = os.path.join(image_dir_path, image_filename)
        main_image_color = load_image(full_image_path, mode='color')
        main_image_gray = convert_to_grayscale(main_image_color)

        if main_image_color is None or main_image_gray is None:
            logger.warning(f"Failed to load image {full_image_path} for final particle crops.")
            continue

        frame_num = extract_frame_from_filename(image_filename)
        if frame_num is None:
            logger.warning(f"Could not parse frame from filename {image_filename} for final particle crops.")
            continue

        for idx, row in group.iterrows():
            track_label = _format_track_label(row, fallback_suffix=idx)
            x_raw, y_raw = row.get('X'), row.get('Y')
            x_int = int(x_raw) if pd.notna(x_raw) else 0
            y_int = int(y_raw) if pd.notna(y_raw) else 0
            crop_filename = build_crop_filename(sample_id, track_label, frame_num, x_int, y_int)

            crop_gray, _ = crop_particle_from_arrays(
                crop_config,
                main_image_color,
                main_image_gray,
                row
            )

            if crop_gray is None:
                logger.warning('Failed to create final crop for %s.', crop_filename)
                continue

            saved_crop_path = os.path.join(final_dir, crop_filename)
            try:
                cv2.imwrite(saved_crop_path, crop_gray)
                updated_df.at[idx, 'final_crop_filename'] = os.path.basename(saved_crop_path)
            except Exception as exc:  # noqa: BLE001
                logger.error('Failed to save final crop %s: %s', saved_crop_path, exc)

    return updated_df.reset_index(drop=True)
