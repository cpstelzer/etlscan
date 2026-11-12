# src/stages/stage_02_characterization.py
import glob
import logging
import math
import numbers
import os
import warnings
from typing import Optional, List, NamedTuple, Tuple, Dict, Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import skimage.util
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

from src.utils import (
    load_dataframe_from_csv, save_dataframe_to_csv, build_crop_filename
)
from src.utils.acquisition_metadata import get_scan_extent_mm
from src.utils.config_loader import get_unified_segmentation_params
from src.utils.path_helpers import extract_frame_from_filename
from src.utils.common_exceptions import ConfigError, DataValidationError
from src.core.image_io_operations import load_image, convert_to_grayscale
from src.core.particle_operations import (
    crop_particle_from_arrays,
    detect_cropped_particle_on_boundary,
    normalized_crop_config,
)
from src.core.segmentation import segment_main_body_from_image_float
from src.visualization.figure_style import (
    draw_floor_shadow,
    figure_style,
    plot_focus_points,
    plot_track_line,
    track_figure_style,
)
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors

logger = logging.getLogger(__name__)

FOCUS_Z_COLUMN = 'z_mm'
FOCUS_Z_RECONSTRUCTED_COLUMN = 'z_reconstructed'
FOCUS_X_PIXEL_COLUMN = 'X'
FOCUS_Y_PIXEL_COLUMN = 'Y'
FOCUS_X_MM_COLUMN = 'x_mm'
FOCUS_Y_MM_COLUMN = 'y_mm'
FOCUS_SHOW_PLOTS = False

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

    focus_cfg = s2_config['focussed_particles']
    track_col = focus_cfg['track_column']

    # focus filter
    filtered_df, diagnostics_df = select_focused_tracks(comprehensive_df, focus_cfg)

    # flag in-focus depths that no neighbouring knot of their track supports (same
    # recording frame rate source as the Stage-1 background offset)
    filtered_df = flag_depth_knot_speed_outliers(
        filtered_df, comprehensive_df, focus_cfg, float(config['acquisition']['camera']['fps'])
    )
    # reconstruct continuous per-frame depth (z) from the sparse in-focus points
    comprehensive_df = add_reconstructed_depth(comprehensive_df, filtered_df, focus_cfg)
    # carry that per-frame depth onto the in-focus selections so it also reaches
    # filtered_in_focus.csv and final_particles.csv (a pass-through, not an ML feature)
    filtered_df = attach_reconstructed_depth_to_subset(
        filtered_df, comprehensive_df, track_col, focus_cfg.get('frame_column', 'frame')
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
            get_scan_extent_mm(config, image_dir_path),
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
        if segmentation_params.get('source_image_illumination_type') == 'collimated':
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
    if not focus_cfg.get('enabled', False):
        return None, None

    frame_col = focus_cfg.get('frame_column', 'frame')
    track_col = focus_cfg.get('track_column', 'track_id')
    sharp_col = focus_cfg.get('sharpness_column', 'variance_laplacian')
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

    if focus_cfg.get('save_plots', True):
        focus_plots_dir = os.path.join(output_dir_stage2, 'focussed_particles')
        os.makedirs(focus_plots_dir, exist_ok=True)

        prefix = focus_cfg.get('crop_filename_prefix') or sample_id
        max_tracks = focus_cfg.get('max_tracks_to_plot')
        plot_example_tracks_with_collages(
            comprehensive_df,
            focus_cfg.get('track_column', 'track_id'),
            focus_cfg.get('frame_column', 'frame'),
            focus_cfg.get('sharpness_column', 'variance_laplacian'),
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
                focus_cfg.get('track_column', 'track_id'),
                focus_cfg.get('frame_column', 'frame'),
                FOCUS_Z_COLUMN,
                os.path.join(focus_plots_dir, 'reconstructed_trajectories_3d.png'),
                max_tracks,
                focus_cfg.get('trajectory_smoothing_factor', 0.5),
                focus_cfg.get('min_points_for_spline', 5),
                extent_mm
            )

            plot_xy_reconstructed(
                comprehensive_df,
                filtered_df,
                focus_cfg.get('track_column', 'track_id'),
                focus_cfg.get('frame_column', 'frame'),
                os.path.join(focus_plots_dir, 'reconstructed_trajectories_xy.png'),
                max_tracks,
                extent_mm
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


PEAK_DETECTION_NUMERIC_KEYS = (
    'baseline_window_mm',
    'baseline_min_window_frames',
    'baseline_min_coverage',
    'sparse_baseline_percentile',
    'min_rise_above_baseline',
    'min_fraction_of_strongest_rise',
)
PEAK_DETECTION_KEYS = (*PEAK_DETECTION_NUMERIC_KEYS, 'one_peak_per_sweep_pass')
FOCUS_PEAK_EDGE_COLUMN = 'focus_peak_at_track_edge'


class FocusSweep(NamedTuple):
    """Focus-sweep geometry of one recording, derived from its per-frame focal depth."""

    step_mm: float              # spacing of the focal planes
    pass_of_frame: pd.Series    # frame number -> index of its monotonic sweep pass


def _peak_detection_settings(focus_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the validated ``focussed_particles.peak_detection`` settings (fail fast)."""
    peak_cfg = focus_cfg.get('peak_detection')
    if not isinstance(peak_cfg, dict):
        raise ConfigError(
            "stage_02_characterization.focussed_particles.peak_detection must be a mapping "
            f"with the keys {', '.join(PEAK_DETECTION_KEYS)}."
        )
    missing = [key for key in PEAK_DETECTION_KEYS if key not in peak_cfg]
    if missing:
        raise ConfigError(f"Missing focussed_particles.peak_detection keys: {', '.join(missing)}")
    if not isinstance(peak_cfg['one_peak_per_sweep_pass'], bool):
        raise ConfigError(
            "peak_detection.one_peak_per_sweep_pass must be true or false, "
            f"got {peak_cfg['one_peak_per_sweep_pass']!r}"
        )

    settings: Dict[str, Any] = {key: float(peak_cfg[key]) for key in PEAK_DETECTION_NUMERIC_KEYS}
    settings['one_peak_per_sweep_pass'] = peak_cfg['one_peak_per_sweep_pass']
    if settings['baseline_window_mm'] <= 0:
        raise ConfigError(f"peak_detection.baseline_window_mm must be > 0, got {settings['baseline_window_mm']}")
    if settings['baseline_min_window_frames'] < 3 or not float(settings['baseline_min_window_frames']).is_integer():
        raise ConfigError(
            "peak_detection.baseline_min_window_frames must be an integer >= 3, "
            f"got {peak_cfg['baseline_min_window_frames']!r}"
        )
    if not 0 < settings['baseline_min_coverage'] <= 1:
        raise ConfigError(f"peak_detection.baseline_min_coverage must be in (0, 1], got {settings['baseline_min_coverage']}")
    if not 0 <= settings['sparse_baseline_percentile'] <= 100:
        raise ConfigError(
            f"peak_detection.sparse_baseline_percentile must be in [0, 100], got {settings['sparse_baseline_percentile']}"
        )
    if settings['min_rise_above_baseline'] < 0:
        raise ConfigError(f"peak_detection.min_rise_above_baseline must be >= 0, got {settings['min_rise_above_baseline']}")
    if not 0 <= settings['min_fraction_of_strongest_rise'] <= 1:
        raise ConfigError(
            "peak_detection.min_fraction_of_strongest_rise must be in [0, 1], "
            f"got {settings['min_fraction_of_strongest_rise']}"
        )
    return settings


def focus_sweep_geometry(df: pd.DataFrame, frame_col: str, z_col: str) -> FocusSweep:
    """Focal-plane spacing and sweep passes of one recording, from its per-frame z.

    The step is the median |dz| between consecutive frames, i.e. the recording's
    own scan step (flybacks and turnarounds are rare and do not move the median).
    A pass is a run of frames in which the focal plane moves in one direction; a
    new pass starts wherever the direction reverses (zigzag turnaround, sawtooth
    flyback). Raises when z never changes, because the peak detector measures its
    baseline window in focal travel.
    """
    frame_z = df.groupby(frame_col)[z_col].first().dropna().sort_index()
    z_values = frame_z.to_numpy(dtype=float)
    dz = np.diff(z_values)
    consecutive = np.diff(frame_z.index.to_numpy(dtype=float)) == 1
    steps = np.abs(dz[consecutive])
    steps = steps[steps > 0]
    if steps.size == 0:
        raise DataValidationError(
            f"Focus-peak detection needs a focus sweep, but '{z_col}' never changes between "
            "consecutive frames of this recording."
        )

    direction = np.sign(dz)
    for i in range(1, direction.size):      # a repeated plane keeps the current direction
        if direction[i] == 0:
            direction[i] = direction[i - 1]
    frame_direction = np.concatenate([direction[:1], direction])  # first frame joins the first step
    pass_index = np.concatenate([[0], np.cumsum(frame_direction[1:] != frame_direction[:-1])])
    return FocusSweep(float(np.median(steps)), pd.Series(pass_index, index=frame_z.index))


def _track_frame_grid(frames: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Place one track's values on a gap-free frame grid (unobserved frames = NaN).

    Returns the grid and, for every input row, its position on the grid. A frame
    holding several detections of the track (Stage 1 occasionally emits
    near-identical duplicates) takes the largest of their values.
    """
    frame_numbers = np.rint(np.asarray(frames, dtype=float)).astype(int)
    positions = frame_numbers - frame_numbers.min()
    grid = np.full(positions.max() + 1, np.nan)
    np.fmax.at(grid, positions, np.asarray(values, dtype=float))
    return grid, positions


def estimate_sharpness_baseline(
    frames: np.ndarray,
    values: np.ndarray,
    focal_step_mm: float,
    focus_cfg: Dict[str, Any]
) -> np.ndarray:
    """Out-of-focus sharpness level of one track, aligned with ``values``.

    The baseline is a centred running median over ``baseline_window_mm`` of focal
    travel (at least ``baseline_min_window_frames`` frames, made odd). A median
    ignores a focus peak that fills less than half the window and follows sudden
    steps, so the level jumps at a sawtooth flyback and the broad humps at zigzag
    turnarounds (both from the brightness-dependent noise floor of the crop) stay
    in the baseline instead of becoming peaks. Windows in which fewer than
    ``baseline_min_coverage`` of the frames hold a detection (tracks of objects
    detected only near their focal plane) cannot show the out-of-focus level and
    use the track's ``sparse_baseline_percentile`` instead. Unobserved values
    (NaN) give a NaN baseline.
    """
    settings = _peak_detection_settings(focus_cfg)
    values = np.asarray(values, dtype=float)
    baseline = np.full(values.shape, np.nan)
    observed = np.isfinite(values)
    if not observed.any():
        return baseline
    if not np.isfinite(focal_step_mm) or focal_step_mm <= 0:
        raise DataValidationError(f"focal_step_mm must be a positive number, got {focal_step_mm!r}")

    grid, positions = _track_frame_grid(np.asarray(frames)[observed], values[observed])
    window = max(int(settings['baseline_min_window_frames']),
                 int(round(settings['baseline_window_mm'] / focal_step_mm)))
    window += 1 - window % 2
    half = window // 2

    padded = np.concatenate([np.full(half, np.nan), grid, np.full(half, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    coverage = np.sum(np.isfinite(windows), axis=1) / window
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN windows -> NaN
        running_median = np.nanmedian(windows, axis=1)
    sparse_level = np.percentile(grid[np.isfinite(grid)], settings['sparse_baseline_percentile'])
    grid_baseline = np.where(coverage >= settings['baseline_min_coverage'], running_median, sparse_level)

    baseline[observed] = grid_baseline[positions]
    return baseline


def detect_peaks_in_track(
    track_df: pd.DataFrame,
    sharp_col: str,
    frame_col: str,
    sweep: FocusSweep,
    focus_cfg: Dict[str, Any]
) -> np.ndarray:
    """Identify positional indices of focus peaks within a single track.

    Each frame's rise is its sharpness minus the track baseline
    (:func:`estimate_sharpness_baseline`). Candidates are local maxima of the
    rise on the frame grid; unobserved frames and the two track ends count as
    lower than any detection, so a track that starts or ends in focus keeps its
    peak. A candidate is a focus peak when its rise is at least
    ``min_rise_above_baseline`` times its baseline (a scale-free gate, unlike an
    absolute sharpness floor) and at least ``min_fraction_of_strongest_rise``
    of the track's strongest rise. With ``one_peak_per_sweep_pass`` only the
    strongest peak of each sweep pass is kept: one object crosses the focal
    plane once per pass, so a weaker peak in the same pass is another object in
    the crop (e.g. an in-focus fibre at a different depth).

    The returned indices are positional into ``track_df`` as passed; callers
    must supply the track sorted by frame.
    """
    settings = _peak_detection_settings(focus_cfg)
    values = pd.to_numeric(track_df[sharp_col], errors='coerce').to_numpy(dtype=float)
    frames = track_df[frame_col].to_numpy()
    baseline = estimate_sharpness_baseline(frames, values, sweep.step_mm, focus_cfg)
    rise = values - baseline
    observed = np.isfinite(rise)
    if not observed.any():
        return np.array([], dtype=int)

    grid, positions = _track_frame_grid(frames[observed], rise[observed])
    searchable = np.concatenate([[-np.inf], np.where(np.isfinite(grid), grid, -np.inf), [-np.inf]])
    candidate_grid, _ = find_peaks(searchable)
    candidate_grid -= 1
    observed_rows = np.flatnonzero(observed)
    by_rise = np.argsort(rise[observed_rows], kind='stable')   # duplicates: the larger rise wins
    row_of_grid = dict(zip(positions[by_rise], observed_rows[by_rise]))
    candidates = np.array([row_of_grid[p] for p in candidate_grid if p in row_of_grid], dtype=int)
    if candidates.size == 0:
        return np.array([], dtype=int)

    strongest = rise[candidates].max()
    if strongest <= 0:
        return np.array([], dtype=int)
    keep = (
        (rise[candidates] > 0)
        & (rise[candidates] >= settings['min_rise_above_baseline'] * baseline[candidates])
        & (rise[candidates] >= settings['min_fraction_of_strongest_rise'] * strongest)
    )
    peaks = candidates[keep]
    if settings['one_peak_per_sweep_pass'] and peaks.size > 1:
        peak_pass = sweep.pass_of_frame.reindex(np.rint(frames[peaks].astype(float))).to_numpy()
        if np.isnan(peak_pass.astype(float)).any():
            raise DataValidationError("A focus peak lies in a frame without a sweep-pass assignment.")
        strongest_first = np.argsort(-rise[peaks], kind='stable')
        _, first_of_pass = np.unique(peak_pass[strongest_first], return_index=True)
        peaks = peaks[strongest_first[first_of_pass]]
    return np.sort(peaks)


def filter_focused_frames(
    df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    sharp_col: str,
    z_col: str,
    focus_cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select each track's focus-peak frames and record summary stats.

    Selected rows carry ``focus_peak_at_track_edge``: True when the peak is the
    track's first or last frame with a sharpness value, i.e. the object entered
    or left the recording at its sharpest, so the true maximum may lie outside
    the track.
    """
    filtered_rows: List[pd.DataFrame] = []
    diagnostics: List[dict] = []
    sweep = focus_sweep_geometry(df, frame_col, z_col)
    duplicate_frames = int(df.duplicated(subset=[track_col, frame_col]).sum())
    if duplicate_frames:
        logger.warning(
            f"{duplicate_frames} track frames hold more than one detection; "
            "focus-peak detection uses the sharper detection of each such frame."
        )

    for tid, group in df.groupby(track_col):
        group_sorted = group.sort_values(frame_col)
        peak_idx = detect_peaks_in_track(group_sorted, sharp_col, frame_col, sweep, focus_cfg)
        if len(peak_idx) > 0:
            selected = group_sorted.iloc[peak_idx].copy()
            observed_rows = np.flatnonzero(
                np.isfinite(pd.to_numeric(group_sorted[sharp_col], errors='coerce').to_numpy(dtype=float))
            )
            selected[FOCUS_PEAK_EDGE_COLUMN] = np.isin(peak_idx, [observed_rows[0], observed_rows[-1]])
            filtered_rows.append(selected)
            diagnostics.append({
                track_col: tid,
                'n_frames': group_sorted.shape[0],
                'n_selected': selected.shape[0]
            })
        else:
            diagnostics.append({
                track_col: tid,
                'n_frames': group_sorted.shape[0],
                'n_selected': 0
            })

    filtered_df = (
        pd.concat(filtered_rows, ignore_index=True) if filtered_rows
        else pd.DataFrame(columns=[*df.columns, FOCUS_PEAK_EDGE_COLUMN])
    )
    diagnostics_df = pd.DataFrame(diagnostics)
    return filtered_df, diagnostics_df


def plot_track_with_collage(
    df: pd.DataFrame,
    track_id: int,
    track_col: str,
    frame_col: str,
    sharp_col: str,
    z_col: str,
    focus_cfg: dict,
    sweep: FocusSweep,
    crops_dir: Optional[str],
    prefix: str,
    outpath: str
) -> Optional[str]:
    """Plot sharpness and its baseline vs. frame for one track, plus a crop collage."""
    track_df = df[df[track_col] == track_id].sort_values(frame_col)
    if track_df.empty:
        return None

    peak_idx = detect_peaks_in_track(track_df, sharp_col, frame_col, sweep, focus_cfg)
    sharpness = pd.to_numeric(track_df[sharp_col], errors='coerce').to_numpy(dtype=float)
    baseline = estimate_sharpness_baseline(
        track_df[frame_col].to_numpy(), sharpness, sweep.step_mm, focus_cfg
    )

    fig = plt.figure(figsize=(8, 8))
    grid_spec = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax_sharpness = fig.add_subplot(grid_spec[0])

    ax_sharpness.plot(track_df[frame_col], sharpness, color='tab:blue')
    ax_sharpness.plot(track_df[frame_col], baseline, color='0.35', linestyle='--')
    ax_sharpness.set_xlabel('Frame')
    # No legend: on the crowded twin-axis plot it would hide the peak labels.
    ax_sharpness.set_ylabel('Sharpness (dashed: baseline)', color='tab:blue')
    ax_sharpness.tick_params(axis='y', labelcolor='tab:blue')

    ax_z = ax_sharpness.twinx()
    ax_z.plot(track_df[frame_col], track_df[z_col], color='tab:orange')
    ax_z.set_ylabel('z (mm)', color='tab:orange')
    ax_z.tick_params(axis='y', labelcolor='tab:orange')

    maxima_frames: List[Tuple[int, int, Optional[int], Optional[int]]] = []
    x_pix_col = FOCUS_X_PIXEL_COLUMN
    y_pix_col = FOCUS_Y_PIXEL_COLUMN

    for idx, peak_position in enumerate(peak_idx):
        peak_row = track_df.iloc[peak_position]
        ax_z.plot(
            peak_row[frame_col], peak_row[z_col],
            color='red', marker='o', linestyle='none', zorder=5, markeredgecolor='black'
        )
        offset_x = 2
        offset_y = 0.02 * (track_df[z_col].max() - track_df[z_col].min() + 1e-9)
        ax_z.text(
            peak_row[frame_col] + offset_x,
            peak_row[z_col] + offset_y,
            str(idx + 1),
            color='black',
            fontweight='bold'
        )

        x_pix_val = peak_row.get(x_pix_col, np.nan)
        y_pix_val = peak_row.get(y_pix_col, np.nan)
        maxima_frames.append((idx + 1, int(peak_row[frame_col]), x_pix_val, y_pix_val))

    ax_sharpness.set_title(f'Track {track_id} ({len(track_df)} frames)')

    ax_collage = fig.add_subplot(grid_spec[1])
    ax_collage.axis('off')

    saved_imgs: List[Tuple[int, np.ndarray]] = []
    crops_available = crops_dir is not None and os.path.isdir(crops_dir)

    if crops_available:
        for label, frame_int, x_pix, y_pix in maxima_frames:
            if any(pd.isna(val) for val in (x_pix, y_pix)):
                continue
            pattern = os.path.join(
                crops_dir,
                f"{prefix}_track{track_id}_frame{frame_int}_X{int(x_pix)}_Y{int(y_pix)}.png"
            )
            files = glob.glob(pattern)
            if files:
                img = mpimg.imread(files[0])
                saved_imgs.append((label, img))

    if saved_imgs:
        cols = min(5, len(saved_imgs))
        rows = math.ceil(len(saved_imgs) / cols)
        collage_grid = GridSpecFromSubplotSpec(rows, cols, subplot_spec=grid_spec[1])
        for i, (label, img) in enumerate(saved_imgs):
            r = i // cols
            c = i % cols
            ax_img = fig.add_subplot(collage_grid[r, c])
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax_img.imshow(img, cmap='gray', aspect='equal')
            else:
                ax_img.imshow(img, aspect='equal')
            ax_img.axis('off')
            ax_img.text(
                0.04,
                0.96,
                str(label),
                transform=ax_img.transAxes,
                color='yellow',
                fontweight='bold',
                ha='left',
                va='top',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round')
            )
    else:
        ax_collage.text(0.5, 0.5, 'No crop images available', ha='center', va='center')

    plt.tight_layout()
    try:
        plt.savefig(outpath)
        if FOCUS_SHOW_PLOTS:
            plt.show()
    finally:
        plt.close(fig)

    return outpath


def plot_example_tracks_with_collages(
    df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    sharp_col: str,
    z_col: str,
    focus_cfg: dict,
    crops_dir: Optional[str],
    prefix: str,
    outdir: str,
    n_tracks: Optional[int]
) -> List[str]:
    """Create diagnostic plots with collages for the longest tracks."""
    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    track_ids = track_lengths.index if n_tracks is None else track_lengths.index[:n_tracks]
    sweep = focus_sweep_geometry(df, frame_col, z_col)

    saved_paths: List[str] = []
    for tid in track_ids:
        outpath = os.path.join(outdir, f"track_{tid}_diagnostic_with_collage.png")
        plotted = plot_track_with_collage(
            df,
            tid,
            track_col,
            frame_col,
            sharp_col,
            z_col,
            focus_cfg,
            sweep,
            crops_dir,
            prefix,
            outpath
        )
        if plotted:
            saved_paths.append(plotted)
    return saved_paths


def reconstruct_track_depth(
    frames: np.ndarray,
    knot_frames: np.ndarray,
    knot_z: np.ndarray,
    smoothing_factor: float,
    min_points_for_spline: int
) -> np.ndarray:
    """Estimate per-frame depth (z) along a track from sparse in-focus measurements.

    For tracks with at least ``min_points_for_spline`` in-focus points a cubic
    smoothing spline (:class:`scipy.interpolate.UnivariateSpline`) is fitted to
    the measured ``(knot_frames, knot_z)`` pairs and evaluated at every frame,
    clamped to the measured frame span to avoid extrapolation artefacts. The
    smoothing budget ``s = smoothing_factor * m * Var(z)`` (m = number of knots)
    is scaled by the spread of z, making ``smoothing_factor`` unitless and
    data-scale independent (0 interpolates, larger smooths more). Shorter tracks
    lack support for a meaningful spline and are assigned a constant depth equal
    to the mean of their in-focus measurements. Returns an array aligned with
    ``frames`` (all NaN when there are no knots).
    """
    frames = np.asarray(frames, dtype=float)
    knot_frames = np.asarray(knot_frames, dtype=float)
    knot_z = np.asarray(knot_z, dtype=float)

    if knot_z.size == 0:
        return np.full(frames.shape, np.nan, dtype=float)

    if knot_z.size >= min_points_for_spline and knot_z.size >= 2:
        spline = UnivariateSpline(
            knot_frames,
            knot_z,
            k=min(3, knot_z.size - 1),
            s=smoothing_factor * knot_z.size * float(np.var(knot_z))
        )
        frame_eval = np.clip(frames, knot_frames[0], knot_frames[-1])
        return spline(frame_eval)

    return np.full(frames.shape, float(np.mean(knot_z)), dtype=float)


DEPTH_KNOT_SPEED_GATE_KEYS = ('max_vertical_speed_mm_s', 'tolerance_planes')
DEPTH_KNOT_SPEED_OUTLIER_COLUMN = 'depth_knot_speed_outlier'


def _depth_knot_speed_gate_settings(focus_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return the validated ``focussed_particles.depth_knot_speed_gate`` settings (fail fast)."""
    gate_cfg = focus_cfg.get('depth_knot_speed_gate')
    if not isinstance(gate_cfg, dict):
        raise ConfigError(
            "stage_02_characterization.focussed_particles.depth_knot_speed_gate must be a mapping "
            f"with the keys {', '.join(DEPTH_KNOT_SPEED_GATE_KEYS)}."
        )
    missing = [key for key in DEPTH_KNOT_SPEED_GATE_KEYS if key not in gate_cfg]
    if missing:
        raise ConfigError(f"Missing focussed_particles.depth_knot_speed_gate keys: {', '.join(missing)}")
    settings = {key: float(gate_cfg[key]) for key in DEPTH_KNOT_SPEED_GATE_KEYS}
    if not settings['max_vertical_speed_mm_s'] > 0:
        raise ConfigError(
            f"depth_knot_speed_gate.max_vertical_speed_mm_s must be > 0, got {gate_cfg['max_vertical_speed_mm_s']!r}"
        )
    if not settings['tolerance_planes'] >= 0:
        raise ConfigError(f"depth_knot_speed_gate.tolerance_planes must be >= 0, got {gate_cfg['tolerance_planes']!r}")
    return settings


def find_depth_knot_speed_outliers(
    knot_frames: np.ndarray,
    knot_z: np.ndarray,
    frame_rate_fps: float,
    focal_step_mm: float,
    focus_cfg: Dict[str, Any]
) -> np.ndarray:
    """Flag the depth knots of one track that no neighbouring knot supports.

    ``knot_frames`` (strictly increasing) and ``knot_z`` hold one knot per frame. Two
    neighbouring knots disagree when their depths differ by more than
    ``tolerance_planes`` focal planes plus ``max_vertical_speed_mm_s`` times the time
    between them (``focussed_particles.depth_knot_speed_gate``). A knot that disagrees
    with each of its neighbours (an end knot has one) is flagged: such a lone depth is a
    focus peak of another object in the crop, e.g. the well rim, or a frame of another
    animal after a Stage-1 identity swap, not a dive and return of the tracked animal.
    Consecutive knots that agree with each other are kept, so a swapped segment of two
    or more knots stays; the knots alone cannot tell it from a real excursion. Tracks
    with fewer than 3 knots have no neighbour majority and get no flag.

    Returns a boolean array aligned with ``knot_z`` (True = outlier).

    Raises:
        DataValidationError: mismatched inputs, frames not strictly increasing, or a
            non-positive frame rate or focal step.
    """
    settings = _depth_knot_speed_gate_settings(focus_cfg)
    knot_frames = np.asarray(knot_frames, dtype=float)
    knot_z = np.asarray(knot_z, dtype=float)
    if knot_frames.shape != knot_z.shape or knot_z.ndim != 1:
        raise DataValidationError(
            f"knot_frames and knot_z must be 1D arrays of one length, got {knot_frames.shape} and {knot_z.shape}"
        )
    if not (np.isfinite(frame_rate_fps) and frame_rate_fps > 0):
        raise DataValidationError(f"frame_rate_fps must be a positive number, got {frame_rate_fps!r}")
    if not (np.isfinite(focal_step_mm) and focal_step_mm > 0):
        raise DataValidationError(f"focal_step_mm must be a positive number, got {focal_step_mm!r}")
    if np.any(np.diff(knot_frames) <= 0):
        raise DataValidationError("Depth knots must be sorted by frame, one knot per frame.")

    outlier = np.zeros(knot_z.shape, dtype=bool)
    if knot_z.size < 3:
        return outlier
    allowed_change_mm = (
        settings['tolerance_planes'] * focal_step_mm
        + settings['max_vertical_speed_mm_s'] * np.diff(knot_frames) / frame_rate_fps
    )
    gap_disagrees = np.abs(np.diff(knot_z)) > allowed_change_mm   # gap i lies between knots i and i + 1
    disagrees_with_previous = np.concatenate([[True], gap_disagrees])  # a missing neighbour cannot support
    disagrees_with_next = np.concatenate([gap_disagrees, [True]])
    return disagrees_with_previous & disagrees_with_next


def flag_depth_knot_speed_outliers(
    filtered_df: Optional[pd.DataFrame],
    comprehensive_df: pd.DataFrame,
    focus_cfg: Dict[str, Any],
    frame_rate_fps: float
) -> Optional[pd.DataFrame]:
    """Return the in-focus selections with ``depth_knot_speed_outlier`` added (a copy).

    A track's depth knots are its selected frames, each at the mean ``z_mm`` of that
    frame's selections (as in :func:`add_reconstructed_depth`); every selection of a
    frame whose knot :func:`find_depth_knot_speed_outliers` flags is marked True. The
    plane spacing is the recording's own (:func:`focus_sweep_geometry` over all frames of
    ``comprehensive_df``); ``frame_rate_fps`` is the recording frame rate. Flagged
    selections stay in the selection (they are still focus peaks) but are left out of
    the depth spline. Returns ``filtered_df`` unchanged when it is None or empty.
    """
    if filtered_df is None or filtered_df.empty:
        return filtered_df
    track_col = focus_cfg['track_column']
    frame_col = focus_cfg['frame_column']
    focal_step_mm = focus_sweep_geometry(comprehensive_df, frame_col, FOCUS_Z_COLUMN).step_mm

    flagged_df = filtered_df.copy()
    flagged_df[DEPTH_KNOT_SPEED_OUTLIER_COLUMN] = False
    for _, track_group in flagged_df.groupby(track_col):
        z_knots = track_group.groupby(frame_col)[FOCUS_Z_COLUMN].mean()
        outlier = find_depth_knot_speed_outliers(
            z_knots.index.to_numpy(dtype=float),
            z_knots.to_numpy(dtype=float),
            frame_rate_fps,
            focal_step_mm,
            focus_cfg
        )
        outlier_rows = track_group.index[track_group[frame_col].isin(z_knots.index[outlier])]
        flagged_df.loc[outlier_rows, DEPTH_KNOT_SPEED_OUTLIER_COLUMN] = True

    outliers_df = flagged_df[flagged_df[DEPTH_KNOT_SPEED_OUTLIER_COLUMN]]
    if not outliers_df.empty:
        listed = ', '.join(
            f"track {row[track_col]} frame {row[frame_col]:g} z={row[FOCUS_Z_COLUMN]:g} mm"
            for _, row in outliers_df.iterrows()
        )
        logger.info(
            f"Depth spline: {len(outliers_df)} in-focus selection(s) disagree with every neighbouring depth "
            f"knot of their track by more than the speed gate allows and are left out: {listed}"
        )
    return flagged_df


def add_reconstructed_depth(
    comprehensive_df: pd.DataFrame,
    filtered_df: Optional[pd.DataFrame],
    focus_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Attach a per-frame ``z_reconstructed`` column to the Stage-2 feature table.

    The in-focus selections (``filtered_df``) provide sparse depth knots; depth
    between them is reconstructed per track with :func:`reconstruct_track_depth`
    and evaluated at every frame of the track. Selections flagged by
    :func:`flag_depth_knot_speed_outliers` are not knots; a track whose knots are all
    flagged gets no depth. In-plane (x, y) coordinates are left unaltered. When focus
    analysis produced no selections, or required columns are missing, the input is
    returned unchanged (no column added).

    Raises:
        DataValidationError: the selections lack ``depth_knot_speed_outlier``.
    """
    if filtered_df is None or filtered_df.empty:
        return comprehensive_df

    track_col = focus_cfg.get('track_column', 'track_id')
    frame_col = focus_cfg.get('frame_column', 'frame')
    z_col = FOCUS_Z_COLUMN
    smoothing_factor = float(focus_cfg.get('trajectory_smoothing_factor', 0.5))
    min_points_for_spline = int(focus_cfg.get('min_points_for_spline', 5))

    required = {track_col, frame_col, z_col}
    if not required.issubset(comprehensive_df.columns) or not required.issubset(filtered_df.columns):
        missing = required - (set(comprehensive_df.columns) & set(filtered_df.columns))
        logger.warning("Skipping z reconstruction; required columns missing: %s", ', '.join(sorted(missing)))
        return comprehensive_df
    if DEPTH_KNOT_SPEED_OUTLIER_COLUMN not in filtered_df.columns:
        raise DataValidationError(
            f"In-focus selections lack '{DEPTH_KNOT_SPEED_OUTLIER_COLUMN}'; flag them with "
            "flag_depth_knot_speed_outliers before reconstructing depth."
        )

    result_df = comprehensive_df.copy()
    result_df[FOCUS_Z_RECONSTRUCTED_COLUMN] = np.nan
    knots_by_track = filtered_df[~filtered_df[DEPTH_KNOT_SPEED_OUTLIER_COLUMN].astype(bool)].groupby(track_col)

    for tid, track_group in result_df.groupby(track_col):
        if tid not in knots_by_track.groups:
            continue
        z_knots = knots_by_track.get_group(tid).groupby(frame_col)[z_col].mean()
        if z_knots.empty:
            continue
        frames_sorted = track_group.sort_values(frame_col)
        z_values = reconstruct_track_depth(
            frames_sorted[frame_col].to_numpy(dtype=float),
            z_knots.index.to_numpy(dtype=float),
            z_knots.to_numpy(dtype=float),
            smoothing_factor,
            min_points_for_spline
        )
        result_df.loc[frames_sorted.index, FOCUS_Z_RECONSTRUCTED_COLUMN] = z_values

    return result_df


def attach_reconstructed_depth_to_subset(
    subset_df: pd.DataFrame,
    comprehensive_df: pd.DataFrame,
    track_col: str,
    frame_col: str
) -> pd.DataFrame:
    """Copy the per-frame ``z_reconstructed`` value onto a subset of track rows.

    :func:`add_reconstructed_depth` computes the depth once on the full per-frame
    table; this matches a subset (e.g. the in-focus selections) to
    ``comprehensive_df`` on ``(track_col, frame_col)`` so the reconstructed depth
    travels onto the selected rows and therefore into ``filtered_in_focus.csv``
    and ``final_particles.csv``. Returns ``subset_df`` unchanged when the column
    or the key columns are unavailable.
    """
    if subset_df is None or subset_df.empty:
        return subset_df
    if FOCUS_Z_RECONSTRUCTED_COLUMN not in comprehensive_df.columns:
        return subset_df
    key_cols = {track_col, frame_col}
    if not key_cols.issubset(subset_df.columns) or not key_cols.issubset(comprehensive_df.columns):
        return subset_df

    if FOCUS_Z_RECONSTRUCTED_COLUMN in subset_df.columns:
        subset_df = subset_df.drop(columns=[FOCUS_Z_RECONSTRUCTED_COLUMN])

    depth_lookup = (
        comprehensive_df[[track_col, frame_col, FOCUS_Z_RECONSTRUCTED_COLUMN]]
        .drop_duplicates(subset=[track_col, frame_col])
    )
    return subset_df.merge(depth_lookup, on=[track_col, frame_col], how='left')


def plot_3d_reconstructed(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    z_col: str,
    outpath: str,
    n_tracks: Optional[int],
    smoothing_factor: float,
    min_points_for_spline: int,
    extent_mm: Tuple[float, float, float]
) -> Optional[str]:
    """Render 3D trajectories with a spline-reconstructed depth profile per track.

    The trajectory line follows a continuous depth estimate: the precomputed
    ``z_reconstructed`` column when present (so the figure matches the saved
    CSV), otherwise an on-the-fly reconstruction from the in-focus knots via
    :func:`reconstruct_track_depth`. In-focus points are marked as open rings at
    their measured depth, and each track's grey copy on the floor shows its x/y
    position. ``smoothing_factor`` and ``min_points_for_spline`` control the
    fallback reconstruction. Call inside :func:`track_figure_style`, which sets
    the line and marker sizes.

    Axes are pinned to ``extent_mm`` (the full imaged volume) rather than
    autoscaled to the tracks, so the plot always shows the whole field of view
    and Z sweep and stays comparable across samples.
    """
    x_mm_col = FOCUS_X_MM_COLUMN
    y_mm_col = FOCUS_Y_MM_COLUMN

    if x_mm_col not in df.columns or y_mm_col not in df.columns:
        logger.debug('Skipping 3D plot because x_mm/y_mm columns are missing.')
        return None

    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    track_ids = track_lengths.index if n_tracks is None else track_lengths.index[:n_tracks]

    if track_lengths.empty:
        return None

    tracks = []
    for tid in track_ids:
        track_df = df[df[track_col] == tid].sort_values(frame_col)
        selected_df = filtered_df[filtered_df[track_col] == tid]
        if track_df.empty or selected_df.empty:
            continue

        # Depth line: prefer the persisted per-frame reconstruction so the figure
        # matches the saved CSV; fall back to reconstructing from the in-focus
        # knots if the column is absent (e.g. called in isolation).
        if (FOCUS_Z_RECONSTRUCTED_COLUMN in track_df.columns
                and track_df[FOCUS_Z_RECONSTRUCTED_COLUMN].notna().any()):
            z_line = track_df[FOCUS_Z_RECONSTRUCTED_COLUMN].to_numpy(dtype=float)
        else:
            z_knots = selected_df.groupby(frame_col)[z_col].mean()
            z_line = reconstruct_track_depth(
                track_df[frame_col].to_numpy(dtype=float),
                z_knots.index.to_numpy(dtype=float),
                z_knots.to_numpy(dtype=float),
                smoothing_factor,
                min_points_for_spline
            )
        tracks.append((track_df, selected_df, z_line))

    if not tracks:
        return None

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Floor shadows first, then every line, then every in-focus ring, so no later
    # track's line covers an earlier track's rings.
    for track_df, _, _ in tracks:
        draw_floor_shadow(ax, track_df[x_mm_col].values, track_df[y_mm_col].values)
    for rank, (track_df, _, z_line) in enumerate(tracks):
        plot_track_line(ax, (track_df[x_mm_col].values, track_df[y_mm_col].values, z_line), rank)
    for rank, (_, selected_df, _) in enumerate(tracks):
        plot_focus_points(
            ax,
            (selected_df[x_mm_col].values, selected_df[y_mm_col].values, selected_df[z_col].values),
            rank
        )
    plotted = len(tracks)

    x_max_mm, y_max_mm, z_max_mm = extent_mm
    ax.set_xlim(0, x_max_mm)
    ax.set_ylim(0, y_max_mm)
    ax.set_zlim(0, z_max_mm)
    ax.set_xlabel('x (mm)', labelpad=14)
    ax.set_ylabel('y (mm)', labelpad=14)
    ax.set_zlabel('z (mm)', labelpad=14)
    ax.set_title(f'Reconstructed trajectories of {plotted} tracks')
    plt.tight_layout()
    try:
        # The tight bbox of a 3D axes omits its z label; include it explicitly.
        plt.savefig(outpath, bbox_extra_artists=[ax.zaxis.label])
    finally:
        plt.close(fig)

    return outpath


def plot_xy_reconstructed(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    outpath: str,
    n_tracks: Optional[int],
    extent_mm: Tuple[float, float, float]
) -> Optional[str]:
    """Render the flat XY projection of the reconstructed trajectories.

    Companion to :func:`plot_3d_reconstructed`, plotting the exact same tracks
    and in-focus selections on the true XY plane. The 3D view is subject to
    perspective distortion (a point's screen position depends on its Z), which
    can make the XY layout look shifted versus Stage 1; this projection removes
    that ambiguity so the two stages can be compared point-for-point. Tracks keep
    the colours they have in the 3D view.

    Axes are pinned to the full field of view from ``extent_mm``; the aspect is
    adjusted via the axes box because the ``datalim`` alternative would stretch
    those fixed limits back out to fill the box.
    """
    x_mm_col = FOCUS_X_MM_COLUMN
    y_mm_col = FOCUS_Y_MM_COLUMN

    if x_mm_col not in df.columns or y_mm_col not in df.columns:
        logger.debug('Skipping XY plot because x_mm/y_mm columns are missing.')
        return None

    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    track_ids = track_lengths.index if n_tracks is None else track_lengths.index[:n_tracks]

    if track_lengths.empty:
        return None

    tracks = []
    for tid in track_ids:
        track_df = df[df[track_col] == tid].sort_values(frame_col)
        selected_df = filtered_df[filtered_df[track_col] == tid]
        if not track_df.empty and not selected_df.empty:
            tracks.append((track_df, selected_df))

    if not tracks:
        return None

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # Same ranks, hence colours, as the 3D view; rings drawn after every line.
    for rank, (track_df, _) in enumerate(tracks):
        plot_track_line(ax, (track_df[x_mm_col].values, track_df[y_mm_col].values), rank)
    for rank, (_, selected_df) in enumerate(tracks):
        plot_focus_points(ax, (selected_df[x_mm_col].values, selected_df[y_mm_col].values), rank)
    plotted = len(tracks)

    x_max_mm, y_max_mm, _ = extent_mm
    ax.set_xlim(0, x_max_mm)
    ax.set_ylim(0, y_max_mm)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Reconstructed XY trajectories of {plotted} tracks')
    plt.tight_layout()
    try:
        plt.savefig(outpath)
    finally:
        plt.close(fig)

    return outpath


def plot_diagnostics_summary(diagnostics_df: pd.DataFrame, outpath: str) -> Optional[str]:
    """Summarize track lengths and selected peak counts in a dual-axis plot."""
    if diagnostics_df.empty:
        return None

    df_sorted = diagnostics_df.sort_values('n_frames', ascending=False).reset_index(drop=True)
    fig, ax_frames = plt.subplots(figsize=(8, 4.5))

    ax_frames.plot(
        df_sorted.index,
        df_sorted['n_frames'],
        color='tab:blue',
        marker='o',
        label='Track length'
    )
    ax_frames.set_xlabel('Track (sorted by length)')
    ax_frames.set_ylabel('Track length (frames)', color='tab:blue')
    ax_frames.tick_params(axis='y', labelcolor='tab:blue')

    ax_peaks = ax_frames.twinx()
    ax_peaks.bar(
        df_sorted.index,
        df_sorted['n_selected'],
        alpha=0.4,
        color='tab:orange',
        label='Sharpness maxima'
    )
    ax_peaks.set_ylabel('Sharpness maxima', color='tab:orange')
    ax_peaks.tick_params(axis='y', labelcolor='tab:orange')
    ax_peaks.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Draw the track-length line above the bars of the twin axis.
    ax_frames.set_zorder(ax_peaks.get_zorder() + 1)
    ax_frames.patch.set_visible(False)

    # One legend for both axes, so the two cannot overlap each other.
    frame_handles, frame_labels = ax_frames.get_legend_handles_labels()
    peak_handles, peak_labels = ax_peaks.get_legend_handles_labels()
    ax_frames.legend(frame_handles + peak_handles, frame_labels + peak_labels, loc='upper right', frameon=False)
    ax_frames.set_title('Track lengths and sharpness maxima')
    plt.tight_layout()
    try:
        plt.savefig(outpath)
    finally:
        plt.close(fig)
    return outpath


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
