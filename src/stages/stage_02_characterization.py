# src/stages/stage_02_characterization.py
import os
import cv2
import pandas as pd
import numpy as np
import logging
import skimage.util
import math
import glob
import matplotlib
import numbers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Optional, List, Tuple, Dict, Any

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv
)
from src.utils.config_loader import get_unified_segmentation_params
from src.utils.path_helpers import extract_frame_from_filename
from src.utils.common_exceptions import ConfigError
from src.core.image_io_operations import load_image, convert_to_grayscale
from src.core.particle_operations import (
    crop_particle_from_arrays,
    detect_cropped_particle_on_boundary,
    normalized_crop_config,
)
from src.core.segmentation import segment_main_body_from_image_float
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors

logger = logging.getLogger(__name__)

FOCUS_Z_COLUMN = 'z_mm'
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
    config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Performs detailed characterization for every particle candidate from Stage 1
    using a modular feature calculator architecture.
    """
    logger.info(f"--- Stage 2: Universal Characterization for Sample ID: {sample_id} ---")
    config = config or get_config()
    s2_config = config.get('stage_02_characterization', {})
    segmentation_params = get_unified_segmentation_params(config)
    
    s1_particles_df = load_dataframe_from_csv(input_s1_csv_path)
    if s1_particles_df is None or s1_particles_df.empty:
        logger.warning(f"No particles loaded from Stage 1 CSV: {input_s1_csv_path}. Aborting Stage 2.")
        return None

    cropping_config = s2_config.get('cropping', {})
    if not cropping_config:
        logger.warning("Cropping configuration not found in 'stage_02_characterization'. Using defaults.")
        cropping_config = {'mode': 'rotational_rectangle', 'rotational_padding': 10}

    try:
        normalized_cropping = normalized_crop_config(cropping_config)
        logger.info(f"Initialized ParticleCropper with mode: '{normalized_cropping.get('mode')}'")
    except ConfigError as e:
        logger.error(f"Failed to initialize ParticleCropper: {e}")
        return None

    boundary_threshold_params = {
        'thresholding_mode': segmentation_params.get('thresholding_mode', 'otsu'),
        'otsu_threshold_multiplier': segmentation_params.get('otsu_threshold_multiplier', 0.45),
        'fixed_threshold_value': segmentation_params.get('fixed_threshold_value', 70),
    }
    illumination_mode = str(segmentation_params.get('source_image_illumination_type', 'darkfield')).strip().lower()

    s1_particles_df['s1_original_index'] = s1_particles_df.index

    # crop
    crop_records: List[dict] = []
    for image_filename, group in s1_particles_df.groupby('source_image_filename'):
        logger.info(f"Characterizing {len(group)} particles from image: {image_filename}")
        full_image_path = os.path.join(image_dir_path, image_filename)
        
        main_img_color = load_image(full_image_path, mode='color')
        main_img_gray = convert_to_grayscale(main_img_color)

        if main_img_color is None or main_img_gray is None:
            logger.warning(f"Failed to load image {full_image_path}. Skipping {len(group)} particles.")
            continue

        frame_num = extract_frame_from_filename(image_filename)
        crop_records.extend(
            crop_particles(
                main_img_color,
                main_img_gray,
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

    focus_cfg = s2_config.get('focussed_particles', {})
    track_col = focus_cfg.get('track_column', 'track_id')

    # focus filter
    filtered_df, diagnostics_df = select_focused_tracks(comprehensive_df, focus_cfg)

    final_particles_df = pd.DataFrame()
    if filtered_df is not None and not filtered_df.empty:
        final_particles_df = select_final_particles_by_area(
            filtered_df,
            track_col,
            area_col='main_area_filled'
        )

    # save
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
        bool(cropping_config.get('save_all_particle_crops', True)),
    )

    logger.info(f"--- Stage 2: Characterization for {sample_id} COMPLETED ---")
    return saved_csv_path


def build_crop_filename(
    sample_id: str,
    track_label: str,
    frame_num: int,
    x_value: Any,
    y_value: Any
) -> str:
    """Assemble a descriptive filename for a particle crop using track and frame info."""
    try:
        x_int = int(x_value) if pd.notna(x_value) else 0
    except (TypeError, ValueError):
        x_int = 0

    try:
        y_int = int(y_value) if pd.notna(y_value) else 0
    except (TypeError, ValueError):
        y_int = 0

    return f"{sample_id}_{track_label}_frame{frame_num}_X{x_int}_Y{y_int}.png"


def crop_particles(
    main_img_color: np.ndarray,
    main_img_gray: np.ndarray,
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

        crop_filename = build_crop_filename(
            sample_id,
            track_label,
            frame_num,
            s1_particle_row.get('X'),
            s1_particle_row.get('Y')
        )

        crop_gray, crop_metadata = crop_particle_from_arrays(
            crop_config,
            main_img_color,
            main_img_gray,
            s1_particle_row
        )

        if crop_gray is None:
            logger.warning("Failed to crop particle at index %s. Skipping.", original_index)
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
            logger.error("Failed to save crop to %s: %s", crop_path, exc)


def emit_focus_artifacts(
    filtered_df: Optional[pd.DataFrame],
    diagnostics_df: Optional[pd.DataFrame],
    comprehensive_df: pd.DataFrame,
    focus_cfg: Dict[str, Any],
    crops_dir: Optional[str],
    sample_id: str,
    output_dir_stage2: str
) -> None:
    """Save focus-filtered CSVs and optional visual diagnostics."""
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

        plot_3d_reconstructed(
            comprehensive_df,
            filtered_df,
            focus_cfg.get('track_column', 'track_id'),
            focus_cfg.get('frame_column', 'frame'),
            FOCUS_Z_COLUMN,
            os.path.join(focus_plots_dir, 'reconstructed_trajectories_3d.png'),
            max_tracks
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
    save_all_particle_crops_flag: bool
) -> Optional[str]:
    """Persist Stage 2 outputs, including crops, focus artifacts, and summary CSVs."""
    os.makedirs(output_dir_stage2, exist_ok=True)
    output_filename = f"{sample_id}_stage2_all_particle_features.csv"
    saved_csv_path = save_dataframe_to_csv(comprehensive_df, output_dir_stage2, output_filename)

    crops_dir = None
    if save_all_particle_crops_flag:
        crops_dir = os.path.join(output_dir_stage2, "stage2_all_particle_crops")
        save_all_particle_crops(crops_df, crops_dir)

    emit_focus_artifacts(filtered_df, diagnostics_df, comprehensive_df, focus_cfg, crops_dir, sample_id, output_dir_stage2)
    emit_final_particle_crops(final_particles_df, image_dir_path, crop_config, sample_id, output_dir_stage2)

    return saved_csv_path


def smooth_series(series: pd.Series, window: int) -> np.ndarray:
    """Return a centered rolling mean of the provided series."""
    if window <= 1:
        return series.to_numpy()
    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def detect_peaks_in_track(
    track_df: pd.DataFrame,
    sharp_col: str,
    focus_cfg: dict
) -> np.ndarray:
    """Identify indices of sharpness peaks within a single track."""
    smooth_window = max(1, int(focus_cfg.get('smooth_window', 3)))
    min_sharpness_frac = float(focus_cfg.get('min_sharpness_fraction', 0.3))
    min_prominence_frac = float(focus_cfg.get('min_prominence_fraction', 0.05))
    sharpness_min_threshold = float(focus_cfg.get('sharpness_min_threshold', 0))

    sharp_series = pd.to_numeric(track_df[sharp_col], errors='coerce').fillna(0)
    y = sharp_series.values
    if len(y) == 0:
        return np.array([], dtype=int)

    try:
        max_val = np.nanmax(y)
    except ValueError:
        return np.array([], dtype=int)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.array([], dtype=int)

    threshold = min_sharpness_frac * max_val
    smoothed = smooth_series(pd.Series(y), smooth_window)
    peaks: List[int] = []

    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] > smoothed[i - 1]
            and smoothed[i] > smoothed[i + 1]
            and smoothed[i] >= threshold
            and smoothed[i] >= sharpness_min_threshold
        ):
            local_window = smoothed[max(0, i - 5):min(len(smoothed), i + 6)]
            prominence = smoothed[i] - np.median(local_window)
            if prominence >= min_prominence_frac * max_val:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def filter_focused_frames(
    df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    sharp_col: str,
    focus_cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter each track to frames near sharpness peaks and record summary stats."""
    filtered_rows: List[pd.DataFrame] = []
    diagnostics: List[dict] = []

    for tid, group in df.groupby(track_col):
        group_sorted = group.sort_values(frame_col)
        peak_idx = detect_peaks_in_track(group_sorted, sharp_col, focus_cfg)
        if len(peak_idx) > 0:
            selected = group_sorted.iloc[peak_idx]
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

    filtered_df = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame(columns=df.columns)
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
    crops_dir: Optional[str],
    prefix: str,
    outpath: str
) -> Optional[str]:
    """Plot sharpness vs. frame for one track and optionally append a crop collage."""
    track_df = df[df[track_col] == track_id].sort_values(frame_col)
    if track_df.empty:
        return None

    peak_idx = detect_peaks_in_track(track_df, sharp_col, focus_cfg)

    fig = plt.figure(figsize=(10, 10))
    grid_spec = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax_sharpness = fig.add_subplot(grid_spec[0])

    ax_sharpness.plot(track_df[frame_col], track_df[sharp_col], color='tab:blue', linewidth=1.2)
    ax_sharpness.set_xlabel('Frame')
    ax_sharpness.set_ylabel('Sharpness', color='tab:blue')
    ax_sharpness.tick_params(axis='y', labelcolor='tab:blue')

    ax_z = ax_sharpness.twinx()
    ax_z.plot(track_df[frame_col], track_df[z_col], color='tab:orange', linewidth=1.0)
    ax_z.set_ylabel('z_mm', color='tab:orange')
    ax_z.tick_params(axis='y', labelcolor='tab:orange')

    maxima_frames: List[Tuple[int, int, Optional[int], Optional[int]]] = []
    x_pix_col = FOCUS_X_PIXEL_COLUMN
    y_pix_col = FOCUS_Y_PIXEL_COLUMN

    for idx, peak_position in enumerate(peak_idx):
        peak_row = track_df.iloc[peak_position]
        ax_z.scatter(peak_row[frame_col], peak_row[z_col], color='red', marker='o', s=40, zorder=5)
        offset_x = 2
        offset_y = 0.02 * (track_df[z_col].max() - track_df[z_col].min() + 1e-9)
        ax_z.text(
            peak_row[frame_col] + offset_x,
            peak_row[z_col] + offset_y,
            str(idx + 1),
            color='black',
            fontsize=8
        )

        x_pix_val = peak_row.get(x_pix_col, np.nan)
        y_pix_val = peak_row.get(y_pix_col, np.nan)
        maxima_frames.append((idx + 1, int(peak_row[frame_col]), x_pix_val, y_pix_val))

    ax_sharpness.set_title(f'Track {track_id} (length={len(track_df)})')

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
                2,
                10,
                str(label),
                color='yellow',
                fontsize=10,
                ha='left',
                va='top',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')
            )
    else:
        ax_collage.text(0.5, 0.5, 'No crop images available', ha='center', va='center')

    plt.tight_layout()
    try:
        plt.savefig(outpath, dpi=150)
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
            crops_dir,
            prefix,
            outpath
        )
        if plotted:
            saved_paths.append(plotted)
    return saved_paths


def plot_3d_reconstructed(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    z_col: str,
    outpath: str,
    n_tracks: Optional[int]
) -> Optional[str]:
    """Render 3D trajectories showing selected in-focus points for each track."""
    x_mm_col = FOCUS_X_MM_COLUMN
    y_mm_col = FOCUS_Y_MM_COLUMN

    if x_mm_col not in df.columns or y_mm_col not in df.columns:
        logger.debug('Skipping 3D plot because x_mm/y_mm columns are missing.')
        return None

    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    track_ids = track_lengths.index if n_tracks is None else track_lengths.index[:n_tracks]

    if track_lengths.empty:
        return None

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('tab20')
    ncol = max(1, min(20, len(track_ids)))
    plotted = 0

    for idx, tid in enumerate(track_ids):
        track_df = df[df[track_col] == tid].sort_values(frame_col)
        selected_df = filtered_df[filtered_df[track_col] == tid]
        if track_df.empty or selected_df.empty:
            continue

        x_vals = track_df[x_mm_col].values
        y_vals = track_df[y_mm_col].values
        z_real = selected_df[z_col].mean()
        z_line = np.full_like(x_vals, z_real, dtype=float)

        color = cmap(idx % ncol)
        ax.plot(x_vals, y_vals, z_line, linewidth=0.8, alpha=0.7, color=color)
        ax.scatter(
            selected_df[x_mm_col].values,
            selected_df[y_mm_col].values,
            selected_df[z_col].values,
            marker='o',
            s=18,
            color=color,
            edgecolor='k',
            linewidth=0.3
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel(x_mm_col)
    ax.set_ylabel(y_mm_col)
    ax.set_zlabel(z_col)
    ax.set_title(f'Reconstructed trajectories of {plotted} tracks')
    plt.tight_layout()
    try:
        plt.savefig(outpath, dpi=200)
    finally:
        plt.close(fig)

    return outpath


def plot_diagnostics_summary(diagnostics_df: pd.DataFrame, outpath: str) -> Optional[str]:
    """Summarize track lengths and selected peak counts in a dual-axis plot."""
    if diagnostics_df.empty:
        return None

    df_sorted = diagnostics_df.sort_values('n_frames', ascending=False).reset_index(drop=True)
    fig, ax_frames = plt.subplots(figsize=(12, 6))

    ax_frames.plot(
        df_sorted.index,
        df_sorted['n_frames'],
        color='tab:blue',
        marker='o',
        label='Track length (n_frames)'
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
        label='Sharpness maxima (n_selected)'
    )
    ax_peaks.set_ylabel('Sharpness maxima count', color='tab:orange')
    ax_peaks.tick_params(axis='y', labelcolor='tab:orange')

    ax_frames.legend(loc='upper left')
    ax_peaks.legend(loc='upper right')
    plt.title('Track lengths and sharpness maxima across all tracks')
    plt.tight_layout()
    try:
        plt.savefig(outpath, dpi=150)
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
        logger.info('No valid rows with both track id and %s for final particle selection.', area_col)
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
        main_img_color = load_image(full_image_path, mode='color')
        main_img_gray = convert_to_grayscale(main_img_color)

        if main_img_color is None or main_img_gray is None:
            logger.warning('Failed to load image %s for final particle crops.', full_image_path)
            continue

        frame_num = extract_frame_from_filename(image_filename)
        if frame_num is None:
            logger.warning('Could not parse frame from filename %s for final particle crops.', image_filename)
            continue

        for idx, row in group.iterrows():
            track_label = _format_track_label(row, fallback_suffix=idx)
            crop_filename = build_crop_filename(
                sample_id,
                track_label,
                frame_num,
                row.get('X'),
                row.get('Y')
            )

            crop_gray, _ = crop_particle_from_arrays(
                crop_config,
                main_img_color,
                main_img_gray,
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
