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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Optional, List, Tuple

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv
)
from src.utils.path_helpers import extract_frame_from_filename
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image, convert_to_grayscale
from src.core.particle_operations import ParticleCropper
from src.core.segmentation import segment_main_body_from_image_float
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors

logger = logging.getLogger(__name__)

def run_stage_02_characterization(
    input_s1_csv_path: str,
    image_dir_path: str,
    output_dir_stage2: str,
    sample_id: str
) -> Optional[str]:
    """
    Performs detailed characterization for every particle candidate from Stage 1
    using a modular feature calculator architecture.
    """
    logger.info(f"--- Stage 2: Universal Characterization for Sample ID: {sample_id} ---")
    
    config = get_config()
    s2_config = config.get('stage_02_characterization', {})
    system_config = config.get('system', {})
    
    s1_particles_df = load_dataframe_from_csv(input_s1_csv_path)
    if s1_particles_df is None or s1_particles_df.empty:
        logger.warning(f"No particles loaded from Stage 1 CSV: {input_s1_csv_path}. Aborting Stage 2.")
        return None

    cropping_config = s2_config.get('cropping', {})
    if not cropping_config:
        logger.warning("Cropping configuration not found in 'stage_02_characterization'. Using defaults.")
        cropping_config = {'mode': 'rotational_rectangle', 'rotational_padding': 10}

    save_all_particle_crops = bool(cropping_config.get('save_all_particle_crops', True))
    all_crops_dir = None
    if save_all_particle_crops:
        all_crops_dir = os.path.join(output_dir_stage2, "stage2_all_particle_crops")
        os.makedirs(all_crops_dir, exist_ok=True)
        
    try:
        cropper = ParticleCropper(cropping_config)
        logger.info(f"Initialized ParticleCropper with mode: '{cropping_config.get('mode')}'")
    except ConfigError as e:
        logger.error(f"Failed to initialize ParticleCropper: {e}")
        return None

    all_particle_features = []

    s1_particles_df['s1_original_index'] = s1_particles_df.index
    
    for image_filename, group in s1_particles_df.groupby('source_image_filename'):
        logger.info(f"Characterizing {len(group)} particles from image: {image_filename}")
        full_image_path = os.path.join(image_dir_path, image_filename)
        
        main_img_color = load_image(full_image_path, mode='color')
        main_img_gray = convert_to_grayscale(main_img_color)

        if main_img_color is None or main_img_gray is None:
            logger.warning(f"Failed to load image {full_image_path}. Skipping {len(group)} particles.")
            continue
            
        for index, s1_particle_row in group.iterrows():
            original_index = s1_particle_row['s1_original_index']
            particle_features = {'s1_original_index': original_index}
            
            frame_num = extract_frame_from_filename(image_filename)
            if frame_num is None:
                logger.warning(f"Could not parse frame from {image_filename} for particle index {original_index}. Skipping.")
                continue

            raw_track_id = s1_particle_row.get('track_id')
            if pd.isna(raw_track_id):
                track_id_str = f"untracked{original_index}"
            else:
                track_id_str = f"track{int(raw_track_id)}"

            crop_filename_prefix = f"{sample_id}_{track_id_str}_frame{frame_num}"
            
            crop_gray, saved_crop_path = cropper.crop_single_particle_from_image(
                main_img_color, main_img_gray, s1_particle_row,
                output_dir_for_crop=all_crops_dir if save_all_particle_crops else None,
                crop_filename_prefix=crop_filename_prefix
            )

            if crop_gray is None:
                logger.warning(f"Failed to crop particle at index {original_index}. Skipping.")
                continue

            particle_features['crop_filename'] = (
                os.path.basename(saved_crop_path) if saved_crop_path else None
            )
            
            crop_float_orig = skimage.util.img_as_float(crop_gray)
            raw_mask, filled_mask, _ = segment_main_body_from_image_float(crop_float_orig, system_config)

            if filled_mask is None or raw_mask is None:
                logger.warning(f"High-precision segmentation failed for crop {particle_features['crop_filename']}. Skipping feature calculation.")
                continue

            quality_features = calculate_quality_features(crop_gray, saved_crop_path, s2_config)
            particle_features.update(quality_features)

            intensity_img_for_props = crop_float_orig.copy()
            if system_config.get('source_image_illumination_type') == 'collimated':
                intensity_img_for_props = 1.0 - intensity_img_for_props
            morphology_features = calculate_detailed_body_morphology(raw_mask, filled_mask, intensity_img_for_props)
            renamed_morphology = {k.replace('_s4', ''): v for k, v in morphology_features.items()}
            particle_features.update(renamed_morphology)

            # --- THIS IS THE CORRECTED LINE ---
            shape_features = calculate_shape_descriptors(crop_gray, filled_mask, s2_config)
            particle_features.update(shape_features)

            all_particle_features.append(particle_features)

    if not all_particle_features:
        logger.warning(f"No particles were successfully characterized for sample {sample_id}.")
        return None

    features_df = pd.DataFrame(all_particle_features)
    s1_particles_df_indexed = s1_particles_df.copy()
    
    comprehensive_df = pd.merge(
        s1_particles_df_indexed, features_df, on='s1_original_index', how='right'
    )
    comprehensive_df.drop(columns=['s1_original_index'], inplace=True, errors='ignore')

    output_filename = f"{sample_id}_stage2_all_particle_features.csv"
    saved_csv_path = save_dataframe_to_csv(comprehensive_df, output_dir_stage2, output_filename)

    run_focussed_particles_analysis(
        comprehensive_df,
        output_dir_stage2,
        sample_id,
        s2_config,
        all_crops_dir
    )

    logger.info(f"--- Stage 2: Characterization for {sample_id} COMPLETED ---")
    return saved_csv_path


def smooth_series(series: pd.Series, window: int) -> np.ndarray:
    if window <= 1:
        return series.to_numpy()
    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def detect_peaks_in_track(
    track_df: pd.DataFrame,
    sharp_col: str,
    focus_cfg: dict
) -> np.ndarray:
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
    x_pix_col = focus_cfg.get('x_pixel_column', 'X')
    y_pix_col = focus_cfg.get('y_pixel_column', 'Y')

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
        if focus_cfg.get('show_plots', False):
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
    focus_cfg: dict,
    outpath: str,
    n_tracks: Optional[int]
) -> Optional[str]:
    x_mm_col = focus_cfg.get('x_mm_column', 'x_mm')
    y_mm_col = focus_cfg.get('y_mm_column', 'y_mm')

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


def run_focussed_particles_analysis(
    comprehensive_df: pd.DataFrame,
    output_dir_stage2: str,
    sample_id: str,
    stage2_config: dict,
    crops_dir: Optional[str]
) -> None:
    focus_cfg = stage2_config.get('focussed_particles', {})
    if not focus_cfg.get('enabled', False):
        return

    frame_col = focus_cfg.get('frame_column', 'frame')
    track_col = focus_cfg.get('track_column', 'track_id')
    sharp_col = focus_cfg.get('sharpness_column', 'variance_laplacian')
    z_col = focus_cfg.get('z_column', 'z_mm')

    required_cols = {frame_col, track_col, sharp_col, z_col}
    missing_cols = [col for col in required_cols if col not in comprehensive_df.columns]
    if missing_cols:
        logger.warning(
            "Skipping focussed particle analysis because required columns are missing: %s",
            ', '.join(missing_cols)
        )
        return

    focus_plots_dir = os.path.join(output_dir_stage2, 'focussed_particles')
    os.makedirs(focus_plots_dir, exist_ok=True)

    filtered_df, diagnostics_df = filter_focused_frames(
        comprehensive_df,
        track_col,
        frame_col,
        sharp_col,
        focus_cfg
    )

    save_dataframe_to_csv(filtered_df, output_dir_stage2, 'filtered_in_focus.csv')
    save_dataframe_to_csv(diagnostics_df, output_dir_stage2, 'diagnostics_selected_peaks.csv')

    if focus_cfg.get('save_plots', True):
        prefix = focus_cfg.get('crop_filename_prefix') or sample_id
        max_tracks = focus_cfg.get('max_tracks_to_plot')
        plot_example_tracks_with_collages(
            comprehensive_df,
            track_col,
            frame_col,
            sharp_col,
            z_col,
            focus_cfg,
            crops_dir,
            prefix,
            focus_plots_dir,
            max_tracks
        )

        plot_3d_reconstructed(
            comprehensive_df,
            filtered_df,
            track_col,
            frame_col,
            z_col,
            focus_cfg,
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