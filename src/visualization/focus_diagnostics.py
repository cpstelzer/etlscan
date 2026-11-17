# src/visualization/focus_diagnostics.py
"""Stage-2 focus diagnostics: per-track sharpness curves with crop collages, and the
summary of track lengths and focus peaks. Draw and save inside ``figure_style(config)``."""
import glob
import math
import os
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

from src.core.focus_peak_detection import (
    FocusSweep,
    detect_peaks_in_track,
    estimate_sharpness_baseline,
    focus_sweep_geometry,
)
from src.core.track_columns import FOCUS_X_PIXEL_COLUMN, FOCUS_Y_PIXEL_COLUMN
from src.visualization.figure_style import figure_size


FOCUS_SHOW_PLOTS = False


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

    fig = plt.figure(figsize=figure_size('focus_diagnostic'))
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


def plot_diagnostics_summary(diagnostics_df: pd.DataFrame, outpath: str) -> Optional[str]:
    """Summarize track lengths and selected peak counts in a dual-axis plot."""
    if diagnostics_df.empty:
        return None

    df_sorted = diagnostics_df.sort_values('n_frames', ascending=False).reset_index(drop=True)
    fig, ax_frames = plt.subplots(figsize=figure_size('diagnostics_summary'))

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
