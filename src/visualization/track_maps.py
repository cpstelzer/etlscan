# src/visualization/track_maps.py
"""Maps of particle tracks: the Stage-1 3D / XY figures of the detected tracks and the
Stage-2 3D / XY / side views with the reconstructed depth.

All of them draw through ``plot_track_line``, ``plot_focus_points`` and
``draw_floor_shadow`` and are created and saved inside ``track_figure_style(config)``, so
they look alike and a track keeps its colour across the views of one stage.
"""
import logging
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.depth_reconstruction import DEPTH_KNOT_SPEED_OUTLIER_COLUMN, reconstruct_track_depth
from src.core.track_columns import FOCUS_X_MM_COLUMN, FOCUS_Y_MM_COLUMN, FOCUS_Z_RECONSTRUCTED_COLUMN
from src.visualization.figure_style import draw_floor_shadow, figure_size, plot_focus_points, plot_track_line

logger = logging.getLogger(__name__)


def build_tracking_figure(
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
    fig = plt.figure(figsize=figure_size('track_map_3d'))
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


def build_tracking_xy_figure(
    tracked_df: pd.DataFrame,
    extent_mm: Tuple[float, float, float]
) -> Optional[plt.Figure]:
    """Create a 2D XY-projection figure of all tracked particle paths.

    Companion to :func:`build_tracking_figure`. The 3D view is subject to
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
    fig = plt.figure(figsize=figure_size('track_map_xy'))
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


SIDE_VIEW_HORIZONTAL_AXES = ('x', 'y')


def _tracks_to_plot(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    n_tracks: Optional[int]
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return the tracks the reconstructed-trajectory maps draw, in rank (= colour) order.

    The ``n_tracks`` longest tracks (all when None) that have in-focus selections, each
    as (its frames sorted by frame, its selections). Every view takes its tracks from
    here, so a track has the same colour in the 3D, XY, XZ and YZ figures.
    """
    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    track_ids = track_lengths.index if n_tracks is None else track_lengths.index[:n_tracks]
    tracks = []
    for tid in track_ids:
        track_df = df[df[track_col] == tid].sort_values(frame_col)
        selected_df = filtered_df[filtered_df[track_col] == tid]
        if not track_df.empty and not selected_df.empty:
            tracks.append((track_df, selected_df))
    return tracks


def _track_depth_line(
    track_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    frame_col: str,
    z_col: str,
    smoothing_factor: float,
    min_points_for_spline: int
) -> np.ndarray:
    """Return one track's per-frame depth for the views that show depth (3D, XZ, YZ).

    The persisted ``z_reconstructed`` when the table has it, so the figures match the
    saved CSV: NaN, hence no line, for a track whose depth knots the speed gate all left
    out. Otherwise (e.g. called in isolation) it is reconstructed here from the same
    knots :func:`add_reconstructed_depth` uses, via :func:`reconstruct_track_depth`.
    """
    if FOCUS_Z_RECONSTRUCTED_COLUMN in track_df.columns:
        return track_df[FOCUS_Z_RECONSTRUCTED_COLUMN].to_numpy(dtype=float)
    if DEPTH_KNOT_SPEED_OUTLIER_COLUMN in selected_df.columns:
        selected_df = selected_df[~selected_df[DEPTH_KNOT_SPEED_OUTLIER_COLUMN].astype(bool)]
    z_knots = selected_df.groupby(frame_col)[z_col].mean()
    return reconstruct_track_depth(
        track_df[frame_col].to_numpy(dtype=float),
        z_knots.index.to_numpy(dtype=float),
        z_knots.to_numpy(dtype=float),
        smoothing_factor,
        min_points_for_spline
    )


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

    The trajectory line follows a continuous depth estimate (:func:`_track_depth_line`):
    the precomputed ``z_reconstructed`` column when present (so the figure matches the
    saved CSV), otherwise an on-the-fly reconstruction from the in-focus knots via
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

    tracks = [
        (track_df, selected_df, _track_depth_line(
            track_df, selected_df, frame_col, z_col, smoothing_factor, min_points_for_spline
        ))
        for track_df, selected_df in _tracks_to_plot(df, filtered_df, track_col, frame_col, n_tracks)
    ]
    if not tracks:
        return None

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=figure_size('track_map_3d'))
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

    tracks = _tracks_to_plot(df, filtered_df, track_col, frame_col, n_tracks)
    if not tracks:
        return None

    fig = plt.figure(figsize=figure_size('track_map_xy'))
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


def plot_side_view_reconstructed(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    track_col: str,
    frame_col: str,
    z_col: str,
    outpath: str,
    n_tracks: Optional[int],
    smoothing_factor: float,
    min_points_for_spline: int,
    extent_mm: Tuple[float, float, float],
    horizontal_axis: str
) -> Optional[str]:
    """Render a side view of the reconstructed trajectories: depth (z) against x or y.

    Companion to :func:`plot_3d_reconstructed`: the same tracks, colours, depth lines
    and in-focus rings, projected onto the XZ (``horizontal_axis='x'``) or YZ (``'y'``)
    plane. In the 3D view perspective hides at which depth each track swims and how far
    its in-focus depths lie from its depth line; a side view shows both at true scale.

    Axes are pinned to the matching sides of ``extent_mm`` with equal x/z scale; the
    aspect is adjusted via the axes box, as in :func:`plot_xy_reconstructed`. Call
    inside :func:`track_figure_style`, which sets the line and marker sizes.

    Raises:
        ValueError: ``horizontal_axis`` is not 'x' or 'y'.
    """
    if horizontal_axis not in SIDE_VIEW_HORIZONTAL_AXES:
        raise ValueError(f"horizontal_axis must be one of {SIDE_VIEW_HORIZONTAL_AXES}, got {horizontal_axis!r}")
    horizontal_col = FOCUS_X_MM_COLUMN if horizontal_axis == 'x' else FOCUS_Y_MM_COLUMN

    if horizontal_col not in df.columns:
        logger.debug(f'Skipping {horizontal_axis.upper()}Z plot because {horizontal_col} is missing.')
        return None

    tracks = _tracks_to_plot(df, filtered_df, track_col, frame_col, n_tracks)
    if not tracks:
        return None

    fig = plt.figure(figsize=figure_size('track_map_side'))
    ax = fig.add_subplot(111)

    # Same ranks, hence colours, as the 3D view; rings drawn after every line.
    for rank, (track_df, selected_df) in enumerate(tracks):
        z_line = _track_depth_line(track_df, selected_df, frame_col, z_col, smoothing_factor, min_points_for_spline)
        plot_track_line(ax, (track_df[horizontal_col].values, z_line), rank)
    for rank, (_, selected_df) in enumerate(tracks):
        plot_focus_points(ax, (selected_df[horizontal_col].values, selected_df[z_col].values), rank)

    x_max_mm, y_max_mm, z_max_mm = extent_mm
    ax.set_xlim(0, x_max_mm if horizontal_axis == 'x' else y_max_mm)
    ax.set_ylim(0, z_max_mm)
    ax.set_xlabel(f'{horizontal_axis} (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Reconstructed {horizontal_axis.upper()}Z trajectories of {len(tracks)} tracks')
    plt.tight_layout()
    try:
        plt.savefig(outpath)
    finally:
        plt.close(fig)

    return outpath
