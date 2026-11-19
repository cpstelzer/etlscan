# src/core/depth_reconstruction.py
"""Depth (z) of a track over time, reconstructed from its in-focus frames.

The focal depth of each in-focus selection is a knot; knots that no neighbouring knot
supports are flagged (``focussed_particles.depth_knot_speed_gate``) and the remaining ones
are joined by a smoothing spline into a per-frame depth.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from src.core.focus_peak_detection import FOCUS_CONFIG_PATH, focus_sweep_geometry
from src.core.track_columns import FOCUS_Z_COLUMN, FOCUS_Z_RECONSTRUCTED_COLUMN
from src.utils.common_exceptions import ConfigError, DataValidationError
from src.utils.config_loader import require_config_keys

logger = logging.getLogger(__name__)


DEPTH_KNOT_SPEED_GATE_KEYS = ('max_vertical_speed_mm_s', 'tolerance_planes')
DEPTH_KNOT_SPEED_OUTLIER_COLUMN = 'depth_knot_speed_outlier'


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


def _depth_knot_speed_gate_settings(focus_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return the validated ``focussed_particles.depth_knot_speed_gate`` settings (fail fast)."""
    require_config_keys(focus_cfg, ('depth_knot_speed_gate',), FOCUS_CONFIG_PATH)
    gate_cfg = require_config_keys(
        focus_cfg['depth_knot_speed_gate'], DEPTH_KNOT_SPEED_GATE_KEYS,
        f"{FOCUS_CONFIG_PATH}.depth_knot_speed_gate",
    )
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

    track_col = focus_cfg['track_column']
    frame_col = focus_cfg['frame_column']
    z_col = FOCUS_Z_COLUMN
    smoothing_factor = float(focus_cfg['trajectory_smoothing_factor'])
    min_points_for_spline = int(focus_cfg['min_points_for_spline'])

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
