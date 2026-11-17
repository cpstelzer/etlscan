# src/core/focus_peak_detection.py
"""Focus peaks of a track: where its sharpness rises above its own out-of-focus baseline.

A track's sharpness is compared with a running baseline of its out-of-focus frames, never
with an absolute value; the settings come from
``stage_02_characterization.focussed_particles.peak_detection``.
"""
import logging
import warnings
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.utils.common_exceptions import ConfigError, DataValidationError
from src.utils.config_loader import require_config_keys

logger = logging.getLogger(__name__)


FOCUS_CONFIG_PATH = 'stage_02_characterization.focussed_particles'
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
    require_config_keys(focus_cfg, ('peak_detection',), FOCUS_CONFIG_PATH)
    peak_cfg = require_config_keys(
        focus_cfg['peak_detection'], PEAK_DETECTION_KEYS, f"{FOCUS_CONFIG_PATH}.peak_detection"
    )
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
