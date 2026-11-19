# tests/test_stage02_focus_peaks.py
"""Regression tests for Stage-2 focus-peak detection.

Each synthetic track mimics a failure seen on axial_localization_exp2 (plate 4)
or on moving-organism recordings: out of focus the sharpness is a noise floor
that follows the crop brightness, so it steps up at a sawtooth flyback and forms
a broad hump at a zigzag turnaround; real focus peaks are narrow in z.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.core.focus_peak_detection import (
    FOCUS_PEAK_EDGE_COLUMN,
    detect_peaks_in_track,
    estimate_sharpness_baseline,
    filter_focused_frames,
    focus_sweep_geometry,
)
from src.utils.common_exceptions import ConfigError, DataValidationError

SHARP = 'sharpness'
CFG = {
    'peak_detection': {
        'baseline_window_mm': 4.0,
        'baseline_min_window_frames': 5,
        'baseline_min_coverage': 0.5,
        'sparse_baseline_percentile': 20,
        'min_rise_above_baseline': 0.15,
        'min_fraction_of_strongest_rise': 0.15,
        'one_peak_per_sweep_pass': True,
    }
}
STEP_MM = 0.2
TOP_MM = 14.6
FOCUS_Z = 12.4


def _cfg(**overrides):
    cfg = copy.deepcopy(CFG)
    cfg['peak_detection'].update(overrides)
    return cfg


def sawtooth_z(n_sweeps, step=STEP_MM, top=TOP_MM):
    planes = np.round(np.arange(0.0, top + 1e-9, step), 3)
    return np.tile(planes, n_sweeps)


def zigzag_z(n_passes, step=STEP_MM, top=TOP_MM):
    up = np.round(np.arange(0.0, top + 1e-9, step), 3)
    z = [up]
    for i in range(1, n_passes):
        z.append(up[::-1][1:] if i % 2 else up[1:])
    return np.concatenate(z)


def noise_floor(z):
    """Out-of-focus level: bright, large defocused crop far from focus (z ~ 0),
    compact dark crop near focus (variance_laplacian-like magnitudes)."""
    return 85.0 + 25.0 * np.clip((10.0 - z) / 10.0, 0.0, 1.0)


def focus_bump(z, focus_z, amplitude, sigma_mm=0.3):
    return amplitude * np.exp(-0.5 * ((z - focus_z) / sigma_mm) ** 2)


def make_track(z, sharpness, frames=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    frames = np.arange(1, len(z) + 1) if frames is None else np.asarray(frames)
    values = np.asarray(sharpness, dtype=float) + (rng.normal(0, noise, len(z)) if noise else 0.0)
    return pd.DataFrame({'frame': frames, 'z_mm': z, SHARP: values, 'track_id': 1})


def peak_z(df, cfg=CFG, sweep_df=None):
    sweep = focus_sweep_geometry(sweep_df if sweep_df is not None else df, 'frame', 'z_mm')
    idx = detect_peaks_in_track(df, SHARP, 'frame', sweep, cfg)
    return df['z_mm'].to_numpy()[idx]


def test_sawtooth_flyback_step_is_not_a_peak():
    # Track 5 (plate4_well20): the floor jumps 85 -> 110 at every flyback.
    z = sawtooth_z(6)
    df = make_track(z, noise_floor(z) + focus_bump(z, FOCUS_Z, 160.0), noise=1.5)
    found = peak_z(df)
    assert len(found) == 6
    assert np.all(np.abs(found - FOCUS_Z) <= STEP_MM + 1e-9)


def test_zigzag_turnaround_hump_is_not_a_peak():
    # Track 34 (plate4_well21): broad hump of the floor at the z ~ 0 turnaround.
    z = zigzag_z(11)
    df = make_track(z, noise_floor(z) + focus_bump(z, FOCUS_Z, 220.0), noise=1.5, seed=1)
    found = peak_z(df)
    assert len(found) == 11          # one crossing per pass
    assert np.all(np.abs(found - FOCUS_Z) <= STEP_MM + 1e-9)


def test_dark_track_detected_like_a_bright_one():
    # Track 33: a darker crop scales floor and peak down; an absolute floor of 100
    # missed it. The detector is scale-free.
    z = zigzag_z(5)
    bright = make_track(z, noise_floor(z) + focus_bump(z, FOCUS_Z, 160.0), noise=1.0, seed=2)
    dark = bright.assign(**{SHARP: bright[SHARP] * 0.4})
    assert np.array_equal(peak_z(dark), peak_z(bright))
    assert len(peak_z(dark)) == 5


def test_contaminant_in_the_same_pass_is_dropped():
    # Track 23 (plate4_well60): a fibre in the crop is in focus at z = 14.4, in the
    # same pass as the organism's own, stronger focus peak.
    z = sawtooth_z(6, step=0.6, top=14.4)
    sharpness = noise_floor(z) + focus_bump(z, 12.0, 125.0) + focus_bump(z, 14.4, 30.0)
    df = make_track(z, sharpness, noise=1.0, seed=3)
    assert np.allclose(peak_z(df), 12.0)
    without_pass_rule = peak_z(df, _cfg(one_peak_per_sweep_pass=False))
    assert np.isclose(without_pass_rule, 14.4).any()


def test_weaker_crossing_in_another_pass_is_kept():
    # Moving organisms cross focus with different sharpness in different passes;
    # a crossing at 25% of the strongest one is a real peak.
    z = sawtooth_z(3)
    sharpness = noise_floor(z)
    sweep_len = len(z) // 3
    amplitudes = [400.0, 100.0, 400.0]
    for i, amp in enumerate(amplitudes):
        part = slice(i * sweep_len, (i + 1) * sweep_len)
        sharpness[part] += focus_bump(z[part], 8.0, amp)
    df = make_track(z, sharpness, noise=1.0, seed=4)
    assert len(peak_z(df)) == 3


def test_noise_only_track_has_no_peak():
    z = sawtooth_z(4)
    df = make_track(z, noise_floor(z), noise=1.5, seed=5)
    assert peak_z(df).size == 0


def test_track_detected_only_near_focus_keeps_one_peak_per_burst():
    # External-BG tracks of objects visible only near their focal plane: the
    # sparse frames cannot show the out-of-focus level -> track-percentile baseline.
    z_all = sawtooth_z(6)
    frames_all = np.arange(1, len(z_all) + 1)
    near = np.abs(z_all - FOCUS_Z) <= 1.0
    z = z_all[near]
    df = make_track(z, 85.0 + focus_bump(z, FOCUS_Z, 160.0), frames=frames_all[near], noise=1.0, seed=6)
    sweep_df = pd.DataFrame({'frame': frames_all, 'z_mm': z_all})
    found = peak_z(df, sweep_df=sweep_df)
    assert len(found) == 6
    assert np.all(np.abs(found - FOCUS_Z) <= STEP_MM + 1e-9)


def test_peaks_at_track_ends_are_kept_and_flagged():
    z = sawtooth_z(3)
    sharpness = noise_floor(z) + focus_bump(z, 8.0, 200.0)
    df = make_track(z, sharpness, noise=1.0, seed=7)
    first_peak = int(np.argmin(np.abs(z[:74] - 8.0)))
    last_peak = int(np.argmin(np.abs(z[148:] - 8.0))) + 148
    trimmed = df.iloc[first_peak:last_peak + 1]
    filtered, diagnostics = filter_focused_frames(trimmed, 'track_id', 'frame', SHARP, 'z_mm', CFG)
    assert len(filtered) == 3
    assert filtered[FOCUS_PEAK_EDGE_COLUMN].tolist() == [True, False, True]
    assert diagnostics['n_selected'].tolist() == [3]


def test_no_peaks_gives_empty_selection_with_edge_column():
    z = sawtooth_z(2)
    df = make_track(z, noise_floor(z), noise=0.5, seed=8)
    filtered, _ = filter_focused_frames(df, 'track_id', 'frame', SHARP, 'z_mm', CFG)
    assert filtered.empty
    assert FOCUS_PEAK_EDGE_COLUMN in filtered.columns


def test_baseline_follows_the_flyback_step():
    z = sawtooth_z(2)
    df = make_track(z, noise_floor(z) + focus_bump(z, FOCUS_Z, 160.0))
    values = df[SHARP].to_numpy()
    baseline = estimate_sharpness_baseline(df['frame'].to_numpy(), values, STEP_MM, CFG)
    after_flyback = slice(74, 74 + 6)   # first frames of the second sweep (z = 0 .. 1)
    relative_rise = (values[after_flyback] - baseline[after_flyback]) / baseline[after_flyback]
    assert np.all(relative_rise < 0.5 * CFG['peak_detection']['min_rise_above_baseline'])
    at_focus = int(np.argmin(np.abs(z[:74] - FOCUS_Z)))
    assert abs(baseline[at_focus] - 85.0) < 2.0


def test_empty_and_all_nan_tracks_return_empty():
    z = sawtooth_z(1)
    sweep = focus_sweep_geometry(make_track(z, z), 'frame', 'z_mm')
    empty = pd.DataFrame({'frame': [], 'z_mm': [], SHARP: []})
    assert detect_peaks_in_track(empty, SHARP, 'frame', sweep, CFG).size == 0
    all_nan = make_track(z, np.full(len(z), np.nan))
    assert detect_peaks_in_track(all_nan, SHARP, 'frame', sweep, CFG).size == 0


def test_returns_sorted_positional_int_indices():
    z = sawtooth_z(3)
    df = make_track(z, noise_floor(z) + focus_bump(z, FOCUS_Z, 160.0)).iloc[10:].reset_index(drop=True)
    sweep = focus_sweep_geometry(df, 'frame', 'z_mm')
    idx = detect_peaks_in_track(df, SHARP, 'frame', sweep, CFG)
    assert idx.dtype.kind == 'i'
    assert np.all(np.diff(idx) > 0)
    assert np.allclose(df['z_mm'].to_numpy()[idx], FOCUS_Z)


def test_focus_sweep_geometry_step_and_passes():
    sawtooth = focus_sweep_geometry(make_track(sawtooth_z(2), np.zeros(148)), 'frame', 'z_mm')
    assert np.isclose(sawtooth.step_mm, STEP_MM)
    assert sawtooth.pass_of_frame.iloc[0] == sawtooth.pass_of_frame.iloc[73]
    assert sawtooth.pass_of_frame.iloc[75] != sawtooth.pass_of_frame.iloc[73]

    z = zigzag_z(3)
    zigzag = focus_sweep_geometry(make_track(z, np.zeros(len(z))), 'frame', 'z_mm')
    assert zigzag.pass_of_frame.nunique() == 3
    top = int(np.argmax(z))
    assert zigzag.pass_of_frame.iloc[top] == zigzag.pass_of_frame.iloc[top - 1]
    assert zigzag.pass_of_frame.iloc[top + 1] != zigzag.pass_of_frame.iloc[top]


def test_recording_without_focus_sweep_raises():
    df = make_track(np.full(20, 5.0), np.arange(20.0))
    with pytest.raises(DataValidationError, match='focus sweep'):
        focus_sweep_geometry(df, 'frame', 'z_mm')


def test_duplicate_detections_in_a_frame_use_the_sharper_one():
    # Stage 1 occasionally emits two near-identical detections of a track in one frame.
    z = sawtooth_z(1)
    df = make_track(z, noise_floor(z) + focus_bump(z, FOCUS_Z, 160.0))
    peak_row = int(np.argmin(np.abs(z - FOCUS_Z)))
    weaker_copy = df.iloc[[peak_row]].assign(**{SHARP: 90.0})
    duplicated = pd.concat([weaker_copy, df]).sort_values('frame', kind='stable').reset_index(drop=True)
    sweep = focus_sweep_geometry(duplicated, 'frame', 'z_mm')
    idx = detect_peaks_in_track(duplicated, SHARP, 'frame', sweep, CFG)
    assert len(idx) == 1
    assert duplicated[SHARP].iloc[idx[0]] == df[SHARP].iloc[peak_row]


@pytest.mark.parametrize('key', list(CFG['peak_detection']))
def test_missing_peak_detection_key_raises(key):
    cfg = copy.deepcopy(CFG)
    del cfg['peak_detection'][key]
    z = sawtooth_z(1)
    df = make_track(z, noise_floor(z))
    sweep = focus_sweep_geometry(df, 'frame', 'z_mm')
    with pytest.raises(ConfigError, match=key):
        detect_peaks_in_track(df, SHARP, 'frame', sweep, cfg)


@pytest.mark.parametrize('key, value', [
    ('baseline_window_mm', 0),
    ('baseline_min_window_frames', 2.5),
    ('baseline_min_coverage', 0),
    ('sparse_baseline_percentile', 101),
    ('min_rise_above_baseline', -0.1),
    ('min_fraction_of_strongest_rise', 1.5),
    ('one_peak_per_sweep_pass', 'yes'),
])
def test_invalid_peak_detection_value_raises(key, value):
    z = sawtooth_z(1)
    df = make_track(z, noise_floor(z))
    sweep = focus_sweep_geometry(df, 'frame', 'z_mm')
    with pytest.raises(ConfigError, match=key):
        detect_peaks_in_track(df, SHARP, 'frame', sweep, _cfg(**{key: value}))
