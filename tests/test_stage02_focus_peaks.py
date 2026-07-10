# tests/test_stage02_focus_peaks.py
"""Regression tests for Stage-2 sharpness-peak detection (detect_peaks_in_track).

These lock in the fix for closely-spaced / fast-focus-sweep peaks that the
previous rolling-mean + fixed-window-median-prominence detector rejected.
"""
import numpy as np
import pandas as pd

from src.stages.stage_02_characterization import detect_peaks_in_track

SHARP = 'variance_laplacian'

BASE_CFG = {
    'smooth_window': 1,
    'min_sharpness_fraction': 0.15,
    'min_prominence_fraction': 0.05,
    'sharpness_min_threshold': 90,
    'min_peak_distance_frames': 2,
}


def make_track(peak_specs, sigma, n=90, baseline=95.0, noise=0.0, seed=0):
    """Build a frame-sorted track with Gaussian sharpness bumps."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    y = np.full(n, baseline, dtype=float)
    for frame, amp in peak_specs:
        y += amp * np.exp(-0.5 * ((x - frame) / sigma) ** 2)
    if noise:
        y = y + rng.normal(0, noise, n)
    y = np.clip(y, 0, None)
    return pd.DataFrame({'frame': x, SHARP: y})


def detected_frames(df, cfg=None):
    cfg = cfg if cfg is not None else BASE_CFG
    idx = detect_peaks_in_track(df, SHARP, cfg)
    return np.sort(df['frame'].to_numpy()[idx])


def _all_matched(detected, expected, tol=1):
    return all(any(abs(d - e) <= tol for d in detected) for e in expected)


def test_closely_spaced_varying_height_all_detected():
    # Fast sweep: 14 narrow peaks ~5 frames apart with varying heights.
    frames = list(range(6, 73, 5))
    heights = np.random.default_rng(42).uniform(160, 650, len(frames))
    df = make_track(list(zip(frames, heights)), sigma=1.0, noise=10.0, seed=42)
    det = detected_frames(df)
    assert _all_matched(det, frames, tol=1)
    assert len(det) == len(frames)


def test_broad_sparse_track_unchanged():
    # TS1-style slow sweep: broad, well-separated peaks (must not regress).
    frames = [12, 34, 56]
    df = make_track([(f, 600) for f in frames], sigma=2.2, noise=8.0, seed=1)
    det = detected_frames(df)
    assert _all_matched(det, frames, tol=1)
    assert len(det) == 3


def test_single_frame_spikes_detected():
    # 1-frame delta spikes that the old centered mean flattened into plateaus.
    df = pd.DataFrame({'frame': np.arange(30), SHARP: np.full(30, 95.0)})
    spikes = [6, 13, 20]
    for f in spikes:
        df.loc[f, SHARP] = 700.0
    assert list(detected_frames(df)) == spikes


def test_very_dense_peaks_detected():
    # Spacing 3 (>= distance 2): all peaks survive.
    frames = list(range(10, 82, 3))
    df = make_track([(f, 600) for f in frames], sigma=0.8, noise=0.0, seed=5)
    det = detected_frames(df)
    assert _all_matched(det, frames, tol=1)


def test_short_peaks_retained_with_low_fraction():
    # One tall peak + three peaks at 20% of it: kept under the relaxed fraction.
    specs = [(15, 900), (30, 180), (45, 180), (60, 180)]
    df = make_track(specs, sigma=1.5, noise=0.0, seed=7)
    det = detected_frames(df)
    assert _all_matched(det, [15, 30, 45, 60], tol=1)
    assert len(det) == 4


def test_min_peak_distance_merges_close_peaks():
    # Two local maxima 3 frames apart: distance=2 keeps both, distance=4 merges.
    df = pd.DataFrame({'frame': np.arange(20), SHARP: np.full(20, 95.0)})
    df.loc[9, SHARP] = 600.0
    df.loc[12, SHARP] = 500.0
    near = detect_peaks_in_track(df, SHARP, dict(BASE_CFG, min_peak_distance_frames=2))
    assert len(near) == 2
    far = detect_peaks_in_track(df, SHARP, dict(BASE_CFG, min_peak_distance_frames=4))
    assert len(far) == 1
    assert df['frame'].to_numpy()[far][0] == 9


def test_returns_positional_int_indices():
    frames = [10, 30, 50]
    df = make_track([(f, 600) for f in frames], sigma=1.5, noise=0.0, seed=3)
    idx = detect_peaks_in_track(df, SHARP, BASE_CFG)
    assert idx.dtype.kind == 'i'
    assert _all_matched(np.sort(df['frame'].to_numpy()[idx]), frames, tol=1)


def test_empty_track_returns_empty():
    df = pd.DataFrame({'frame': [], SHARP: []})
    assert detect_peaks_in_track(df, SHARP, BASE_CFG).size == 0


def test_all_nan_returns_empty():
    df = pd.DataFrame({'frame': range(10), SHARP: [np.nan] * 10})
    assert detect_peaks_in_track(df, SHARP, BASE_CFG).size == 0


def test_flat_below_floor_returns_empty():
    df = pd.DataFrame({'frame': range(10), SHARP: [50.0] * 10})
    assert detect_peaks_in_track(df, SHARP, BASE_CFG).size == 0
