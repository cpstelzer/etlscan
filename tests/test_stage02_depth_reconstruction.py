# tests/test_stage02_depth_reconstruction.py
"""Tests for Stage-2 3D depth (z) reconstruction from sparse in-focus points.

Covers the cubic-smoothing-spline reconstruction (restored from the
publication-3d-plot-experimental branch) and its persistence as a per-frame
``z_reconstructed`` column.
"""
import numpy as np
import pandas as pd

from src.stages.stage_02_characterization import (
    reconstruct_track_depth,
    add_reconstructed_depth,
    attach_reconstructed_depth_to_subset,
    FOCUS_Z_RECONSTRUCTED_COLUMN,
)


def test_rising_depth_yields_rising_reconstruction():
    # An organism swimming upward: in-focus depths increase over frames.
    knot_frames = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    knot_z = np.array([0.5, 0.8, 1.2, 1.6, 2.1, 2.6], dtype=float)
    frames = np.arange(10, 61, dtype=float)

    z = reconstruct_track_depth(frames, knot_frames, knot_z, 0.5, 5)

    assert z.shape == frames.shape
    assert np.all(np.isfinite(z))
    assert z[-1] > z[0] + 1.0                      # clear upward trend preserved
    assert z.min() >= knot_z.min() - 0.5           # smoothing stays near measured span
    assert z.max() <= knot_z.max() + 0.5


def test_evaluation_clamped_outside_knot_span():
    knot_frames = np.array([20, 30, 40, 50, 60], dtype=float)
    knot_z = np.array([1.0, 1.2, 1.4, 1.6, 1.8], dtype=float)
    frames = np.array([10, 20, 60, 70], dtype=float)  # 10 < first knot, 70 > last

    z = reconstruct_track_depth(frames, knot_frames, knot_z, 0.5, 5)
    z_at_first = reconstruct_track_depth(np.array([20.0]), knot_frames, knot_z, 0.5, 5)[0]
    z_at_last = reconstruct_track_depth(np.array([60.0]), knot_frames, knot_z, 0.5, 5)[0]

    assert np.isclose(z[0], z_at_first)   # frame 10 clamped to first knot frame
    assert np.isclose(z[-1], z_at_last)   # frame 70 clamped to last knot frame


def test_few_points_fall_back_to_constant_mean():
    knot_frames = np.array([10, 20, 30], dtype=float)
    knot_z = np.array([1.0, 2.0, 3.0], dtype=float)
    frames = np.arange(10, 31, dtype=float)

    z = reconstruct_track_depth(frames, knot_frames, knot_z, 0.5, 5)

    assert np.allclose(z, 2.0)  # mean of the in-focus depths


def test_no_knots_returns_all_nan():
    frames = np.arange(5, dtype=float)
    z = reconstruct_track_depth(frames, np.array([]), np.array([]), 0.5, 5)
    assert z.shape == frames.shape
    assert np.all(np.isnan(z))


def _cfg():
    return {
        'track_column': 'track_id',
        'frame_column': 'frame',
        'trajectory_smoothing_factor': 0.5,
        'min_points_for_spline': 5,
    }


def test_add_reconstructed_depth_adds_per_frame_column():
    # Track 1: rising, >= 5 knots -> splined. Track 2: 3 knots -> constant mean.
    rows = [{'track_id': 1, 'frame': f, 'z_mm': np.nan} for f in range(10, 61)]
    rows += [{'track_id': 2, 'frame': f, 'z_mm': np.nan} for f in (10, 20, 30)]
    comp = pd.DataFrame(rows)

    knots = [{'track_id': 1, 'frame': f, 'z_mm': z}
             for f, z in zip([10, 20, 30, 40, 50, 60], [0.5, 0.9, 1.3, 1.8, 2.3, 2.8])]
    knots += [{'track_id': 2, 'frame': f, 'z_mm': z}
              for f, z in zip([10, 20, 30], [1.0, 2.0, 3.0])]
    filtered = pd.DataFrame(knots)

    out = add_reconstructed_depth(comp, filtered, _cfg())

    assert FOCUS_Z_RECONSTRUCTED_COLUMN in out.columns
    t1 = out[out['track_id'] == 1].sort_values('frame')
    assert t1[FOCUS_Z_RECONSTRUCTED_COLUMN].notna().all()
    assert t1[FOCUS_Z_RECONSTRUCTED_COLUMN].iloc[-1] > t1[FOCUS_Z_RECONSTRUCTED_COLUMN].iloc[0] + 1.0
    t2 = out[out['track_id'] == 2]
    assert np.allclose(t2[FOCUS_Z_RECONSTRUCTED_COLUMN].to_numpy(), 2.0)


def test_add_reconstructed_depth_noop_when_no_selections():
    comp = pd.DataFrame({'track_id': [1, 1], 'frame': [1, 2], 'z_mm': [np.nan, np.nan]})
    out = add_reconstructed_depth(comp, pd.DataFrame(), _cfg())
    assert FOCUS_Z_RECONSTRUCTED_COLUMN not in out.columns


def test_add_reconstructed_depth_leaves_xy_unaltered():
    rows = [{'track_id': 1, 'frame': f, 'z_mm': np.nan, 'x_mm': 5.0 + f, 'y_mm': 7.0 - f}
            for f in range(10, 61)]
    comp = pd.DataFrame(rows)
    knots = pd.DataFrame([{'track_id': 1, 'frame': f, 'z_mm': z}
                          for f, z in zip([10, 20, 30, 40, 50, 60],
                                          [0.5, 0.9, 1.3, 1.8, 2.3, 2.8])])
    out = add_reconstructed_depth(comp, knots, _cfg())
    pd.testing.assert_series_equal(out['x_mm'], comp['x_mm'])
    pd.testing.assert_series_equal(out['y_mm'], comp['y_mm'])


def test_attach_reconstructed_depth_to_subset_carries_column():
    # per-frame table with reconstructed depth already computed
    comp = pd.DataFrame([{'track_id': 1, 'frame': f, 'z_mm': np.nan} for f in range(10, 61)])
    knots = pd.DataFrame([{'track_id': 1, 'frame': f, 'z_mm': z}
                          for f, z in zip([10, 20, 30, 40, 50, 60],
                                          [0.5, 0.9, 1.3, 1.8, 2.3, 2.8])])
    comp = add_reconstructed_depth(comp, knots, _cfg())

    subset = knots.copy()  # the in-focus selections, lacking the column
    assert FOCUS_Z_RECONSTRUCTED_COLUMN not in subset.columns

    out = attach_reconstructed_depth_to_subset(subset, comp, 'track_id', 'frame')

    assert FOCUS_Z_RECONSTRUCTED_COLUMN in out.columns
    assert len(out) == len(subset)
    # subset values equal the per-frame reconstruction at those frames
    check = out.merge(comp[['track_id', 'frame', FOCUS_Z_RECONSTRUCTED_COLUMN]],
                      on=['track_id', 'frame'], suffixes=('_sub', '_comp'))
    assert np.allclose(check[f'{FOCUS_Z_RECONSTRUCTED_COLUMN}_sub'],
                       check[f'{FOCUS_Z_RECONSTRUCTED_COLUMN}_comp'])


def test_attach_reconstructed_depth_noop_when_column_absent():
    comp = pd.DataFrame({'track_id': [1], 'frame': [10], 'z_mm': [1.0]})  # no z_reconstructed
    subset = pd.DataFrame({'track_id': [1], 'frame': [10]})
    out = attach_reconstructed_depth_to_subset(subset, comp, 'track_id', 'frame')
    assert FOCUS_Z_RECONSTRUCTED_COLUMN not in out.columns
    assert len(out) == 1
