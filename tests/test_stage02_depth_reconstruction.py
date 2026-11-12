# tests/test_stage02_depth_reconstruction.py
"""Tests for Stage-2 3D depth (z) reconstruction from sparse in-focus points.

Covers the cubic-smoothing-spline reconstruction (restored from the
publication-3d-plot-experimental branch), its persistence as a per-frame
``z_reconstructed`` column, and the speed gate that keeps lone implausible depth
knots (false focus peaks, identity swaps) out of the spline.
"""
import numpy as np
import pandas as pd
import pytest

from src.stages.stage_02_characterization import (
    reconstruct_track_depth,
    add_reconstructed_depth,
    attach_reconstructed_depth_to_subset,
    find_depth_knot_speed_outliers,
    flag_depth_knot_speed_outliers,
    DEPTH_KNOT_SPEED_OUTLIER_COLUMN,
    FOCUS_Z_RECONSTRUCTED_COLUMN,
)
from src.utils.common_exceptions import ConfigError, DataValidationError


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
        'depth_knot_speed_gate': {'max_vertical_speed_mm_s': 1.5, 'tolerance_planes': 2},
    }


def _unflagged(knots_df):
    """In-focus selections as flag_depth_knot_speed_outliers leaves knots it keeps."""
    return knots_df.assign(**{DEPTH_KNOT_SPEED_OUTLIER_COLUMN: False})


def test_add_reconstructed_depth_adds_per_frame_column():
    # Track 1: rising, >= 5 knots -> splined. Track 2: 3 knots -> constant mean.
    rows = [{'track_id': 1, 'frame': f, 'z_mm': np.nan} for f in range(10, 61)]
    rows += [{'track_id': 2, 'frame': f, 'z_mm': np.nan} for f in (10, 20, 30)]
    comp = pd.DataFrame(rows)

    knots = [{'track_id': 1, 'frame': f, 'z_mm': z}
             for f, z in zip([10, 20, 30, 40, 50, 60], [0.5, 0.9, 1.3, 1.8, 2.3, 2.8])]
    knots += [{'track_id': 2, 'frame': f, 'z_mm': z}
              for f, z in zip([10, 20, 30], [1.0, 2.0, 3.0])]
    filtered = _unflagged(pd.DataFrame(knots))

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
    knots = _unflagged(pd.DataFrame([{'track_id': 1, 'frame': f, 'z_mm': z}
                                     for f, z in zip([10, 20, 30, 40, 50, 60],
                                                     [0.5, 0.9, 1.3, 1.8, 2.3, 2.8])]))
    out = add_reconstructed_depth(comp, knots, _cfg())
    pd.testing.assert_series_equal(out['x_mm'], comp['x_mm'])
    pd.testing.assert_series_equal(out['y_mm'], comp['y_mm'])


def test_attach_reconstructed_depth_to_subset_carries_column():
    # per-frame table with reconstructed depth already computed
    comp = pd.DataFrame([{'track_id': 1, 'frame': f, 'z_mm': np.nan} for f in range(10, 61)])
    knots = _unflagged(pd.DataFrame([{'track_id': 1, 'frame': f, 'z_mm': z}
                                     for f, z in zip([10, 20, 30, 40, 50, 60],
                                                     [0.5, 0.9, 1.3, 1.8, 2.3, 2.8])]))
    comp = add_reconstructed_depth(comp, knots, _cfg())

    subset = knots.drop(columns=DEPTH_KNOT_SPEED_OUTLIER_COLUMN)  # in-focus selections, lacking the column
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


# --- depth-knot speed gate -------------------------------------------------------

FPS = 25.0
STEP_MM = 0.4


def _outliers(frames, z):
    return find_depth_knot_speed_outliers(
        np.asarray(frames, dtype=float), np.asarray(z, dtype=float), FPS, STEP_MM, _cfg()
    ).tolist()


def test_lone_end_knot_is_flagged():
    # Track starts on the sharp well rim near the sweep start, then follows an animal at ~9 mm.
    assert _outliers([4, 34, 80, 89, 137], [1.6, 8.8, 9.6, 9.2, 10.0]) == [True, False, False, False, False]


def test_lone_interior_knot_is_flagged():
    # One knot 7 mm off its neighbours 30 frames (1.2 s) away on either side.
    assert _outliers([22, 35, 78, 108, 138, 145], [8.8, 8.4, 8.8, 1.6, 10.4, 9.2]) == [
        False, False, False, True, False, False
    ]


def test_slow_drift_and_turnaround_pairs_are_kept():
    # 0.3 mm/s drift, and knot pairs one plane-pair apart across a sweep turnaround.
    frames = np.arange(0, 301, 30)
    assert not any(_outliers(frames, 2.0 + 0.3 * frames / FPS))
    assert not any(_outliers([137, 145, 193, 201], [10.0, 9.2, 10.0, 9.2]))


def test_agreeing_knots_are_kept_even_far_from_the_rest():
    # Two consecutive knots support each other (e.g. a swapped segment): the gate cannot tell.
    assert not any(_outliers([100, 130, 160, 170, 200, 230], [4.8, 4.8, 9.6, 9.6, 4.8, 4.8]))


def test_tracks_with_fewer_than_three_knots_are_not_gated():
    assert _outliers([4, 34], [1.6, 8.8]) == [False, False]


def test_speed_gate_validates_input():
    with pytest.raises(DataValidationError):
        _outliers([34, 4, 80], [8.8, 1.6, 9.6])                     # not sorted by frame
    with pytest.raises(DataValidationError):
        find_depth_knot_speed_outliers(np.arange(3.0), np.ones(3), 0.0, STEP_MM, _cfg())
    with pytest.raises(ConfigError):
        find_depth_knot_speed_outliers(np.arange(3.0), np.ones(3), FPS, STEP_MM, {})
    with pytest.raises(ConfigError):
        bad_cfg = dict(_cfg(), depth_knot_speed_gate={'max_vertical_speed_mm_s': 0, 'tolerance_planes': 2})
        find_depth_knot_speed_outliers(np.arange(3.0), np.ones(3), FPS, STEP_MM, bad_cfg)


def _zigzag_track(frames):
    """Per-frame table of one track on a 0.4 mm zigzag sweep 0..11.2 mm (period 56 frames)."""
    z = STEP_MM * (28 - np.abs(np.asarray(frames) % 56 - 28))
    return pd.DataFrame({'track_id': 1, 'frame': frames, 'z_mm': z})


def test_flagged_knot_stays_selected_but_does_not_bend_the_depth():
    comp = _zigzag_track(np.arange(1, 200))
    # In focus at 4.8 mm on every pass, plus one frame on the rim at the sweep start (z 0).
    knot_frames = [12, 44, 68, 100, 112, 124, 156]
    filtered = comp[comp['frame'].isin(knot_frames)].copy()

    flagged = flag_depth_knot_speed_outliers(filtered, comp, _cfg(), FPS)
    out = add_reconstructed_depth(comp, flagged, _cfg())

    assert flagged.loc[flagged['frame'] == 112, DEPTH_KNOT_SPEED_OUTLIER_COLUMN].tolist() == [True]
    assert flagged[DEPTH_KNOT_SPEED_OUTLIER_COLUMN].sum() == 1
    assert len(flagged) == len(filtered)                         # still an in-focus selection
    z_line = out.set_index('frame')[FOCUS_Z_RECONSTRUCTED_COLUMN]
    assert abs(z_line.loc[112] - 4.8) < 0.1
    assert np.allclose(z_line.loc[12:156], 4.8, atol=0.1)


def test_flag_passes_empty_selections_through():
    comp = _zigzag_track(np.arange(1, 60))
    assert flag_depth_knot_speed_outliers(None, comp, _cfg(), FPS) is None
    empty = comp.iloc[0:0]
    assert flag_depth_knot_speed_outliers(empty, comp, _cfg(), FPS) is empty


def test_add_reconstructed_depth_requires_the_speed_flag():
    comp = _zigzag_track(np.arange(1, 60))
    with pytest.raises(DataValidationError, match=DEPTH_KNOT_SPEED_OUTLIER_COLUMN):
        add_reconstructed_depth(comp, comp.iloc[[5, 10, 20]], _cfg())
