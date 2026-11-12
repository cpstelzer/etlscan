# tests/test_track_figure_style.py
"""Tests for the shared track-map style (Stage-1 tracks_3d/xy, Stage-2 reconstructed_trajectories_3d/xy).

Dozens of overlapping trajectories with ~10 in-focus points each read as clutter at the
general figure sizes, so the track maps draw thinner lines, small open rings and a grey
floor shadow per track, from one shared style.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.stages.stage_01_initial_detection import _build_tracking_figure, _build_tracking_xy_figure
from src.stages.stage_02_characterization import plot_3d_reconstructed, plot_xy_reconstructed
from src.visualization.figure_style import track_figure_style
from tests.test_figure_style import STYLE_CONFIG

EXTENT_MM = (15.0, 12.0, 3.2)
TRACK_SIZES = (STYLE_CONFIG['figure_style']['track_line_width'], STYLE_CONFIG['figure_style']['track_marker_size'])


@pytest.fixture
def tracks_df():
    """Three short tracks, each with z rising so the depth line differs from the floor."""
    rows = []
    for track_id in range(3):
        for frame in range(8):
            rows.append({
                'track_id': track_id, 'frame': frame,
                'x_mm': 2.0 + 4 * track_id + 0.1 * frame,
                'y_mm': 3.0 + 0.2 * frame,
                'z_mm': 1.0 + 0.2 * frame,
                'z_reconstructed': 1.0 + 0.2 * frame,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def saved_axes(monkeypatch):
    """Capture the axes of Stage-2 plots, which save and close internally."""
    captured = []
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: captured.append(plt.gcf().axes[0]))
    return captured


def _is_ring(line):
    return line.get_marker() == 'o' and line.get_linestyle() == 'None'


def test_track_figure_style_sets_track_sizes_and_clear_3d_panes():
    line_width_before = plt.rcParams['lines.linewidth']

    with track_figure_style(STYLE_CONFIG):
        assert (plt.rcParams['lines.linewidth'], plt.rcParams['lines.markersize']) == TRACK_SIZES
        assert plt.rcParams['savefig.dpi'] == 321  # the rest of the figure style still applies
        for axis in 'xyz':
            assert plt.rcParams[f'axes3d.{axis}axis.panecolor'][3] == 0.0

    assert plt.rcParams['lines.linewidth'] == line_width_before


def test_reconstructed_3d_draws_shadows_then_lines_then_rings(tracks_df, saved_axes, tmp_path):
    selected_df = tracks_df[tracks_df['frame'] % 3 == 0]
    with track_figure_style(STYLE_CONFIG):
        plot_3d_reconstructed(
            tracks_df, selected_df, 'track_id', 'frame', 'z_mm',
            str(tmp_path / 'tracks_3d.png'), None, 0.5, 5, EXTENT_MM
        )

    lines = saved_axes[0].lines
    shadows, track_lines, rings = lines[:3], lines[3:6], lines[6:]
    assert len(rings) == 3

    # Shadows lie on the z = 0 floor, below every track line.
    assert all((shadow.get_data_3d()[2] == 0).all() for shadow in shadows)
    # Rings come after every line, so no line covers them, and are open rings in their line's colour.
    for line, ring in zip(track_lines, rings):
        assert not _is_ring(line) and _is_ring(ring)
        assert ring.get_markerfacecolor() == 'white'
        assert ring.get_markeredgecolor() == line.get_color()
        assert (line.get_linewidth(), ring.get_markersize()) == TRACK_SIZES
    # Three tracks, three distinct colours.
    assert len({line.get_color() for line in track_lines}) == 3


def test_reconstructed_xy_keeps_the_3d_colours(tracks_df, saved_axes, tmp_path):
    with track_figure_style(STYLE_CONFIG):
        plot_3d_reconstructed(
            tracks_df, tracks_df, 'track_id', 'frame', 'z_mm',
            str(tmp_path / 'tracks_3d.png'), None, 0.5, 5, EXTENT_MM
        )
        plot_xy_reconstructed(
            tracks_df, tracks_df, 'track_id', 'frame', str(tmp_path / 'tracks_xy.png'), None, EXTENT_MM
        )

    colours_3d = [line.get_color() for line in saved_axes[0].lines[3:6]]
    colours_xy = [line.get_color() for line in saved_axes[1].lines[:3]]
    assert colours_xy == colours_3d


def test_stage1_track_maps_share_the_track_style(tracks_df):
    with track_figure_style(STYLE_CONFIG):
        fig_3d = _build_tracking_figure(tracks_df, EXTENT_MM)
        fig_xy = _build_tracking_xy_figure(tracks_df, EXTENT_MM)

    lines_3d = fig_3d.axes[0].lines
    assert all((shadow.get_data_3d()[2] == 0).all() for shadow in lines_3d[:3])
    colours_3d = [line.get_color() for line in lines_3d[3:]]
    colours_xy = [line.get_color() for line in fig_xy.axes[0].lines]
    assert colours_xy == colours_3d and len(set(colours_3d)) == 3
    assert {line.get_linewidth() for line in fig_xy.axes[0].lines} == {TRACK_SIZES[0]}
    plt.close(fig_3d)
    plt.close(fig_xy)
