# tests/test_tracking_plot_extent.py
"""Tests that the Stage-1/Stage-2 tracking plots span the full imaged volume.

Matplotlib autoscales to the data it is given, so a well whose organisms sit in
one corner and barely move rendered zoomed onto that corner: the particle-free
part of the arena was cropped away and no two wells shared a scale. The plots now
pin their axes to the extent of the recording, read from its own frames.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from src.utils.acquisition_metadata import get_scan_extent_mm
from src.utils.common_exceptions import DataValidationError
from src.stages.stage_01_initial_detection import (
    _build_tracking_figure,
    _build_tracking_xy_figure,
)
from src.stages.stage_02_characterization import (
    plot_3d_reconstructed,
    plot_side_view_reconstructed,
    plot_xy_reconstructed,
)

# Deliberately non-square in x/y so an axis mix-up cannot pass unnoticed.
EXTENT_MM = (15.0, 12.0, 3.2)


@pytest.fixture
def corner_tracks_df():
    """Two barely-moving tracks parked in one corner of a much larger arena."""
    rows = []
    for track_id, (center_x, center_y) in enumerate([(3.1, 8.2), (3.2, 8.3)]):
        for frame in range(12):
            rows.append({
                'track_id': track_id,
                'frame': frame,
                'x_mm': center_x + frame * 0.001,
                'y_mm': center_y + frame * 0.001,
                'z_mm': 1.5 + frame * 0.001,
            })
    return pd.DataFrame(rows)


def _limits_at_draw(ax):
    """Read axis limits after a draw, since set_aspect only resolves then."""
    ax.figure.canvas.draw()
    return ax.get_xlim(), ax.get_ylim()


# The recording's pixel scale (read_recording_parameters): 2.0 um pixels at 0.5x = 0.004 mm.
# No config is handed in: the camera ROI and scan height in it describe the rig as set up
# now, not the recording.
PIXEL_TO_MM_FACTOR = 0.004


def _write_frame(directory, frame, fd_mm, width_px, height_px):
    name = f"20251007_plate0_well19_f{frame:05d}_dpt{-2.0 + fd_mm / 2.25:.3f}_fd{fd_mm:.2f}mm.png"
    Image.new('L', (width_px, height_px)).save(os.path.join(directory, name))


def test_scan_extent_derived_from_recorded_frames(tmp_path):
    # Zigzag sweep: the deepest plane is reached mid-run, not at the last frame.
    for frame, fd_mm in enumerate([0.0, 0.6, 1.2, 0.6, 0.0]):
        _write_frame(tmp_path, frame, fd_mm, width_px=1000, height_px=500)
    (tmp_path / 'metadata_log.csv').write_text('# not a frame\n')

    # 2.0 um at 0.5x = 4 um = 0.004 mm per pixel, over 1000x500 px frames; deepest plane 1.2 mm.
    assert get_scan_extent_mm(str(tmp_path), PIXEL_TO_MM_FACTOR) == pytest.approx((4.0, 2.0, 1.2))


def test_scan_extent_without_frames_fails(tmp_path):
    with pytest.raises(DataValidationError, match="No acquisition frames"):
        get_scan_extent_mm(str(tmp_path), PIXEL_TO_MM_FACTOR)


def test_stage1_3d_figure_spans_full_extent(corner_tracks_df):
    fig = _build_tracking_figure(corner_tracks_df, EXTENT_MM)
    ax = fig.axes[0]

    xlim, ylim = _limits_at_draw(ax)
    assert xlim == pytest.approx((0.0, 15.0))
    assert ylim == pytest.approx((0.0, 12.0))
    assert ax.get_zlim() == pytest.approx((0.0, 3.2))
    plt.close(fig)


def test_stage1_xy_figure_spans_full_extent(corner_tracks_df):
    fig = _build_tracking_xy_figure(corner_tracks_df, EXTENT_MM)
    ax = fig.axes[0]

    # Equal aspect must not claw back the fixed limits: adjustable='datalim'
    # would rescale them at draw time to fill the (non-square) axes box.
    xlim, ylim = _limits_at_draw(ax)
    assert xlim == pytest.approx((0.0, 15.0))
    assert ylim == pytest.approx((0.0, 12.0))
    plt.close(fig)


@pytest.fixture
def drawn_limits(monkeypatch):
    """Capture the drawn axis limits of Stage-2 plots, which save and close internally."""
    captured = []

    def spy_savefig(*args, **kwargs):
        ax = plt.gcf().axes[0]
        ax.figure.canvas.draw()
        captured.append((ax.get_xlim(), ax.get_ylim()))

    monkeypatch.setattr(plt, 'savefig', spy_savefig)
    return captured


def test_stage2_3d_reconstructed_spans_full_extent(corner_tracks_df, drawn_limits, tmp_path):
    plot_3d_reconstructed(
        corner_tracks_df, corner_tracks_df, 'track_id', 'frame', 'z_mm',
        str(tmp_path / 'tracks_3d.png'), None, 0.5, 5, EXTENT_MM
    )

    xlim, ylim = drawn_limits[0]
    assert xlim == pytest.approx((0.0, 15.0))
    assert ylim == pytest.approx((0.0, 12.0))


def test_stage2_xy_reconstructed_spans_full_extent(corner_tracks_df, drawn_limits, tmp_path):
    plot_xy_reconstructed(
        corner_tracks_df, corner_tracks_df, 'track_id', 'frame',
        str(tmp_path / 'tracks_xy.png'), None, EXTENT_MM
    )

    xlim, ylim = drawn_limits[0]
    assert xlim == pytest.approx((0.0, 15.0))
    assert ylim == pytest.approx((0.0, 12.0))


@pytest.mark.parametrize("horizontal_axis, horizontal_max_mm", [('x', 15.0), ('y', 12.0)])
def test_stage2_side_views_span_full_extent(corner_tracks_df, drawn_limits, tmp_path,
                                            horizontal_axis, horizontal_max_mm):
    plot_side_view_reconstructed(
        corner_tracks_df, corner_tracks_df, 'track_id', 'frame', 'z_mm',
        str(tmp_path / f'tracks_{horizontal_axis}z.png'), None, 0.5, 5, EXTENT_MM, horizontal_axis
    )

    # Equal x/z scale is set via the box, so the fixed limits survive the draw.
    xlim, ylim = drawn_limits[0]
    assert xlim == pytest.approx((0.0, horizontal_max_mm))
    assert ylim == pytest.approx((0.0, 3.2))
