# src/visualization/figure_style.py
"""Shared matplotlib style for every figure the pipeline writes.

All sizes come from ``config['figure_style']``; plot code sets no font sizes, line
widths, marker sizes or dpi of its own, so every figure follows the same style.

The track maps (Stage-1 ``*_tracks_3d/xy``, Stage-2 ``reconstructed_trajectories_3d/xy/xz/yz``)
use :func:`track_figure_style` instead and draw through the ``plot_track_line`` /
``plot_focus_points`` / ``draw_floor_shadow`` helpers below, so they all look alike.
"""
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.utils.common_exceptions import ConfigError

_FIGURE_STYLE_KEYS = (
    'dpi', 'font_family', 'font_size', 'tick_label_size', 'title_size', 'line_width',
    'marker_size', 'marker_edge_width', 'axes_line_width', 'tick_length',
    'track_line_width', 'track_marker_size',
)

# Fifteen hues of similar weight: tab10, then the darkest step of five tab20b hues. tab20's
# light/dark pairs of one hue made neighbouring tracks look like one track, and its pastels
# faded on white. Past fifteen tracks the colours repeat.
_TRACK_COLORS = (
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
)
_TRACK_LINE_ALPHA = 0.9
_TRACK_GRID_COLOR = '0.88'    # faint 3D grid, so the tracks rather than the box carry the figure
_FLOOR_SHADOW_COLOR = '0.78'  # darker than the grid, so the shadows stand out on the floor
_TRANSPARENT = (1.0, 1.0, 1.0, 0.0)


def _figure_style_rc_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate ``config['figure_style']`` into matplotlib rcParams (ConfigError if incomplete)."""
    style_config = config.get('figure_style')
    if style_config is None:
        raise ConfigError("Config section 'figure_style' is missing.")
    missing = [key for key in _FIGURE_STYLE_KEYS if key not in style_config]
    if missing:
        raise ConfigError(f"Config section 'figure_style' is missing keys: {missing}")

    return {
        'savefig.dpi': style_config['dpi'],
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': list(style_config['font_family']),
        'font.size': style_config['font_size'],
        'axes.labelsize': style_config['font_size'],
        'axes.titlesize': style_config['title_size'],
        'xtick.labelsize': style_config['tick_label_size'],
        'ytick.labelsize': style_config['tick_label_size'],
        'legend.fontsize': style_config['tick_label_size'],
        'lines.linewidth': style_config['line_width'],
        'lines.markersize': style_config['marker_size'],
        'lines.markeredgewidth': style_config['marker_edge_width'],
        'patch.linewidth': style_config['marker_edge_width'],
        'axes.linewidth': style_config['axes_line_width'],
        'xtick.major.width': style_config['axes_line_width'],
        'ytick.major.width': style_config['axes_line_width'],
        'xtick.major.size': style_config['tick_length'],
        'ytick.major.size': style_config['tick_length'],
    }


@contextmanager
def figure_style(config: Dict[str, Any]) -> Iterator[None]:
    """Apply the configured figure style to every figure created and saved inside the block."""
    with plt.rc_context(_figure_style_rc_params(config)):
        yield


@contextmanager
def track_figure_style(config: Dict[str, Any]) -> Iterator[None]:
    """Apply the figure style with the track-map line and marker sizes, white 3D panes and a faint 3D grid."""
    rc_params = _figure_style_rc_params(config)
    style_config = config['figure_style']
    rc_params.update({
        'lines.linewidth': style_config['track_line_width'],
        'lines.markersize': style_config['track_marker_size'],
        'grid.color': _TRACK_GRID_COLOR,
        'axes3d.xaxis.panecolor': _TRANSPARENT,
        'axes3d.yaxis.panecolor': _TRANSPARENT,
        'axes3d.zaxis.panecolor': _TRANSPARENT,
    })
    with plt.rc_context(rc_params):
        yield


def _track_color(rank: int) -> str:
    return _TRACK_COLORS[rank % len(_TRACK_COLORS)]


def plot_track_line(ax: Axes, coords: Sequence[Any], rank: int) -> None:
    """Draw one trajectory, ``coords`` = (x, y) or (x, y, z), in the colour of its rank among the plotted tracks."""
    ax.plot(*coords, color=_track_color(rank), alpha=_TRACK_LINE_ALPHA)


def plot_focus_points(ax: Axes, coords: Sequence[Any], rank: int) -> None:
    """Mark a track's in-focus measurements as open rings in the colour of the track's line.

    Open rings read as measurements along the line instead of covering it. In 3D they sit
    off the smoothed depth line by up to about a focal-plane step, which filled markers
    turned into apparent scatter.
    """
    ax.plot(
        *coords, marker='o', linestyle='none', markerfacecolor='white',
        markeredgecolor=_track_color(rank)
    )


def draw_floor_shadow(ax: Axes, x_mm: Sequence[float], y_mm: Sequence[float]) -> None:
    """Draw a 3D track's grey copy on the z = 0 floor, so its x/y position can be read off the 3D view.

    Drawn at the grid's line weight: like the grid, it is background for the tracks.
    """
    ax.plot(
        x_mm, y_mm, np.zeros(len(x_mm)), color=_FLOOR_SHADOW_COLOR,
        linewidth=plt.rcParams['grid.linewidth']
    )
