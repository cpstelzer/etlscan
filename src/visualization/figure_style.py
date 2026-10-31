# src/visualization/figure_style.py
"""Shared matplotlib style for every figure the pipeline writes.

All sizes come from ``config['figure_style']``; plot code sets no font sizes, line
widths, marker sizes or dpi of its own, so every figure follows the same style.
"""
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import matplotlib.pyplot as plt

from src.utils.common_exceptions import ConfigError

_FIGURE_STYLE_KEYS = (
    'dpi', 'font_family', 'font_size', 'tick_label_size', 'title_size', 'line_width',
    'marker_size', 'marker_edge_width', 'axes_line_width', 'tick_length',
)


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
