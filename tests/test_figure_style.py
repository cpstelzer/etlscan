# tests/test_figure_style.py
"""Tests for the shared figure style (config['figure_style']) and the Stage-4 overlay that uses it."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage.measure

from src.utils.common_exceptions import ConfigError
from src.utils.config_loader import load_config_once
from src.visualization.feature_visualization import visualize_ml_features
from src.visualization.figure_style import figure_style

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

STYLE_CONFIG = {
    'figure_style': {
        'dpi': 321,
        'font_family': ['DejaVu Sans'],
        'font_size': 17,
        'tick_label_size': 13,
        'title_size': 19,
        'line_width': 2.7,
        'marker_size': 6.5,
        'marker_edge_width': 0.9,
        'axes_line_width': 1.4,
        'tick_length': 5.5,
    }
}


def test_figure_style_applies_config_inside_block_and_restores_after():
    line_width_before = plt.rcParams['lines.linewidth']

    with figure_style(STYLE_CONFIG):
        assert plt.rcParams['savefig.dpi'] == 321
        assert plt.rcParams['axes.labelsize'] == 17
        assert plt.rcParams['xtick.labelsize'] == 13
        assert plt.rcParams['legend.fontsize'] == 13
        assert plt.rcParams['axes.titlesize'] == 19
        assert plt.rcParams['lines.linewidth'] == 2.7
        assert plt.rcParams['lines.markersize'] == 6.5
        assert plt.rcParams['lines.markeredgewidth'] == 0.9
        assert plt.rcParams['axes.linewidth'] == 1.4
        assert plt.rcParams['ytick.major.size'] == 5.5

    assert plt.rcParams['lines.linewidth'] == line_width_before


def test_figure_style_fails_fast_without_section():
    with pytest.raises(ConfigError, match="figure_style"):
        with figure_style({}):
            pass


def test_figure_style_names_missing_keys():
    incomplete = {'figure_style': {k: v for k, v in STYLE_CONFIG['figure_style'].items() if k != 'line_width'}}
    with pytest.raises(ConfigError, match="line_width"):
        with figure_style(incomplete):
            pass


@pytest.mark.parametrize("config_name", ["config.yml", "config_external_bg.yml"])
def test_repo_configs_define_a_complete_figure_style(config_name):
    with figure_style(load_config_once(str(CONFIG_DIR / config_name))):
        pass


def test_stage4_overlay_outlines_regions_and_legends_every_element(monkeypatch, tmp_path):
    """visualize_ml_features logs instead of raising, so check it reaches savefig with all overlays."""
    image = np.full((80, 50), 200, dtype=np.uint8)
    rows, cols = np.ogrid[:80, :50]
    body_mask = ((rows - 40) / 30.0) ** 2 + ((cols - 25) / 12.0) ** 2 <= 1
    egg_mask = body_mask & (rows > 58)
    image[body_mask] = 60
    main_props = skimage.measure.regionprops(body_mask.astype(int))[0]
    egg_props = skimage.measure.regionprops(egg_mask.astype(int))[0]

    saved_legends = []

    def spy_savefig(*args, **kwargs):
        legend = plt.gcf().axes[0].get_legend()
        saved_legends.append([text.get_text() for text in legend.get_texts()])

    monkeypatch.setattr(plt, 'savefig', spy_savefig)
    with figure_style(STYLE_CONFIG):
        visualize_ml_features(
            image_path_or_array=image,
            main_props=main_props,
            egg_area_props=egg_props,
            output_visualization_path=str(tmp_path / "overlay.png"),
            ml_feature_params={'egg_posterior_pixel_fraction': 0.35},
        )

    assert saved_legends == [[
        "Main body", "Egg area", "Major axis", "Posterior end", "Posterior 35% area"
    ]]
