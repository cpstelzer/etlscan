"""Config variants via ``base_config``: a variant holds only the values it changes and is
deep-merged over the config it names, so it cannot drift from the main config."""
import copy
from pathlib import Path

import pytest

from src.utils.common_exceptions import ConfigError
from src.utils.config_loader import load_config_once

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_variant_overrides_only_its_values(tmp_path):
    (tmp_path / "base.yml").write_text("a:\n  b: 1\n  c: 2\nd: [1, 2]\n", encoding="utf-8")
    (tmp_path / "variant.yml").write_text('base_config: "base.yml"\na:\n  c: 3\n', encoding="utf-8")

    config = load_config_once(str(tmp_path / "variant.yml"))

    assert config["a"] == {"b": 1, "c": 3}
    assert config["d"] == [1, 2]
    assert "base_config" not in config


def test_circular_base_config_raises(tmp_path):
    (tmp_path / "one.yml").write_text('base_config: "two.yml"\nx: 1\n', encoding="utf-8")
    (tmp_path / "two.yml").write_text('base_config: "one.yml"\ny: 2\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular base_config"):
        load_config_once(str(tmp_path / "one.yml"))


def test_external_bg_variant_differs_from_main_only_in_stage2_peak_gating():
    main = load_config_once(str(CONFIG_DIR / "config.yml"))
    variant = load_config_once(str(CONFIG_DIR / "config_external_bg.yml"))

    expected = copy.deepcopy(main)
    expected["stage_02_characterization"]["focussed_particles"]["sharpness_min_threshold"] = 0
    expected["stage_02_characterization"]["focussed_particles"]["min_prominence_fraction"] = 0.2
    assert variant == expected
