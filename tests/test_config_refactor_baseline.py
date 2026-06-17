"""Behavioral invariants locked across the config-streamlining refactor.

These tests assert on *resolved outputs* (merged hardware values, physical well
coordinates, realtime config field values, unified segmentation params, resolved
paths) rather than on the YAML structure.  The config layout is being
reorganized, but every value an operator or the hardware ultimately sees must
stay identical.  If a phase changes one of these numbers, that is a behavior
change and must be reviewed deliberately -- not absorbed silently.
"""
from __future__ import annotations

import os

import pytest
import yaml

from src.utils import config_loader
from src.utils.config_loader import get_unified_segmentation_params
from src.acquire.config import get_plate_settings
from src.acquire.plate_layout import (
    calculate_coords_24wellplate,
    calculate_coords_96wellplate,
)
from src.utils.path_helpers import get_output_dir_for_sample, get_project_base_dir

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")


def _load_with_profile(profile_name: str) -> dict:
    """Load the real config and merge a chosen hardware profile (bypasses lru_cache)."""
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    config_loader._replace_null_with_none(cfg)
    config_loader._assemble_system_view(cfg)
    cfg["active_hardware_profile"] = profile_name
    config_loader._apply_hardware_profile(cfg)
    config_loader._link_stage_controller(cfg)
    config_loader._link_acquisition_output_root(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Hardware-profile merge (Phase 1 / 3 sensitive)
# ---------------------------------------------------------------------------
def test_hardware_merge_s5vpj6415():
    acq = _load_with_profile("S5VPJ6415")["acquisition"]
    assert acq["camera"]["exposure_us"] == 750
    assert acq["roi"]["width"] == 4512 and acq["roi"]["height"] == 4512
    assert acq["focusing"]["scan_ratio"] == 4.89
    assert acq["focusing"]["scan_height_mm"] == 11
    assert acq["focusing"]["lens_controller_port"] == "COM10"
    # plate.type is decoupled from the hardware profile: it stays the base value.
    assert acq["plate"]["type"] == 96
    # Zaber stage settings merged from the hardware profile.
    sc = acq["stage_controller"]
    assert sc["serial_port"] == "COM4"
    assert sc["device_x_number"] == 1 and sc["device_y_number"] == 2
    assert sc["microstep_size_um"] == 0.15625
    assert sc["position_tolerance_microsteps"] == 10
    # raw_data_root is injected as the acquisition output root (single source of truth).
    assert acq["paths"]["output_folder_base"] == "C:/wellscanner/test/"


def test_hardware_merge_ts1():
    acq = _load_with_profile("TS1_0638_70_EL")["acquisition"]
    assert acq["camera"]["exposure_us"] == 750
    assert acq["roi"]["width"] == 3600 and acq["roi"]["height"] == 3600
    assert acq["focusing"]["scan_ratio"] == 2.25
    assert acq["focusing"]["scan_height_mm"] == 11.2
    assert acq["focusing"]["lens_controller_port"] == "COM5"
    # plate.type is decoupled: even under the (24-well) TS1 profile it stays the
    # base run-level selection (96) unless explicitly overridden for a run.
    assert acq["plate"]["type"] == 96
    assert acq["stage_controller"]["serial_port"] == "COM4"
    # TS1 supplies the 24-well origin via profile override.
    assert acq["plate"]["definitions"]["24"]["origin_x"] == 6.387


# ---------------------------------------------------------------------------
# Physical well coordinates -- the values the stage actually drives to.
# ---------------------------------------------------------------------------
def test_plate_coordinates_s5vpj6415_96well():
    acq = _load_with_profile("S5VPJ6415")["acquisition"]
    plate_type, settings = get_plate_settings(acq)
    assert plate_type == "96"
    assert settings["start_x"] == 20.3 and settings["start_y"] == 14.5
    assert settings["spacing"] == 9.0
    assert settings["origin_x"] == 17.096 and settings["origin_y"] == 14.981

    df = calculate_coords_96wellplate(settings)
    assert len(df) == 96
    first, last = df.iloc[0].to_dict(), df.iloc[-1].to_dict()
    assert first["wellname"] == "A1"
    assert first["X"] == pytest.approx(3.204) and first["Y"] == pytest.approx(0.0)
    assert first["negY"] is True
    assert last["wellname"] == "H12"
    assert last["X"] == pytest.approx(66.204) and last["Y"] == pytest.approx(98.519)


def test_plate_coordinates_ts1_24well():
    acq = _load_with_profile("TS1_0638_70_EL")["acquisition"]
    # Select the 24-well plate at run level (decoupled from the profile).
    acq["plate"]["type"] = 24
    plate_type, settings = get_plate_settings(acq)
    assert plate_type == "24"
    assert settings["start_x"] == 15.74 and settings["start_y"] == 18.88
    assert settings["spacing"] == 18.0
    assert settings["origin_x"] == 6.387 and settings["origin_y"] == 17.963

    df = calculate_coords_24wellplate(settings)
    assert len(df) == 24
    first, last = df.iloc[0].to_dict(), df.iloc[-1].to_dict()
    assert first["wellname"] == "A1"
    assert first["X"] == pytest.approx(9.353) and first["Y"] == pytest.approx(0.917)
    assert last["wellname"] == "D6"
    assert last["X"] == pytest.approx(63.353) and last["Y"] == pytest.approx(90.917)


# ---------------------------------------------------------------------------
# Realtime config resolved values (Phase 2/4/5/6 sensitive)
# ---------------------------------------------------------------------------
def _load_realtime_config_module():
    # Load src/realtime/config.py directly to avoid the package __init__ chain,
    # which imports the (heavy, optionally-installed) realtime pipeline modules.
    import importlib.util
    import sys

    rt_config_path = os.path.join(os.path.dirname(__file__), "..", "src", "realtime", "config.py")
    spec = importlib.util.spec_from_file_location("rtcfg_baseline", rt_config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rtcfg_baseline"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_realtime_config_resolved_values():
    load_realtime_config = _load_realtime_config_module().load_realtime_config

    rt = load_realtime_config(CONFIG_PATH)
    # hardware-derived (active profile S5VPJ6415)
    assert rt.exposure_us == 750.0
    assert rt.roi_width == 4512 and rt.roi_height == 4512
    assert rt.scan_ratio == 4.89
    assert rt.scan_height_mm == 11.0
    assert rt.lens_controller_port == "COM10"
    # realtime-specific
    assert rt.output_root_dir == "data/output/realtime"
    assert rt.target_fps == 5.0
    assert rt.min_particle_area == 5000.0
    assert rt.max_stage2_candidates_per_frame == 200
    assert rt.start_dpt == pytest.approx(-1.074)
    assert rt.focus_pattern == "zigzag"
    assert rt.scan_step_um == 150.0
    # shared thresholding pulled from canonical section
    assert rt.stage2_segmentation_params["otsu_threshold_multiplier"] == 0.7
    assert rt.stage1_config["global_threshold_config"]["fixed_threshold_value"] == 70


# ---------------------------------------------------------------------------
# Unified segmentation params (Phase 6 sensitive -- egg params currently merged)
# ---------------------------------------------------------------------------
def test_unified_segmentation_params_snapshot():
    cfg = _load_with_profile("S5VPJ6415")
    params = get_unified_segmentation_params(cfg)
    # global thresholding + segmentation
    assert params["thresholding_mode"] == "otsu"
    assert params["otsu_threshold_multiplier"] == 0.7
    assert params["fixed_threshold_value"] == 70
    assert params["source_image_illumination_type"] == "collimated"
    assert params["min_significant_area"] == 300
    # egg-detection keys are decoupled: they must NOT leak into the shared params.
    assert "egg_min_area" not in params
    assert "egg_max_area" not in params
    assert "egg_min_solidity" not in params


# ---------------------------------------------------------------------------
# Output path resolution (Phase 2 sensitive) -- the final string must not move.
# ---------------------------------------------------------------------------
def test_output_path_resolution_structure(tmp_path):
    cfg = {"paths": {"pipeline_base": str(tmp_path)}}
    out = get_output_dir_for_sample(cfg, sample_subdir_name="sampleA", batch_name="batch1")
    expected = os.path.join(str(tmp_path), "data", "output", "batch1", "sampleA")
    assert out == expected
    assert get_project_base_dir(cfg) == str(tmp_path)


def test_legacy_project_base_dir_fallback(tmp_path):
    # Configs not yet migrated to paths.pipeline_base still resolve.
    cfg = {"project_base_dir": str(tmp_path)}
    assert get_project_base_dir(cfg) == str(tmp_path)
