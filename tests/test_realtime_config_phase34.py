from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RT_CONFIG_PATH = REPO_ROOT / "src" / "realtime" / "config.py"


def _load_rt_config_module():
    spec = importlib.util.spec_from_file_location("rtcfg_test", RT_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rtcfg_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_realtime_loader_uses_pipeline_for_shared_analysis(tmp_path: Path) -> None:
    ws_cfg = {
        "focusing": {
            "pattern": "zigzag",
            "scan_height_mm": 11.2,
            "scan_step_um": 350,
            "scan_ratio": 2.25,
            "lens_settle_time_sec": 0.025,
            "large_jump_threshold_dpt": -3.0,
            "large_jump_settle_sec": 0.040,
            "lens_controller_port": "COM5",
        },
        "realtime_mode": {
            "enabled": True,
            "runtime": {"target_fps": 5, "max_processing_latency_ms": 250, "queue_size": 2},
            "background": {"bgi_min_age_s": 2.0, "bgi_max_age_s": 4.0},
            "acquisition": {"source": "camera", "camera_index": 0, "retrieve_timeout_margin_ms": 150},
            "stage1": {"foreground_diff_threshold": 10, "max_foreground_fraction": 0.08},
            "stage2": {"min_particle_area": 500, "max_candidates_per_frame": 200},
            "persistence": {"save_accepted_crops": True, "image_format": "png", "png_compression": 3},
            "logging": {"metric_print_interval_s": 5, "etl_status_interval_frames": 30, "focus_rejection_log_interval_frames": 30},
        },
    }
    pipeline_cfg = {
        "system": {"source_image_illumination_type": "collimated"},
        "stage_01_initial_detection": {
            "black_plate_well_analysis": {
                "enable_brightness_adjustment": True,
                "well_identification": {
                    "low_threshold_value": 10,
                    "min_expected_diameter_px": 1500,
                    "max_expected_diameter_px": 4000,
                    "min_circularity": 0.6,
                },
            },
            "unsubtracted_particle_analysis": {"local_otsu_roi_size": 150},
        },
        "stage_02_characterization": {
            "cropping": {"mode": "rotational_rectangle", "rotational_padding": 30},
            "focussed_particles": {"sharpness_min_threshold": 80},
        },
        "stage_03_filtering_and_classification": {
            "quality_flags_filter": {"filter_black_mask": True, "filter_on_boundary": True}
        },
    }

    ws_path = tmp_path / "ws_config.yaml"
    pipeline_path = tmp_path / "pipeline_config.yml"
    ws_path.write_text(yaml.safe_dump(ws_cfg), encoding="utf-8")
    pipeline_path.write_text(yaml.safe_dump(pipeline_cfg), encoding="utf-8")

    mod = _load_rt_config_module()
    cfg = mod.load_realtime_config(str(ws_path), pipeline_config_path=str(pipeline_path))

    assert cfg.stage1_config["source_image_illumination"] == "collimated"
    assert cfg.stage1_config["local_otsu_roi_size"] == 150
    assert cfg.stage2_crop_config["rotational_padding"] == 30
    assert cfg.min_focus_score == 80
    assert cfg.stage3_enable_artifact_filter is True
    assert cfg.stage3_enable_boundary_filter is True


def test_realtime_loader_rejects_duplicated_ws_analysis_keys(tmp_path: Path) -> None:
    ws_cfg = {
        "focusing": {
            "pattern": "zigzag",
            "scan_height_mm": 11.2,
            "scan_step_um": 350,
            "scan_ratio": 2.25,
            "lens_settle_time_sec": 0.025,
            "large_jump_threshold_dpt": -3.0,
            "large_jump_settle_sec": 0.040,
            "lens_controller_port": "COM5",
        },
        "realtime_mode": {
            "enabled": True,
            "runtime": {"target_fps": 5, "max_processing_latency_ms": 250, "queue_size": 2},
            "background": {"bgi_min_age_s": 2.0, "bgi_max_age_s": 4.0},
            "acquisition": {"source": "camera", "camera_index": 0, "retrieve_timeout_margin_ms": 150},
            "stage1": {"source_image_illumination": "collimated"},
        },
    }
    pipeline_cfg = {
        "system": {"source_image_illumination_type": "collimated"},
    }

    ws_path = tmp_path / "ws_config.yaml"
    pipeline_path = tmp_path / "pipeline_config.yml"
    ws_path.write_text(yaml.safe_dump(ws_cfg), encoding="utf-8")
    pipeline_path.write_text(yaml.safe_dump(pipeline_cfg), encoding="utf-8")

    mod = _load_rt_config_module()
    with pytest.raises(ValueError, match="Duplicated realtime analysis keys"):
        mod.load_realtime_config(str(ws_path), pipeline_config_path=str(pipeline_path))
