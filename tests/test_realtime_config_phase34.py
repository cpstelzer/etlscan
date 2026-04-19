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


def _base_unified_config(**overrides):
    cfg = {
        "acquisition": {
            "camera": {"exposure_us": 1234},
            "roi": {"width": 3333, "height": 2222},
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
        },
        "system": {
            "source_image_illumination_type": "collimated",
            "thresholding": {"thresholding_mode": "otsu", "otsu_threshold_multiplier": 0.7, "fixed_threshold_value": 70},
            "segmentation": {"min_noise_size": 99, "closing_selem_size": 3},
        },
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
            "well_masking": {},
            "particle_analysis": {},
            "particle_comparison": {},
            "unsubtracted_particle_analysis": {"local_otsu_roi_size": 150},
        },
        "stage_02_characterization": {
            "cropping": {"mode": "rotational_rectangle", "rotational_padding": 30, "adaptive_padding_factor": 0.2},
            "focussed_particles": {"sharpness_min_threshold": 80},
            "quality_features": {
                "black_mask_check": {
                    "min_area_ratio": 0.05,
                    "max_black_pixel_value": 15,
                    "min_edge_thickness_fraction": 0.2,
                }
            },
        },
        "stage_03_filtering_and_classification": {
            "quality_flags_filter": {"filter_black_mask": True, "filter_on_boundary": True}
        },
        "stage_04_biological_analysis": {
            "egg_detection": {
                "gaussian_sigma": 1.0,
                "egg_posterior_pixel_fraction": 0.35,
                "egg_threshold_quantile": 0.8,
                "egg_min_area": 50,
                "egg_max_area": 5000,
                "egg_min_solidity": 0.60,
                "egg_max_solidity": 0.99,
                "egg_min_eccentricity": 0.2,
                "orientation_diff_tolerance_deg": 85,
            }
        },
        "realtime_mode": {
            "enabled": True,
            "default_grayscale": True,
            "output_root_dir": "output/realtime",
            "runtime": {
                "target_fps": 5,
                "max_processing_latency_ms": 250,
                "drop_frame_if_backlogged": True,
                "queue_size": 2,
            },
            "background": {"bgi_min_age_s": 2.0, "bgi_max_age_s": 4.0},
            "stage1": {"reacquire_detection_threshold": 3},
            "stage2": {"min_particle_area": 500, "max_candidates_per_frame": 200},
            "persistence": {"save_accepted_crops": True, "image_format": "png", "png_compression": 3},
            "logging": {
                "metric_print_interval_s": 5,
                "etl_status_interval_frames": 30,
                "focus_rejection_log_interval_frames": 30,
            },
            "acquisition": {
                "source": "camera",
                "camera_index": 0,
                "input_dir": None,
                "max_frames": None,
                "flat_field_dir": None,
                "retrieve_timeout_margin_ms": 150,
                "adaptive_enabled": False,
                "adaptive_window_fraction": None,
                "adaptive_min_window_points": None,
                "adaptive_dispersion_coeff": None,
                "etl_control_enabled": True,
                "start_dpt": -2.0,
            },
        },
    }
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_realtime_loader_uses_pipeline_for_shared_analysis(tmp_path: Path) -> None:
    cfg = _base_unified_config()
    config_path = _write_config(tmp_path, cfg)

    mod = _load_rt_config_module()
    rt_cfg = mod.load_realtime_config(str(config_path))

    assert rt_cfg.stage1_config["source_image_illumination"] == "collimated"
    assert rt_cfg.stage1_config["unsubtracted_analysis_config"]["local_otsu_roi_size"] == 150
    assert rt_cfg.stage2_crop_config["rotational_padding"] == 30
    assert rt_cfg.stage2_segmentation_params["otsu_threshold_multiplier"] == 0.7
    assert rt_cfg.stage2_segmentation_params["closing_selem_size"] == 3
    assert rt_cfg.stage2_quality_features_config["black_mask_check"]["max_black_pixel_value"] == 15
    assert rt_cfg.stage1_config["global_threshold_config"]["otsu_threshold_multiplier"] == 0.7
    assert rt_cfg.min_focus_score == 80
    assert rt_cfg.stage3_enable_artifact_filter is True
    assert rt_cfg.stage3_enable_boundary_filter is True
    assert rt_cfg.exposure_us == 1234
    assert rt_cfg.roi_width == 3333
    assert rt_cfg.roi_height == 2222


def test_realtime_loader_requires_black_mask_config(tmp_path: Path) -> None:
    cfg = _base_unified_config()
    cfg["stage_02_characterization"]["quality_features"] = {"black_mask_check": {}}
    config_path = _write_config(tmp_path, cfg)

    mod = _load_rt_config_module()
    with pytest.raises(ValueError, match="black_mask_check"):
        mod.load_realtime_config(str(config_path))


def test_realtime_loader_requires_adaptive_params_when_enabled(tmp_path: Path) -> None:
    cfg = _base_unified_config()
    cfg["realtime_mode"]["acquisition"]["adaptive_enabled"] = True
    config_path = _write_config(tmp_path, cfg)

    mod = _load_rt_config_module()
    with pytest.raises(ValueError, match="adaptive_window_fraction"):
        mod.load_realtime_config(str(config_path))
