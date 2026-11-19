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
            "camera": {
                "exposure_us": 1234,
                "pixel_format_preference": ["Mono8", "Mono12"],
                "max_num_buffer_min": 50,
                "retrieve_timeout_margin_ms": 150,
                "trigger_mode": "software_grid",
                "trigger_ready_lead_ms": 8,
                "post_exposure_margin_ms": 2,
                "timer_busy_wait_ms": 2,
                "thread_switch_interval_ms": 0.5,
                "clock_sync_samples": 20,
                "trigger_late_report_ms": 2,
                "timestamp_mismatch_ms": 10,
            },
            "roi": {"width": 3333, "height": 2222},
            "focusing": {
                "pattern": "zigzag",
                "scan_height_mm": 11.2,
                "scan_step_um": 350,
                "scan_ratio": 2.25,
                "lens_settle_time_sec": 0.025,
                "prefocus_settle_sec": 1.0,
                "min_settle_ms": 10,
                "lens_temperature_interval_sec": 1.0,
                "large_jump_threshold_dpt": -3.0,
                "large_jump_settle_sec": 0.040,
                "lens_controller_port": "COM5",
            },
        },
        "system": {
            "source_image_illumination_type": "collimated",
            "thresholding": {"thresholding_mode": "otsu", "otsu_threshold_multiplier": 0.7, "fixed_threshold_value": 70},
            "subtracted_foreground": {"foreground_diff_threshold": 10, "max_foreground_fraction": 0.08},
            "segmentation": {"min_noise_size": 99, "closing_selem_size": 3},
        },
        "stage_01_initial_detection": {
            "black_plate_well_analysis": {
                "enable_brightness_adjustment": True,
                "rim_exclusion_px": 60,
                "well_identification": {
                    "morph_kernel_size": 9,
                    "low_threshold_value": 10,
                    "min_expected_diameter_px": 1500,
                    "max_expected_diameter_px": 4000,
                    "min_fill_ratio_to_enclosing_circle": 0.8,
                    "center_offset_fraction_threshold": 0.1,
                    "fallback_full_frame_on_failure": True,
                },
                "brightness_peak_calculation": {"min_well_pixels": 100, "percentile_for_peak_proxy": 75},
                "brightness_matching": {
                    "max_peak_difference_allowed": 15,
                    "adjustment_method": "global_shift",
                    "target_peak_offset_after_shift": 3,
                },
            },
            "well_masking": {
                "create_mask": False, "auto_center_mask": True, "radius_px": 2000, "radius_fraction": 0.75,
            },
            "particle_analysis": {
                "apply_convolution": False,
                "convolution_kernel": None,
                "min_microorganism_area": 300,
                "microorganism_blur_kernel": 5,
            },
            "particle_comparison": {"search_radius_factor": 2.0, "default_particle_area": 2000},
            "unsubtracted_particle_analysis": {
                "local_otsu_roi_size": 150,
                "min_particle_area": 600,
                "adaptive_padding_factor": 0.2,
                "background_floor_percentile": 25,
                "max_roi_foreground_fraction": 0.85,
            },
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
            }
        },
        "realtime_mode": {
            "enabled": True,
            "default_grayscale": True,
            "output_root_dir": "output/realtime",
            "runtime": {
                "target_fps": 5,
                "drop_frame_if_backlogged": True,
                "queue_size": 2,
                "max_consecutive_missing_frames": 15,
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
    assert rt_cfg.stage1_config["subtracted_foreground_config"]["foreground_diff_threshold"] == 10
    assert rt_cfg.stage1_config["subtracted_foreground_config"]["max_foreground_fraction"] == 0.08
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


def test_realtime_loader_reads_the_shared_trigger_mode(tmp_path: Path) -> None:
    mod = _load_rt_config_module()
    rt_cfg = mod.load_realtime_config(str(_write_config(tmp_path, _base_unified_config())))
    assert rt_cfg.acquisition_config["camera"]["trigger_mode"] == "software_grid"

    cfg = _base_unified_config(**{"acquisition.camera.trigger_mode": "external"})
    (tmp_path / "invalid").mkdir()   # load_config_once caches by path
    with pytest.raises(ValueError, match="trigger_mode"):
        mod.load_realtime_config(str(_write_config(tmp_path / "invalid", cfg)))


def test_realtime_loader_requires_grid_settings_only_for_the_grid_loop(tmp_path: Path) -> None:
    mod = _load_rt_config_module()
    cfg = _base_unified_config()
    del cfg["acquisition"]["focusing"]["min_settle_ms"]
    with pytest.raises(KeyError, match="min_settle_ms"):
        mod.load_realtime_config(str(_write_config(tmp_path, cfg)))

    cfg["acquisition"]["camera"]["trigger_mode"] = "free_run"
    (tmp_path / "free_run").mkdir()   # load_config_once caches by path
    mod.load_realtime_config(str(_write_config(tmp_path / "free_run", cfg)))


def test_realtime_loader_requires_the_stage1_detection_keys(tmp_path: Path) -> None:
    mod = _load_rt_config_module()
    cfg = _base_unified_config()
    del cfg["stage_01_initial_detection"]["black_plate_well_analysis"]["brightness_matching"]["adjustment_method"]
    with pytest.raises(Exception, match="brightness_matching.adjustment_method"):
        mod.load_realtime_config(str(_write_config(tmp_path, cfg)))


def test_realtime_loader_checks_the_camera_setup_keys_for_a_camera_session(tmp_path: Path) -> None:
    mod = _load_rt_config_module()
    cfg = _base_unified_config()
    del cfg["acquisition"]["focusing"]["prefocus_settle_sec"]
    with pytest.raises(Exception, match="acquisition.focusing.prefocus_settle_sec"):
        mod.load_realtime_config(str(_write_config(tmp_path, cfg)))

    cfg["realtime_mode"]["acquisition"]["source"] = "directory"   # frames from disk: no camera is set up
    (tmp_path / "directory").mkdir()
    assert mod.load_realtime_config(str(_write_config(tmp_path / "directory", cfg))).max_consecutive_missing_frames == 15
