from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Optional

import yaml

from src.utils.config_loader import load_config_once
from src.utils.config_loader import get_unified_segmentation_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeConfig:
    enabled: bool = False
    default_grayscale: bool = True
    output_root_dir: str = "data/output/realtime"
    target_fps: float = 5.0
    max_processing_latency_ms: int = 250
    drop_frame_if_backlogged: bool = False
    queue_size: int = 2
    bgi_min_age_s: float = 2.0
    bgi_max_age_s: float = 4.0
    stage1_config: Dict[str, Any] = field(default_factory=dict)
    stage2_crop_config: Dict[str, Any] = field(default_factory=lambda: {"mode": "rotational_rectangle", "rotational_padding": 10})
    stage2_segmentation_params: Dict[str, Any] = field(default_factory=dict)
    stage2_quality_features_config: Dict[str, Any] = field(default_factory=dict)
    min_focus_score: float = 0.0
    min_particle_area: float = 500.0
    max_stage2_candidates_per_frame: int = 200
    stage3_enable_artifact_filter: bool = True
    stage3_enable_boundary_filter: bool = False
    save_accepted_crops: bool = True
    image_format: str = "png"
    png_compression: int = 3
    metric_print_interval_s: float = 5.0
    etl_status_interval_frames: int = 30
    focus_rejection_log_interval_frames: int = 30
    acquisition_source: str = "camera"
    camera_index: int = 0
    input_dir: Optional[str] = None
    max_frames: Optional[int] = None
    flat_field_dir: Optional[str] = None
    retrieve_timeout_margin_ms: int = 150
    exposure_us: float = 750.0
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None

    # ETL control (wired from src/acquire focusing config)
    etl_control_enabled: bool = True
    start_dpt: float = -2.0
    focus_pattern: str = "zigzag"
    scan_height_mm: float = 11.2
    scan_step_um: float = 350.0
    scan_ratio: float = 2.25
    lens_settle_time_sec: float = 0.025
    large_jump_threshold_dpt: float = -3.0
    large_jump_settle_sec: float = 0.04
    lens_controller_port: Optional[str] = "COM5"




def _default_black_plate_well_analysis() -> Dict[str, Any]:
    return {
        "enable_brightness_adjustment": True,
        "well_identification": {
            "low_threshold_value": 10,
            "min_expected_diameter_px": 1500,
            "max_expected_diameter_px": 4000,
            "min_circularity": 0.6,
        },
        "brightness_peak_calculation": {
            "min_intensity_for_peak_search": 75,
            "percentile_for_peak_proxy": 75,
        },
        "brightness_matching": {
            "max_peak_difference_allowed": 15,
            "adjustment_method": "global_shift",
            "target_peak_offset_after_shift": 3,
        },
    }


def _resolve_stage1_config(pipeline_cfg: Dict[str, Any], rt: Dict[str, Any]) -> Dict[str, Any]:
    stage1_cfg = dict((rt.get("stage1", {}) or {}))
    system_cfg = pipeline_cfg.get("system", {}) or {}
    stage1_root = pipeline_cfg.get("stage_01_initial_detection", {}) or {}
    legacy_black_plate = stage1_root.get("black_plate_well_analysis", {}) or {}

    stage1_cfg["source_image_illumination"] = str(
        system_cfg.get("source_image_illumination_type", stage1_cfg.get("source_image_illumination", "collimated"))
    )

    if legacy_black_plate:
        stage1_cfg["config_black_plate_well"] = legacy_black_plate
        stage1_cfg["enable_brightness_adjustment"] = bool(
            legacy_black_plate.get("enable_brightness_adjustment", True)
        )
    else:
        defaults = _default_black_plate_well_analysis()
        stage1_cfg["config_black_plate_well"] = defaults
        stage1_cfg["enable_brightness_adjustment"] = bool(
            defaults.get("enable_brightness_adjustment", True)
        )

    unsub_cfg = stage1_root.get("unsubtracted_particle_analysis", {}) or {}
    if "local_otsu_roi_size" in unsub_cfg:
        stage1_cfg["local_otsu_roi_size"] = int(unsub_cfg["local_otsu_roi_size"])

    # Reuse offline Stage-1 parameter namespaces directly for realtime.
    stage1_cfg["system_cfg"] = system_cfg
    stage1_cfg["well_masking"] = stage1_root.get("well_masking", {}) or {}
    stage1_cfg["particle_analysis_subtracted_config"] = stage1_root.get("particle_analysis", {}) or {}
    stage1_cfg["particle_comparison_config"] = stage1_root.get("particle_comparison", {}) or {}
    stage1_cfg["unsubtracted_analysis_config"] = stage1_root.get("unsubtracted_particle_analysis", {}) or {}
    stage1_cfg["global_threshold_config"] = system_cfg.get("thresholding", {}) or {}

    return stage1_cfg

def _must_be_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"realtime_mode.{name} must be > 0, got {value}")


def _must_be_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"realtime_mode.{name} must be >= 0, got {value}")


def _reject_duplicated_realtime_analysis_keys(rt: Dict[str, Any]) -> None:
    stage1 = rt.get("stage1", {}) or {}
    stage2 = rt.get("stage2", {}) or {}
    stage3 = rt.get("stage3", {}) or {}

    forbidden = []
    if "source_image_illumination" in stage1:
        forbidden.append("realtime_mode.stage1.source_image_illumination")
    if "enable_brightness_adjustment" in stage1:
        forbidden.append("realtime_mode.stage1.enable_brightness_adjustment")
    if "config_black_plate_well" in stage1:
        forbidden.append("realtime_mode.stage1.config_black_plate_well")
    if "local_otsu_roi_size" in stage1:
        forbidden.append("realtime_mode.stage1.local_otsu_roi_size")
    if "crop_config" in stage2:
        forbidden.append("realtime_mode.stage2.crop_config")
    if "min_focus_score" in stage2:
        forbidden.append("realtime_mode.stage2.min_focus_score")
    if "enable_artifact_filter" in stage3:
        forbidden.append("realtime_mode.stage3.enable_artifact_filter")
    if "enable_boundary_filter" in stage3:
        forbidden.append("realtime_mode.stage3.enable_boundary_filter")

    if forbidden:
        raise ValueError(
            "Duplicated realtime analysis keys are no longer supported in ws_config.yaml. "
            "Move these to config/pipeline_config.yml and remove them from realtime_mode: "
            + ", ".join(forbidden)
        )


def load_realtime_config(config_path: str, pipeline_config_path: str = "config/pipeline_config.yml") -> RealtimeConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        ws_cfg = yaml.safe_load(handle) or {}

    pipeline_cfg = load_config_once(pipeline_config_path)

    rt = ws_cfg.get("realtime_mode", {}) or {}
    _reject_duplicated_realtime_analysis_keys(rt)
    runtime = rt.get("runtime", {}) or {}
    background = rt.get("background", {}) or {}
    stage2 = rt.get("stage2", {}) or {}
    persistence = rt.get("persistence", {}) or {}
    logging_cfg = rt.get("logging", {}) or {}
    acquisition_cfg = rt.get("acquisition", {}) or {}
    camera_cfg = ws_cfg.get("camera", {}) or {}
    roi_cfg = ws_cfg.get("roi", {}) or {}
    focusing_cfg = ws_cfg.get("focusing", {}) or {}
    stage2_offline = pipeline_cfg.get("stage_02_characterization", {}) or {}
    stage3_offline = pipeline_cfg.get("stage_03_filtering_and_classification", {}) or {}
    unified_segmentation_params = get_unified_segmentation_params(pipeline_cfg)

    stage1_cfg = _resolve_stage1_config(pipeline_cfg, rt)

    stage2_cropping = stage2_offline.get("cropping", {}) or {}
    stage2_focus_cfg = stage2_offline.get("focussed_particles", {}) or {}
    stage3_quality_cfg = stage3_offline.get("quality_flags_filter", {}) or {}

    logger.info(
        "Realtime config sources: runtime/acquisition from %s, shared analysis from %s",
        config_path,
        pipeline_config_path,
    )

    parsed = RealtimeConfig(
        enabled=bool(rt.get("enabled", False)),
        default_grayscale=bool(rt.get("default_grayscale", True)),
        output_root_dir=str(rt.get("output_root_dir", "data/output/realtime")),
        target_fps=float(runtime.get("target_fps", 5.0)),
        max_processing_latency_ms=int(runtime.get("max_processing_latency_ms", 250)),
        drop_frame_if_backlogged=bool(runtime.get("drop_frame_if_backlogged", False)),
        queue_size=int(runtime.get("queue_size", 2)),
        bgi_min_age_s=float(background.get("bgi_min_age_s", 2.0)),
        bgi_max_age_s=float(background.get("bgi_max_age_s", 4.0)),
        stage1_config=stage1_cfg,
        stage2_crop_config=stage2_cropping or {"mode": "rotational_rectangle", "rotational_padding": 30},
        stage2_segmentation_params=unified_segmentation_params,
        stage2_quality_features_config=stage2_offline.get("quality_features", {}) or {},
        min_focus_score=float(stage2_focus_cfg.get("sharpness_min_threshold", 80.0)),
        min_particle_area=float(stage2.get("min_particle_area", 500.0)),
        max_stage2_candidates_per_frame=int(stage2.get("max_candidates_per_frame", 200)),
        stage3_enable_artifact_filter=bool(stage3_quality_cfg.get("filter_black_mask", True)),
        stage3_enable_boundary_filter=bool(stage3_quality_cfg.get("filter_on_boundary", True)),
        save_accepted_crops=bool(persistence.get("save_accepted_crops", True)),
        image_format=str(persistence.get("image_format", "png")).lower(),
        png_compression=int(persistence.get("png_compression", 3)),
        metric_print_interval_s=float(logging_cfg.get("metric_print_interval_s", 5.0)),
        etl_status_interval_frames=int(logging_cfg.get("etl_status_interval_frames", 30)),
        focus_rejection_log_interval_frames=int(logging_cfg.get("focus_rejection_log_interval_frames", 30)),
        acquisition_source=str(acquisition_cfg.get("source", "camera")).lower(),
        camera_index=int(acquisition_cfg.get("camera_index", 0)),
        input_dir=acquisition_cfg.get("input_dir"),
        max_frames=acquisition_cfg.get("max_frames"),
        flat_field_dir=acquisition_cfg.get("flat_field_dir"),
        retrieve_timeout_margin_ms=int(acquisition_cfg.get("retrieve_timeout_margin_ms", 150)),
        exposure_us=float(camera_cfg.get("exposure_us", 750.0)),
        roi_width=int(roi_cfg["width"]) if roi_cfg.get("width") is not None else None,
        roi_height=int(roi_cfg["height"]) if roi_cfg.get("height") is not None else None,
        etl_control_enabled=bool(acquisition_cfg.get("etl_control_enabled", True)),
        start_dpt=float(acquisition_cfg.get("start_dpt", -2.0)),
        focus_pattern=str(focusing_cfg.get("pattern", "zigzag")),
        scan_height_mm=float(focusing_cfg.get("scan_height_mm", 11.2)),
        scan_step_um=float(focusing_cfg.get("scan_step_um", 350.0)),
        scan_ratio=float(focusing_cfg.get("scan_ratio", 2.25)),
        lens_settle_time_sec=float(focusing_cfg.get("lens_settle_time_sec", 0.025)),
        large_jump_threshold_dpt=float(focusing_cfg.get("large_jump_threshold_dpt", -3.0)),
        large_jump_settle_sec=float(focusing_cfg.get("large_jump_settle_sec", 0.04)),
        lens_controller_port=focusing_cfg.get("lens_controller_port", "COM5"),
    )

    _must_be_positive("runtime.target_fps", parsed.target_fps)
    _must_be_nonnegative("runtime.max_processing_latency_ms", parsed.max_processing_latency_ms)
    if parsed.queue_size < 1:
        raise ValueError("realtime_mode.runtime.queue_size must be >= 1")
    _must_be_positive("background.bgi_min_age_s", parsed.bgi_min_age_s)
    _must_be_positive("background.bgi_max_age_s", parsed.bgi_max_age_s)
    if parsed.bgi_max_age_s <= parsed.bgi_min_age_s:
        raise ValueError("realtime_mode.background.bgi_max_age_s must be > bgi_min_age_s")
    _must_be_nonnegative("stage2.min_focus_score", parsed.min_focus_score)
    _must_be_nonnegative("stage2.min_particle_area", parsed.min_particle_area)
    if parsed.max_stage2_candidates_per_frame < 1:
        raise ValueError("realtime_mode.stage2.max_candidates_per_frame must be >= 1")
    black_mask_cfg = parsed.stage2_quality_features_config.get("black_mask_check")
    if not isinstance(black_mask_cfg, dict):
        raise ValueError(
            "pipeline stage_02_characterization.quality_features.black_mask_check must be defined for realtime mode"
        )
    required_black_mask_keys = (
        "min_area_ratio",
        "max_black_pixel_value",
        "min_edge_thickness_fraction",
    )
    missing_black_mask_keys = [k for k in required_black_mask_keys if k not in black_mask_cfg]
    if missing_black_mask_keys:
        raise ValueError(
            "Missing required keys in stage_02_characterization.quality_features.black_mask_check: "
            + ", ".join(missing_black_mask_keys)
        )
    if parsed.image_format not in {"png", "jpg", "jpeg", "bmp"}:
        raise ValueError("realtime_mode.persistence.image_format must be one of png/jpg/jpeg/bmp")
    _must_be_nonnegative("acquisition.retrieve_timeout_margin_ms", parsed.retrieve_timeout_margin_ms)
    _must_be_positive("camera.exposure_us", parsed.exposure_us)
    if parsed.roi_width is not None and parsed.roi_width < 1:
        raise ValueError("roi.width must be >= 1 when provided")
    if parsed.roi_height is not None and parsed.roi_height < 1:
        raise ValueError("roi.height must be >= 1 when provided")
    if parsed.etl_status_interval_frames < 1:
        raise ValueError("realtime_mode.logging.etl_status_interval_frames must be >= 1")
    if parsed.focus_rejection_log_interval_frames < 1:
        raise ValueError("realtime_mode.logging.focus_rejection_log_interval_frames must be >= 1")

    if abs(parsed.scan_ratio) < 1e-9:
        raise ValueError("focusing.scan_ratio must not be 0 for ETL stepping")
    _must_be_positive("focusing.scan_height_mm", parsed.scan_height_mm)
    _must_be_positive("focusing.scan_step_um", parsed.scan_step_um)
    _must_be_nonnegative("focusing.lens_settle_time_sec", parsed.lens_settle_time_sec)
    _must_be_nonnegative("focusing.large_jump_settle_sec", parsed.large_jump_settle_sec)

    return parsed
