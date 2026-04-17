from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Optional

from src.utils.config_loader import load_config_once, get_unified_segmentation_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeConfig:
    enabled: bool
    default_grayscale: bool
    output_root_dir: str
    target_fps: float
    max_processing_latency_ms: int
    drop_frame_if_backlogged: bool
    queue_size: int
    bgi_min_age_s: float
    bgi_max_age_s: float
    stage1_reacquire_detection_threshold: int
    stage1_config: Dict[str, Any]
    stage2_crop_config: Dict[str, Any]
    stage2_segmentation_params: Dict[str, Any]
    stage2_quality_features_config: Dict[str, Any]
    min_focus_score: float
    min_particle_area: float
    max_stage2_candidates_per_frame: int
    stage3_enable_artifact_filter: bool
    stage3_enable_boundary_filter: bool
    save_accepted_crops: bool
    image_format: str
    png_compression: int
    metric_print_interval_s: float
    etl_status_interval_frames: int
    focus_rejection_log_interval_frames: int
    acquisition_source: str
    camera_index: int
    input_dir: Optional[str]
    max_frames: Optional[int]
    flat_field_dir: Optional[str]
    retrieve_timeout_margin_ms: int
    adaptive_enabled: bool
    adaptive_window_fraction: Optional[float]
    adaptive_min_window_points: Optional[int]
    adaptive_dispersion_coeff: Optional[int]
    exposure_us: float
    roi_width: Optional[int]
    roi_height: Optional[int]
    etl_control_enabled: bool
    start_dpt: float
    focus_pattern: str
    scan_height_mm: float
    scan_step_um: float
    scan_ratio: float
    lens_settle_time_sec: float
    large_jump_threshold_dpt: float
    large_jump_settle_sec: float
    lens_controller_port: Optional[str]


def _resolve_stage1_config(pipeline_cfg: Dict[str, Any], rt: Dict[str, Any]) -> Dict[str, Any]:
    stage1_cfg = dict(rt["stage1"])
    system_cfg = pipeline_cfg["system"]
    stage1_root = pipeline_cfg["stage_01_initial_detection"]
    black_plate = stage1_root["black_plate_well_analysis"]

    stage1_cfg["source_image_illumination"] = str(system_cfg["source_image_illumination_type"])
    stage1_cfg["config_black_plate_well"] = black_plate
    stage1_cfg["enable_brightness_adjustment"] = bool(black_plate["enable_brightness_adjustment"])

    stage1_cfg["system_cfg"] = system_cfg
    stage1_cfg["well_masking"] = stage1_root["well_masking"]
    stage1_cfg["particle_analysis_subtracted_config"] = stage1_root["particle_analysis"]
    stage1_cfg["particle_comparison_config"] = stage1_root["particle_comparison"]
    stage1_cfg["unsubtracted_analysis_config"] = stage1_root["unsubtracted_particle_analysis"]
    stage1_cfg["global_threshold_config"] = system_cfg["thresholding"]

    return stage1_cfg


def _must_be_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"realtime_mode.{name} must be > 0, got {value}")


def _must_be_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"realtime_mode.{name} must be >= 0, got {value}")


def load_realtime_config(config_path: str = "config/config.yml") -> RealtimeConfig:
    pipeline_cfg = load_config_once(config_path)

    rt = pipeline_cfg["realtime_mode"]
    runtime = rt["runtime"]
    background = rt["background"]
    stage1 = rt["stage1"]
    stage2 = rt["stage2"]
    persistence = rt["persistence"]
    logging_cfg = rt["logging"]
    acquisition_rt = rt["acquisition"]

    acquisition_cfg = pipeline_cfg["acquisition"]
    camera_cfg = acquisition_cfg["camera"]
    roi_cfg = acquisition_cfg["roi"]
    focusing_cfg = acquisition_cfg["focusing"]

    stage2_offline = pipeline_cfg["stage_02_characterization"]
    stage3_offline = pipeline_cfg["stage_03_filtering_and_classification"]
    unified_segmentation_params = get_unified_segmentation_params(pipeline_cfg)

    stage1_cfg = _resolve_stage1_config(pipeline_cfg, rt)

    stage2_cropping = stage2_offline["cropping"]
    stage2_focus_cfg = stage2_offline["focussed_particles"]
    stage3_quality_cfg = stage3_offline["quality_flags_filter"]

    logger.info("Realtime config loaded from %s", config_path)

    parsed = RealtimeConfig(
        enabled=bool(rt["enabled"]),
        default_grayscale=bool(rt["default_grayscale"]),
        output_root_dir=str(rt["output_root_dir"]),
        target_fps=float(runtime["target_fps"]),
        max_processing_latency_ms=int(runtime["max_processing_latency_ms"]),
        drop_frame_if_backlogged=bool(runtime["drop_frame_if_backlogged"]),
        queue_size=int(runtime["queue_size"]),
        bgi_min_age_s=float(background["bgi_min_age_s"]),
        bgi_max_age_s=float(background["bgi_max_age_s"]),
        stage1_reacquire_detection_threshold=int(stage1["reacquire_detection_threshold"]),
        stage1_config=stage1_cfg,
        stage2_crop_config=stage2_cropping,
        stage2_segmentation_params=unified_segmentation_params,
        stage2_quality_features_config=stage2_offline["quality_features"],
        min_focus_score=float(stage2_focus_cfg["sharpness_min_threshold"]),
        min_particle_area=float(stage2["min_particle_area"]),
        max_stage2_candidates_per_frame=int(stage2["max_candidates_per_frame"]),
        stage3_enable_artifact_filter=bool(stage3_quality_cfg["filter_black_mask"]),
        stage3_enable_boundary_filter=bool(stage3_quality_cfg["filter_on_boundary"]),
        save_accepted_crops=bool(persistence["save_accepted_crops"]),
        image_format=str(persistence["image_format"]).lower(),
        png_compression=int(persistence["png_compression"]),
        metric_print_interval_s=float(logging_cfg["metric_print_interval_s"]),
        etl_status_interval_frames=int(logging_cfg["etl_status_interval_frames"]),
        focus_rejection_log_interval_frames=int(logging_cfg["focus_rejection_log_interval_frames"]),
        acquisition_source=str(acquisition_rt["source"]).lower(),
        camera_index=int(acquisition_rt["camera_index"]),
        input_dir=acquisition_rt["input_dir"],
        max_frames=acquisition_rt["max_frames"],
        flat_field_dir=acquisition_rt["flat_field_dir"],
        retrieve_timeout_margin_ms=int(acquisition_rt["retrieve_timeout_margin_ms"]),
        adaptive_enabled=bool(acquisition_rt["adaptive_enabled"]),
        adaptive_window_fraction=(
            float(acquisition_rt["adaptive_window_fraction"])
            if acquisition_rt["adaptive_window_fraction"] is not None
            else None
        ),
        adaptive_min_window_points=(
            int(acquisition_rt["adaptive_min_window_points"])
            if acquisition_rt["adaptive_min_window_points"] is not None
            else None
        ),
        adaptive_dispersion_coeff=(
            int(acquisition_rt["adaptive_dispersion_coeff"])
            if acquisition_rt["adaptive_dispersion_coeff"] is not None
            else None
        ),
        exposure_us=float(camera_cfg["exposure_us"]),
        roi_width=int(roi_cfg["width"]) if roi_cfg.get("width") is not None else None,
        roi_height=int(roi_cfg["height"]) if roi_cfg.get("height") is not None else None,
        etl_control_enabled=bool(acquisition_rt["etl_control_enabled"]),
        start_dpt=float(acquisition_rt["start_dpt"]),
        focus_pattern=str(focusing_cfg["pattern"]),
        scan_height_mm=float(focusing_cfg["scan_height_mm"]),
        scan_step_um=float(focusing_cfg["scan_step_um"]),
        scan_ratio=float(focusing_cfg["scan_ratio"]),
        lens_settle_time_sec=float(focusing_cfg["lens_settle_time_sec"]),
        large_jump_threshold_dpt=float(focusing_cfg["large_jump_threshold_dpt"]),
        large_jump_settle_sec=float(focusing_cfg["large_jump_settle_sec"]),
        lens_controller_port=focusing_cfg["lens_controller_port"],
    )

    _must_be_positive("runtime.target_fps", parsed.target_fps)
    _must_be_nonnegative("runtime.max_processing_latency_ms", parsed.max_processing_latency_ms)
    if parsed.queue_size < 1:
        raise ValueError("realtime_mode.runtime.queue_size must be >= 1")
    _must_be_positive("background.bgi_min_age_s", parsed.bgi_min_age_s)
    _must_be_positive("background.bgi_max_age_s", parsed.bgi_max_age_s)
    if parsed.bgi_max_age_s <= parsed.bgi_min_age_s:
        raise ValueError("realtime_mode.background.bgi_max_age_s must be > bgi_min_age_s")
    if parsed.stage1_reacquire_detection_threshold < 1:
        raise ValueError("realtime_mode.stage1.reacquire_detection_threshold must be >= 1")
    _must_be_nonnegative("stage2.min_focus_score", parsed.min_focus_score)
    _must_be_nonnegative("stage2.min_particle_area", parsed.min_particle_area)
    if parsed.max_stage2_candidates_per_frame < 1:
        raise ValueError("realtime_mode.stage2.max_candidates_per_frame must be >= 1")
    black_mask_cfg = parsed.stage2_quality_features_config["black_mask_check"]
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
    if parsed.adaptive_enabled:
        if parsed.adaptive_window_fraction is None:
            raise ValueError("realtime_mode.acquisition.adaptive_window_fraction must be set when adaptive_enabled=true")
        if parsed.adaptive_min_window_points is None:
            raise ValueError("realtime_mode.acquisition.adaptive_min_window_points must be set when adaptive_enabled=true")
        if parsed.adaptive_dispersion_coeff is None:
            raise ValueError("realtime_mode.acquisition.adaptive_dispersion_coeff must be set when adaptive_enabled=true")
        if not (0 < parsed.adaptive_window_fraction <= 1.0):
            raise ValueError("realtime_mode.acquisition.adaptive_window_fraction must be in (0, 1]")
        if parsed.adaptive_min_window_points < 3:
            raise ValueError("realtime_mode.acquisition.adaptive_min_window_points must be >= 3")
        if parsed.adaptive_min_window_points % 2 == 0:
            raise ValueError("realtime_mode.acquisition.adaptive_min_window_points must be odd")
        if parsed.adaptive_dispersion_coeff < 1:
            raise ValueError("realtime_mode.acquisition.adaptive_dispersion_coeff must be >= 1")
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
