from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


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
    min_focus_score: float = 0.0
    min_particle_area: float = 500.0
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


def _must_be_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"realtime_mode.{name} must be > 0, got {value}")


def _must_be_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"realtime_mode.{name} must be >= 0, got {value}")


def load_realtime_config(config_path: str) -> RealtimeConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    rt = cfg.get("realtime_mode", {}) or {}
    runtime = rt.get("runtime", {}) or {}
    background = rt.get("background", {}) or {}
    stage2 = rt.get("stage2", {}) or {}
    stage3 = rt.get("stage3", {}) or {}
    persistence = rt.get("persistence", {}) or {}
    logging_cfg = rt.get("logging", {}) or {}
    acquisition_cfg = rt.get("acquisition", {}) or {}
    focusing_cfg = cfg.get("focusing", {}) or {}

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
        stage1_config=rt.get("stage1", {}) or {},
        stage2_crop_config=stage2.get("crop_config", {"mode": "rotational_rectangle", "rotational_padding": 10}),
        min_focus_score=float(stage2.get("min_focus_score", 0.0)),
        min_particle_area=float(stage2.get("min_particle_area", 500.0)),
        stage3_enable_artifact_filter=bool(stage3.get("enable_artifact_filter", True)),
        stage3_enable_boundary_filter=bool(stage3.get("enable_boundary_filter", False)),
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
    if parsed.image_format not in {"png", "jpg", "jpeg", "bmp"}:
        raise ValueError("realtime_mode.persistence.image_format must be one of png/jpg/jpeg/bmp")
    _must_be_nonnegative("acquisition.retrieve_timeout_margin_ms", parsed.retrieve_timeout_margin_ms)
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
