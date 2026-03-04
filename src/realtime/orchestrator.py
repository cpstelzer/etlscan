from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

import pandas as pd

from src.core.image_io_operations import FlatFieldCorrectionAssets, load_flat_field_assets, load_image, save_image

from .acquisition import FramePacket, build_focus_cycle_pattern, build_frame_source
from .config import RealtimeConfig
from .monitoring import RuntimeMonitor
from .persistence import save_accepted_crops
from .session import create_session
from .stage1_rt import run_stage1_realtime
from .stage2_rt import run_stage2_realtime
from .stage3_rt import run_stage3_realtime

logger = logging.getLogger(__name__)


def _start_acquisition_thread(config: RealtimeConfig, out_queue: queue.Queue[Optional[FramePacket]], monitor: RuntimeMonitor) -> threading.Thread:
    source = build_frame_source(
        config.acquisition_source,
        input_dir=config.input_dir,
        camera_index=config.camera_index,
        target_fps=config.target_fps,
        max_frames=config.max_frames,
        retrieve_timeout_margin_ms=config.retrieve_timeout_margin_ms,
        exposure_us=config.exposure_us,
        roi_width=config.roi_width,
        roi_height=config.roi_height,
        etl_control_enabled=config.etl_control_enabled,
        start_dpt=config.start_dpt,
        focus_pattern=config.focus_pattern,
        scan_height_mm=config.scan_height_mm,
        scan_step_um=config.scan_step_um,
        scan_ratio=config.scan_ratio,
        lens_settle_time_sec=config.lens_settle_time_sec,
        large_jump_threshold_dpt=config.large_jump_threshold_dpt,
        large_jump_settle_sec=config.large_jump_settle_sec,
        lens_controller_port=config.lens_controller_port,
    )

    def _run() -> None:
        for packet in source:
            monitor.counters.frames_acquired += 1
            try:
                out_queue.put(packet, timeout=0.01)
            except queue.Full:
                if config.drop_frame_if_backlogged:
                    monitor.counters.frames_dropped_backpressure += 1
                else:
                    out_queue.put(packet)
        out_queue.put(None)

    thread = threading.Thread(target=_run, name="realtime-acquisition", daemon=True)
    thread.start()
    return thread


def _load_optional_ffc_assets(config: RealtimeConfig) -> Optional[FlatFieldCorrectionAssets]:
    if not config.flat_field_dir:
        return None
    return load_flat_field_assets(config.flat_field_dir, mode="gray")


def _make_csv_friendly(df: pd.DataFrame) -> pd.DataFrame:
    csv_df = df.copy()
    drop_cols = [
        col
        for col in ["crop_gray", "raw_mask", "filled_mask", "crop_metadata"]
        if col in csv_df.columns
    ]
    if drop_cols:
        csv_df = csv_df.drop(columns=drop_cols)
    return csv_df


def _bg_filename(diopter: float) -> str:
    return f"bg_dpt{diopter:.3f}.png".replace("-", "m").replace("+", "p")


def run_realtime_session(config: RealtimeConfig, plate: int, well: int) -> str:
    session = create_session(config, plate=plate, well=well)
    logger.info("Realtime session started: %s", session.output_dir)
    logger.info("Accepted crops directory: %s", os.path.abspath(session.accepted_dir))
    logger.info("Background image directory: %s", os.path.abspath(session.bg_dir))

    if not config.save_accepted_crops:
        logger.warning("realtime_mode.persistence.save_accepted_crops=false, so accepted crops will NOT be written to disk.")
    logger.info(
        "Realtime filters: artifact_filter=%s boundary_filter=%s",
        config.stage3_enable_artifact_filter,
        config.stage3_enable_boundary_filter,
    )

    use_etl_bg_library = config.acquisition_source == "camera" and config.etl_control_enabled
    if not use_etl_bg_library:
        raise RuntimeError(
            "Realtime mode requires ETL-controlled camera acquisition with fixed bg_images library. "
            "Set realtime_mode.acquisition.source=camera and etl_control_enabled=true."
        )

    expected_bg_steps = 0
    if use_etl_bg_library:
        focus_cycle = build_focus_cycle_pattern(
            start_dpt=config.start_dpt,
            pattern=config.focus_pattern,
            scan_height_mm=config.scan_height_mm,
            scan_step_um=config.scan_step_um,
            scan_ratio=config.scan_ratio,
        )
        # Only unique ETL steps need one background each. Zigzag cycles contain mirrored duplicates.
        expected_bg_steps = len({f"{d:.3f}" for d in focus_cycle})
        logger.info(
            "Background mode: fixed ETL-step library from initial sweep (%d unique ETL-step images in bg_images/).",
            expected_bg_steps,
        )

    logger.info(
        "Realtime acquisition policy: target_fps=%.2f drop_frame_if_backlogged=%s",
        config.target_fps,
        config.drop_frame_if_backlogged,
    )
    if not config.drop_frame_if_backlogged:
        logger.info("No-intentional-skip mode enabled: queue backpressure will block instead of dropping software frames.")

    ffc_assets = _load_optional_ffc_assets(config)
    monitor = RuntimeMonitor(config.metric_print_interval_s)
    accepted_csv_path = os.path.join(session.output_dir, "accepted_particles.csv")
    accepted_csv_header_written = False
    accepted_rows_count = 0
    last_focus_reject_warn_frame = -10**9
    last_boundary_reject_warn_frame = -10**9

    bg_library_paths: dict[str, str] = {}
    bg_library_ready = False

    frame_queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=config.queue_size)
    acq_thread = _start_acquisition_thread(config, frame_queue, monitor)

    try:
        while True:
            packet = frame_queue.get()
            if packet is None:
                break

            frame_start = time.monotonic()
            if packet.diopter is not None or packet.focusing_distance_mm is not None:
                if packet.frame_index % config.etl_status_interval_frames == 0:
                    logger.info(
                        "ETL status (frame=%d): diopters=%s, focusing_distance_mm=%s",
                        packet.frame_index,
                        f"{packet.diopter:.3f}" if packet.diopter is not None else "n/a",
                        f"{packet.focusing_distance_mm:.3f}" if packet.focusing_distance_mm is not None else "n/a",
                    )
            if packet.diopter is None:
                raise RuntimeError(
                    "ETL telemetry missing for camera frame. Fixed bg_images subtraction requires per-frame diopter values."
                )

            key = f"dpt_{packet.diopter:.3f}"
            if key not in bg_library_paths:
                filename = _bg_filename(packet.diopter)
                saved_bg = save_image(packet.image_gray, session.bg_dir, filename)
                if saved_bg:
                    bg_library_paths[key] = saved_bg
                    logger.info("Captured background reference %d/%d for %s -> %s", len(bg_library_paths), expected_bg_steps, key, saved_bg)

            if not bg_library_ready:
                if len(bg_library_paths) >= expected_bg_steps:
                    bg_library_ready = True
                    logger.info("Background reference library complete (%d unique ETL steps). Realtime processing now active.", expected_bg_steps)
                else:
                    monitor.counters.frames_skipped_no_bg += 1
                    monitor.maybe_log()
                    continue

            if key not in bg_library_paths:
                raise RuntimeError(
                    f"Missing background image for ETL step {key} after library initialization."
                )

            bg_path_for_frame = bg_library_paths[key]
            logger.info(
                "BG match: frame=%d fg_diopter=%.3f bg_image=%s",
                packet.frame_index,
                float(packet.diopter),
                os.path.basename(bg_path_for_frame),
            )

            bg_img = load_image(bg_path_for_frame, mode="gray")
            if bg_img is None:
                monitor.counters.frames_skipped_no_bg += 1
                monitor.maybe_log()
                continue

            s1 = run_stage1_realtime(
                packet.image_gray,
                bg_img,
                frame_index=packet.frame_index,
                timestamp_s=packet.timestamp_s,
                config=config.stage1_config,
                ffc_assets=ffc_assets,
            )
            logger.info("Frame %d: Stage-1 detections=%d", packet.frame_index, len(s1.detections_df))
            monitor.counters.stage1_detections += len(s1.detections_df)

            s2 = run_stage2_realtime(packet.image_gray, s1.detections_df, packet.frame_index, {
                "crop_config": config.stage2_crop_config,
                "segmentation_params": config.stage2_segmentation_params,
                "quality_features": config.stage2_quality_features_config,
                "min_focus_score": config.min_focus_score,
                "min_particle_area": config.min_particle_area,
                "max_candidates_per_frame": config.max_stage2_candidates_per_frame,
                "source_image_illumination": config.stage1_config.get("source_image_illumination", "darkfield"),
            })
            monitor.counters.stage2_candidates += s2.pre_threshold_count
            monitor.counters.stage2_rejected_focus += s2.rejected_focus_count
            monitor.counters.stage2_rejected_area += s2.rejected_area_count

            s3 = run_stage3_realtime(
                s2.particles_df,
                {
                    "enable_artifact_filter": config.stage3_enable_artifact_filter,
                    "enable_boundary_filter": config.stage3_enable_boundary_filter,
                },
            )
            monitor.counters.stage3_rejected_artifact += s3.rejected_artifact_count
            monitor.counters.stage3_rejected_boundary += s3.rejected_boundary_count

            if (
                s2.pre_threshold_count > 0
                and s2.rejected_focus_count == s2.pre_threshold_count
                and len(s3.accepted_df) == 0
                and (packet.frame_index - last_focus_reject_warn_frame) >= config.focus_rejection_log_interval_frames
            ):
                logger.warning(
                    "All %d Stage-2 candidates rejected by focus threshold (min_focus_score=%.3f, observed variance_laplacian range=[%s, %s]).",
                    s2.pre_threshold_count,
                    float(config.min_focus_score),
                    f"{s2.focus_min:.3f}" if s2.focus_min is not None else "n/a",
                    f"{s2.focus_max:.3f}" if s2.focus_max is not None else "n/a",
                )
                last_focus_reject_warn_frame = packet.frame_index

            if (
                s2.pre_threshold_count > 0
                and s3.rejected_boundary_count == s2.pre_threshold_count
                and len(s3.accepted_df) == 0
                and config.stage3_enable_boundary_filter
                and (packet.frame_index - last_boundary_reject_warn_frame) >= config.focus_rejection_log_interval_frames
            ):
                logger.warning(
                    "All %d Stage-2 candidates were rejected by boundary filter (mode=%s).",
                    s2.pre_threshold_count,
                    str(config.stage1_config.get("source_image_illumination", "darkfield")).strip().lower(),
                )
                last_boundary_reject_warn_frame = packet.frame_index

            saved_count = 0
            accepted_for_csv = _make_csv_friendly(s3.accepted_df.reset_index(drop=True))
            if config.save_accepted_crops:
                saved_paths = save_accepted_crops(s3.accepted_df, session, config.image_format, config.png_compression)
                saved_count = len(saved_paths)
                if not accepted_for_csv.empty:
                    accepted_for_csv["saved_crop_path"] = None
                    n = min(len(accepted_for_csv), len(saved_paths))
                    if n > 0:
                        accepted_for_csv.loc[: n - 1, "saved_crop_path"] = saved_paths[:n]
            if not accepted_for_csv.empty:
                accepted_for_csv.to_csv(
                    accepted_csv_path,
                    mode="a",
                    header=not accepted_csv_header_written,
                    index=False,
                )
                accepted_csv_header_written = True
                accepted_rows_count += len(accepted_for_csv)

            monitor.counters.saved_crops += saved_count
            monitor.counters.accepted_particles += len(s3.accepted_df)
            monitor.counters.frames_processed += 1
            monitor.maybe_log()

            elapsed_ms = (time.monotonic() - frame_start) * 1000.0
            if elapsed_ms > config.max_processing_latency_ms:
                logger.warning("Frame %d processing latency %.2fms > budget %dms", packet.frame_index, elapsed_ms, config.max_processing_latency_ms)
    except KeyboardInterrupt:
        logger.info("Realtime session interrupted by user; finalizing outputs written so far.")
    finally:
        acq_thread.join(timeout=2.0)
        monitor.maybe_log(force=True)

    if not accepted_csv_header_written:
        pd.DataFrame().to_csv(accepted_csv_path, index=False)
    logger.info("Accepted particle dataframe written to: %s (rows=%d)", accepted_csv_path, accepted_rows_count)

    logger.info("Realtime session finished. output_dir=%s", session.output_dir)
    return session.output_dir
