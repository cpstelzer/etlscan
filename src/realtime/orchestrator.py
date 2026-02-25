from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

import pandas as pd

from src.core.image_io_operations import FlatFieldCorrectionAssets, load_flat_field_assets

from .acquisition import FramePacket, build_frame_source
from .background_buffer import BackgroundBuffer
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


def run_realtime_session(config: RealtimeConfig, plate: int, well: int) -> str:
    session = create_session(config, plate=plate, well=well)
    logger.info("Realtime session started: %s", session.output_dir)
    logger.info("Accepted crops directory: %s", os.path.abspath(session.accepted_dir))

    if not config.save_accepted_crops:
        logger.warning("realtime_mode.persistence.save_accepted_crops=false, so accepted crops will NOT be written to disk.")
    logger.info(
        "Realtime filters: artifact_filter=%s boundary_filter=%s",
        config.stage3_enable_artifact_filter,
        config.stage3_enable_boundary_filter,
    )

    ffc_assets = _load_optional_ffc_assets(config)
    bg_buffer = BackgroundBuffer(config.bgi_min_age_s, config.bgi_max_age_s)
    monitor = RuntimeMonitor(config.metric_print_interval_s)
    accepted_rows_for_csv: list[pd.DataFrame] = []
    warned_etl_unavailable = False

    frame_queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=config.queue_size)
    acq_thread = _start_acquisition_thread(config, frame_queue, monitor)

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
        elif not warned_etl_unavailable and config.acquisition_source == "camera":
            logger.info(
                "ETL status unavailable for source=camera in realtime mode (no live ETL telemetry wired). "
                "If you replay acquisition filenames with _dpt..._fd...mm, ETL values will be shown.")
            warned_etl_unavailable = True

        bg_buffer.update(packet.image_gray, packet.timestamp_s)
        bg_img = bg_buffer.get_valid_aged_background(packet.timestamp_s)
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
        monitor.counters.stage1_detections += len(s1.detections_df)

        s2 = run_stage2_realtime(packet.image_gray, s1.detections_df, packet.frame_index, {
            "crop_config": config.stage2_crop_config,
            "min_focus_score": config.min_focus_score,
            "min_particle_area": config.min_particle_area,
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
        ):
            logger.warning(
                "All %d Stage-2 candidates rejected by focus threshold (min_focus_score=%.3f, observed variance_laplacian range=[%s, %s]). "
                "Try lowering realtime_mode.stage2.min_focus_score.",
                s2.pre_threshold_count,
                float(config.min_focus_score),
                f"{s2.focus_min:.3f}" if s2.focus_min is not None else "n/a",
                f"{s2.focus_max:.3f}" if s2.focus_max is not None else "n/a",
            )

        if (
            s2.pre_threshold_count > 0
            and s3.rejected_boundary_count == s2.pre_threshold_count
            and len(s3.accepted_df) == 0
            and config.stage3_enable_boundary_filter
        ):
            logger.warning(
                "All %d Stage-2 candidates were rejected by boundary filter. "
                "This often indicates illumination mismatch for boundary detection; current mode=%s. "
                "You can temporarily set realtime_mode.stage3.enable_boundary_filter=false to confirm flow.",
                s2.pre_threshold_count,
                str(config.stage1_config.get("source_image_illumination", "darkfield")).strip().lower(),
            )

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
            accepted_rows_for_csv.append(accepted_for_csv)

        monitor.counters.saved_crops += saved_count
        monitor.counters.accepted_particles += len(s3.accepted_df)
        monitor.counters.frames_processed += 1
        monitor.maybe_log()

        elapsed_ms = (time.monotonic() - frame_start) * 1000.0
        if elapsed_ms > config.max_processing_latency_ms:
            logger.warning("Frame %d processing latency %.2fms > budget %dms", packet.frame_index, elapsed_ms, config.max_processing_latency_ms)

    acq_thread.join(timeout=2.0)
    monitor.maybe_log(force=True)

    accepted_csv_path = os.path.join(session.output_dir, "accepted_particles.csv")
    if accepted_rows_for_csv:
        accepted_df = pd.concat(accepted_rows_for_csv, ignore_index=True)
    else:
        accepted_df = pd.DataFrame()
    accepted_df.to_csv(accepted_csv_path, index=False)
    logger.info("Accepted particle dataframe written to: %s (rows=%d)", accepted_csv_path, len(accepted_df))

    logger.info("Realtime session finished. output_dir=%s", session.output_dir)
    return session.output_dir
