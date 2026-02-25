from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

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


def run_realtime_session(config: RealtimeConfig, plate: int, well: int) -> str:
    session = create_session(config, plate=plate, well=well)
    logger.info("Realtime session started: %s", session.output_dir)

    ffc_assets = _load_optional_ffc_assets(config)
    bg_buffer = BackgroundBuffer(config.bgi_min_age_s, config.bgi_max_age_s)
    monitor = RuntimeMonitor(config.metric_print_interval_s)

    frame_queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=config.queue_size)
    acq_thread = _start_acquisition_thread(config, frame_queue, monitor)

    while True:
        packet = frame_queue.get()
        if packet is None:
            break

        frame_start = time.monotonic()
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
        s2 = run_stage2_realtime(packet.image_gray, s1.detections_df, packet.frame_index, {
            "crop_config": config.stage2_crop_config,
            "min_focus_score": config.min_focus_score,
            "min_particle_area": config.min_particle_area,
        })
        s3 = run_stage3_realtime(
            s2.particles_df,
            {
                "enable_artifact_filter": config.stage3_enable_artifact_filter,
                "enable_boundary_filter": config.stage3_enable_boundary_filter,
            },
        )

        if config.save_accepted_crops:
            save_accepted_crops(s3.accepted_df, session, config.image_format, config.png_compression)
        monitor.counters.accepted_particles += len(s3.accepted_df)
        monitor.counters.frames_processed += 1
        monitor.maybe_log()

        elapsed_ms = (time.monotonic() - frame_start) * 1000.0
        if elapsed_ms > config.max_processing_latency_ms:
            logger.warning("Frame %d processing latency %.2fms > budget %dms", packet.frame_index, elapsed_ms, config.max_processing_latency_ms)

    acq_thread.join(timeout=2.0)
    logger.info("Realtime session finished. output_dir=%s", session.output_dir)
    return session.output_dir
