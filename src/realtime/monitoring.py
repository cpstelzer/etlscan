from __future__ import annotations

from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class RuntimeCounters:
    frames_acquired: int = 0
    frames_processed: int = 0
    frames_skipped_no_bg: int = 0
    frames_dropped_backpressure: int = 0
    stage1_detections: int = 0
    stage2_candidates: int = 0
    stage2_rejected_focus: int = 0
    stage2_rejected_area: int = 0
    stage3_rejected_artifact: int = 0
    stage3_rejected_boundary: int = 0
    accepted_particles: int = 0
    saved_crops: int = 0


class RuntimeMonitor:
    def __init__(self, print_interval_s: float) -> None:
        self.print_interval_s = print_interval_s
        self.last_print_s = time.monotonic()
        self.counters = RuntimeCounters()

    def maybe_log(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print_s < self.print_interval_s:
            return
        self.last_print_s = now
        c = self.counters
        logger.info(
            "Realtime metrics: acquired=%d processed=%d no_bg=%d dropped=%d "
            "s1=%d s2_candidates=%d rej_focus=%d rej_area=%d "
            "rej_artifact=%d rej_boundary=%d accepted=%d saved=%d",
            c.frames_acquired,
            c.frames_processed,
            c.frames_skipped_no_bg,
            c.frames_dropped_backpressure,
            c.stage1_detections,
            c.stage2_candidates,
            c.stage2_rejected_focus,
            c.stage2_rejected_area,
            c.stage3_rejected_artifact,
            c.stage3_rejected_boundary,
            c.accepted_particles,
            c.saved_crops,
        )
