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
    accepted_particles: int = 0


class RuntimeMonitor:
    def __init__(self, print_interval_s: float) -> None:
        self.print_interval_s = print_interval_s
        self.last_print_s = time.monotonic()
        self.counters = RuntimeCounters()

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now - self.last_print_s < self.print_interval_s:
            return
        self.last_print_s = now
        c = self.counters
        logger.info(
            "Realtime metrics: acquired=%d processed=%d no_bg=%d dropped=%d accepted=%d",
            c.frames_acquired,
            c.frames_processed,
            c.frames_skipped_no_bg,
            c.frames_dropped_backpressure,
            c.accepted_particles,
        )
