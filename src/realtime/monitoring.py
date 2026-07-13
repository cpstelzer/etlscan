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
    frames_over_latency_budget: int = 0
    max_latency_ms: float = 0.0
    stage1_detections: int = 0
    stage2_candidates: int = 0
    stage2_rejected_focus: int = 0
    stage2_rejected_area: int = 0
    stage3_rejected_artifact: int = 0
    stage3_rejected_boundary: int = 0
    accepted_particles: int = 0
    saved_crops: int = 0


class RuntimeMonitor:
    def __init__(self, print_interval_s: float, target_fps: float) -> None:
        self.print_interval_s = print_interval_s
        self.last_print_s = time.monotonic()
        self.counters = RuntimeCounters()
        # Per-frame budget to keep up with the camera, derived from target_fps
        # (the interval between incoming frames). No separate config knob.
        self.per_frame_budget_ms = 1000.0 / target_fps if target_fps > 0 else float("inf")
        # Snapshots/peaks for computing per-interval deltas in the backlog warning.
        self._last_over_budget = 0
        self._last_dropped = 0
        self._interval_peak_latency_ms = 0.0

    def record_latency(self, elapsed_ms: float) -> None:
        """Record one frame's end-to-end processing latency.

        Measured on the consumer side, so this detects "falling behind" in both
        drop and blocking (no-drop) modes, independent of the queue policy.
        """
        if elapsed_ms > self.per_frame_budget_ms:
            self.counters.frames_over_latency_budget += 1
        if elapsed_ms > self.counters.max_latency_ms:
            self.counters.max_latency_ms = elapsed_ms
        if elapsed_ms > self._interval_peak_latency_ms:
            self._interval_peak_latency_ms = elapsed_ms

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

        # Rate-limited "falling behind" warning (one per logging interval), in
        # place of the old per-frame latency warning. Fires whenever frames in
        # this interval exceeded the derived budget or were dropped.
        over_budget_delta = c.frames_over_latency_budget - self._last_over_budget
        dropped_delta = c.frames_dropped_backpressure - self._last_dropped
        if over_budget_delta > 0 or dropped_delta > 0:
            logger.warning(
                "Falling behind: %d frame(s) over %.0fms budget (peak %.0fms), "
                "%d dropped this interval. Processing slower than target_fps.",
                over_budget_delta,
                self.per_frame_budget_ms,
                self._interval_peak_latency_ms,
                dropped_delta,
            )
        self._last_over_budget = c.frames_over_latency_budget
        self._last_dropped = c.frames_dropped_backpressure
        self._interval_peak_latency_ms = 0.0
