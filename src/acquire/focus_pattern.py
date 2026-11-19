"""ETL focus-sweep pattern generation, shared by batch and realtime acquisition.

One focus cycle is the list of diopter values the liquid lens steps through
before the cycle repeats:

``sawtooth``
    The upward ramp only: N frames per cycle, every plane visited once.
``zigzag``
    Up and back down without repeating the end planes: 2(N - 1) frames per
    cycle, inner planes visited twice and the two end planes once.

Both the batch acquisition (``wellscanner_acquisition_module.run_acquisition``)
and the realtime frame source build their cycle here, so the planes the lens
visits and any frame-count arithmetic derived from them cannot drift apart.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_focus_cycle_from_up_values(pattern: str, diopter_values_up: list[float]) -> list[float]:
    """Turn an upward diopter ramp into one cycle of the requested pattern.

    Raises:
        ValueError: ``pattern`` is neither ``zigzag`` nor ``sawtooth``.
    """
    normalized_pattern = pattern.strip().lower()
    if normalized_pattern == "sawtooth":
        return diopter_values_up
    if normalized_pattern != "zigzag":
        raise ValueError(f"Unknown focus pattern {pattern!r}: acquisition.focusing.pattern must be 'zigzag' or 'sawtooth'.")
    diopter_values_down: list[float] = []
    if len(diopter_values_up) > 1:
        diopter_values_down = diopter_values_up[-2:0:-1]
    return diopter_values_up + diopter_values_down


def build_focus_cycle_pattern(
    start_dpt: float,
    pattern: str,
    scan_height_mm: float,
    scan_step_um: float,
    scan_ratio: float,
) -> list[float]:
    """Build one focus cycle (diopters, rounded to 3 decimals) for the ETL sweep.

    The ramp runs from ``start_dpt`` over ``scan_height_mm`` in steps of
    ``scan_step_um``; the last plane is the step nearest the range end, so it
    can overshoot ``scan_height_mm`` by up to half a step.

    Raises:
        ValueError: if ``scan_ratio`` or the resulting diopter step is zero.
    """
    if abs(scan_ratio) < 1e-9:
        raise ValueError("Scan ratio cannot be zero.")
    end_dpt = start_dpt + scan_height_mm / scan_ratio
    step_dpt = abs((scan_step_um / 1000.0) / scan_ratio)
    if step_dpt < 1e-9:
        raise ValueError("Calculated diopter step is zero, cannot generate a pattern.")
    if end_dpt < start_dpt:
        step_dpt = -step_dpt

    diopter_values_up = np.arange(start_dpt, end_dpt + (step_dpt / 2.0), step_dpt)
    if diopter_values_up.size == 0:
        diopter_values_up = np.array([start_dpt])
    return build_focus_cycle_from_up_values(pattern, diopter_values_up.round(3).tolist())


def count_frames_for_plane_visits(focus_cycle: list[float], plane_visits: int) -> int:
    """Return the fewest frames after which every plane of the cycle was visited ``plane_visits`` times.

    Counted by stepping through the repeated cycle from its first frame, so it
    is exact for any pattern. Closed forms for N planes and k visits:
    sawtooth ``k * N``; zigzag ``(2k - 1) * (N - 1) + 1`` (the top end plane,
    visited once per cycle, is the last to reach k).

    Raises:
        ValueError: if ``focus_cycle`` is empty or ``plane_visits`` < 1.
    """
    if not focus_cycle:
        raise ValueError("Focus cycle is empty.")
    if plane_visits < 1:
        raise ValueError(f"plane_visits must be >= 1, got {plane_visits}.")
    remaining_visits = {dpt: plane_visits for dpt in focus_cycle}
    frame_count = 0
    while remaining_visits:
        dpt = focus_cycle[frame_count % len(focus_cycle)]
        frame_count += 1
        if dpt in remaining_visits:
            remaining_visits[dpt] -= 1
            if remaining_visits[dpt] == 0:
                del remaining_visits[dpt]
    return frame_count
