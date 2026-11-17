from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import os
import re
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np

from src.core.image_io_operations import load_image
from src.acquire.focus_pattern import build_focus_cycle_from_up_values, build_focus_cycle_pattern
from src.acquire.hardware_setup import (
    close_camera,
    connect_lens,
    disconnect_lens,
    open_first_camera,
    prefocus_lens,
    select_pixel_format,
    set_buffer_count,
    set_exposure,
    set_frame_rate,
)
from src.acquire.roi import set_centered_roi
from src.acquire.triggered_acquisition import (
    TRIGGER_MODE_SOFTWARE_GRID,
    FixedGridTriggerLoop,
    GrabbedFrame,
    GridTimingParams,
    build_camera_clock_map,
    camera_tick_frequency_hz,
    camera_ticks_to_ns,
    grab_timestamp_ticks,
    sample_camera_clock,
    summarize_grid_recording,
)

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    frame_index: int
    timestamp_s: float
    image_gray: np.ndarray
    diopter: Optional[float] = None
    focusing_distance_mm: Optional[float] = None
    focus_round_id: Optional[int] = None
    camera_timestamp_ns: Optional[int] = None   # exposure time on the camera clock (camera sources)




class AcquisitionControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._repeat_same_diopter_requests = 0
        self._pending_adaptive_center_diopter: Optional[float] = None
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask the camera source to stop after the current frame (the session is ending)."""
        with self._lock:
            self._stop_requested = True

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def request_repeat_same_diopter_once(self) -> None:
        with self._lock:
            self._repeat_same_diopter_requests += 1

    def consume_repeat_same_diopter_request(self) -> bool:
        with self._lock:
            if self._repeat_same_diopter_requests <= 0:
                return False
            self._repeat_same_diopter_requests -= 1
            return True

    def submit_adaptive_center_diopter(self, diopter: float) -> None:
        with self._lock:
            self._pending_adaptive_center_diopter = float(diopter)

    def consume_adaptive_center_diopter(self) -> Optional[float]:
        with self._lock:
            dpt = self._pending_adaptive_center_diopter
            self._pending_adaptive_center_diopter = None
            return dpt

class FrameSource:
    def __iter__(self) -> Iterator[FramePacket]:
        raise NotImplementedError


_FILENAME_DPT_FD_RE = re.compile(
    r"_dpt(?P<dpt>[-+]?[0-9]*\.?[0-9]+)_fd(?P<fd_mm>[-+]?[0-9]*\.?[0-9]+)mm",
    flags=re.IGNORECASE,
)


def _parse_etl_from_filename(path: str) -> Tuple[Optional[float], Optional[float]]:
    name = os.path.basename(path)
    m = _FILENAME_DPT_FD_RE.search(name)
    if not m:
        return None, None
    try:
        return float(m.group("dpt")), float(m.group("fd_mm"))
    except ValueError:
        return None, None


def build_unique_diopter_steps(focus_cycle: list[float]) -> list[float]:
    unique: list[float] = []
    seen: set[str] = set()
    for dpt in focus_cycle:
        key = f"{float(dpt):.3f}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(round(float(dpt), 3))
    return unique


def _nearest_index_for_diopter(diopter_steps: list[float], center_dpt: float) -> int:
    if not diopter_steps:
        raise ValueError("Cannot pick nearest index from empty diopter_steps")
    return min(range(len(diopter_steps)), key=lambda i: abs(diopter_steps[i] - center_dpt))


def select_adaptive_window_steps(
    diopter_steps: list[float],
    center_dpt: float,
    window_fraction: float,
    min_window_points: int,
    dispersion_coeff: int,
) -> list[float]:
    if not diopter_steps:
        return []
    n = len(diopter_steps)
    target_points = int(np.ceil(n * window_fraction))
    window_points = max(min_window_points, target_points)
    window_points = min(window_points, n)
    if window_points % 2 == 0:
        if window_points < n:
            window_points += 1
        else:
            window_points -= 1

    center_idx = _nearest_index_for_diopter(diopter_steps, center_dpt)
    selected = {center_idx}
    radius = 1
    while len(selected) < window_points:
        left = center_idx - radius * dispersion_coeff
        right = center_idx + radius * dispersion_coeff
        if left >= 0:
            selected.add(left)
            if len(selected) >= window_points:
                break
        if right < n:
            selected.add(right)
            if len(selected) >= window_points:
                break
        if left < 0 and right >= n:
            for offset in range(1, n):
                i_left = center_idx - offset
                i_right = center_idx + offset
                if i_left >= 0:
                    selected.add(i_left)
                    if len(selected) >= window_points:
                        break
                if i_right < n:
                    selected.add(i_right)
                    if len(selected) >= window_points:
                        break
            break
        radius += 1

    return [diopter_steps[i] for i in sorted(selected)]


class DirectoryFrameSource(FrameSource):
    def __init__(self, input_dir: str, target_fps: float, max_frames: Optional[int] = None) -> None:
        self._paths = [
            os.path.join(input_dir, name)
            for name in sorted(os.listdir(input_dir))
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        self._target_period_s = 1.0 / max(target_fps, 1.0)
        self._max_frames = max_frames

    def __iter__(self) -> Iterator[FramePacket]:
        start = time.monotonic()
        for idx, path in enumerate(self._paths):
            if self._max_frames is not None and idx >= int(self._max_frames):
                break
            img = load_image(path, mode="gray")
            if img is None:
                continue
            dpt, fd_mm = _parse_etl_from_filename(path)
            yield FramePacket(
                frame_index=idx,
                timestamp_s=time.monotonic(),
                image_gray=img,
                diopter=dpt,
                focusing_distance_mm=fd_mm,
            )
            next_t = start + (idx + 1) * self._target_period_s
            sleep_s = max(0.0, next_t - time.monotonic())
            if sleep_s > 0:
                time.sleep(sleep_s)


def _to_uint8_grayscale(raw_frame: np.ndarray) -> np.ndarray:
    if raw_frame.ndim == 3:
        raw_frame = np.mean(raw_frame, axis=2)
    if raw_frame.dtype == np.uint8:
        return raw_frame
    if raw_frame.dtype == np.uint16:
        return np.right_shift(raw_frame, 8).astype(np.uint8)
    frame = np.asarray(raw_frame, dtype=np.float32)
    frame = np.clip(frame, 0.0, 255.0)
    return frame.astype(np.uint8)




@dataclass(frozen=True)
class _FocusEntry:
    diopter: Optional[float]
    focusing_distance_mm: Optional[float]
    focus_round_id: int
    repeat: bool


class _FocusPlanner:
    """Lens setting of every realtime frame: steps through the focus cycle and applies the
    repeat-same-diopter and adaptive-window requests of the AcquisitionControl. Shared by the
    free-run and the fixed-grid loop, so both step through the planes the same way."""

    def __init__(
        self,
        base_focus_cycle: list[float],
        unique_diopter_steps: list[float],
        lens_enabled: bool,
        start_dpt: float,
        scan_ratio: float,
        focus_pattern: str,
        adaptive_enabled: bool,
        adaptive_window_fraction: Optional[float],
        adaptive_min_window_points: Optional[int],
        adaptive_dispersion_coeff: Optional[int],
        acquisition_control: Optional[AcquisitionControl],
    ) -> None:
        self._base_focus_cycle = list(base_focus_cycle)
        self._active_focus_cycle = list(base_focus_cycle)
        self._unique_diopter_steps = unique_diopter_steps
        self._lens_enabled = lens_enabled
        self._start_dpt = start_dpt
        self._scan_ratio = scan_ratio
        self._focus_pattern = focus_pattern
        self._adaptive_enabled = adaptive_enabled
        self._adaptive_window_fraction = adaptive_window_fraction
        self._adaptive_min_window_points = adaptive_min_window_points
        self._adaptive_dispersion_coeff = adaptive_dispersion_coeff
        self._acquisition_control = acquisition_control
        self._focus_step_idx = 0
        self._focus_round_id = 0

    def next_entry(self, advance: bool) -> _FocusEntry:
        """Setting of the next frame; ``advance`` is False for the first frame of a session."""
        repeat_same_diopter = bool(
            self._acquisition_control and self._acquisition_control.consume_repeat_same_diopter_request()
        )
        if advance and not repeat_same_diopter:
            self._focus_step_idx += 1
            if self._focus_step_idx >= len(self._active_focus_cycle):
                self._focus_step_idx = 0
                self._focus_round_id += 1

        if self._adaptive_enabled and self._acquisition_control is not None:
            adaptive_center = self._acquisition_control.consume_adaptive_center_diopter()
            if adaptive_center is not None and self._unique_diopter_steps:
                adaptive_steps = select_adaptive_window_steps(
                    self._unique_diopter_steps,
                    adaptive_center,
                    float(self._adaptive_window_fraction),
                    int(self._adaptive_min_window_points),
                    int(self._adaptive_dispersion_coeff),
                )
                if adaptive_steps:
                    self._active_focus_cycle = build_focus_cycle_from_up_values(self._focus_pattern, adaptive_steps)
                    self._focus_step_idx = 0
                    self._focus_round_id += 1
                    logger.info(
                        "Adaptive focusing engaged round %d: center=%.3f, selected_steps=%s",
                        self._focus_round_id,
                        adaptive_center,
                        ",".join(f"{x:.3f}" for x in adaptive_steps),
                    )
                else:
                    self._active_focus_cycle = list(self._base_focus_cycle)

        current_diopter = None
        current_fd_mm = None
        if self._lens_enabled:
            current_diopter = float(self._active_focus_cycle[self._focus_step_idx % len(self._active_focus_cycle)])
            current_fd_mm = (current_diopter - self._start_dpt) * self._scan_ratio
        return _FocusEntry(current_diopter, current_fd_mm, self._focus_round_id, repeat_same_diopter)


class _FrameHandoff:
    """Bounded hand-off of grid frames from the camera's grabber thread to the generator.

    ``offer(item, drop_when_full=True)`` refuses a frame while ``capacity`` frames wait;
    without dropping, every frame is kept and the trigger thread calls ``wait_for_space``
    before it commands the next frame (only frames already in flight exceed the capacity).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: deque = deque()
        self._condition = threading.Condition()
        self._closed = False

    def offer(self, item: tuple, drop_when_full: bool) -> bool:
        with self._condition:
            if self._closed or (drop_when_full and len(self._items) >= self._capacity):
                return False
            self._items.append(item)
            self._condition.notify_all()
            return True

    def wait_for_space(self, should_stop: Callable[[], bool], poll_s: float) -> bool:
        """Block while ``capacity`` frames wait (checking ``should_stop`` every ``poll_s``);
        True if it had to wait."""
        with self._condition:
            waited = False
            while len(self._items) >= self._capacity and not self._closed and not should_stop():
                waited = True
                self._condition.wait(timeout=poll_s)
            return waited

    def take(self) -> Optional[tuple]:
        """Next frame; None once the hand-off is closed and empty."""
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if not self._items:
                return None
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class CameraFrameSource(FrameSource):
    """Realtime camera source using pypylon and explicit ETL command-per-frame stepping.

    The loop follows the shared ``acquisition.camera.trigger_mode``: ``software_grid`` runs
    the fixed-grid trigger loop of the batch acquisition (src/acquire/triggered_acquisition.py)
    on a background thread at ``target_fps``, so every frame's diopter is the setting it was
    exposed under; ``free_run`` keeps the legacy loop (lens command, settle sleep, next
    free-run frame). Without ETL control there are no lens commands to synchronise and the
    camera always free-runs.
    """

    def __init__(
        self,
        target_fps: float,
        max_frames: Optional[int],
        retrieve_timeout_margin_ms: int,
        adaptive_enabled: bool,
        adaptive_window_fraction: Optional[float],
        adaptive_min_window_points: Optional[int],
        adaptive_dispersion_coeff: Optional[int],
        exposure_us: float,
        roi_width: Optional[int],
        roi_height: Optional[int],
        *,
        etl_control_enabled: bool,
        start_dpt: float,
        focus_pattern: str,
        scan_height_mm: float,
        scan_step_um: float,
        scan_ratio: float,
        lens_settle_time_sec: float,
        large_jump_threshold_dpt: float,
        large_jump_settle_sec: float,
        lens_controller_port: Optional[str],
        acquisition_config: Dict[str, Any],
        queue_size: int,
        drop_frame_if_backlogged: bool,
        max_consecutive_missing_frames: int,
        acquisition_control: Optional[AcquisitionControl] = None,
    ) -> None:
        self._target_fps = target_fps
        self._max_frames = max_frames
        self._retrieve_timeout_margin_ms = retrieve_timeout_margin_ms
        self._adaptive_enabled = adaptive_enabled
        self._adaptive_window_fraction = adaptive_window_fraction
        self._adaptive_min_window_points = adaptive_min_window_points
        self._adaptive_dispersion_coeff = adaptive_dispersion_coeff
        self._exposure_us = exposure_us
        self._roi_width = roi_width
        self._roi_height = roi_height
        self._etl_control_enabled = etl_control_enabled
        self._start_dpt = start_dpt
        self._focus_pattern = focus_pattern
        self._scan_height_mm = scan_height_mm
        self._scan_step_um = scan_step_um
        self._scan_ratio = scan_ratio
        self._lens_settle_time_sec = lens_settle_time_sec
        self._large_jump_threshold_dpt = large_jump_threshold_dpt
        self._large_jump_settle_sec = large_jump_settle_sec
        self._lens_controller_port = lens_controller_port
        self._acquisition_config = acquisition_config
        self._queue_size = queue_size
        self._drop_frame_if_backlogged = drop_frame_if_backlogged
        # The session stops when this many frames in a row did not arrive (free-run: retrieve
        # timeouts; fixed grid: commanded frames still without a frame).
        self._max_consecutive_missing_frames = max_consecutive_missing_frames
        self._acquisition_control = acquisition_control

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            from pypylon import pylon
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pypylon is required for realtime camera acquisition") from exc

        lens = None
        ecc1c = None
        focus_cycle = [round(self._start_dpt, 3)]
        unique_diopter_steps = [round(self._start_dpt, 3)]
        if self._etl_control_enabled:
            try:
                import optoICC
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("optoICC is required for ETL-controlled realtime camera acquisition") from exc

            if not self._lens_controller_port:
                raise RuntimeError("focusing.lens_controller_port must be set when ETL control is enabled")
            ecc1c, lens = connect_lens(optoICC, self._lens_controller_port, emit=logger.info)

            focus_cycle = build_focus_cycle_pattern(
                start_dpt=self._start_dpt,
                pattern=self._focus_pattern,
                scan_height_mm=self._scan_height_mm,
                scan_step_um=self._scan_step_um,
                scan_ratio=self._scan_ratio,
            )
            if not focus_cycle:
                focus_cycle = [round(self._start_dpt, 3)]
            unique_diopter_steps = build_unique_diopter_steps(focus_cycle)

            prefocus_lens(
                lens, self._start_dpt, self._acquisition_config["focusing"]["prefocus_settle_sec"], emit=logger.info
            )

        planner = _FocusPlanner(
            focus_cycle,
            unique_diopter_steps,
            lens is not None,
            self._start_dpt,
            self._scan_ratio,
            self._focus_pattern,
            self._adaptive_enabled,
            self._adaptive_window_fraction,
            self._adaptive_min_window_points,
            self._adaptive_dispersion_coeff,
            self._acquisition_control,
        )

        camera = open_first_camera(pylon, emit=logger.info)

        try:
            camera_config = self._acquisition_config["camera"]
            select_pixel_format(camera, camera_config["pixel_format_preference"], False, emit=logger.info)
            actual_exposure_us = float(set_exposure(camera, self._exposure_us, emit=logger.info))
            set_buffer_count(camera, camera_config["max_num_buffer_min"], self._target_fps, emit=logger.info)

            if (self._roi_width is None) ^ (self._roi_height is None):
                raise ValueError("Realtime ROI requires both width and height when either is set.")
            if self._roi_width is not None and self._roi_height is not None:
                actual_w, actual_h = set_centered_roi(
                    camera,
                    self._roi_width,
                    self._roi_height,
                    emit=lambda msg: logger.info(msg),
                )
                logger.info(
                    "Realtime camera centered ROI set to %sx%s (requested %dx%d)",
                    actual_w,
                    actual_h,
                    self._roi_width,
                    self._roi_height,
                )

            if hasattr(camera, "AcquisitionMode"):
                symbols = list(camera.AcquisitionMode.Symbolics)
                if "Continuous" in symbols:
                    camera.AcquisitionMode.SetValue("Continuous")
            if hasattr(camera, "TriggerMode"):
                symbols = list(camera.TriggerMode.Symbolics)
                if "Off" in symbols:
                    camera.TriggerMode.SetValue("Off")
            set_frame_rate(camera, float(self._target_fps), emit=logger.info)

            retrieve_timeout = int(1000 / max(self._target_fps, 1.0) + self._retrieve_timeout_margin_ms)
            trigger_mode = self._acquisition_config["camera"]["trigger_mode"]
            if lens is not None and trigger_mode == TRIGGER_MODE_SOFTWARE_GRID:
                yield from self._iterate_with_grid_trigger(pylon, camera, lens, planner, actual_exposure_us,
                                                           retrieve_timeout)
            else:
                if lens is None and trigger_mode == TRIGGER_MODE_SOFTWARE_GRID:
                    logger.info("ETL control disabled: no lens commands to synchronise, the camera free-runs.")
                yield from self._iterate_with_free_run(pylon, camera, lens, planner, retrieve_timeout)
        finally:
            close_camera(camera, emit=logger.info, restore_full_roi=False)
            if ecc1c is not None:
                disconnect_lens(ecc1c, emit=logger.info)

    def _stop_requested(self, entry_index: int) -> bool:
        if self._max_frames is not None and entry_index >= int(self._max_frames):
            return True
        return bool(self._acquisition_control and self._acquisition_control.stop_requested())

    def _iterate_with_free_run(self, pylon: Any, camera: Any, lens: Any, planner: _FocusPlanner,
                               retrieve_timeout: int) -> Iterator[FramePacket]:
        """Legacy loop: lens command, settle sleep, next frame of the free-running camera."""
        tick_hz = camera_tick_frequency_hz(camera)
        consecutive_timeouts = 0
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

        frame_idx = 0
        previous_diopter = self._start_dpt
        while camera.IsGrabbing():
            if self._stop_requested(frame_idx):
                break

            entry = planner.next_entry(advance=frame_idx > 0)
            if lens is not None and not entry.repeat:
                diopter_change = entry.diopter - previous_diopter
                settle_time_this_frame = (
                    self._large_jump_settle_sec
                    if diopter_change < self._large_jump_threshold_dpt
                    else self._lens_settle_time_sec
                )
                lens.StaticInput.SetAsInput()
                lens.StaticInput.SetFocalPower(entry.diopter)
                if settle_time_this_frame > 0:
                    time.sleep(settle_time_this_frame)
                previous_diopter = entry.diopter

            try:
                grab_result = camera.RetrieveResult(retrieve_timeout, pylon.TimeoutHandling_ThrowException)
            except Exception as exc:  # noqa: BLE001
                consecutive_timeouts += 1
                logger.warning(
                    "Camera RetrieveResult timeout/failure (count=%d, timeout=%dms): %s",
                    consecutive_timeouts,
                    retrieve_timeout,
                    exc,
                )
                if consecutive_timeouts >= self._max_consecutive_missing_frames:
                    raise RuntimeError("Camera acquisition repeatedly timed out; stopping realtime session.") from exc
                continue

            try:
                consecutive_timeouts = 0
                if not grab_result.GrabSucceeded():
                    continue
                raw_frame = grab_result.Array
                if raw_frame is None:
                    continue
                image_gray = _to_uint8_grayscale(raw_frame)
                yield FramePacket(
                    frame_index=frame_idx,
                    timestamp_s=time.monotonic(),
                    image_gray=image_gray,
                    diopter=entry.diopter,
                    focusing_distance_mm=entry.focusing_distance_mm,
                    focus_round_id=entry.focus_round_id,
                    camera_timestamp_ns=camera_ticks_to_ns(grab_timestamp_ticks(grab_result), tick_hz),
                )
                frame_idx += 1
            finally:
                grab_result.Release()

    def _iterate_with_grid_trigger(self, pylon: Any, camera: Any, lens: Any, planner: _FocusPlanner,
                                   exposure_us: float, retrieve_timeout: int) -> Iterator[FramePacket]:
        """Fixed-grid trigger loop on a background thread; frames reach the generator through
        a hand-off of ``queue_size`` frames. With ``drop_frame_if_backlogged`` a full hand-off
        drops the frame; without, the loop waits before the next lens command and then starts
        a new grid (no catch-up burst). A request of the AcquisitionControl applies from the
        next lens command, which follows right after the current exposure."""
        camera_config = self._acquisition_config["camera"]
        params = GridTimingParams.from_config(self._acquisition_config, self._target_fps, exposure_us,
                                              retrieve_timeout)
        period_s = 1.0 / params.fps
        loop = FixedGridTriggerLoop(camera, lens, params, emit=logger.warning)
        handoff = _FrameHandoff(self._queue_size)
        tick_hz = camera_tick_frequency_hz(camera)
        entries: dict[int, _FocusEntry] = {}
        counts = {"frames_seen": 0, "frames_dropped": 0}
        session_stop = threading.Event()
        outcome: dict[str, Any] = {}

        def should_stop(entry_index: int) -> bool:
            return session_stop.is_set() or self._stop_requested(entry_index)

        def next_diopter(entry_index: int) -> Optional[float]:
            if should_stop(entry_index):
                return None
            if len(entries) - counts["frames_seen"] >= self._max_consecutive_missing_frames:
                raise RuntimeError("Camera delivered no frames for the last triggers; stopping realtime session.")
            if not self._drop_frame_if_backlogged:
                if handoff.wait_for_space(lambda: should_stop(entry_index), period_s):
                    loop.request_grid_restart()
                if should_stop(entry_index):
                    return None
            entry = planner.next_entry(advance=entry_index > 0)
            entries[entry_index] = entry
            return entry.diopter

        def on_frame(frame: GrabbedFrame, image_array: Optional[np.ndarray]) -> None:
            counts["frames_seen"] += 1
            entry = entries.get(frame.entry_index) if frame.entry_index is not None else None
            if entry is None or image_array is None:
                return
            item = (frame.entry_index, time.monotonic(), image_array, entry,
                    camera_ticks_to_ns(frame.camera_ticks, tick_hz))
            if not handoff.offer(item, drop_when_full=self._drop_frame_if_backlogged):
                counts["frames_dropped"] += 1
                if counts["frames_dropped"] == 1:
                    logger.warning(f"Realtime frame hand-off full ({self._queue_size} frames): dropping frames "
                                   f"while processing is behind (drop_frame_if_backlogged).")

        def run_loop() -> None:
            try:
                outcome["recording"] = loop.run(next_diopter, on_frame, initial_diopter=self._start_dpt,
                                                grab_strategy=pylon.GrabStrategy_OneByOne)
            except BaseException as exc:  # noqa: BLE001 - re-raised by the generator
                outcome["error"] = exc
            finally:
                handoff.close()

        clock_samples_before = sample_camera_clock(camera, int(camera_config["clock_sync_samples"]))
        logger.info(f"Realtime fixed-grid software trigger at {params.fps:g} fps "
                    f"(lens settle >= {params.min_settle_s * 1000:g} ms).")
        loop_thread = threading.Thread(target=run_loop, name="realtime-grid-trigger", daemon=True)
        loop_thread.start()
        try:
            while True:
                item = handoff.take()
                if item is None:
                    break
                entry_index, timestamp_s, image_array, entry, camera_timestamp_ns = item
                yield FramePacket(
                    frame_index=entry_index,
                    timestamp_s=timestamp_s,
                    image_gray=_to_uint8_grayscale(image_array),
                    diopter=entry.diopter,
                    focusing_distance_mm=entry.focusing_distance_mm,
                    focus_round_id=entry.focus_round_id,
                    camera_timestamp_ns=camera_timestamp_ns,
                )
        finally:
            session_stop.set()
            handoff.close()
            loop_thread.join()
            recording = outcome.get("recording")
            if recording is not None:
                clock_map = build_camera_clock_map(
                    clock_samples_before, sample_camera_clock(camera, int(camera_config["clock_sync_samples"])), tick_hz)
                summary = summarize_grid_recording(recording, clock_map, tick_hz,
                                                   camera_config["trigger_late_report_ms"],
                                                   camera_config["timestamp_mismatch_ms"])
                logger.info(f"Realtime recording check: {summary.format(camera_config['trigger_late_report_ms'])}; "
                            f"frames dropped for slow processing {counts['frames_dropped']}")
        if "error" in outcome:
            raise RuntimeError("Fixed-grid realtime acquisition failed; stopping realtime session.") from outcome["error"]


def build_frame_source(
    source: str,
    *,
    input_dir: Optional[str],
    camera_index: int,
    target_fps: float,
    max_frames: Optional[int],
    retrieve_timeout_margin_ms: int,
    adaptive_enabled: bool,
    adaptive_window_fraction: Optional[float],
    adaptive_min_window_points: Optional[int],
    adaptive_dispersion_coeff: Optional[int],
    exposure_us: float,
    roi_width: Optional[int],
    roi_height: Optional[int],
    etl_control_enabled: bool,
    start_dpt: float,
    focus_pattern: str,
    scan_height_mm: float,
    scan_step_um: float,
    scan_ratio: float,
    lens_settle_time_sec: float,
    large_jump_threshold_dpt: float,
    large_jump_settle_sec: float,
    lens_controller_port: Optional[str],
    acquisition_config: Dict[str, Any],
    queue_size: int,
    drop_frame_if_backlogged: bool,
    max_consecutive_missing_frames: int,
    acquisition_control: Optional[AcquisitionControl] = None,
) -> FrameSource:
    if source == "directory":
        if not input_dir:
            raise ValueError("realtime_mode.acquisition.input_dir is required when source=directory")
        return DirectoryFrameSource(input_dir=input_dir, target_fps=target_fps, max_frames=max_frames)
    _ = camera_index
    return CameraFrameSource(
        target_fps=target_fps,
        max_frames=max_frames,
        retrieve_timeout_margin_ms=retrieve_timeout_margin_ms,
        adaptive_enabled=adaptive_enabled,
        adaptive_window_fraction=adaptive_window_fraction,
        adaptive_min_window_points=adaptive_min_window_points,
        adaptive_dispersion_coeff=adaptive_dispersion_coeff,
        exposure_us=exposure_us,
        roi_width=roi_width,
        roi_height=roi_height,
        etl_control_enabled=etl_control_enabled,
        start_dpt=start_dpt,
        focus_pattern=focus_pattern,
        scan_height_mm=scan_height_mm,
        scan_step_um=scan_step_um,
        scan_ratio=scan_ratio,
        lens_settle_time_sec=lens_settle_time_sec,
        large_jump_threshold_dpt=large_jump_threshold_dpt,
        large_jump_settle_sec=large_jump_settle_sec,
        lens_controller_port=lens_controller_port,
        acquisition_config=acquisition_config,
        queue_size=queue_size,
        drop_frame_if_backlogged=drop_frame_if_backlogged,
        max_consecutive_missing_frames=max_consecutive_missing_frames,
        acquisition_control=acquisition_control,
    )
