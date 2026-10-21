"""Fixed-grid, software-triggered ETL acquisition loop and its timing helpers.

Why this exists: in free-run at 25 fps a full 4512 x 4512 frame needs ~35 ms to reach the
PC, so the frame returned after a lens command was exposed *before* that command and shows
the plane commanded for the previous frame (rig diagnosis, Oct 2026). Here every frame is
exposed by its own software trigger, on an absolute time grid ``t0 + i / fps``:

    lens command (entry i) --(>= min settle)--> WaitForFrameTriggerReady (lead before due)
    --> wait until due --> ExecuteSoftwareTrigger --> wait exposure + margin --> entry i + 1

A late trigger fires at once and the following triggers stay on the grid. A grabber thread
issues one ``RetrieveResult`` per fired trigger and hands each frame, with the pattern entry
it belongs to (by block ID: the first block ID is the first fired trigger), to a callback,
so no image work runs in the trigger loop.

Used by the batch acquisition (``wellscanner_acquisition_module.acquire_and_save_frames``);
built to be reused by the realtime camera source (``next_diopter`` may choose each entry on
the fly).
"""
from __future__ import annotations

import os
import queue
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

pc = time.perf_counter

TRIGGER_MODE_SOFTWARE_GRID = "software_grid"
TRIGGER_MODE_FREE_RUN = "free_run"
TRIGGER_MODES = (TRIGGER_MODE_SOFTWARE_GRID, TRIGGER_MODE_FREE_RUN)

# Facts, not settings:
_WINDOWS_TIMER_RESOLUTION_MS = 1        # finest period timeBeginPeriod accepts
_SFNC_TIMESTAMP_TICK_HZ = 1e9           # SFNC cameras (Basler ace 2) count timestamps in ns
_GEV1_BLOCK_ID_PERIOD = 0xFFFF          # GigE Vision 1.x block IDs run 1..65535, then wrap to 1
_LENS_TEMPERATURE_PLAUSIBLE_C = (-30.0, 120.0)   # rejects calls that return something else
# optoICC calls that may return the lens temperature, tried in this order (Optotune documents
# controller.TemperatureManager.GetDeviceTemperature() for C#; the Python names vary).
_LENS_TEMPERATURE_CALLS = (
    "ecc.TemperatureManager.GetDeviceTemperature",
    "lens.TemperatureManager.GetDeviceTemperature",
    "ecc.Lens.TemperatureManager.GetDeviceTemperature",
    "ecc.TemperatureManager.GetTemperature",
    "lens.TemperatureManager.GetTemperature",
    "lens.GetTemperature",
    "ecc.GetTemperature",
)


# ---------------------------------------------------------------------------
# Camera trigger configuration
# ---------------------------------------------------------------------------
def configure_software_trigger(camera: Any) -> None:
    """One exposure per ``ExecuteSoftwareTrigger``; no internal frame-rate limit."""
    camera.AcquisitionFrameRateEnable.SetValue(False)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Software")


def restore_free_run(camera: Any) -> None:
    """Back to free-run at the camera's ``AcquisitionFrameRate`` (its value is left as set)."""
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("Off")
    camera.AcquisitionFrameRateEnable.SetValue(True)


# ---------------------------------------------------------------------------
# Windows timing
# ---------------------------------------------------------------------------
def _windows_multimedia_timer() -> Optional[Any]:
    """winmm on Windows (timeBeginPeriod / timeEndPeriod), None on other systems."""
    if os.name != "nt":
        return None
    import ctypes
    return ctypes.windll.winmm


@contextmanager
def precise_timing(switch_interval_s: float) -> Iterator[None]:
    """While a trigger loop runs: 1 ms system timer resolution (Windows ``time.sleep``
    granularity is ~15.6 ms otherwise) and a short GIL switch interval, so the grabber and
    saver threads cannot hold up a trigger for the default 5 ms."""
    old_switch_interval = sys.getswitchinterval()
    winmm = _windows_multimedia_timer()
    sys.setswitchinterval(switch_interval_s)
    if winmm is not None:
        winmm.timeBeginPeriod(_WINDOWS_TIMER_RESOLUTION_MS)
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(_WINDOWS_TIMER_RESOLUTION_MS)
        sys.setswitchinterval(old_switch_interval)


def sleep_until(deadline_s: float, busy_wait_s: float) -> None:
    """Sleep until ``perf_counter() >= deadline_s``; busy-wait the last ``busy_wait_s``."""
    while True:
        remaining_s = deadline_s - pc()
        if remaining_s <= 0:
            return
        if remaining_s > busy_wait_s:
            time.sleep(remaining_s - busy_wait_s)
        else:
            while pc() < deadline_s:
                pass
            return


# ---------------------------------------------------------------------------
# Camera clock: exposure timestamps on the PC clock
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockSample:
    """One camera-clock latch with the PC times just before and after it."""
    host_before_s: float
    host_after_s: float
    camera_ticks: int


def latch_camera_clock(camera: Any) -> Optional[int]:
    """Camera timestamp ticks latched now (SFNC, then GigE node names); None if unsupported."""
    for command_name, value_name in (("TimestampLatch", "TimestampLatchValue"),
                                     ("GevTimestampControlLatch", "GevTimestampValue")):
        try:
            getattr(camera, command_name).Execute()
            return int(getattr(camera, value_name).GetValue())
        except Exception:  # noqa: BLE001 - node absent on this camera: try the other naming
            continue
    return None


def sample_camera_clock(camera: Any, n_samples: int) -> List[ClockSample]:
    """Latch the camera clock ``n_samples`` times (empty list if the camera cannot latch)."""
    samples: List[ClockSample] = []
    for _ in range(n_samples):
        host_before_s = pc()
        ticks = latch_camera_clock(camera)
        host_after_s = pc()
        if ticks is None:
            break
        samples.append(ClockSample(host_before_s, host_after_s, ticks))
    return samples


def camera_tick_frequency_hz(camera: Any) -> float:
    """GigE ``GevTimestampTickFrequency``; SFNC cameras without that node count in ns."""
    try:
        return float(camera.GevTimestampTickFrequency.GetValue())
    except Exception:  # noqa: BLE001 - SFNC camera (no Gev node): timestamps are in ns
        return _SFNC_TIMESTAMP_TICK_HZ


def grab_timestamp_ticks(grab_result: Any) -> Optional[int]:
    """Exposure timestamp of a grab result in camera ticks, None if it has none."""
    for attribute_name in ("TimeStamp", "GetTimeStamp"):
        try:
            value = getattr(grab_result, attribute_name)
            return int(value() if callable(value) else value)
        except Exception:  # noqa: BLE001 - attribute naming differs between pypylon versions
            continue
    return None


def camera_ticks_to_ns(camera_ticks: Optional[int], tick_hz: float) -> Optional[int]:
    """Camera timestamp in ns (None stays None)."""
    return None if camera_ticks is None else int(round(camera_ticks * 1e9 / tick_hz))


def grab_block_id(grab_result: Any) -> Optional[int]:
    """Block ID (frame counter of the camera stream) of a grab result, None if it has none."""
    try:
        return int(grab_result.GetBlockID())
    except Exception:  # noqa: BLE001 - no block ID (e.g. failed grab)
        return None


@dataclass(frozen=True)
class CameraClockMap:
    """Maps camera ticks to ``perf_counter`` seconds.

    The offset (camera seconds - PC seconds) is the median over the latches taken before
    and after a recording; between the two it is interpolated linearly in camera time, so a
    slow drift of the two clocks does not shift the mapped exposure times.
    """
    tick_hz: float
    anchor_camera_s: Tuple[float, ...]
    anchor_offset_s: Tuple[float, ...]

    def host_time_s(self, camera_ticks: int) -> float:
        camera_s = camera_ticks / self.tick_hz
        offset_s = float(np.interp(camera_s, self.anchor_camera_s, self.anchor_offset_s))
        return camera_s - offset_s


def build_camera_clock_map(samples_before: Sequence[ClockSample], samples_after: Sequence[ClockSample],
                           tick_hz: float) -> Optional[CameraClockMap]:
    """Clock map from the latches around a recording; None if the camera gave none."""
    anchors = []
    for samples in (samples_before, samples_after):
        if not samples:
            continue
        camera_s = [s.camera_ticks / tick_hz for s in samples]
        offsets = [c - 0.5 * (s.host_before_s + s.host_after_s) for c, s in zip(camera_s, samples)]
        anchors.append((statistics.median(camera_s), statistics.median(offsets)))
    if not anchors:
        return None
    anchors.sort()
    return CameraClockMap(tick_hz, tuple(a[0] for a in anchors), tuple(a[1] for a in anchors))


# ---------------------------------------------------------------------------
# Lens temperature
# ---------------------------------------------------------------------------
def _call_returning_float(fn: Callable[[], Any]) -> float:
    value = fn()
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _public_attribute_names(obj: Any) -> List[str]:
    try:
        return [name for name in dir(obj) if not name.startswith("_")]
    except Exception:  # noqa: BLE001 - objects of the vendor library may refuse dir()
        return []


def _attribute_or_none(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001 - probing: a vendor property may raise anything
        return None


def _resolve_call(path: str, roots: Dict[str, Any]) -> Any:
    root_name, *attribute_names = path.split(".")
    obj = roots[root_name]
    for attribute_name in attribute_names:
        obj = _attribute_or_none(obj, attribute_name)
    return obj


def find_lens_temperature_reader(ecc: Any, lens: Any) -> Tuple[Optional[Callable[[], float]], Optional[str]]:
    """Find an optoICC call that returns the lens temperature in degrees C.

    Tries the known call names, then every callable whose name contains 'temp' up to two
    attribute levels below the controller and the lens. Returns ``(reader, call path)``, or
    ``(None, None)`` if no call returns a plausible temperature (the controller may not
    report one; the caller then records no temperature).
    """
    roots = {"ecc": ecc, "lens": lens}
    low_c, high_c = _LENS_TEMPERATURE_PLAUSIBLE_C

    def _plausible(fn: Any) -> bool:
        try:
            return callable(fn) and low_c < _call_returning_float(fn) < high_c
        except Exception:  # noqa: BLE001 - probing: any failure means "not this call"
            return False

    for path in _LENS_TEMPERATURE_CALLS:
        fn = _resolve_call(path, roots)
        if _plausible(fn):
            return (lambda fn=fn: _call_returning_float(fn)), path
    for root_name, root in roots.items():
        for name in _public_attribute_names(root):
            if "temp" not in name.lower():
                continue
            obj = _attribute_or_none(root, name)
            candidates = [(f"{root_name}.{name}", obj)] + [
                (f"{root_name}.{name}.{sub}", _attribute_or_none(obj, sub))
                for sub in _public_attribute_names(obj) if "temp" in sub.lower()]
            for path, fn in candidates:
                if path not in _LENS_TEMPERATURE_CALLS and _plausible(fn):
                    return (lambda fn=fn: _call_returning_float(fn)), path
    return None, None


def read_lens_temperature(reader: Optional[Callable[[], float]]) -> Optional[float]:
    """One temperature reading, None without a reader or if the read fails."""
    if reader is None:
        return None
    try:
        return reader()
    except Exception:  # noqa: BLE001 - a failed read leaves the value empty
        return None


# ---------------------------------------------------------------------------
# Fixed-grid trigger loop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridTimingParams:
    """Timing of the fixed-grid loop (seconds); built from the config by ``from_config``."""
    fps: float
    exposure_s: float
    min_settle_s: float
    large_jump_settle_s: float
    large_jump_threshold_dpt: float
    trigger_ready_lead_s: float
    post_exposure_margin_s: float
    busy_wait_s: float
    switch_interval_s: float
    retrieve_timeout_ms: int
    lens_temperature_every_frames: int

    @classmethod
    def from_config(cls, acquisition_config: Dict[str, Any], fps: float, exposure_us: float,
                    retrieve_timeout_ms: int) -> "GridTimingParams":
        """Read the ``acquisition`` config; ``fps`` and ``exposure_us`` are the values the
        camera accepted, ``retrieve_timeout_ms`` bounds every camera wait."""
        focusing_config = acquisition_config["focusing"]
        camera_config = acquisition_config["camera"]
        if fps <= 0:
            raise ValueError(f"Fixed-grid acquisition needs a positive frame rate, got {fps}.")
        temperature_interval_s = float(focusing_config["lens_temperature_interval_sec"])
        if temperature_interval_s <= 0:
            raise ValueError("focusing.lens_temperature_interval_sec must be positive.")
        return cls(
            fps=float(fps),
            exposure_s=float(exposure_us) / 1e6,
            min_settle_s=float(focusing_config["min_settle_ms"]) / 1000.0,
            large_jump_settle_s=float(focusing_config["large_jump_settle_sec"]),
            large_jump_threshold_dpt=float(focusing_config["large_jump_threshold_dpt"]),
            trigger_ready_lead_s=float(camera_config["trigger_ready_lead_ms"]) / 1000.0,
            post_exposure_margin_s=float(camera_config["post_exposure_margin_ms"]) / 1000.0,
            busy_wait_s=float(camera_config["timer_busy_wait_ms"]) / 1000.0,
            switch_interval_s=float(camera_config["thread_switch_interval_ms"]) / 1000.0,
            retrieve_timeout_ms=int(retrieve_timeout_ms),
            lens_temperature_every_frames=max(1, int(round(fps * temperature_interval_s))),
        )


@dataclass
class GridTrigger:
    """One focus-pattern entry of a grid recording; times are ``perf_counter`` seconds."""
    entry_index: int
    diopter: float
    large_jump: bool
    lens_cmd_start_s: float
    lens_cmd_end_s: float
    grid_time_s: Optional[float] = None
    trigger_time_s: Optional[float] = None      # just before ExecuteSoftwareTrigger
    loop_end_s: Optional[float] = None          # post-exposure wait done
    lens_temp_c: Optional[float] = None
    grid_restart: bool = False                  # a new grid starts at this entry (after a pause)
    error: Optional[str] = None

    @property
    def trigger_late_ms(self) -> Optional[float]:
        if self.trigger_time_s is None or self.grid_time_s is None:
            return None
        return 1000.0 * (self.trigger_time_s - self.grid_time_s)


@dataclass
class GrabbedFrame:
    """One ``RetrieveResult`` of a grid recording that returned a frame."""
    entry_index: Optional[int]                  # None: block ID outside the fired triggers
    block_id: Optional[int]
    camera_ticks: Optional[int]
    retrieve_start_s: float
    retrieve_end_s: float
    grab_ok: bool


@dataclass
class GridRecording:
    """What a grid loop did: one trigger record per entry, frames in arrival order."""
    triggers: List[GridTrigger] = field(default_factory=list)
    frames: List[GrabbedFrame] = field(default_factory=list)
    retrieve_timeouts: int = 0
    grid_restarts: int = 0
    callback_errors: List[str] = field(default_factory=list)

    @property
    def triggers_fired(self) -> int:
        return sum(1 for t in self.triggers if t.trigger_time_s is not None)


class FixedGridTriggerLoop:
    """Software-triggered acquisition on a fixed time grid (see the module docstring).

    ``run`` configures the software trigger, sends ``SetAsInput()`` once, records until
    ``next_diopter(i)`` returns None, and restores free-run before it returns. Per entry the
    lens gets only ``SetFocalPower()``. A late trigger keeps the grid (the following
    triggers catch up, limited by the camera's minimum frame period); after an intentional
    pause, ``next_diopter`` calls ``request_grid_restart`` and the grid starts anew instead.
    """

    def __init__(self, camera: Any, lens: Any, params: GridTimingParams,
                 lens_temperature_reader: Optional[Callable[[], float]] = None,
                 emit: Callable[[str], None] = print) -> None:
        self._camera = camera
        self._lens = lens
        self._params = params
        self._temperature_reader = lens_temperature_reader
        self._emit = emit
        self._grid_restart_requested = False

    def request_grid_restart(self) -> None:
        """Start a new grid at the entry being requested instead of catching up the old one.

        For callers whose ``next_diopter`` paused on purpose (realtime: waiting for a slow
        consumer); call it from ``next_diopter``, before it returns the entry.
        """
        self._grid_restart_requested = True

    def run(self, next_diopter: Callable[[int], Optional[float]],
            on_frame: Callable[[GrabbedFrame, Optional[np.ndarray]], None],
            initial_diopter: float, grab_strategy: Any) -> GridRecording:
        """Record entries ``0, 1, ...`` until ``next_diopter`` returns None.

        ``on_frame(frame, image_array)`` runs on the grabber thread for every frame received
        (``image_array`` is None for a failed grab) and must return quickly. A large downward
        jump is measured from ``initial_diopter`` for entry 0 (where the lens was pre-focused).
        """
        from pypylon import pylon

        if grab_strategy != pylon.GrabStrategy_OneByOne:
            raise ValueError("Fixed-grid acquisition maps frames to triggers in arrival order and "
                             "needs camera.grab_strategy 'OneByOne'.")
        recording = GridRecording()
        fired_entries: List[int] = []          # entry index of every fired trigger, in order
        retrieve_requests: "queue.Queue[Optional[int]]" = queue.Queue()
        camera = self._camera
        configure_software_trigger(camera)
        self._lens.StaticInput.SetAsInput()
        try:
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            grabber = threading.Thread(target=self._grab_frames, name="grid-grabber", daemon=True,
                                       args=(pylon, recording, fired_entries, retrieve_requests, on_frame))
            grabber.start()
            try:
                self._trigger_loop(pylon, recording, fired_entries, retrieve_requests,
                                   next_diopter, initial_diopter)
            finally:
                retrieve_requests.put(None)
                grabber.join()
        finally:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                restore_free_run(camera)
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask a recording error
                self._emit(f"Error restoring free-run camera settings: {exc}")
        return recording

    def _trigger_loop(self, pylon: Any, recording: GridRecording, fired_entries: List[int],
                      retrieve_requests: "queue.Queue[Optional[int]]",
                      next_diopter: Callable[[int], Optional[float]], initial_diopter: float) -> None:
        params, camera = self._params, self._camera
        static_input = self._lens.StaticInput
        period_s = 1.0 / params.fps
        post_exposure_s = params.exposure_s + params.post_exposure_margin_s
        grid_start_s = None
        previous_diopter = initial_diopter
        entry_index = 0
        with precise_timing(params.switch_interval_s):
            while True:
                diopter = next_diopter(entry_index)
                if diopter is None:
                    return
                large_jump = (diopter - previous_diopter) < params.large_jump_threshold_dpt
                cmd_start_s = pc()
                try:
                    static_input.SetFocalPower(diopter)
                    command_error = None
                except Exception as exc:  # noqa: BLE001 - controller error: skip this frame
                    command_error = f"lens command failed: {exc}"
                cmd_end_s = pc()
                trigger = GridTrigger(entry_index, float(diopter), large_jump, cmd_start_s, cmd_end_s,
                                      error=command_error)
                recording.triggers.append(trigger)
                previous_diopter = diopter
                if self._temperature_reader is not None and entry_index % params.lens_temperature_every_frames == 0:
                    trigger.lens_temp_c = self._read_temperature_in_loop()
                restarted_grid_start_s = cmd_end_s + params.min_settle_s - entry_index * period_s
                if grid_start_s is None:
                    grid_start_s = restarted_grid_start_s
                elif self._grid_restart_requested and restarted_grid_start_s > grid_start_s:
                    # only a pause that made this entry late moves the grid, and only later
                    grid_start_s = restarted_grid_start_s
                    trigger.grid_restart = True
                    recording.grid_restarts += 1
                self._grid_restart_requested = False
                trigger.grid_time_s = grid_start_s + entry_index * period_s
                settle_s = params.large_jump_settle_s if large_jump else params.min_settle_s
                fire_at_s = max(trigger.grid_time_s, cmd_end_s + settle_s)
                if command_error is None:
                    self._fire(pylon, trigger, fire_at_s, fired_entries, retrieve_requests)
                if trigger.error is not None:
                    self._emit(f"Frame {entry_index}: {trigger.error}; no frame recorded for this entry.")
                else:
                    sleep_until(pc() + post_exposure_s, params.busy_wait_s)   # lens must not move during the exposure
                trigger.loop_end_s = pc()
                entry_index += 1

    def _fire(self, pylon: Any, trigger: GridTrigger, fire_at_s: float, fired_entries: List[int],
              retrieve_requests: "queue.Queue[Optional[int]]") -> None:
        params, camera = self._params, self._camera
        try:
            sleep_until(fire_at_s - params.trigger_ready_lead_s, params.busy_wait_s)
            camera.WaitForFrameTriggerReady(params.retrieve_timeout_ms, pylon.TimeoutHandling_ThrowException)
            sleep_until(fire_at_s, params.busy_wait_s)
        except Exception as exc:  # noqa: BLE001 - camera not ready: skip, keep the grid
            trigger.error = f"camera not ready for the trigger: {exc}"
            return
        fired_entries.append(trigger.entry_index)
        trigger.trigger_time_s = pc()
        try:
            camera.ExecuteSoftwareTrigger()
        except Exception as exc:  # noqa: BLE001 - trigger not executed: skip, keep the grid
            fired_entries.pop()
            trigger.trigger_time_s = None
            trigger.error = f"software trigger failed: {exc}"
            return
        retrieve_requests.put(trigger.entry_index)

    def _read_temperature_in_loop(self) -> Optional[float]:
        """Read in the settling time; a failing reader is not asked again in this recording."""
        try:
            return self._temperature_reader()
        except Exception as exc:  # noqa: BLE001 - the recording goes on without temperature
            self._emit(f"Warning: lens temperature read failed ({exc}); no further reads in this recording.")
            self._temperature_reader = None
            return None

    def _grab_frames(self, pylon: Any, recording: GridRecording, fired_entries: List[int],
                     retrieve_requests: "queue.Queue[Optional[int]]",
                     on_frame: Callable[[GrabbedFrame, Optional[np.ndarray]], None]) -> None:
        first_block_id: List[Optional[int]] = [None]
        while retrieve_requests.get() is not None:
            self._retrieve_one(pylon, recording, fired_entries, first_block_id, on_frame)
        # A frame that arrived after its RetrieveResult timed out is still in the queue.
        while len(recording.frames) < len(fired_entries):
            if not self._retrieve_one(pylon, recording, fired_entries, first_block_id, on_frame):
                break

    def _retrieve_one(self, pylon: Any, recording: GridRecording, fired_entries: List[int],
                      first_block_id: List[Optional[int]],
                      on_frame: Callable[[GrabbedFrame, Optional[np.ndarray]], None]) -> bool:
        retrieve_start_s = pc()
        grab_result = self._camera.RetrieveResult(self._params.retrieve_timeout_ms, pylon.TimeoutHandling_Return)
        retrieve_end_s = pc()
        if grab_result is None or not grab_result.IsValid():
            recording.retrieve_timeouts += 1
            return False
        try:
            grab_ok = bool(grab_result.GrabSucceeded())
            block_id = grab_block_id(grab_result)
            frame = GrabbedFrame(_entry_for_block_id(block_id, first_block_id, fired_entries), block_id,
                                 grab_timestamp_ticks(grab_result), retrieve_start_s, retrieve_end_s, grab_ok)
            recording.frames.append(frame)
            try:
                on_frame(frame, grab_result.Array if grab_ok else None)
            except Exception as exc:  # noqa: BLE001 - keep retrieving; the caller sees the error
                recording.callback_errors.append(f"block {block_id}: {type(exc).__name__}: {exc}")
        finally:
            grab_result.Release()
        return True


def _entry_for_block_id(block_id: Optional[int], first_block_id: List[Optional[int]],
                        fired_entries: List[int]) -> Optional[int]:
    """Pattern entry of a frame: its block-ID offset from the first frame indexes the fired
    triggers (a skipped trigger therefore does not shift the frames after it)."""
    if block_id is None:
        return None
    if first_block_id[0] is None:
        first_block_id[0] = block_id
    offset = block_id - first_block_id[0]
    if offset < 0:
        offset += _GEV1_BLOCK_ID_PERIOD
    return fired_entries[offset] if offset < len(fired_entries) else None


# ---------------------------------------------------------------------------
# Recording checks
# ---------------------------------------------------------------------------
def block_id_steps(block_ids: Sequence[int]) -> List[int]:
    """Differences between consecutive block IDs (arrival order), wrap-around removed."""
    steps = []
    for previous, current in zip(block_ids[:-1], block_ids[1:]):
        step = current - previous
        steps.append(step + _GEV1_BLOCK_ID_PERIOD if step < 0 else step)
    return steps


@dataclass(frozen=True)
class RecordingSummary:
    """Per-recording check printed instead of per-frame timing warnings."""
    frames_expected: int
    frames_received: int
    mean_frame_interval_ms: Optional[float]
    block_id_gaps: int
    triggers_late: Optional[int] = None             # None: free-run (no triggers)
    triggers_late_large_jump: Optional[int] = None
    timestamp_mismatches: Optional[int] = None      # None: no triggers or no camera clock
    exposure_minus_trigger_ms_median: Optional[float] = None
    grid_restarts: int = 0

    def format(self, late_threshold_ms: float) -> str:
        interval = "n/a" if self.mean_frame_interval_ms is None else f"{self.mean_frame_interval_ms:.2f} ms"
        parts = [f"frames received {self.frames_received}/{self.frames_expected}",
                 f"mean frame interval {interval} (camera timestamps)",
                 f"block-ID gaps {self.block_id_gaps}"]
        if self.triggers_late is None:
            parts.append("triggers late: n/a (free-run)")
        else:
            parts.append(f"triggers late by > {late_threshold_ms:g} ms {self.triggers_late}"
                         + (f" (incl. {self.triggers_late_large_jump} delayed after a large lens jump)"
                            if self.triggers_late_large_jump else ""))
        if self.timestamp_mismatches is None:
            parts.append("timestamp mismatches: n/a")
        else:
            parts.append(f"timestamp mismatches {self.timestamp_mismatches} "
                         f"(exposure - trigger median {self.exposure_minus_trigger_ms_median:.2f} ms)")
        if self.grid_restarts:
            parts.append(f"grid restarted {self.grid_restarts} times after a pause")
        return "; ".join(parts)


def summarize_recording(frames_expected: int, block_ids: Sequence[Optional[int]],
                        camera_ticks: Sequence[Optional[int]], tick_hz: float,
                        expected_block_id_steps: Optional[Sequence[int]] = None,
                        trigger_late_ms: Optional[Sequence[float]] = None,
                        late_is_large_jump: Optional[Sequence[bool]] = None,
                        exposure_minus_trigger_ms: Optional[Sequence[float]] = None,
                        late_threshold_ms: Optional[float] = None,
                        mismatch_tolerance_ms: Optional[float] = None) -> RecordingSummary:
    """Check a recording from its received frames, in recording order.

    ``block_ids`` / ``camera_ticks``: one per received frame (None where the camera gave
    none). ``expected_block_id_steps``: the step expected before each frame after the first
    (default 1; the free-run loop discards one frame after a large jump). The mean frame
    interval is the camera-timestamp span over the block-ID span, so a lost or discarded
    frame does not inflate it. ``trigger_late_ms`` / ``late_is_large_jump`` (one per fired
    trigger) and ``exposure_minus_trigger_ms`` (one per received frame with a mapped exposure
    time) are given for triggered recordings only. A frame is a timestamp mismatch when its
    exposure - trigger deviates from the recording's median by more than
    ``mismatch_tolerance_ms``: it was exposed by another trigger than the one it was assigned to.
    """
    known_ids = [b for b in block_ids if b is not None]
    steps = block_id_steps(known_ids)
    if expected_block_id_steps is None or len(known_ids) != len(block_ids):
        expected_block_id_steps = [1] * len(steps)
    gaps = sum(1 for step, expected in zip(steps, expected_block_id_steps) if step != expected)
    mean_interval_ms = None
    known_ticks = [t for t in camera_ticks if t is not None]
    if len(known_ticks) > 1:
        n_intervals = sum(steps) if len(known_ids) == len(camera_ticks) == len(known_ticks) else len(known_ticks) - 1
        if n_intervals > 0:
            mean_interval_ms = 1000.0 * (known_ticks[-1] - known_ticks[0]) / tick_hz / n_intervals
    summary = RecordingSummary(frames_expected, len(block_ids), mean_interval_ms, gaps)
    if trigger_late_ms is None:
        return summary
    late = [i for i, late_ms in enumerate(trigger_late_ms) if late_ms > late_threshold_ms]
    mismatches = median_ms = None
    if exposure_minus_trigger_ms:
        median_ms = statistics.median(exposure_minus_trigger_ms)
        mismatches = sum(1 for d in exposure_minus_trigger_ms if abs(d - median_ms) > mismatch_tolerance_ms)
    return RecordingSummary(summary.frames_expected, summary.frames_received, summary.mean_frame_interval_ms,
                            summary.block_id_gaps, len(late), sum(1 for i in late if late_is_large_jump[i]),
                            mismatches, median_ms)


def summarize_grid_recording(recording: GridRecording, clock_map: Optional[CameraClockMap], tick_hz: float,
                             late_threshold_ms: float, mismatch_tolerance_ms: float) -> RecordingSummary:
    """Recording check of a fixed-grid recording (batch and realtime print the same check).

    Received frames are those mapped to an entry with a successful grab, in entry order;
    exposure - trigger needs the camera clock map (None: no mismatch check).
    """
    frames_by_entry: Dict[int, GrabbedFrame] = {}
    for frame in recording.frames:
        if frame.entry_index is not None and frame.grab_ok:
            frames_by_entry.setdefault(frame.entry_index, frame)
    received = [frames_by_entry[i] for i in sorted(frames_by_entry)]
    fired = [t for t in recording.triggers if t.trigger_time_s is not None]
    exposure_minus_trigger_ms = []
    if clock_map is not None:
        for frame in received:
            trigger = recording.triggers[frame.entry_index]
            if frame.camera_ticks is not None and trigger.trigger_time_s is not None:
                exposure_minus_trigger_ms.append(
                    1000.0 * (clock_map.host_time_s(frame.camera_ticks) - trigger.trigger_time_s))
    summary = summarize_recording(
        len(recording.triggers), [f.block_id for f in received], [f.camera_ticks for f in received], tick_hz,
        trigger_late_ms=[t.trigger_late_ms for t in fired], late_is_large_jump=[t.large_jump for t in fired],
        exposure_minus_trigger_ms=exposure_minus_trigger_ms, late_threshold_ms=late_threshold_ms,
        mismatch_tolerance_ms=mismatch_tolerance_ms)
    return replace(summary, grid_restarts=recording.grid_restarts)
