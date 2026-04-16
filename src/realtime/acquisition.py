from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import os
import re
import time
from typing import Iterator, Optional, Tuple

import numpy as np

from src.core.image_io_operations import load_image
from src.acquire.roi import set_centered_roi

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    frame_index: int
    timestamp_s: float
    image_gray: np.ndarray
    diopter: Optional[float] = None
    focusing_distance_mm: Optional[float] = None
    focus_round_id: Optional[int] = None




class AcquisitionControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._repeat_same_diopter_requests = 0
        self._pending_adaptive_center_diopter: Optional[float] = None

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


def build_focus_cycle_from_up_values(pattern: str, diopter_values_up: list[float]) -> list[float]:
    normalized_pattern = pattern.strip().lower()
    if normalized_pattern == "sawtooth":
        return diopter_values_up
    if normalized_pattern != "zigzag":
        logger.warning("Unknown focus pattern '%s'. Defaulting to 'zigzag'.", pattern)
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
    end_dpt = start_dpt + scan_height_mm / scan_ratio
    step_dpt = (scan_step_um / 1000.0) / scan_ratio
    if abs(step_dpt) < 1e-9:
        return [round(start_dpt, 3)]

    diopter_values_up = np.arange(start_dpt, end_dpt + (step_dpt / 2.0), step_dpt)
    if diopter_values_up.size == 0:
        diopter_values_up = np.array([start_dpt])
    diopter_values_up = diopter_values_up.round(3).tolist()
    return build_focus_cycle_from_up_values(pattern, diopter_values_up)


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


class CameraFrameSource(FrameSource):
    """Realtime camera source using pypylon and explicit ETL command-per-frame stepping."""

    def __init__(
        self,
        target_fps: float,
        max_frames: Optional[int] = None,
        retrieve_timeout_margin_ms: int = 150,
        adaptive_enabled: bool = False,
        adaptive_window_fraction: Optional[float] = None,
        adaptive_min_window_points: Optional[int] = None,
        adaptive_dispersion_coeff: Optional[int] = None,
        exposure_us: float = 750.0,
        roi_width: Optional[int] = None,
        roi_height: Optional[int] = None,
        *,
        etl_control_enabled: bool = True,
        start_dpt: float = -2.0,
        focus_pattern: str = "zigzag",
        scan_height_mm: float = 11.2,
        scan_step_um: float = 350.0,
        scan_ratio: float = 2.25,
        lens_settle_time_sec: float = 0.025,
        large_jump_threshold_dpt: float = -3.0,
        large_jump_settle_sec: float = 0.04,
        lens_controller_port: Optional[str] = "COM5",
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
        self._acquisition_control = acquisition_control

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            from pypylon import pylon
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pypylon is required for realtime camera acquisition") from exc

        lens = None
        ecc1c = None
        focus_cycle = [round(self._start_dpt, 3)]
        base_focus_cycle = [round(self._start_dpt, 3)]
        unique_diopter_steps = [round(self._start_dpt, 3)]
        if self._etl_control_enabled:
            try:
                import optoICC
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("optoICC is required for ETL-controlled realtime camera acquisition") from exc

            if not self._lens_controller_port:
                raise RuntimeError("focusing.lens_controller_port must be set when ETL control is enabled")
            ecc1c = optoICC.connectEcc(port=self._lens_controller_port)
            lens = ecc1c.Lens
            if lens is None:
                raise RuntimeError("Failed to get valid Lens object from ECC-1C")

            focus_cycle = build_focus_cycle_pattern(
                start_dpt=self._start_dpt,
                pattern=self._focus_pattern,
                scan_height_mm=self._scan_height_mm,
                scan_step_um=self._scan_step_um,
                scan_ratio=self._scan_ratio,
            )
            if not focus_cycle:
                focus_cycle = [round(self._start_dpt, 3)]
            base_focus_cycle = list(focus_cycle)
            unique_diopter_steps = build_unique_diopter_steps(base_focus_cycle)

            lens.StaticInput.SetAsInput()
            lens.StaticInput.SetFocalPower(self._start_dpt)
            time.sleep(1.0)

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Pylon compatible camera found")

        camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        camera.Open()

        try:
            def _set_node_value_with_clamp(node: object, requested_value: float) -> float:
                value = float(requested_value)
                try:
                    if hasattr(node, "Min"):
                        value = max(value, float(node.Min))
                    if hasattr(node, "Max"):
                        value = min(value, float(node.Max))
                    if hasattr(node, "Inc"):
                        inc = float(node.Inc)
                        min_v = float(node.Min) if hasattr(node, "Min") else 0.0
                        if inc > 0:
                            steps = round((value - min_v) / inc)
                            value = min_v + steps * inc
                    node.SetValue(value)
                    if hasattr(node, "GetValue"):
                        return float(node.GetValue())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to set camera node value to %s: %s", requested_value, exc)
                return value

            if hasattr(camera, "PixelFormat"):
                available_formats = list(camera.PixelFormat.Symbolics)
                if "Mono8" in available_formats:
                    camera.PixelFormat.SetValue("Mono8")
                elif "Mono12" in available_formats:
                    camera.PixelFormat.SetValue("Mono12")

            if hasattr(camera, "ExposureTime"):
                actual_exp = _set_node_value_with_clamp(camera.ExposureTime, self._exposure_us)
                logger.info("Realtime camera exposure set to %.3f us (requested %.3f us)", actual_exp, self._exposure_us)

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
            if hasattr(camera, "AcquisitionFrameRateEnable"):
                camera.AcquisitionFrameRateEnable.SetValue(True)
            if hasattr(camera, "AcquisitionFrameRate"):
                camera.AcquisitionFrameRate.SetValue(float(self._target_fps))

            retrieve_timeout = int(1000 / max(self._target_fps, 1.0) + self._retrieve_timeout_margin_ms)
            consecutive_timeouts = 0
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

            frame_idx = 0
            focus_step_idx = 0
            focus_round_id = 0
            active_focus_cycle = list(focus_cycle)
            previous_diopter = self._start_dpt
            while camera.IsGrabbing():
                if self._max_frames is not None and frame_idx >= int(self._max_frames):
                    break

                repeat_same_diopter = bool(
                    self._acquisition_control and self._acquisition_control.consume_repeat_same_diopter_request()
                )
                if frame_idx > 0 and not repeat_same_diopter:
                    focus_step_idx += 1
                    if focus_step_idx >= len(active_focus_cycle):
                        focus_step_idx = 0
                        focus_round_id += 1

                if self._adaptive_enabled and self._acquisition_control is not None:
                    adaptive_center = self._acquisition_control.consume_adaptive_center_diopter()
                    if adaptive_center is not None and unique_diopter_steps:
                        adaptive_steps = select_adaptive_window_steps(
                            unique_diopter_steps,
                            adaptive_center,
                            float(self._adaptive_window_fraction),
                            int(self._adaptive_min_window_points),
                            int(self._adaptive_dispersion_coeff),
                        )
                        if adaptive_steps:
                            active_focus_cycle = build_focus_cycle_from_up_values(self._focus_pattern, adaptive_steps)
                            focus_step_idx = 0
                            focus_round_id += 1
                            logger.info(
                                "Adaptive focusing engaged round %d: center=%.3f, selected_steps=%s",
                                focus_round_id,
                                adaptive_center,
                                ",".join(f"{x:.3f}" for x in adaptive_steps),
                            )
                        else:
                            active_focus_cycle = list(base_focus_cycle)

                current_diopter = None
                current_fd_mm = None
                if lens is not None:
                    current_diopter = float(active_focus_cycle[focus_step_idx % len(active_focus_cycle)])
                    current_fd_mm = (current_diopter - self._start_dpt) * self._scan_ratio

                    if not repeat_same_diopter:
                        diopter_change = current_diopter - previous_diopter
                        settle_time_this_frame = (
                            self._large_jump_settle_sec
                            if diopter_change < self._large_jump_threshold_dpt
                            else self._lens_settle_time_sec
                        )
                        lens.StaticInput.SetAsInput()
                        lens.StaticInput.SetFocalPower(current_diopter)
                        if settle_time_this_frame > 0:
                            time.sleep(settle_time_this_frame)
                        previous_diopter = current_diopter

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
                    if consecutive_timeouts >= 15:
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
                        diopter=current_diopter,
                        focusing_distance_mm=current_fd_mm,
                        focus_round_id=focus_round_id,
                    )
                    frame_idx += 1
                finally:
                    grab_result.Release()
        finally:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if camera.IsOpen():
                camera.Close()
            if ecc1c is not None:
                try:
                    if hasattr(ecc1c, "disconnect") and callable(ecc1c.disconnect):
                        ecc1c.disconnect()
                    elif hasattr(ecc1c, "close") and callable(ecc1c.close):
                        ecc1c.close()
                except Exception:
                    pass


def build_frame_source(
    source: str,
    *,
    input_dir: Optional[str],
    camera_index: int,
    target_fps: float,
    max_frames: Optional[int],
    retrieve_timeout_margin_ms: int = 150,
    adaptive_enabled: bool = False,
    adaptive_window_fraction: Optional[float] = None,
    adaptive_min_window_points: Optional[int] = None,
    adaptive_dispersion_coeff: Optional[int] = None,
    exposure_us: float = 750.0,
    roi_width: Optional[int] = None,
    roi_height: Optional[int] = None,
    etl_control_enabled: bool = True,
    start_dpt: float = -2.0,
    focus_pattern: str = "zigzag",
    scan_height_mm: float = 11.2,
    scan_step_um: float = 350.0,
    scan_ratio: float = 2.25,
    lens_settle_time_sec: float = 0.025,
    large_jump_threshold_dpt: float = -3.0,
    large_jump_settle_sec: float = 0.04,
    lens_controller_port: Optional[str] = "COM5",
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
        acquisition_control=acquisition_control,
    )
