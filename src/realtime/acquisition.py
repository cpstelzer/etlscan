from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Iterator, Optional, Tuple

import numpy as np

from src.core.image_io_operations import load_image


@dataclass
class FramePacket:
    frame_index: int
    timestamp_s: float
    image_gray: np.ndarray
    diopter: Optional[float] = None
    focusing_distance_mm: Optional[float] = None


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


def _build_focus_cycle_pattern(
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

    if pattern.strip().lower() == "sawtooth":
        return diopter_values_up

    diopter_values_down: list[float] = []
    if len(diopter_values_up) > 1:
        diopter_values_down = diopter_values_up[-2:0:-1]
    return diopter_values_up + diopter_values_down


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


class CameraFrameSource(FrameSource):
    """Realtime camera source using pypylon and explicit ETL command-per-frame stepping."""

    def __init__(
        self,
        target_fps: float,
        max_frames: Optional[int] = None,
        retrieve_timeout_margin_ms: int = 150,
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
    ) -> None:
        self._target_fps = target_fps
        self._max_frames = max_frames
        self._retrieve_timeout_margin_ms = retrieve_timeout_margin_ms
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

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            from pypylon import pylon
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pypylon is required for realtime camera acquisition") from exc

        lens = None
        ecc1c = None
        focus_cycle = [round(self._start_dpt, 3)]
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

            focus_cycle = _build_focus_cycle_pattern(
                start_dpt=self._start_dpt,
                pattern=self._focus_pattern,
                scan_height_mm=self._scan_height_mm,
                scan_step_um=self._scan_step_um,
                scan_ratio=self._scan_ratio,
            )
            if not focus_cycle:
                focus_cycle = [round(self._start_dpt, 3)]

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
            if hasattr(camera, "PixelFormat"):
                available_formats = list(camera.PixelFormat.Symbolics)
                if "Mono8" in available_formats:
                    camera.PixelFormat.SetValue("Mono8")
                elif "Mono12" in available_formats:
                    camera.PixelFormat.SetValue("Mono12")

            if hasattr(camera, "AcquisitionFrameRateEnable"):
                camera.AcquisitionFrameRateEnable.SetValue(True)
            if hasattr(camera, "AcquisitionFrameRate"):
                camera.AcquisitionFrameRate.SetValue(float(self._target_fps))

            retrieve_timeout = int(1000 / max(self._target_fps, 1.0) + self._retrieve_timeout_margin_ms)
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

            frame_idx = 0
            previous_diopter = self._start_dpt
            while camera.IsGrabbing():
                if self._max_frames is not None and frame_idx >= int(self._max_frames):
                    break

                current_diopter = None
                current_fd_mm = None
                if lens is not None:
                    current_diopter = float(focus_cycle[frame_idx % len(focus_cycle)])
                    current_fd_mm = (current_diopter - self._start_dpt) * self._scan_ratio

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

                grab_result = camera.RetrieveResult(retrieve_timeout, pylon.TimeoutHandling_ThrowException)
                try:
                    if not grab_result.GrabSucceeded():
                        continue
                    raw_frame = grab_result.Array
                    if raw_frame is None:
                        continue
                    if raw_frame.ndim == 3:
                        image_gray = np.mean(raw_frame, axis=2).astype(np.uint8)
                    else:
                        image_gray = raw_frame
                    yield FramePacket(
                        frame_index=frame_idx,
                        timestamp_s=time.monotonic(),
                        image_gray=image_gray,
                        diopter=current_diopter,
                        focusing_distance_mm=current_fd_mm,
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
    )
