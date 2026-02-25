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
    """Realtime camera source using the same pypylon grab method as src/acquire."""

    def __init__(
        self,
        target_fps: float,
        max_frames: Optional[int] = None,
        retrieve_timeout_margin_ms: int = 150,
    ) -> None:
        self._target_fps = target_fps
        self._max_frames = max_frames
        self._retrieve_timeout_margin_ms = retrieve_timeout_margin_ms

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            from pypylon import pylon
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pypylon is required for realtime camera acquisition") from exc

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
            while camera.IsGrabbing():
                if self._max_frames is not None and frame_idx >= int(self._max_frames):
                    break

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
                        diopter=None,
                        focusing_distance_mm=None,
                    )
                    frame_idx += 1
                finally:
                    grab_result.Release()
        finally:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if camera.IsOpen():
                camera.Close()


def build_frame_source(
    source: str,
    *,
    input_dir: Optional[str],
    camera_index: int,
    target_fps: float,
    max_frames: Optional[int],
    retrieve_timeout_margin_ms: int = 150,
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
    )
