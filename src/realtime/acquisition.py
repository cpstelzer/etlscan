from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Iterator, Optional

import cv2
import numpy as np

from src.core.image_io_operations import load_image


@dataclass
class FramePacket:
    frame_index: int
    timestamp_s: float
    image_gray: np.ndarray


class FrameSource:
    def __iter__(self) -> Iterator[FramePacket]:
        raise NotImplementedError


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
            yield FramePacket(frame_index=idx, timestamp_s=time.monotonic(), image_gray=img)
            next_t = start + (idx + 1) * self._target_period_s
            sleep_s = max(0.0, next_t - time.monotonic())
            if sleep_s > 0:
                time.sleep(sleep_s)


class CameraFrameSource(FrameSource):
    def __init__(self, camera_index: int, target_fps: float, max_frames: Optional[int] = None) -> None:
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._max_frames = max_frames

    def __iter__(self) -> Iterator[FramePacket]:
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        try:
            frame_idx = 0
            while True:
                if self._max_frames is not None and frame_idx >= int(self._max_frames):
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                yield FramePacket(frame_index=frame_idx, timestamp_s=time.monotonic(), image_gray=gray)
                frame_idx += 1
        finally:
            cap.release()


def build_frame_source(source: str, *, input_dir: Optional[str], camera_index: int, target_fps: float, max_frames: Optional[int]) -> FrameSource:
    if source == "directory":
        if not input_dir:
            raise ValueError("realtime_mode.acquisition.input_dir is required when source=directory")
        return DirectoryFrameSource(input_dir=input_dir, target_fps=target_fps, max_frames=max_frames)
    return CameraFrameSource(camera_index=camera_index, target_fps=target_fps, max_frames=max_frames)
