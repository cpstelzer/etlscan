from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _TimedImage:
    image: np.ndarray
    captured_at_s: float


class BackgroundBuffer:
    def __init__(self, bgi_min_age_s: float, bgi_max_age_s: float) -> None:
        self.bgi_min_age_s = bgi_min_age_s
        self.bgi_max_age_s = bgi_max_age_s
        self.bg_image_aged: Optional[_TimedImage] = None
        self.bg_image_new: Optional[_TimedImage] = None

    def update(self, frame: np.ndarray, timestamp_s: float) -> None:
        if self.bg_image_new is None and self.bg_image_aged is None:
            self.bg_image_new = _TimedImage(frame.copy(), timestamp_s)
            return

        if self.bg_image_aged is None and self.bg_image_new is not None:
            if timestamp_s - self.bg_image_new.captured_at_s >= self.bgi_min_age_s:
                self.bg_image_aged = self.bg_image_new
                self.bg_image_new = None
            return

        if self.bg_image_aged is None:
            return

        aged_age = timestamp_s - self.bg_image_aged.captured_at_s
        if aged_age >= self.bgi_max_age_s:
            self.bg_image_aged = None

        if self.bg_image_aged is not None and self.bg_image_new is None:
            preload_age = self.bgi_max_age_s - self.bgi_min_age_s
            if aged_age >= preload_age:
                self.bg_image_new = _TimedImage(frame.copy(), timestamp_s)

        if self.bg_image_new is not None:
            new_age = timestamp_s - self.bg_image_new.captured_at_s
            if new_age >= self.bgi_min_age_s:
                self.bg_image_aged = self.bg_image_new
                self.bg_image_new = None

    def get_valid_aged_background(self, timestamp_s: float) -> Optional[np.ndarray]:
        if self.bg_image_aged is None:
            return None
        if timestamp_s - self.bg_image_aged.captured_at_s > self.bgi_max_age_s:
            self.bg_image_aged = None
            return None
        return self.bg_image_aged.image
