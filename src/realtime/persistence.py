from __future__ import annotations

import os
from typing import List

import pandas as pd

from src.core.image_io_operations import save_image
from src.utils.filename_helpers import build_crop_filename

from .session import RealtimeSession


def save_accepted_crops(df: pd.DataFrame, session: RealtimeSession, image_format: str, png_compression: int) -> List[str]:
    if df is None or df.empty:
        return []

    saved_paths: List[str] = []
    ext = "jpg" if image_format == "jpeg" else image_format

    for i, row in df.reset_index(drop=True).iterrows():
        crop = row.get("crop_gray")
        if crop is None:
            continue

        frame = int(row.get("frame", 0))
        x = int(row.get("X", 0))
        y = int(row.get("Y", 0))
        prefix = f"{session.session_id}_plate{session.plate}_well{session.well}"
        filename = build_crop_filename(prefix, f"particle{i}", frame, x, y, ext=ext, frame_padding=6)

        # Keep ext controlled by realtime config; writer uses OpenCV if available,
        # otherwise PIL fallback from src.core.image_io_operations.
        saved_path = save_image(crop, session.accepted_dir, filename)
        if saved_path:
            saved_paths.append(saved_path)

    _ = png_compression  # Compression tuning currently only applies when OpenCV is available.
    return saved_paths
