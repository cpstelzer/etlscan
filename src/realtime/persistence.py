from __future__ import annotations

import os
from typing import List

import pandas as pd

from src.core.image_io_operations import save_image

from .session import RealtimeSession


def _crop_filename(session: RealtimeSession, frame_idx: int, particle_idx: int, x: int, y: int, ext: str) -> str:
    return (
        f"{session.session_id}_plate{session.plate}_well{session.well}"
        f"_frame{frame_idx:06d}_particle{particle_idx}_X{x}_Y{y}.{ext}"
    )


def save_accepted_crops(df: pd.DataFrame, session: RealtimeSession, image_format: str = "png", png_compression: int = 3) -> List[str]:
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
        filename = _crop_filename(session, frame, i, x, y, ext)

        # Keep ext controlled by realtime config; writer uses OpenCV if available,
        # otherwise PIL fallback from src.core.image_io_operations.
        saved_path = save_image(crop, session.accepted_dir, filename)
        if saved_path:
            saved_paths.append(saved_path)

    _ = png_compression  # Compression tuning currently only applies when OpenCV is available.
    return saved_paths
