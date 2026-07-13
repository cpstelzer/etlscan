from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os

from .config import RealtimeConfig


@dataclass(frozen=True)
class RealtimeSession:
    session_id: str
    plate: int
    well: int
    output_dir: str
    accepted_dir: str
    bg_dir: str


def create_session(config: RealtimeConfig, plate: int, well: int) -> RealtimeSession:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_root_dir, f"{session_id}_plate{plate}_well{well}")
    accepted_dir = os.path.join(output_dir, "accepted_crops")
    bg_dir = os.path.join(output_dir, "bg_images")
    os.makedirs(accepted_dir, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)
    return RealtimeSession(
        session_id=session_id,
        plate=plate,
        well=well,
        output_dir=output_dir,
        accepted_dir=accepted_dir,
        bg_dir=bg_dir,
    )
