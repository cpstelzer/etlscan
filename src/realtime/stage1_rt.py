from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.image_enhancement import subtract_images
from src.core.image_io_operations import FlatFieldCorrectionAssets, apply_flat_field_correction
from src.stages.stage_01_initial_detection import analyze_subtracted_image, refine_with_local_otsu


@dataclass
class Stage1Result:
    detections_df: pd.DataFrame


def _as_uint8(image_float01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image_float01, 0.0, 1.0) * 255.0).astype(np.uint8)


def run_stage1_realtime(
    fg_img_uint8: np.ndarray,
    bg_img_uint8: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    config: Dict[str, Any],
    ffc_assets: Optional[FlatFieldCorrectionAssets] = None,
) -> Stage1Result:
    if ffc_assets is not None:
        fg_img_uint8 = _as_uint8(apply_flat_field_correction(fg_img_uint8, ffc_assets))
        bg_img_uint8 = _as_uint8(apply_flat_field_correction(bg_img_uint8, ffc_assets))

    subtracted_img_uint8 = subtract_images(fg_img_uint8, bg_img_uint8)
    sub_cfg = config.get("particle_analysis_subtracted_config", {})
    global_cfg = config.get("global_threshold_config", {})
    local_cfg = config.get("unsubtracted_analysis_config", {})

    sub_analysis = analyze_subtracted_image(
        subtracted_img_uint8,
        well_mask_uint8=None,
        particle_analysis_config=sub_cfg,
        global_threshold_config=global_cfg,
        sample_id="REALTIME",
        source_filename=f"realtime_frame_{frame_index:06d}.png",
    )
    refined = refine_with_local_otsu(
        unsub_prepared_uint8=fg_img_uint8,
        candidate_particles=sub_analysis["particles_df"],
        roi_crop_size=int(config.get("local_otsu_roi_size", 150)),
        analysis_config=local_cfg,
        global_threshold_config=global_cfg,
        source_illumination=str(config.get("source_image_illumination", "darkfield")),
    )

    detections = refined["particles_df"].copy()
    if detections.empty:
        return Stage1Result(detections_df=detections)

    detections["frame"] = frame_index
    detections["timestamp_s"] = timestamp_s
    if "track_id" not in detections.columns:
        detections["track_id"] = [f"rt_{frame_index}_{i}" for i in range(len(detections))]
    return Stage1Result(detections_df=detections)
