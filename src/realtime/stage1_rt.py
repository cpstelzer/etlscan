from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from src.core.image_io_operations import FlatFieldCorrectionAssets, apply_flat_field_correction
from src.stages.stage_01_initial_detection import (
    analyze_subtracted_image,
    prepare_masked_foreground,
    refine_with_local_otsu,
)
from src.core.image_enhancement import subtract_images

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    detections_df: pd.DataFrame
    subtracted_img_uint8: Optional[np.ndarray] = None


def _as_uint8(image_float01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image_float01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _has_brightness_config_sections(cfg: Dict[str, Any]) -> bool:
    return all(
        section in cfg and isinstance(cfg.get(section), dict) and bool(cfg.get(section))
        for section in ("well_identification", "brightness_peak_calculation", "brightness_matching")
    )


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

    brightness_cfg = config.get("config_black_plate_well", {})
    enable_brightness_adjustment = bool(config.get("enable_brightness_adjustment", True))
    if enable_brightness_adjustment and not _has_brightness_config_sections(brightness_cfg):
        if frame_index % 60 == 0:
            logger.warning(
                "Realtime Stage-1 brightness adjustment requested but config_black_plate_well is incomplete; "
                "running without brightness matching for this frame."
            )
        enable_brightness_adjustment = False

    prep = prepare_masked_foreground(
        fg_img_uint8,
        bg_img_uint8,
        enable_brightness_adjustment=enable_brightness_adjustment,
        brightness_config=brightness_cfg,
        well_masking_cfg=config.get("well_masking", {}),
        system_cfg=config.get("system_cfg", {}),
        source_illumination=str(config.get("source_image_illumination", "darkfield")).strip().lower(),
    )

    subtracted_img_uint8 = subtract_images(prep["fg_for_sub"], prep["bg_for_sub"])

    if subtracted_img_uint8 is None:
        return Stage1Result(detections_df=pd.DataFrame(), subtracted_img_uint8=None)

    foreground_diff_threshold = int(config.get("foreground_diff_threshold", 10))
    max_foreground_fraction = float(config.get("max_foreground_fraction", 0.08))
    _, motion_mask = cv2.threshold(subtracted_img_uint8, foreground_diff_threshold, 255, cv2.THRESH_BINARY)
    foreground_fraction = float(np.count_nonzero(motion_mask)) / float(motion_mask.size) if motion_mask.size else 0.0

    if foreground_fraction > max_foreground_fraction:
        if frame_index % 30 == 0:
            logger.warning(
                "Frame %d skipped: foreground fraction %.3f exceeds max_foreground_fraction %.3f (possible bg mismatch/illumination drift).",
                frame_index,
                foreground_fraction,
                max_foreground_fraction,
            )
        return Stage1Result(detections_df=pd.DataFrame(), subtracted_img_uint8=subtracted_img_uint8)

    sub_cfg = config.get("particle_analysis_subtracted_config", {})
    global_cfg = config.get("global_threshold_config", {})
    local_cfg = config.get("unsubtracted_analysis_config", {})

    sub_analysis = analyze_subtracted_image(
        subtracted_img_uint8,
        well_mask_uint8=prep.get("well_mask"),
        particle_analysis_config=sub_cfg,
        global_threshold_config=global_cfg,
        sample_id="REALTIME",
        source_filename=f"realtime_frame_{frame_index:06d}.png",
    )
    refined = refine_with_local_otsu(
        unsub_prepared_uint8=prep["unsub_prepared"],
        candidate_particles=sub_analysis["particles_df"],
        adaptive_padding_factor=float(local_cfg.get("adaptive_padding_factor", 0.2)),
        analysis_config=local_cfg,
        global_threshold_config=global_cfg,
        source_illumination=str(config.get("source_image_illumination", "darkfield")).strip().lower(),
    )

    detections = refined["particles_df"].copy()
    if detections.empty:
        return Stage1Result(detections_df=detections, subtracted_img_uint8=subtracted_img_uint8)

    detections["frame"] = frame_index
    detections["timestamp_s"] = timestamp_s
    if "track_id" not in detections.columns:
        detections["track_id"] = [f"rt_{frame_index}_{i}" for i in range(len(detections))]
    return Stage1Result(detections_df=detections, subtracted_img_uint8=subtracted_img_uint8)
