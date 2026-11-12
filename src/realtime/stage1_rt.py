from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.image_io_operations import FlatFieldCorrectionAssets, apply_flat_field_correction
from src.stages.stage_01_initial_detection import (
    analyze_subtracted_image,
    prepare_masked_foreground,
    refine_with_local_otsu,
)
from src.core.image_enhancement import subtract_images, float01_to_uint8

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    detections_df: pd.DataFrame


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
        fg_img_uint8 = float01_to_uint8(apply_flat_field_correction(fg_img_uint8, ffc_assets))
        bg_img_uint8 = float01_to_uint8(apply_flat_field_correction(bg_img_uint8, ffc_assets))

    brightness_cfg = config["config_black_plate_well"]
    enable_brightness_adjustment = bool(config["enable_brightness_adjustment"])
    if enable_brightness_adjustment and not _has_brightness_config_sections(brightness_cfg):
        if frame_index % 60 == 0:
            logger.warning(
                "Realtime Stage-1 brightness adjustment requested but config_black_plate_well is incomplete; "
                "running without brightness matching for this frame."
            )
        enable_brightness_adjustment = False

    source_illumination = str(config["source_image_illumination"]).strip().lower()

    prep = prepare_masked_foreground(
        fg_img_uint8,
        bg_img_uint8,
        enable_brightness_adjustment=enable_brightness_adjustment,
        well_analysis_config=brightness_cfg,
        well_masking_cfg=config["well_masking"],
        system_cfg=config["system_cfg"],
        source_illumination=source_illumination,
    )

    subtracted_img_uint8 = subtract_images(prep["fg_for_sub"], prep["bg_for_sub"])

    if subtracted_img_uint8 is None:
        return Stage1Result(detections_df=pd.DataFrame())

    sub_cfg = config["particle_analysis_subtracted_config"]
    global_cfg = config["global_threshold_config"]
    local_cfg = config["unsubtracted_analysis_config"]

    # The shared FG/BG foreground gate (floor + max-fraction guard) is applied inside
    # analyze_subtracted_image via subtracted_foreground_config; it reports subtraction_failed
    # when a poorly-subtracted frame would flood with false positives.
    sub_analysis = analyze_subtracted_image(
        subtracted_img_uint8,
        well_mask_uint8=prep["subtraction_roi_mask"],
        particle_analysis_config=sub_cfg,
        global_threshold_config=global_cfg,
        sample_id="REALTIME",
        source_filename=f"realtime_frame_{frame_index:06d}.png",
        subtracted_foreground_config=config["subtracted_foreground_config"],
    )
    if sub_analysis.get("subtraction_failed"):
        if frame_index % 30 == 0:
            logger.warning(
                f"Frame {frame_index} skipped: FG/BG subtraction failed (foreground exceeds "
                "max_foreground_fraction; possible bg mismatch/illumination drift)."
            )
        return Stage1Result(detections_df=pd.DataFrame())
    refined = refine_with_local_otsu(
        unsub_prepared_uint8=prep["unsub_prepared"],
        candidate_particles=sub_analysis["particles_df"],
        adaptive_padding_factor=float(local_cfg["adaptive_padding_factor"]),
        analysis_config=local_cfg,
        global_threshold_config=global_cfg,
        source_illumination=source_illumination,
    )

    detections = refined["particles_df"].copy()
    if detections.empty:
        return Stage1Result(detections_df=detections)

    detections["frame"] = frame_index
    detections["timestamp_s"] = timestamp_s
    if "track_id" not in detections.columns:
        detections["track_id"] = [f"rt_{frame_index}_{i}" for i in range(len(detections))]
    return Stage1Result(detections_df=detections)
