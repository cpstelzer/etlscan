from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import skimage.util

from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.particle_operations import normalized_crop_config
from src.stages.stage_02_characterization import crop_particles, segment_particles


@dataclass
class Stage2Result:
    particles_df: pd.DataFrame
    pre_threshold_count: int = 0
    rejected_focus_count: int = 0
    rejected_area_count: int = 0


def run_stage2_realtime(
    fg_gray,
    s1_df: pd.DataFrame,
    frame_index: int,
    stage2_cfg: Dict[str, Any],
) -> Stage2Result:
    if s1_df is None or s1_df.empty:
        return Stage2Result(particles_df=pd.DataFrame())

    working = s1_df.copy()
    working["s1_original_index"] = working.index
    crop_cfg = normalized_crop_config(stage2_cfg.get("crop_config", {"mode": "rotational_rectangle", "rotational_padding": 10}))
    boundary_threshold_params = {
        "thresholding_mode": "otsu",
        "otsu_threshold_multiplier": 0.45,
        "fixed_threshold_value": 70,
    }
    crop_records = crop_particles(
        main_img_color=fg_gray,
        main_img_gray=fg_gray,
        particles_df=working,
        crop_config=crop_cfg,
        sample_id="REALTIME",
        image_filename=f"realtime_frame_{frame_index:06d}.png",
        frame_num=frame_index,
        illumination_mode="darkfield",
        boundary_threshold_params=boundary_threshold_params,
    )
    crops_df = pd.DataFrame(crop_records)
    if crops_df.empty:
        return Stage2Result(particles_df=crops_df)

    segmented_df = segment_particles(crops_df, {"source_image_illumination_type": "darkfield"})
    if segmented_df.empty:
        return Stage2Result(particles_df=segmented_df)

    rows: List[Dict[str, Any]] = []
    min_focus = float(stage2_cfg.get("min_focus_score", 100.0))
    min_area = float(stage2_cfg.get("min_particle_area", 500.0))
    rejected_focus = 0
    rejected_area = 0

    for _, row in segmented_df.iterrows():
        crop_gray = row["crop_gray"]
        raw_mask = row["raw_mask"]
        filled_mask = row["filled_mask"]

        feature_row: Dict[str, Any] = row.to_dict()
        quality = calculate_quality_features(crop_gray, saved_crop_path=None, config={})
        feature_row.update(quality)

        intensity = skimage.util.img_as_float(crop_gray)
        morph = calculate_detailed_body_morphology(raw_mask, filled_mask, intensity)
        feature_row.update({k.replace("_s4", ""): v for k, v in morph.items()})

        if float(feature_row.get("variance_laplacian", 0.0)) <= min_focus:
            rejected_focus += 1
            continue
        if float(feature_row.get("main_area_filled", 0.0)) <= min_area:
            rejected_area += 1
            continue
        rows.append(feature_row)

    return Stage2Result(
        particles_df=pd.DataFrame(rows),
        pre_threshold_count=len(segmented_df),
        rejected_focus_count=rejected_focus,
        rejected_area_count=rejected_area,
    )
