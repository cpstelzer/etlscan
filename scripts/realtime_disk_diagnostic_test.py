"""Disk-based diagnostic orchestrator for realtime stages 1-3.

This script intentionally mirrors the realtime processing flow while sourcing a
single FG/BG image pair from disk (hardwired paths below).

It calls the realtime stage functions directly:
- src.realtime.stage1_rt.run_stage1_realtime
- src.realtime.stage2_rt.run_stage2_realtime
- src.realtime.stage3_rt.run_stage3_realtime

It also emits rich diagnostics and intermediate artifacts to a dedicated output
folder so failures and parameter mismatches are easy to inspect.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import cv2
import numpy as np
import pandas as pd

from src.core.image_enhancement import subtract_images
from src.core.image_io_operations import load_flat_field_assets, apply_flat_field_correction
from src.core.well_analysis import create_well_mask_black_plate, match_well_brightness
from src.realtime.config import load_realtime_config
from src.realtime.stage1_rt import run_stage1_realtime
from src.realtime.stage2_rt import run_stage2_realtime
from src.realtime.stage3_rt import run_stage3_realtime
from src.stages.stage_01_initial_detection import prepare_masked_foreground

# ---------------------------------------------------------------------------
# HARDWIRED INPUT PATHS (edit as needed)
# ---------------------------------------------------------------------------
FG_IMAGE_PATH = Path("data/realtime_diagnostic/fg.png")
BG_IMAGE_PATH = Path("data/realtime_diagnostic/bg.png")
CONFIG_PATH = Path("config/config.yml")

# Dedicated output root for this diagnostic script
DIAGNOSTIC_OUTPUT_ROOT = Path("data/output/realtime_disk_diagnostic")


def _configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "diagnostic.log"

    logger = logging.getLogger("realtime_disk_diagnostic")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _save_image(path: Path, image: np.ndarray, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        logger.warning("Failed to save image: %s", path)
    else:
        logger.debug("Saved image: %s", path)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_detection_overlay(base_gray: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
    overlay = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    if detections is None or detections.empty:
        cv2.putText(overlay, "No detections", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return overlay

    for _, row in detections.iterrows():
        x = int(round(float(row.get("main_X", row.get("X", row.get("x", 0))))))
        y = int(round(float(row.get("main_Y", row.get("Y", row.get("y", 0))))))
        area = float(row.get("main_area_filled", row.get("area", 0.0)))
        radius = int(max(6, min(60, np.sqrt(max(area, 1.0) / np.pi))))
        cv2.circle(overlay, (x, y), radius, (0, 255, 0), 2)

    cv2.putText(
        overlay,
        f"detections={len(detections)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    return overlay


def _save_category_crops(category_name: str, category_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    category_dir = output_dir / "category_crops" / category_name
    category_dir.mkdir(parents=True, exist_ok=True)
    if category_df is None or category_df.empty:
        logger.info("Category '%s': no crops to save", category_name)
        return

    saved = 0
    for i, (_, row) in enumerate(category_df.iterrows(), start=1):
        crop = row.get("crop_gray")
        if isinstance(crop, np.ndarray):
            _save_image(category_dir / f"crop_{i:03d}.png", crop, logger)
            saved += 1
    logger.info("Category '%s': saved %d crops", category_name, saved)


def _row_key(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype=str)
    if "crop_filename" in df.columns:
        return df["crop_filename"].astype(str)
    cols = [c for c in ["frame", "X", "Y", "main_X", "main_Y", "s1_original_index"] if c in df.columns]
    if not cols:
        return pd.Series([str(i) for i in df.index], index=df.index)
    return df[cols].astype(str).agg("|".join, axis=1)


def _diff_df(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty:
        return pd.DataFrame()
    if b is None or b.empty:
        return a.copy()
    b_keys = set(_row_key(b).tolist())
    a_keys = _row_key(a)
    return a.loc[~a_keys.isin(b_keys)].copy()


def run() -> int:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DIAGNOSTIC_OUTPUT_ROOT / run_stamp
    logger = _configure_logging(output_dir)

    logger.info("Starting realtime disk diagnostic run")
    logger.info("FG_IMAGE_PATH=%s", FG_IMAGE_PATH)
    logger.info("BG_IMAGE_PATH=%s", BG_IMAGE_PATH)
    logger.info("CONFIG_PATH=%s", CONFIG_PATH)
    logger.info("OUTPUT_DIR=%s", output_dir)

    if not FG_IMAGE_PATH.exists() or not BG_IMAGE_PATH.exists():
        logger.error("FG/BG input images missing. Please create both hardwired files first.")
        return 2

    cfg = load_realtime_config(str(CONFIG_PATH))
    logger.info("Loaded realtime config successfully")

    fg = cv2.imread(str(FG_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread(str(BG_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if fg is None or bg is None:
        logger.error("Failed to load one or both images as grayscale")
        return 3

    logger.info("Loaded FG/BG images (shape FG=%s BG=%s)", fg.shape, bg.shape)
    _save_image(output_dir / "01_fg_raw.png", fg, logger)
    _save_image(output_dir / "02_bg_raw.png", bg, logger)

    # Flat-field correction using the same logic as realtime stage 1
    ffc_dir = Path(cfg.flat_field_dir) if cfg.flat_field_dir else FG_IMAGE_PATH.parent
    ffc_assets = load_flat_field_assets(str(ffc_dir), mode="gray")
    if ffc_assets is not None:
        logger.info("Flat-field assets found and loaded from %s", ffc_dir)
        fg_ffc = _to_uint8(apply_flat_field_correction(fg, ffc_assets))
        bg_ffc = _to_uint8(apply_flat_field_correction(bg, ffc_assets))
    else:
        logger.info("No flat-field assets found; continuing without correction")
        fg_ffc, bg_ffc = fg.copy(), bg.copy()

    _save_image(output_dir / "03_fg_ffc.png", fg_ffc, logger)
    _save_image(output_dir / "04_bg_ffc.png", bg_ffc, logger)

    # Diagnostic: explicit preparation path before stage-1 wrapper
    prep = prepare_masked_foreground(
        fg_ffc,
        bg_ffc,
        enable_brightness_adjustment=bool(cfg.stage1_config["enable_brightness_adjustment"]),
        well_analysis_config=cfg.stage1_config["config_black_plate_well"],
        well_masking_cfg=cfg.stage1_config["well_masking"],
        system_cfg=cfg.stage1_config["system_cfg"],
        source_illumination=str(cfg.stage1_config["source_image_illumination"]).strip().lower(),
    )
    logger.info("Computed masked FG/BG preparation")

    _save_image(output_dir / "05_fg_for_sub.png", prep["fg_for_sub"], logger)
    _save_image(output_dir / "06_bg_for_sub.png", prep["bg_for_sub"], logger)
    _save_image(output_dir / "07_unsub_prepared.png", prep["unsub_prepared"], logger)

    if prep.get("well_mask") is not None:
        _save_image(output_dir / "08_well_mask.png", prep["well_mask"], logger)

    sub_img = subtract_images(prep["fg_for_sub"], prep["bg_for_sub"])
    if sub_img is None:
        logger.error("subtract_images returned None; aborting")
        return 4
    _save_image(output_dir / "09_subtracted.png", sub_img, logger)

    threshold = int(cfg.stage1_config["foreground_diff_threshold"])
    _, motion_mask = cv2.threshold(sub_img, threshold, 255, cv2.THRESH_BINARY)
    _save_image(output_dir / "10_motion_mask.png", motion_mask, logger)

    # Stage 1 (realtime function)
    logger.info("Running realtime Stage 1")
    s1 = run_stage1_realtime(
        fg,
        bg,
        frame_index=1,
        timestamp_s=0.0,
        config=cfg.stage1_config,
        ffc_assets=ffc_assets,
    )
    logger.info("Stage 1 completed: detections=%d", len(s1.detections_df))
    s1.detections_df.to_csv(output_dir / "11_stage1_detections.csv", index=False)

    s1_overlay = _draw_detection_overlay(fg, s1.detections_df)
    _save_image(output_dir / "12_stage1_detections_overlay.png", s1_overlay, logger)

    # Stage 2 (realtime function)
    logger.info("Running realtime Stage 2")
    s2 = run_stage2_realtime(
        fg,
        s1.detections_df,
        frame_index=1,
        stage2_cfg={
            "crop_config": cfg.stage2_crop_config,
            "segmentation_params": cfg.stage2_segmentation_params,
            "min_focus_score": cfg.min_focus_score,
            "min_particle_area": cfg.min_particle_area,
            "max_candidates_per_frame": cfg.max_stage2_candidates_per_frame,
            "source_image_illumination": cfg.stage1_config["source_image_illumination"],
        },
    )
    logger.info(
        "Stage 2 completed: accepted=%d pre_threshold=%d rejected_focus=%d rejected_area=%d",
        len(s2.particles_df),
        s2.pre_threshold_count,
        s2.rejected_focus_count,
        s2.rejected_area_count,
    )
    s2.particles_df.to_csv(output_dir / "13_stage2_particles.csv", index=False)

    # Save selected Stage-2 image diagnostics where available
    max_save = 50
    for i, (_, row) in enumerate(s2.particles_df.head(max_save).iterrows(), start=1):
        if isinstance(row.get("crop_gray"), np.ndarray):
            _save_image(output_dir / "stage2_crops" / f"crop_{i:03d}.png", row["crop_gray"], logger)
        if isinstance(row.get("raw_mask"), np.ndarray):
            _save_image(output_dir / "stage2_raw_masks" / f"raw_mask_{i:03d}.png", (row["raw_mask"] > 0).astype(np.uint8) * 255, logger)
        if isinstance(row.get("filled_mask"), np.ndarray):
            _save_image(output_dir / "stage2_filled_masks" / f"filled_mask_{i:03d}.png", (row["filled_mask"] > 0).astype(np.uint8) * 255, logger)

    # Stage 3 (realtime function)
    logger.info("Running realtime Stage 3")
    s3 = run_stage3_realtime(
        s2.particles_df,
        {
            "enable_artifact_filter": cfg.stage3_enable_artifact_filter,
            "enable_boundary_filter": cfg.stage3_enable_boundary_filter,
        },
    )
    logger.info(
        "Stage 3 completed: accepted=%d rejected_artifact=%d rejected_boundary=%d",
        len(s3.accepted_df),
        s3.rejected_artifact_count,
        s3.rejected_boundary_count,
    )
    s3.accepted_df.to_csv(output_dir / "14_stage3_accepted.csv", index=False)

    # Build/save category crops while preserving realtime behavior.
    # We only use additional Stage-2 calls with altered thresholds to recover
    # category membership, but still through the same realtime functions.
    stage2_base_cfg = {
        "crop_config": cfg.stage2_crop_config,
        "segmentation_params": cfg.stage2_segmentation_params,
        "min_focus_score": cfg.min_focus_score,
        "min_particle_area": cfg.min_particle_area,
        "max_candidates_per_frame": cfg.max_stage2_candidates_per_frame,
        "source_image_illumination": cfg.stage1_config.get("source_image_illumination", "darkfield"),
    }

    s2_all = run_stage2_realtime(fg, s1.detections_df, frame_index=1, stage2_cfg={**stage2_base_cfg, "min_focus_score": -1.0, "min_particle_area": -1.0})
    s2_focus_only = run_stage2_realtime(fg, s1.detections_df, frame_index=1, stage2_cfg={**stage2_base_cfg, "min_particle_area": -1.0})

    post_artifact = run_stage3_realtime(
        s2.particles_df,
        {
            "enable_artifact_filter": cfg.stage3_enable_artifact_filter,
            "enable_boundary_filter": False,
        },
    ).accepted_df

    diagnostic_categories = {
        "stage1_detections": s2_all.particles_df,
        "stage2_pre_threshold": s2_all.particles_df,
        "stage2_after_threshold": s2.particles_df,
        "stage2_rejected_focus": _diff_df(s2_all.particles_df, s2_focus_only.particles_df),
        "stage2_rejected_area": _diff_df(s2_focus_only.particles_df, s2.particles_df),
        "stage3_accepted": s3.accepted_df,
        "stage3_rejected_artifact": _diff_df(s2.particles_df, post_artifact),
        "stage3_rejected_boundary": _diff_df(post_artifact, s3.accepted_df),
    }
    for category_name, category_df in diagnostic_categories.items():
        _save_category_crops(category_name, category_df, output_dir, logger)

    accepted_overlay = _draw_detection_overlay(fg, s3.accepted_df)
    _save_image(output_dir / "15_stage3_accepted_overlay.png", accepted_overlay, logger)

    # Optional focused well diagnostic using same config sections
    try:
        bp_cfg = cfg.stage1_config["config_black_plate_well"]
        if bp_cfg:
            well_mask, edges = create_well_mask_black_plate(fg, bp_cfg)
            if well_mask is not None:
                _save_image(output_dir / "16_black_plate_well_mask.png", well_mask, logger)
            if edges is not None:
                _save_image(output_dir / "17_black_plate_edges.png", edges, logger)
            _, bg_matched = match_well_brightness(fg, bg, bp_cfg)
            if bg_matched is not None:
                _save_image(output_dir / "18_bg_brightness_matched.png", bg_matched, logger)
    except Exception as exc:  # diagnostics should not kill the core flow
        logger.warning("Optional black-plate diagnostics failed: %s", exc)

    summary = {
        "stage1_detections": len(diagnostic_categories["stage1_detections"]),
        "stage2_pre_threshold": len(diagnostic_categories["stage2_pre_threshold"]),
        "stage2_after_threshold": len(diagnostic_categories["stage2_after_threshold"]),
        "stage2_rejected_focus": len(diagnostic_categories["stage2_rejected_focus"]),
        "stage2_rejected_area": len(diagnostic_categories["stage2_rejected_area"]),
        "stage3_accepted": len(diagnostic_categories["stage3_accepted"]),
        "stage3_rejected_artifact": len(diagnostic_categories["stage3_rejected_artifact"]),
        "stage3_rejected_boundary": len(diagnostic_categories["stage3_rejected_boundary"]),
        "stage2_after_threshold_runtime": len(s2.particles_df),
        "stage2_rejected_focus_runtime": int(s2.rejected_focus_count),
        "stage2_rejected_area_runtime": int(s2.rejected_area_count),
        "stage3_accepted_runtime": len(s3.accepted_df),
        "stage3_rejected_artifact_runtime": int(s3.rejected_artifact_count),
        "stage3_rejected_boundary_runtime": int(s3.rejected_boundary_count),
    }
    pd.Series(summary, name="value").to_csv(output_dir / "19_summary.csv")

    logger.info("Diagnostic run completed successfully")
    logger.info("Artifacts available at: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
