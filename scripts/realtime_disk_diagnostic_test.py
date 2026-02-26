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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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
WS_CONFIG_PATH = Path("src/acquire/ws_config.yaml")
PIPELINE_CONFIG_PATH = Path("config/pipeline_config.yml")

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


def run() -> int:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DIAGNOSTIC_OUTPUT_ROOT / run_stamp
    logger = _configure_logging(output_dir)

    logger.info("Starting realtime disk diagnostic run")
    logger.info("FG_IMAGE_PATH=%s", FG_IMAGE_PATH)
    logger.info("BG_IMAGE_PATH=%s", BG_IMAGE_PATH)
    logger.info("WS_CONFIG_PATH=%s", WS_CONFIG_PATH)
    logger.info("PIPELINE_CONFIG_PATH=%s", PIPELINE_CONFIG_PATH)
    logger.info("OUTPUT_DIR=%s", output_dir)

    if not FG_IMAGE_PATH.exists() or not BG_IMAGE_PATH.exists():
        logger.error("FG/BG input images missing. Please create both hardwired files first.")
        return 2

    cfg = load_realtime_config(str(WS_CONFIG_PATH), pipeline_config_path=str(PIPELINE_CONFIG_PATH))
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
    ffc_assets = load_flat_field_assets(str(FG_IMAGE_PATH.parent), mode="gray")
    if ffc_assets is not None:
        logger.info("Flat-field assets found and loaded from %s", FG_IMAGE_PATH.parent)
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
        enable_brightness_adjustment=bool(cfg.stage1_config.get("enable_brightness_adjustment", True)),
        brightness_config=cfg.stage1_config.get("config_black_plate_well", {}),
        well_masking_cfg=cfg.stage1_config.get("well_masking", {}),
        system_cfg=cfg.stage1_config.get("system_cfg", {}),
        source_illumination=str(cfg.stage1_config.get("source_image_illumination", "darkfield")).strip().lower(),
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

    threshold = int(cfg.stage1_config.get("foreground_diff_threshold", 10))
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
            "min_focus_score": cfg.min_focus_score,
            "min_particle_area": cfg.min_particle_area,
            "max_candidates_per_frame": cfg.max_stage2_candidates_per_frame,
            "source_image_illumination": cfg.stage1_config.get("source_image_illumination", "darkfield"),
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

    accepted_overlay = _draw_detection_overlay(fg, s3.accepted_df)
    _save_image(output_dir / "15_stage3_accepted_overlay.png", accepted_overlay, logger)

    # Optional focused well diagnostic using same config sections
    try:
        bp_cfg = cfg.stage1_config.get("config_black_plate_well", {})
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
        "stage1_detections": len(s1.detections_df),
        "stage2_pre_threshold": int(s2.pre_threshold_count),
        "stage2_after_threshold": len(s2.particles_df),
        "stage2_rejected_focus": int(s2.rejected_focus_count),
        "stage2_rejected_area": int(s2.rejected_area_count),
        "stage3_accepted": len(s3.accepted_df),
        "stage3_rejected_artifact": int(s3.rejected_artifact_count),
        "stage3_rejected_boundary": int(s3.rejected_boundary_count),
    }
    pd.Series(summary, name="value").to_csv(output_dir / "19_summary.csv")

    logger.info("Diagnostic run completed successfully")
    logger.info("Artifacts available at: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
