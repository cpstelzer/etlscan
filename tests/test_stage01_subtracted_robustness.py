"""Regression tests for Stage-1 background-subtracted detection robustness.

Guards the false-positive flood where global Otsu on the near-zero subtracted image
collapses into the noise floor when a FG/BG pair subtracts poorly, emitting hundreds
of noise-floor "particles" across the whole well. The shared gate
(`assess_subtracted_foreground`, used by both the batch and realtime pipelines) enforces
a fixed foreground-difference floor plus a max-foreground-fraction guard that discards a
frame whose subtraction failed.
"""
import sys
import types

import cv2
import numpy as np
import pytest

# trackpy is only used by the tracking step (absent from the lightweight test env).
# Use the real module when installed; only stub it when missing, so we never shadow
# a real install for tests that do exercise tracking (test isolation).
try:
    import trackpy  # noqa: F401,E402
except ImportError:
    sys.modules.setdefault("trackpy", types.ModuleType("trackpy"))

from src.core.segmentation import assess_subtracted_foreground  # noqa: E402
from src.stages.stage_01_initial_detection import analyze_subtracted_image  # noqa: E402

OTSU_CFG = {"thresholding_mode": "otsu", "otsu_threshold_multiplier": 0.7, "fixed_threshold_value": 70}
PARTICLE_CFG = {
    "min_microorganism_area": 500, "microorganism_blur_kernel": 5,
    "apply_convolution": False, "convolution_kernel": None,
}
SUBFG_CFG = {"foreground_diff_threshold": 10, "max_foreground_fraction": 0.08}


def _analyze(img):
    return analyze_subtracted_image(
        img, None, PARTICLE_CFG, OTSU_CFG, "SAMPLE", "frame.png",
        subtracted_foreground_config=SUBFG_CFG,
    )


def test_assess_subtracted_foreground_verdict():
    # Shared helper: 30% of the area above the floor => failed; a sparse blob => usable.
    flooded = np.zeros((100, 100), dtype=np.uint8)
    flooded[:, :30] = 15
    mask, fraction, failed = assess_subtracted_foreground(flooded, 10, 0.08, area_px=flooded.size)
    assert failed is True and fraction == pytest.approx(0.30)
    assert mask.dtype == bool and mask[:, :30].all() and not mask[:, 30:].any()

    sparse = np.zeros((100, 100), dtype=np.uint8)
    sparse[:5, :5] = 15
    _, fraction2, failed2 = assess_subtracted_foreground(sparse, 10, 0.08, area_px=sparse.size)
    assert failed2 is False and fraction2 == pytest.approx(0.0025)


def test_noise_floor_flood_yields_no_candidates():
    # Near-zero subtracted image, all sub-floor noise (0..8), no real signal. Bare Otsu
    # would split this noise and flood; the fixed floor rejects every pixel.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 9, size=(1000, 1000), dtype=np.uint8)
    result = _analyze(img)
    assert result["particles"] == []
    assert result["subtraction_failed"] is False  # nothing above floor, but not a "failure"


def test_real_bright_particles_survive():
    # Clean background (0) with two compact bright blobs well above the floor.
    img = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.circle(img, (300, 300), 20, 40, -1)
    cv2.circle(img, (700, 600), 20, 40, -1)
    result = _analyze(img)
    assert len(result["particles"]) == 2
    assert result["subtraction_failed"] is False


def test_guard_rejects_failed_subtraction():
    # A failed subtraction leaves a large region above the floor (~30% of the area).
    # The max-foreground-fraction guard must discard the whole frame.
    img = np.zeros((1000, 1000), dtype=np.uint8)
    img[:, :300] = 15
    result = _analyze(img)
    assert result["particles"] == []
    assert result["subtraction_failed"] is True
