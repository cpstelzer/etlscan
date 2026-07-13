"""Regression tests for Stage-1 unsubtracted local-Otsu background handling.

Guards the mid-gray-background flooding bug: each candidate ROI used to be
thresholded into a solid square (circularity == pi/4, solidity 1.0) because the
sub-1.0 Otsu multiplier dropped *below* an elevated (inverted) background floor,
labelling the whole square ROI as foreground. The fix (`_suppress_roi_background`
plus a foreground-fraction guard) pulls the local background floor to ~0 so the
multiplier behaves regardless of absolute background brightness.
"""
import sys
import types

import numpy as np
import pytest

# trackpy is only used by the tracking step (not the detection helpers under test)
# and is absent from the lightweight test env; stub it so the module imports.
sys.modules.setdefault("trackpy", types.ModuleType("trackpy"))

from src.stages.stage_01_initial_detection import (  # noqa: E402
    _analyze_unsubtracted_locally,
    _suppress_roi_background,
)

OTSU_CFG = {
    "thresholding_mode": "otsu",
    "otsu_threshold_multiplier": 0.7,
    "fixed_threshold_value": 70,
}
ANALYSIS_CFG = {
    "min_particle_area": 200,
    "background_floor_percentile": 25,
    "max_roi_foreground_fraction": 0.85,
}
QUARTER_PI = np.pi / 4  # circularity fingerprint of a filled axis-aligned square


def test_suppress_roi_background_zeros_elevated_floor():
    roi = np.full((50, 50), 0.4, dtype=np.float64)
    roi[20:30, 20:30] = 0.9  # bright (post-inversion) particle patch
    out = _suppress_roi_background(roi, 25.0)
    assert out.min() == pytest.approx(0.0, abs=1e-9)     # background pulled to zero
    assert out[25, 25] == pytest.approx(0.5, abs=1e-6)   # particle signal preserved
    assert out.max() <= 1.0


def _mid_gray_frame(with_particle: bool) -> np.ndarray:
    """A mid-gray (~151/255) collimated background, optionally with a dark particle."""
    rng = np.random.default_rng(0)
    img = np.full((600, 600), 151, dtype=np.int16)
    img += rng.normal(0, 3, img.shape).astype(np.int16)  # non-uniform (uniform => empty-mask guard)
    if with_particle:
        yy, xx = np.ogrid[:600, :600]
        disk = (yy - 300) ** 2 + (xx - 300) ** 2 <= 60 ** 2
        img[disk] = 80                                    # dark, compact, in-focus particle
    return np.clip(img, 0, 255).astype(np.uint8)


def _candidate():
    return [{"X": 300, "Y": 300, "bounding_w": 200, "bounding_h": 200}]


def test_pure_midgray_background_does_not_flood_to_square():
    frame = _mid_gray_frame(with_particle=False)
    particles, _, _ = _analyze_unsubtracted_locally(
        frame, _candidate(), 0.2, ANALYSIS_CFG, OTSU_CFG, "collimated"
    )
    # Empty background must not surface as a solid-square flood.
    for p in particles:
        assert abs(p["circularity"] - QUARTER_PI) > 0.05


def test_dark_particle_detected_as_blob_not_square():
    frame = _mid_gray_frame(with_particle=True)
    particles, _, _ = _analyze_unsubtracted_locally(
        frame, _candidate(), 0.2, ANALYSIS_CFG, OTSU_CFG, "collimated"
    )
    assert particles, "expected the dark particle to be detected"
    p = particles[0]
    roi_area = (200 * 1.4) ** 2                           # padded square-ROI area
    assert p["area"] < 0.5 * roi_area                    # not flooded to the ROI square
    assert abs(p["circularity"] - QUARTER_PI) > 0.05     # compact blob, not a square
