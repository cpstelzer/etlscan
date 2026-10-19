"""Tests for src/core/well_analysis.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

from src.core.well_analysis import (
    create_well_mask_black_plate,
    create_well_mask_with_fallback,
    create_full_frame_bright_mask,
    calculate_well_brightness_peak,
    match_low_frequency_illumination,
    match_well_brightness,
)
from src.utils.common_exceptions import ImageProcessingError


@pytest.fixture()
def well_config():
    return {
        "low_threshold_value": 10,
        "min_expected_diameter_px": 100,
        "max_expected_diameter_px": 190,
        "min_fill_ratio_to_enclosing_circle": 0.80,
        "center_offset_fraction_threshold": 0.15,
    }


@pytest.fixture()
def full_analysis_config():
    return {
        "well_identification": {
            "low_threshold_value": 10,
            "min_expected_diameter_px": 100,
            "max_expected_diameter_px": 190,
            "min_fill_ratio_to_enclosing_circle": 0.80,
            "center_offset_fraction_threshold": 0.1,
        },
        "brightness_peak_calculation": {
            "min_intensity_for_peak_search": 75,
            "percentile_for_peak_proxy": 75,
        },
        "brightness_matching": {
            "max_peak_difference_allowed": 10,
            "adjustment_method": "global_shift",
            "target_peak_offset_after_shift": 3,
        },
    }


@pytest.fixture()
def fg_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, 180, -1)
    cv2.circle(img, (80, 80), 10, 30, -1)
    return img


@pytest.fixture()
def bg_image_brighter():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, 210, -1)
    return img


@pytest.fixture()
def bg_image_similar():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, 183, -1)
    return img


def test_create_well_mask_returns_mask_and_props(fg_image, well_config):
    mask, props = create_well_mask_black_plate(fg_image, well_config)
    assert mask is not None
    assert props is not None
    assert mask.shape == fg_image.shape
    assert "center_x" in props
    assert "radius_px" in props


def test_create_well_mask_none_input(well_config):
    mask, props = create_well_mask_black_plate(None, well_config)
    assert mask is None
    assert props is None


def test_create_well_mask_rejects_non_2d(well_config):
    img_3d = np.zeros((100, 100, 3), dtype=np.uint8)
    mask, props = create_well_mask_black_plate(img_3d, well_config)
    assert mask is None


def test_calculate_well_brightness_peak_returns_float(fg_image, well_config):
    mask, _ = create_well_mask_black_plate(fg_image, well_config)
    assert mask is not None
    peak = calculate_well_brightness_peak(fg_image, mask)
    assert peak is not None
    assert isinstance(peak, float)
    assert peak > 0


def test_calculate_well_brightness_peak_none_inputs():
    img = np.zeros((100, 100), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    assert calculate_well_brightness_peak(None, mask) is None
    assert calculate_well_brightness_peak(img, None) is None


def test_match_well_brightness_brighter_bg_is_adjusted(
    fg_image, bg_image_brighter, full_analysis_config
):
    adj_fg, adj_bg = match_well_brightness(fg_image, bg_image_brighter, full_analysis_config)
    # The returned arrays should be valid uint8 images of the same shape.
    assert adj_fg.shape == fg_image.shape
    assert adj_bg.shape == bg_image_brighter.shape
    assert adj_fg.dtype == np.uint8
    assert adj_bg.dtype == np.uint8


def test_match_well_brightness_similar_bg_unchanged(
    fg_image, bg_image_similar, full_analysis_config
):
    adj_fg, adj_bg = match_well_brightness(fg_image, bg_image_similar, full_analysis_config)
    # Peak difference is within tolerance; background should not change.
    assert np.array_equal(adj_bg, bg_image_similar)


def test_match_well_brightness_missing_config_returns_originals(fg_image, bg_image_brighter):
    adj_fg, adj_bg = match_well_brightness(fg_image, bg_image_brighter, {})
    assert np.array_equal(adj_fg, fg_image)
    assert np.array_equal(adj_bg, bg_image_brighter)


# --- Full-frame fallback (no discrete well present) ---------------------------

@pytest.fixture()
def full_frame_fg():
    """Bright sample that fills the whole frame (no black plate border)."""
    img = np.full((200, 200), 150, dtype=np.uint8)
    cv2.circle(img, (80, 80), 8, 30, -1)  # a dark feature
    return img


@pytest.fixture()
def full_frame_bg_brighter():
    return np.full((200, 200), 210, dtype=np.uint8)


def test_create_full_frame_bright_mask_thresholds_dark_pixels():
    img = np.full((50, 50), 100, dtype=np.uint8)
    img[0:10, 0:10] = 2  # below low_threshold_value
    mask = create_full_frame_bright_mask(img, low_threshold_value=10)
    assert mask is not None
    assert mask.shape == img.shape
    assert mask[25, 25] == 255          # bright region kept
    assert mask[5, 5] == 0              # dark corner excluded


def test_create_full_frame_bright_mask_invalid_input():
    assert create_full_frame_bright_mask(None) is None
    assert create_full_frame_bright_mask(np.zeros((4, 4, 3), dtype=np.uint8)) is None


def test_match_well_brightness_fallback_normalizes_when_no_well(
    full_frame_fg, full_frame_bg_brighter, full_analysis_config
):
    # max_expected_diameter_px (190) is smaller than the full-frame enclosing
    # circle, so circle detection fails and the fallback should engage.
    adj_fg, adj_bg = match_well_brightness(
        full_frame_fg, full_frame_bg_brighter, full_analysis_config
    )
    # BG (210) is brighter than FG (150) beyond tolerance, so it must be dimmed.
    assert not np.array_equal(adj_bg, full_frame_bg_brighter)
    assert adj_bg.mean() < full_frame_bg_brighter.mean()
    assert adj_bg.dtype == np.uint8


def test_match_well_brightness_fallback_can_be_disabled(
    full_frame_fg, full_frame_bg_brighter, full_analysis_config
):
    full_analysis_config["well_identification"]["fallback_full_frame_on_failure"] = False
    adj_fg, adj_bg = match_well_brightness(
        full_frame_fg, full_frame_bg_brighter, full_analysis_config
    )
    # Fallback disabled -> original images returned unchanged.
    assert np.array_equal(adj_fg, full_frame_fg)
    assert np.array_equal(adj_bg, full_frame_bg_brighter)


# --- Shared well mask with fallback ---------------------------------------------

def test_create_well_mask_with_fallback_prefers_detected_well(fg_image, well_config):
    expected_mask, _ = create_well_mask_black_plate(fg_image, well_config)
    assert np.array_equal(create_well_mask_with_fallback(fg_image, well_config), expected_mask)


def test_create_well_mask_with_fallback_uses_full_frame_mask(full_frame_fg, full_analysis_config):
    well_identification = full_analysis_config["well_identification"]
    mask = create_well_mask_with_fallback(full_frame_fg, well_identification)
    assert np.array_equal(mask, create_full_frame_bright_mask(full_frame_fg, well_identification["low_threshold_value"]))


def test_create_well_mask_with_fallback_disabled_returns_none(full_frame_fg, full_analysis_config):
    well_identification = dict(full_analysis_config["well_identification"], fallback_full_frame_on_failure=False)
    assert create_well_mask_with_fallback(full_frame_fg, well_identification) is None


# --- Low-frequency illumination match (external background) --------------------

def _gradient_pair_with_blob():
    """BG flat at 180; FG darkened by a 0..25 vertical gradient plus one small dark organism."""
    bg = np.full((400, 400), 180, dtype=np.uint8)
    gradient = np.linspace(0, 25, 400, dtype=np.float32)[:, None]
    fg = np.clip(bg.astype(np.float32) - gradient, 0, 255).astype(np.uint8)
    cv2.circle(fg, (200, 200), 8, 60, -1)
    return fg, bg


def test_match_low_frequency_illumination_removes_gradient_keeps_organism():
    fg, bg = _gradient_pair_with_blob()
    # Collimated subtraction (inverted FG minus inverted BG) == BG - FG, clipped at 0.
    raw_residual = cv2.subtract(bg, fg)
    assert (raw_residual >= 10).mean() > 0.4  # the gradient alone would flood a >=10 gate

    adjusted_bg = match_low_frequency_illumination(fg, bg, downsample_factor=4, median_kernel_px=15)
    residual = cv2.subtract(adjusted_bg, fg)

    away_from_organism = np.ones_like(residual, dtype=bool)
    away_from_organism[180:221, 180:221] = False
    assert residual[away_from_organism].max() < 10
    assert residual[200, 200] > 80


def test_match_low_frequency_illumination_rejects_invalid_input():
    fg, bg = _gradient_pair_with_blob()
    with pytest.raises(ImageProcessingError):
        match_low_frequency_illumination(fg, bg[:-1], downsample_factor=4, median_kernel_px=15)
    with pytest.raises(ImageProcessingError):
        match_low_frequency_illumination(fg, bg, downsample_factor=4, median_kernel_px=14)
