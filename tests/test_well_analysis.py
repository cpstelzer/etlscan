"""Tests for src/core/well_analysis.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

from src.core.well_analysis import (
    create_well_mask_black_plate,
    calculate_well_brightness_peak,
    match_well_brightness,
)


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
