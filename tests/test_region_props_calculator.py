"""Tests for src/core/region_props_calculator.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

from src.core.region_props_calculator import (
    find_contours,
    calculate_contour_measurements,
    filter_contours_by_area,
)


@pytest.fixture()
def two_shape_image():
    """Binary image containing a rectangle and a circle."""
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (50, 50), 255, -1)
    cv2.circle(img, (70, 70), 15, 255, -1)
    return img


def test_find_contours_returns_two_shapes(two_shape_image):
    contours = find_contours(two_shape_image, mode="external", method="simple")
    assert contours is not None
    assert len(contours) == 2


def test_find_contours_none_image():
    assert find_contours(None) is None


def test_find_contours_wrong_dtype():
    img = np.zeros((50, 50), dtype=np.float32)
    assert find_contours(img) is None


def test_calculate_contour_measurements_returns_dict(two_shape_image):
    contours = find_contours(two_shape_image, mode="external", method="simple")
    assert contours is not None
    for cnt in contours:
        measurements = calculate_contour_measurements(cnt)
        assert measurements is not None
        assert isinstance(measurements, dict)
        assert "X" in measurements
        assert "Y" in measurements
        assert "area" in measurements


def test_filter_contours_by_area_keeps_large(two_shape_image):
    contours = find_contours(two_shape_image, mode="external", method="simple")
    assert contours is not None
    # Rectangle ~30×30=900px, circle ~π×15²≈707px; threshold 800 keeps only rectangle.
    large = filter_contours_by_area(contours, min_area=800)
    assert len(large) == 1


def test_filter_contours_by_area_keeps_all(two_shape_image):
    contours = find_contours(two_shape_image, mode="external", method="simple")
    assert contours is not None
    result = filter_contours_by_area(contours, min_area=100)
    assert len(result) == 2


def test_filter_contours_by_area_keeps_none(two_shape_image):
    contours = find_contours(two_shape_image, mode="external", method="simple")
    assert contours is not None
    result = filter_contours_by_area(contours, min_area=1000)
    assert len(result) == 0
