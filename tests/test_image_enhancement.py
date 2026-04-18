"""Tests for src/core/image_enhancement.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

from src.core.image_enhancement import (
    subtract_images,
    apply_threshold,
    apply_blur,
    apply_morphological_operation,
)


@pytest.fixture()
def square_images():
    """Two small grayscale images: a larger square and a smaller inner square."""
    img_a = np.zeros((50, 50), dtype=np.uint8)
    img_a[10:30, 10:30] = 150

    img_b = np.zeros((50, 50), dtype=np.uint8)
    img_b[15:25, 15:25] = 50

    return img_a, img_b


def test_subtract_images_returns_array(square_images):
    img_a, img_b = square_images
    result = subtract_images(img_a, img_b)
    assert result is not None
    assert result.shape == img_a.shape


def test_subtract_images_none_inputs():
    assert subtract_images(None, np.zeros((10, 10), dtype=np.uint8)) is None
    assert subtract_images(np.zeros((10, 10), dtype=np.uint8), None) is None


def test_subtract_images_shape_mismatch():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((20, 20), dtype=np.uint8)
    assert subtract_images(a, b) is None


def test_apply_threshold_binary(square_images):
    img_a, _ = square_images
    result = apply_threshold(img_a, 100, "binary")
    assert result is not None
    assert result.shape == img_a.shape


def test_apply_threshold_otsu_binary(square_images):
    img_a, _ = square_images
    result = apply_threshold(img_a, 0, "otsu_binary")
    assert result is not None


def test_apply_threshold_binary_inv(square_images):
    img_a, _ = square_images
    result = apply_threshold(img_a, 100, "binary_inv")
    assert result is not None


def test_apply_threshold_unsupported_type(square_images):
    img_a, _ = square_images
    assert apply_threshold(img_a, 100, "unknown_type") is None


def test_apply_threshold_none_image():
    assert apply_threshold(None, 100) is None


def test_apply_blur_median(square_images):
    img_a, _ = square_images
    result = apply_blur(img_a, 3, "median")
    assert result is not None
    assert result.shape == img_a.shape


def test_apply_blur_gaussian(square_images):
    img_a, _ = square_images
    result = apply_blur(img_a, 3, "gaussian")
    assert result is not None


def test_apply_blur_even_kernel_rejected(square_images):
    img_a, _ = square_images
    assert apply_blur(img_a, 4, "median") is None


def test_apply_blur_none_image():
    assert apply_blur(None, 3) is None


def test_apply_morphological_erode(square_images):
    img_a, _ = square_images
    thresh = apply_threshold(img_a, 100, "binary")
    result = apply_morphological_operation(thresh, 3, "erode", "rect")
    assert result is not None


def test_apply_morphological_dilate(square_images):
    img_a, _ = square_images
    thresh = apply_threshold(img_a, 100, "binary")
    result = apply_morphological_operation(thresh, 3, "dilate", "rect")
    assert result is not None


def test_apply_morphological_open_close(square_images):
    img_a, _ = square_images
    thresh = apply_threshold(img_a, 100, "binary")
    assert apply_morphological_operation(thresh, 3, "open", "ellipse") is not None
    assert apply_morphological_operation(thresh, 3, "close", "cross") is not None


def test_apply_morphological_unsupported_operation(square_images):
    img_a, _ = square_images
    thresh = apply_threshold(img_a, 100, "binary")
    assert apply_morphological_operation(thresh, 3, "unsupported_op") is None


def test_apply_morphological_none_image():
    assert apply_morphological_operation(None, 3, "erode") is None
