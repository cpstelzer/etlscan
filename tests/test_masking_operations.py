"""Tests for src/core/masking_operations.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

from src.core.masking_operations import (
    create_circular_mask,
    apply_mask,
    mask_well_area,
    detect_black_mask_artifact,
)


@pytest.fixture()
def circle_image():
    """Grayscale image with a bright circle on a black background."""
    img = np.zeros((200, 250), dtype=np.uint8)
    cv2.circle(img, (125, 100), 70, 150, -1)
    return img


def test_create_circular_mask_shape_and_dtype():
    shape = (100, 100)
    mask = create_circular_mask(shape, center=(50, 50), radius=30)
    assert mask is not None
    assert mask.shape == shape
    assert mask.dtype == np.uint8


def test_apply_mask_zeros_outside_circle():
    shape = (100, 100)
    img = np.full(shape, 128, dtype=np.uint8)
    mask = create_circular_mask(shape, center=(50, 50), radius=30)
    masked = apply_mask(img, mask)
    assert masked is not None
    # Corners should be zeroed by the circular mask.
    assert masked[0, 0] == 0


def test_apply_mask_none_inputs():
    img = np.full((50, 50), 100, dtype=np.uint8)
    mask = np.ones((50, 50), dtype=np.uint8) * 255
    assert apply_mask(None, mask) is None
    assert apply_mask(img, None) is None


def test_apply_mask_shape_mismatch():
    img = np.full((50, 50), 100, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    assert apply_mask(img, mask) is None


def test_mask_well_area_auto_center(circle_image):
    mask_config = {"auto_center_mask": True, "radius_px": 100, "radius_fraction": 0.8}
    system_config = {}
    result_image, binary_mask = mask_well_area(circle_image, mask_config, system_config)
    assert result_image is not None
    assert binary_mask is not None
    assert result_image.shape == circle_image.shape


def test_mask_well_area_manual_center(circle_image):
    mask_config = {"auto_center_mask": False, "radius_px": 60, "radius_fraction": 1.0}
    system_config = {"image_width_px": 250, "image_height_px": 200}
    result_image, binary_mask = mask_well_area(circle_image, mask_config, system_config)
    assert result_image is not None
    assert binary_mask is not None


def test_mask_well_area_missing_radius_fraction(circle_image):
    mask_config = {"auto_center_mask": True, "radius_px": 100}
    result_image, binary_mask = mask_well_area(circle_image, mask_config, {})
    assert result_image is None
    assert binary_mask is None


def test_detect_black_mask_artifact_detects_corner(tmp_path):
    img = np.full((100, 100), 128, dtype=np.uint8)
    img[0:30, 0:30] = 0
    path = str(tmp_path / "corner.png")
    cv2.imwrite(path, img)
    assert detect_black_mask_artifact(path, min_area_ratio=0.05) is True


def test_detect_black_mask_artifact_clean_image(tmp_path):
    img = np.full((100, 100), 128, dtype=np.uint8)
    path = str(tmp_path / "clean.png")
    cv2.imwrite(path, img)
    assert detect_black_mask_artifact(path, min_area_ratio=0.05) is False


def test_detect_black_mask_artifact_in_memory_array():
    img = np.full((100, 100), 128, dtype=np.uint8)
    img[0:30, 0:30] = 0
    assert detect_black_mask_artifact(None, image_gray=img, min_area_ratio=0.05) is True
