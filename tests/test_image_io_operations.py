"""Tests for src/core/image_io_operations.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import os
import tempfile

import numpy as np
import pytest

from src.core.image_io_operations import (
    load_image,
    save_image,
    convert_to_grayscale,
)


@pytest.fixture()
def color_image():
    """A simple 100x100 BGR image: left half blue, right half red."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = [255, 0, 0]   # blue in BGR
    img[:, 50:] = [0, 0, 255]   # red in BGR
    return img


@pytest.fixture()
def saved_color_image(color_image, tmp_path):
    """Write the color image to a temp file and return its path."""
    path = save_image(color_image, str(tmp_path), "color_test.png")
    assert path is not None, "Fixture setup failed: could not save test image."
    return path


def test_save_image_returns_path(color_image, tmp_path):
    path = save_image(color_image, str(tmp_path), "out.png")
    assert path is not None
    assert os.path.exists(path)


def test_save_image_none_array(tmp_path):
    assert save_image(None, str(tmp_path), "out.png") is None


def test_load_image_color_roundtrip(saved_color_image, color_image):
    loaded = load_image(saved_color_image, mode="color")
    assert loaded is not None
    assert loaded.shape == color_image.shape


def test_load_image_gray_is_2d(saved_color_image):
    loaded = load_image(saved_color_image, mode="gray")
    assert loaded is not None
    assert loaded.ndim == 2


def test_load_image_nonexistent_returns_none():
    assert load_image("/tmp/does_not_exist_xyz.png") is None


def test_convert_to_grayscale_from_bgr(color_image):
    gray = convert_to_grayscale(color_image)
    assert gray is not None
    assert gray.ndim == 2
    assert gray.shape == color_image.shape[:2]


def test_convert_to_grayscale_already_gray():
    gray_in = np.full((50, 50), 128, dtype=np.uint8)
    gray_out = convert_to_grayscale(gray_in)
    assert gray_out is not None
    assert gray_out.ndim == 2


def test_convert_to_grayscale_none_returns_none():
    assert convert_to_grayscale(None) is None
