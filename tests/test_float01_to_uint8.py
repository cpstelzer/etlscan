"""Tests for float01_to_uint8 added to src/core/image_enhancement.py."""
import numpy as np
import pytest

from src.core.image_enhancement import float01_to_uint8


def test_zero_maps_to_zero():
    img = np.zeros((4, 4), dtype=np.float32)
    result = float01_to_uint8(img)
    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_one_maps_to_255():
    img = np.ones((4, 4), dtype=np.float64)
    result = float01_to_uint8(img)
    assert result.dtype == np.uint8
    assert np.all(result == 255)


def test_half_maps_to_128():
    img = np.full((4, 4), 0.5, dtype=np.float32)
    result = float01_to_uint8(img)
    # 0.5 * 255 = 127.5 → rounded to 128
    assert result.dtype == np.uint8
    assert np.all(result == 128)


def test_values_above_one_are_clamped_to_255():
    img = np.array([[1.5, 2.0]], dtype=np.float32)
    result = float01_to_uint8(img)
    assert np.all(result == 255)


def test_values_below_zero_are_clamped_to_zero():
    img = np.array([[-0.5, -1.0]], dtype=np.float32)
    result = float01_to_uint8(img)
    assert np.all(result == 0)


def test_output_shape_matches_input():
    img = np.random.rand(10, 20).astype(np.float32)
    result = float01_to_uint8(img)
    assert result.shape == img.shape


def test_round_trip_approximate():
    # float → uint8 → float should be within 1/255 of original
    original = np.linspace(0.0, 1.0, 256, dtype=np.float64).reshape(16, 16)
    uint8 = float01_to_uint8(original)
    recovered = uint8.astype(np.float64) / 255.0
    assert np.allclose(original, recovered, atol=1.0 / 255.0)
