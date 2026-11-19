"""Tests for src/core/feature_calculators/bio_features_calculator.py."""
import numpy as np
import pytest

from src.core.feature_calculators.bio_features_calculator import calculate_posterior_attributes


# ---------------------------------------------------------------------------
# calculate_posterior_attributes
# ---------------------------------------------------------------------------

class _FakeProps:
    """Minimal stand-in for skimage RegionProperties."""
    def __init__(self):
        # Simple 20×20 square body centred at (10,10)
        rows, cols = np.meshgrid(range(20), range(20), indexing='ij')
        self.coords = np.column_stack([rows.ravel(), cols.ravel()])
        self.area = len(self.coords)
        self.centroid = (10.0, 10.0)
        self.orientation = 0.0  # horizontal


def test_calculate_posterior_attributes_returns_triple():
    props = _FakeProps()
    result = calculate_posterior_attributes(props, (20, 20), 0.3)
    assert len(result) == 3


def test_calculate_posterior_attributes_attaches_axis_vector():
    props = _FakeProps()
    annotated, _, _ = calculate_posterior_attributes(props, (20, 20), 0.3)
    assert hasattr(annotated, '_axis_vector_pointing_posterior')
    assert annotated._axis_vector_pointing_posterior.shape == (2,)


def test_calculate_posterior_attributes_ratio_in_range():
    props = _FakeProps()
    _, ratio, _ = calculate_posterior_attributes(props, (20, 20), 0.3)
    assert 0.0 <= ratio <= 1.0


def test_calculate_posterior_attributes_missing_orientation_defaults():
    class _NoOrientation:
        area = 100
        coords = np.zeros((100, 2))
        centroid = (5.0, 5.0)
        # no 'orientation' attribute

    props = _NoOrientation()
    annotated, ratio, coords = calculate_posterior_attributes(props, (10, 10), 0.3)
    assert ratio == 0.5
    assert coords is None


def test_calculate_posterior_attributes_out_of_range_fraction_clamps():
    props = _FakeProps()
    # fraction > 1 should be clamped to 0.45 without raising
    annotated, ratio, coords = calculate_posterior_attributes(props, (20, 20), 1.5)
    assert 0.0 <= ratio <= 1.0
