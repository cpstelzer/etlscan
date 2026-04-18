"""Tests for src/core/sharpness_assessment.py.

Migrated from the __main__ block that previously lived in the source file.
"""
import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

from src.core.sharpness_assessment import (
    calculate_sharpness_metrics,
    normalize_and_combine_sharpness,
    AVAILABLE_SHARPNESS_METRICS,
)


@pytest.fixture()
def sharp_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    return img


@pytest.fixture()
def blurry_image(sharp_image):
    return cv2.GaussianBlur(sharp_image, (15, 15), 0)


def test_sharp_greater_than_blurry_variance_laplacian(sharp_image, blurry_image):
    sharp_scores = calculate_sharpness_metrics(sharp_image)
    blurry_scores = calculate_sharpness_metrics(blurry_image)
    assert sharp_scores["variance_laplacian"] > blurry_scores["variance_laplacian"]


def test_sharp_greater_than_blurry_tenengrad(sharp_image, blurry_image):
    sharp_scores = calculate_sharpness_metrics(sharp_image)
    blurry_scores = calculate_sharpness_metrics(blurry_image)
    assert sharp_scores["tenengrad_variance"] > blurry_scores["tenengrad_variance"]


def test_small_image_returns_all_nan():
    scores = calculate_sharpness_metrics(np.zeros((2, 2), dtype=np.uint8))
    assert all(np.isnan(v) for v in scores.values())


def test_none_image_returns_all_nan():
    scores = calculate_sharpness_metrics(None)
    assert all(np.isnan(v) for v in scores.values())


def test_returns_all_expected_keys(sharp_image):
    scores = calculate_sharpness_metrics(sharp_image)
    assert set(scores.keys()) == set(AVAILABLE_SHARPNESS_METRICS)


def test_normalize_and_combine_adds_norm_and_combined_columns():
    df = pd.DataFrame({
        "variance_laplacian": [100.0, 200.0, 300.0],
        "tenengrad_variance": [5000.0, 6000.0, 8000.0],
    })
    result = normalize_and_combine_sharpness(df, ["variance_laplacian", "tenengrad_variance"])
    assert "variance_laplacian_norm" in result.columns
    assert "tenengrad_variance_norm" in result.columns
    assert "combined_sharpness" in result.columns


def test_normalize_min_is_zero_max_is_one():
    df = pd.DataFrame({"variance_laplacian": [100.0, 200.0, 300.0]})
    result = normalize_and_combine_sharpness(df, ["variance_laplacian"])
    assert np.isclose(result["variance_laplacian_norm"].min(), 0.0)
    assert np.isclose(result["variance_laplacian_norm"].max(), 1.0)


def test_normalize_nan_input_propagates():
    df = pd.DataFrame({"variance_laplacian": [100.0, np.nan, 300.0]})
    result = normalize_and_combine_sharpness(df, ["variance_laplacian"])
    assert np.isnan(result.loc[1, "variance_laplacian_norm"])


def test_normalize_single_unique_value_gives_nan():
    # A metric with only one unique value has no meaningful range; the code
    # emits NaN because it cannot normalize to [0, 1] without division by zero.
    df = pd.DataFrame({"variance_laplacian": [100.0, 100.0]})
    result = normalize_and_combine_sharpness(df, ["variance_laplacian"])
    assert result["variance_laplacian_norm"].isnull().all()


def test_normalize_all_nan_column():
    df = pd.DataFrame({"variance_laplacian": [np.nan, np.nan]})
    result = normalize_and_combine_sharpness(df, ["variance_laplacian"])
    assert result["variance_laplacian_norm"].isnull().all()
    assert result["combined_sharpness"].isnull().all()


def test_normalize_empty_dataframe():
    df = pd.DataFrame({"variance_laplacian": pd.Series([], dtype=float)})
    result = normalize_and_combine_sharpness(df, ["variance_laplacian"])
    assert result.empty or result["combined_sharpness"].isnull().all()
