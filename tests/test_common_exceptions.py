"""Tests for src/utils/common_exceptions.py — exception hierarchy and contracts."""
import pytest

from src.utils.common_exceptions import (
    ConfigError,
    ProcessingError,
    ImageProcessingError,
    MLModelError,
    DataValidationError,
)


# --- Hierarchy checks ---

def test_config_error_is_exception():
    assert issubclass(ConfigError, Exception)


def test_processing_error_is_runtime_error():
    assert issubclass(ProcessingError, RuntimeError)


def test_image_processing_error_is_processing_error():
    assert issubclass(ImageProcessingError, ProcessingError)


def test_ml_model_error_is_processing_error():
    assert issubclass(MLModelError, ProcessingError)


def test_data_validation_error_is_value_error():
    assert issubclass(DataValidationError, ValueError)


# --- Catching via parent class ---

def test_image_processing_error_caught_as_processing_error():
    with pytest.raises(ProcessingError):
        raise ImageProcessingError("bad image")


def test_ml_model_error_caught_as_processing_error():
    with pytest.raises(ProcessingError):
        raise MLModelError("model failed")


def test_ml_model_error_caught_as_runtime_error():
    with pytest.raises(RuntimeError):
        raise MLModelError("model failed")


def test_data_validation_error_caught_as_value_error():
    with pytest.raises(ValueError):
        raise DataValidationError("missing column")


# --- Message round-trip ---

def test_exception_messages_are_preserved():
    msg = "configuration key missing"
    exc = ConfigError(msg)
    assert str(exc) == msg


def test_ml_model_error_message_preserved():
    msg = "model weights not found"
    exc = MLModelError(msg)
    assert str(exc) == msg
