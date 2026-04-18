"""Tests for src/utils/filename_helpers.py."""
import pytest

from src.utils.filename_helpers import build_crop_filename


def test_basic_offline_format():
    result = build_crop_filename("20240101_plate1_well3", "track42", 7, 120, 85)
    assert result == "20240101_plate1_well3_track42_frame7_X120_Y85.png"


def test_default_extension_is_png():
    result = build_crop_filename("sid", "p1", 1, 0, 0)
    assert result.endswith(".png")


def test_custom_extension():
    result = build_crop_filename("sid", "p1", 1, 0, 0, ext="jpg")
    assert result.endswith(".jpg")


def test_frame_padding_zero():
    result = build_crop_filename("sid", "p1", 5, 0, 0, frame_padding=0)
    assert "_frame5_" in result


def test_frame_padding_six():
    result = build_crop_filename("sid", "p1", 5, 0, 0, frame_padding=6)
    assert "_frame000005_" in result


def test_realtime_format():
    prefix = "sess1_plate2_well5"
    result = build_crop_filename(prefix, "particle3", 100, 200, 300, ext="png", frame_padding=6)
    assert result == "sess1_plate2_well5_particle3_frame000100_X200_Y300.png"


def test_zero_coordinates():
    result = build_crop_filename("p", "t", 0, 0, 0)
    assert result == "p_t_frame0_X0_Y0.png"


def test_large_frame_number_with_padding():
    result = build_crop_filename("p", "t", 1000000, 0, 0, frame_padding=6)
    assert "_frame1000000_" in result
