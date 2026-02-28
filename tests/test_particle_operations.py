import numpy as np
import pandas as pd
import pytest
cv2 = pytest.importorskip("cv2", reason="OpenCV is required for particle cropping utilities.", exc_type=ImportError)


from src.core.particle_operations import (
    compute_square_for_rotated_rectangle,
    crop_particle_from_arrays,
    detect_cropped_particle_on_boundary,
)


def test_compute_square_for_rotated_rectangle_builds_square_and_clips_to_image():
    """Square helper returns a square crop and respects image bounds."""
    center = (100.0, 150.0)
    width_with_padding = 60.0
    height_with_padding = 40.0
    angle_degrees = 45.0
    image_shape = (300, 300)

    x1, y1, x2, y2 = compute_square_for_rotated_rectangle(
        center,
        width_with_padding,
        height_with_padding,
        angle_degrees,
        image_shape
    )

    assert x2 - x1 == y2 - y1
    assert 0 <= x1 < x2 <= image_shape[1]
    assert 0 <= y1 < y2 <= image_shape[0]

    edge_center = (5.0, 5.0)
    x1_edge, y1_edge, x2_edge, y2_edge = compute_square_for_rotated_rectangle(
        edge_center,
        10.0,
        10.0,
        0.0,
        image_shape
    )
    assert x1_edge == 0
    assert y1_edge == 0
    assert x2_edge - x1_edge == y2_edge - y1_edge


def test_centroid_square_rr_mode_returns_square_crop():
    """centroid_square_rr mode yields a square crop and metadata entry."""
    crop_config = {'mode': 'centroid_square_rr', 'rotational_padding': 5}
    full_image = np.zeros((120, 120), dtype=np.uint8)
    particle_data = pd.Series({
        'X': 60.0,
        'Y': 60.0,
        'rot_rect_width': 30.0,
        'rot_rect_height': 20.0,
        'rot_rect_angle': 30.0,
    })

    crop, metadata = crop_particle_from_arrays(
        crop_config,
        full_image,
        full_image,
        particle_data
    )

    assert crop is not None
    assert crop.shape[0] == crop.shape[1]
    assert metadata.get('mode') == 'centroid_square_rr'
    assert 'square_bbox' in metadata


def test_boundary_detection_ignores_disconnected_edge_speck():
    """Disconnected edge specks should not flag the crop as boundary-touching."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    cv2.ellipse(img, (60, 60), (18, 35), 0, 0, 360, 20, -1)
    img[0:2, 5:7] = 10  # small disconnected speck on the top edge

    result = detect_cropped_particle_on_boundary(
        image_crop_gray=img,
        crop_filename="synthetic_disconnected.png",
        illumination_mode_of_crop="collimated",
        threshold_params={
            "thresholding_mode": "fixed",
            "fixed_threshold_value": 128,
        },
    )

    assert result is False


def test_boundary_detection_flags_when_edge_artifact_connects_to_main_particle():
    """If an edge artifact is connected to the main body, boundary flag should remain true."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    cv2.ellipse(img, (60, 60), (18, 35), 0, 0, 360, 20, -1)
    cv2.line(img, (60, 0), (60, 25), 20, 2)  # connected to the top of the main body

    result = detect_cropped_particle_on_boundary(
        image_crop_gray=img,
        crop_filename="synthetic_connected.png",
        illumination_mode_of_crop="collimated",
        threshold_params={
            "thresholding_mode": "fixed",
            "fixed_threshold_value": 128,
        },
    )

    assert result is True
