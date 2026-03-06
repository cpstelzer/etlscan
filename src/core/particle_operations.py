# src/core/particle_operations.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import skimage.util
import shutil
import os

from src.utils.common_exceptions import ConfigError
from src.core.segmentation import generate_binary_mask_from_float_image

logger = logging.getLogger(__name__)

def crop_rotated(
    image: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[int, int],
    angle: float,
    padding: int = 10
) -> np.ndarray:
    """Rotate an image around a center and extract a padded rectangular crop."""
    # ... (This helper function remains unchanged) ...
    w, h = size
    w_padded, h_padded = w + 2 * padding, h + 2 * padding
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    img_rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    
    x_start = int(center[0] - w_padded / 2)
    y_start = int(center[1] - h_padded / 2)
    x_end = x_start + int(w_padded)
    y_end = y_start + int(h_padded)
    
    x_start_clipped = max(0, x_start)
    y_start_clipped = max(0, y_start)
    x_end_clipped = min(img_rotated.shape[1], x_end)
    y_end_clipped = min(img_rotated.shape[0], y_end)
    
    cropped = img_rotated[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]
    
    return cropped


def normalized_crop_config(crop_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize crop configuration values for particle cropping."""
    mode = crop_config.get('mode', 'rotational_rectangle')
    if mode not in [
        'rotational_rectangle',
        'centroid_square',
        'centroid_square_rr',
        'bounding_box',
        'bounding_box_rr'
    ]:
        raise ConfigError(f"Invalid ParticleCropper mode: {mode}")

    adaptive_padding_factor = float(crop_config.get('adaptive_padding_factor', 0.2))
    if adaptive_padding_factor < 0:
        raise ConfigError(f"adaptive_padding_factor must be >= 0, got {adaptive_padding_factor}")

    return {
        'mode': mode,
        'adaptive_padding_factor': adaptive_padding_factor,
        'bbox_column_mapping': crop_config.get('bbox_column_mapping', {
            'x': 'bounding_x', 'y': 'bounding_y', 'w': 'bounding_w', 'h': 'bounding_h'
        }),
    }




def compute_adaptive_padding(width: float, height: float, factor: float) -> Tuple[int, int]:
    """Return per-axis padding derived from particle dimensions and a factor."""
    pad_x = max(int(np.ceil(max(width, 1.0) * factor)), 1)
    pad_y = max(int(np.ceil(max(height, 1.0) * factor)), 1)
    return pad_x, pad_y

def compute_aabb_for_rotated_rectangle(
    center: Tuple[float, float],
    width_with_padding: float,
    height_with_padding: float,
    angle_degrees: float,
    image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Calculate the smallest axis-aligned box that covers a padded rotated rectangle.

    This finds the half-width/half-height of the rotated rectangle projected on the
    x and y axes, then converts those extents into integer crop coordinates that are
    clipped to the image boundaries.
    """
    cx, cy = center
    hw, hh = width_with_padding / 2.0, height_with_padding / 2.0

    angle_rad = np.deg2rad(angle_degrees)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))

    aabb_half_w = hw * cos_a + hh * sin_a
    aabb_half_h = hw * sin_a + hh * cos_a

    x_min = int(np.floor(cx - aabb_half_w))
    x_max = int(np.ceil(cx + aabb_half_w))
    y_min = int(np.floor(cy - aabb_half_h))
    y_max = int(np.ceil(cy + aabb_half_h))

    img_h, img_w = image_shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return x_min, y_min, x_max, y_max


def compute_square_for_rotated_rectangle(
    center: Tuple[float, float],
    width_with_padding: float,
    height_with_padding: float,
    angle_degrees: float,
    image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Build the smallest upright square that fully covers a padded, rotated rectangle.

    In plain terms: take the rectangle dimensions (with padding), rotate them by the given
    angle, find how far the corners stretch horizontally and vertically, pick the larger
    reach so the square covers every corner, and clip the square so it stays inside the image.
    The result is returned as integer (x_min, y_min, x_max, y_max) coordinates ready for
    NumPy slicing.
    """
    cx, cy = center
    hw, hh = width_with_padding / 2.0, height_with_padding / 2.0

    angle_rad = np.deg2rad(angle_degrees)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))

    aabb_half_w = hw * cos_a + hh * sin_a
    aabb_half_h = hw * sin_a + hh * cos_a
    square_half_size = max(aabb_half_w, aabb_half_h)

    x_min = int(np.floor(cx - square_half_size))
    x_max = int(np.ceil(cx + square_half_size))
    y_min = int(np.floor(cy - square_half_size))
    y_max = int(np.ceil(cy + square_half_size))

    img_h, img_w = image_shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return x_min, y_min, x_max, y_max


def crop_particle_from_arrays(
    crop_config: Dict[str, Any],
    full_color_image: np.ndarray,
    full_gray_image: np.ndarray,
    particle_data: pd.Series
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Pure functional cropper that applies the provided configuration to in-memory arrays.

    Args:
        crop_config: Dictionary describing the crop strategy.
        full_color_image: Unused currently, reserved for future use.
        full_gray_image: Grayscale image array.
        particle_data: Row of particle metadata.

    Returns:
        Tuple of the cropped grayscale array (or None) and metadata about the crop.
    """
    normalized = normalized_crop_config(crop_config)
    metadata: Dict[str, Any] = {
        'mode': normalized['mode'],
    }

    if normalized['mode'] == 'centroid_square':
        try:
            cx, cy = int(particle_data['X']), int(particle_data['Y'])
            h_img, w_img = full_gray_image.shape
            bb_w = max(float(particle_data.get('bounding_w', 1.0)), 1.0)
            bb_h = max(float(particle_data.get('bounding_h', 1.0)), 1.0)
            pad_x, pad_y = compute_adaptive_padding(bb_w, bb_h, normalized['adaptive_padding_factor'])
            half_size = int(np.ceil(max(bb_w + 2 * pad_x, bb_h + 2 * pad_y) / 2.0))

            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w_img, cx + half_size)
            y2 = min(h_img, cy + half_size)

            cropped_gray = full_gray_image[y1:y2, x1:x2]
            metadata.update({
                'center': (cx, cy),
                'adaptive_padding': (pad_x, pad_y),
            })

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Missing/invalid data for centroid_square crop: {e}")
            return None, metadata

    elif normalized['mode'] == 'rotational_rectangle':
        try:
            center = (particle_data['X'], particle_data['Y'])
            width = particle_data['rot_rect_width']
            height = particle_data['rot_rect_height']
            angle = particle_data['rot_rect_angle']

            if width > height:
                width, height = height, width
                angle += 90

            pad_x, pad_y = compute_adaptive_padding(width, height, normalized['adaptive_padding_factor'])
            padded_size = (int(width + 2 * pad_x), int(height + 2 * pad_y))
            cropped_gray = crop_rotated(full_gray_image, center, padded_size, angle, padding=0)
            metadata.update({
                'center': center,
                'size': padded_size,
                'angle': angle,
                'adaptive_padding': (pad_x, pad_y),
            })

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Missing/invalid data for rotational_rectangle crop: {e}")
            return None, metadata

    elif normalized['mode'] == 'bounding_box':
        try:
            mapping = normalized['bbox_column_mapping']
            x_col, y_col = mapping['x'], mapping['y']
            w_col, h_col = mapping['w'], mapping['h']
            x, y = int(particle_data[x_col]), int(particle_data[y_col])
            w, h = int(particle_data[w_col]), int(particle_data[h_col])
            
            pad_x, pad_y = compute_adaptive_padding(w, h, normalized['adaptive_padding_factor'])
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(full_gray_image.shape[1], x + w + pad_x)
            y2 = min(full_gray_image.shape[0], y + h + pad_y)

            cropped_gray = full_gray_image[y1:y2, x1:x2]
            metadata.update({
                'bbox': (x, y, w, h),
                'adaptive_padding': (pad_x, pad_y),
            })

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Missing/invalid data for bounding_box crop: {e}")
            return None, metadata

    elif normalized['mode'] == 'bounding_box_rr':
        try:
            center = (particle_data['X'], particle_data['Y'])
            width = float(particle_data['rot_rect_width'])
            height = float(particle_data['rot_rect_height'])
            angle = float(particle_data['rot_rect_angle'])

            pad_x, pad_y = compute_adaptive_padding(width, height, normalized['adaptive_padding_factor'])
            padded_width = width + 2 * pad_x
            padded_height = height + 2 * pad_y

            x1, y1, x2, y2 = compute_aabb_for_rotated_rectangle(
                center,
                padded_width,
                padded_height,
                angle,
                full_gray_image.shape
            )

            cropped_gray = full_gray_image[y1:y2, x1:x2]
            metadata.update({
                'center': center,
                'angle': angle,
                'padded_size': (padded_width, padded_height),
                'adaptive_padding': (pad_x, pad_y),
                'aabb': (x1, y1, x2 - x1, y2 - y1),
            })

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Missing/invalid data for bounding_box_rr crop: {e}")
            return None, metadata

    elif normalized['mode'] == 'centroid_square_rr':
        try:
            center = (float(particle_data['X']), float(particle_data['Y']))
            width = float(particle_data['rot_rect_width'])
            height = float(particle_data['rot_rect_height'])
            angle = float(particle_data['rot_rect_angle'])

            pad_x, pad_y = compute_adaptive_padding(width, height, normalized['adaptive_padding_factor'])
            padded_width = width + 2 * pad_x
            padded_height = height + 2 * pad_y

            x1, y1, x2, y2 = compute_square_for_rotated_rectangle(
                center,
                padded_width,
                padded_height,
                angle,
                full_gray_image.shape
            )

            cropped_gray = full_gray_image[y1:y2, x1:x2]
            metadata.update({
                'center': center,
                'angle': angle,
                'padded_size': (padded_width, padded_height),
                'adaptive_padding': (pad_x, pad_y),
                'square_bbox': (x1, y1, x2 - x1, y2 - y1),
            })

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Missing/invalid data for centroid_square_rr crop: {e}")
            return None, metadata
    
    else:
        return None, metadata

    if cropped_gray is None or cropped_gray.size == 0:
        return None, metadata

    return cropped_gray, metadata


class ParticleCropper:
    """Utility for cropping particles from full images based on configuration."""

    def __init__(self, crop_config: Dict[str, Any]):
        """Store the normalized cropping configuration for reuse."""
        normalized = normalized_crop_config(crop_config)
        self.mode = normalized['mode']
        self._config = normalized

    def crop_single_particle_from_image(
        self,
        full_color_image: np.ndarray,
        full_gray_image: np.ndarray,
        particle_data: pd.Series,
        output_dir_for_crop: Optional[str] = None,
        crop_filename_prefix: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Crop one particle and optionally save it with a generated filename."""
        
        cropped_gray, _ = crop_particle_from_arrays(
            self._config,
            full_color_image,
            full_gray_image,
            particle_data
        )

        if cropped_gray is None or cropped_gray.size == 0:
            return None, None

        saved_path = None
        if output_dir_for_crop and crop_filename_prefix:
            final_filename = f"{crop_filename_prefix}_X{int(particle_data.get('X',0))}_Y{int(particle_data.get('Y',0))}.png"
            saved_path = os.path.join(output_dir_for_crop, final_filename)
            try:
                cv2.imwrite(saved_path, cropped_gray)
            except Exception as e:
                logger.error(f"Failed to save cropped image to {saved_path}: {e}")
                return cropped_gray, None
        
        return cropped_gray, saved_path


def detect_cropped_particle_on_boundary(
    # ... (This function remains unchanged) ...
    image_crop_gray: np.ndarray,
    crop_filename: str,
    illumination_mode_of_crop: str,
    threshold_params: Dict[str, Any]
) -> bool:
    """Determine whether the main segmented particle touches the crop border.

    Boundary dust/scratches can produce tiny disconnected foreground blobs on the
    crop edge. To avoid false positives, this check evaluates only the largest
    connected foreground component (the main particle candidate).
    """
    h, w = image_crop_gray.shape
    if h < 3 or w < 3: return True
    
    img_float = skimage.util.img_as_float(image_crop_gray)
    
    if illumination_mode_of_crop == 'collimated':
        img_float_inverted = 1.0 - img_float
        binary_mask = generate_binary_mask_from_float_image(img_float_inverted, threshold_params)
    else: # darkfield
        binary_mask = generate_binary_mask_from_float_image(img_float, threshold_params)
    
    if binary_mask is None:
        logger.warning(f"Boundary check failed for {crop_filename} due to mask generation error.")
        return True 

    binary_uint8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)
    if num_labels <= 1:
        return False

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_label = 1 + int(np.argmax(component_areas))
    main_particle_mask = labels == largest_component_label

    if (
        main_particle_mask[0, :].any() or
        main_particle_mask[-1, :].any() or
        main_particle_mask[:, 0].any() or
        main_particle_mask[:, -1].any()
    ):
        return True
        
    return False

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    # ... (ParticleCropper tests can remain unchanged if you wish to keep them) ...
    # Example from ParticleCropper test:
    main_h, main_w = 400, 500
    main_color_img = np.random.randint(0, 256, (main_h, main_w, 3), dtype=np.uint8)
    cv2.ellipse(main_color_img, (250, 200), (100, 50), 25, 0, 360, (255, 0, 0), -1)
    main_gray_img = cv2.cvtColor(main_color_img, cv2.COLOR_BGR2GRAY)
    
    # MODIFIED: Added test data for rotational crop
    particles = pd.DataFrame([
        {'X': 50, 'Y': 50, 'particle_id': 1, 'bounding_x': 40, 'bounding_y': 40, 'bounding_w': 20, 'bounding_h': 20, 'rot_rect_width': 20, 'rot_rect_height': 22, 'rot_rect_angle': 15.0},
        {'X': 250, 'Y': 200, 'particle_id': 2, 'bounding_x': 150, 'bounding_y': 150, 'bounding_w': 200, 'bounding_h': 100, 'rot_rect_width': 200, 'rot_rect_height': 100, 'rot_rect_angle': 25.0} # Wide rectangle
    ])
    
    temp_crop_output_dir = "temp_particle_ops_test_rotational" 
    os.makedirs(temp_crop_output_dir, exist_ok=True)

    logger.info("\n--- Testing Rotational Crop Mode ---")
    cropper_config_rot = {'mode': 'rotational_rectangle', 'adaptive_padding_factor': 0.2}
    rot_cropper = ParticleCropper(cropper_config_rot)
    crop_rot, path_rot = rot_cropper.crop_single_particle_from_image(main_color_img, main_gray_img, particles.iloc[1])
    assert crop_rot is not None and path_rot is not None
    logger.info(f"Rotational crop successful. Saved to {path_rot}")
    # Verify the orientation (height > width)
    assert crop_rot.shape[0] > crop_rot.shape[1], "Rotational crop orientation logic failed."
    logger.info("Rotational crop orientation test passed.")


    logger.info("\n--- Testing Boundary Crop Detection (Consolidated Thresholding Logic) ---")
    
    crop_dark_particle_bright_bg_distinct = np.full((50, 50), 250, dtype=np.uint8) 
    cv2.rectangle(crop_dark_particle_bright_bg_distinct, (0, 10), (30, 40), 10, -1) 
    test_path1_distinct = os.path.join(temp_crop_output_dir, "dark_particle_touching_consolidated.png")
    cv2.imwrite(test_path1_distinct, crop_dark_particle_bright_bg_distinct)
    
    # Define threshold params for the test
    otsu_thresh_params_test = {
        'thresholding_mode':"otsu",
        'otsu_threshold_multiplier':0.45, 
    }
    is_touching_consolidated_otsu = detect_cropped_particle_on_boundary(
        crop_dark_particle_bright_bg_distinct, test_path1_distinct, 
        illumination_mode_of_crop="collimated", 
        threshold_params=otsu_thresh_params_test
    )
    assert is_touching_consolidated_otsu is True, "Boundary crop (distinct dark particle, collimated, consolidated Otsu) detection failed"
    logger.info("Boundary crop (distinct dark particle, collimated, consolidated Otsu) detection PASSED.")


    if os.path.exists(temp_crop_output_dir):
        shutil.rmtree(temp_crop_output_dir)
        logger.info(f"Cleaned up {temp_crop_output_dir}")
