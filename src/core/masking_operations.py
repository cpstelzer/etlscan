# src/core/masking_operations.py
import cv2
import os
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from src.utils.common_exceptions import ImageProcessingError
from .region_props_calculator import find_contours # For mask_well_area

logger = logging.getLogger(__name__)

def create_circular_mask(image_shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
    """
    Creates a binary circular mask.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies a binary mask to an image.
    """
    if image is None or mask is None:
        logger.error("Image or mask is None, cannot apply mask.")
        return None
    if image.shape[:2] != mask.shape[:2]:
        logger.error(f"Image shape {image.shape[:2]} and mask shape {mask.shape[:2]} must match.")
        return None
    if mask.dtype != np.uint8:
        logger.warning("Mask dtype is not uint8. This might lead to unexpected behavior.")
    
    try:
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        logger.debug("Mask applied successfully.")
        return masked_image
    except Exception as e:
        logger.exception(f"Error applying mask: {e}")
        return None

def mask_well_area(
    image_gray: np.ndarray,
    mask_config: Dict[str, Any],
    system_params: Dict[str, Any] # For fixed centering/radius if auto is false
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Defines an ROI for the well and masks the image.
    Adapted from wellcounter_imaging_module.mask_well_area.
    The radius calculation is modified to strictly follow the original logic:
    it always uses the configured 'radius_px' and 'radius_fraction',
    ignoring any auto-detected radius for sizing.

    Args:
        image_gray (np.ndarray): Input grayscale image.
        mask_config (Dict[str, Any]): Configuration for masking (e.g., auto_center_mask, radius_px, radius_fraction).
        system_params (Dict[str, Any]): System parameters (e.g., image_width_px).

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (masked_result_image, binary_mask_used) or (None, None) on error.
    """
    if image_gray is None:
        logger.error("Input image for well masking is None.")
        return None, None
    if image_gray.ndim != 2:
        logger.error("Well masking requires a grayscale image.")
        return None, None

    img_h, img_w = image_gray.shape
    center: Optional[Tuple[int, int]] = None
    radius: Optional[int] = None

    try:
        # Determine center coordinates
        if mask_config.get('auto_center_mask', True):
            # Threshold to find the well area
            _, threshold_img = cv2.threshold(image_gray, 5, 255, cv2.THRESH_BINARY)
            contours = find_contours(threshold_img, mode="external", method="simple")
            if not contours:
                logger.warning("No contours found for auto-centering mask. Falling back to image center.")
                center = (img_w // 2, img_h // 2)
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                # auto_radius from minEnclosingCircle is NOT used for final mask radius sizing
                # as per original wim logic, only (cx, cy) is used.
                (cx_auto, cy_auto), _ = cv2.minEnclosingCircle(largest_contour)
                center = (int(cx_auto), int(cy_auto))
        else: # Manual centering
            center_x = system_params.get('image_width_px', img_w) // 2
            center_y = system_params.get('image_height_px', img_h) // 2
            center = (center_x, center_y)

        # Determine radius (always uses configured radius_px and radius_fraction)
        config_radius_px = mask_config.get("radius_px")
        if config_radius_px is None:
            logger.error("'radius_px' not found in mask_config. Cannot determine mask radius.")
            return None, None
        
        radius_fraction = mask_config.get('radius_fraction')
        if radius_fraction is None:
            logger.error("'radius_fraction' not found in mask_config for well masking. "
                         "Ensure it's set (e.g., to 0.67 to match original behavior).")
            return None, None
            
        radius = int(config_radius_px * radius_fraction)

        if radius <= 0:
            logger.error(f"Calculated mask radius ({radius}) is not positive. Cannot create mask.")
            return None, None

        logger.debug(f"Creating well mask with center {center} and radius {radius}.")
        well_binary_mask = create_circular_mask(image_gray.shape, center, radius)
        
        result_image = apply_mask(image_gray, well_binary_mask)
        if result_image is None:
            return None, None # Error applying mask

        return result_image, well_binary_mask

    except Exception as e:
        logger.exception(f"Error during well area masking: {e}")
        return None, None


def detect_black_mask_artifact(
    image_path: Optional[str], # Original wum function took path, can be adapted for array
    image_gray: Optional[np.ndarray] = None,
    min_area_ratio: float = 0.05
    ) -> bool:
    """
    Detects if an image has a large, contiguous area of exactly zero intensity
    touching two adjacent borders (i.e., a corner black mask).
    Adapted from wellscanner_util_module.detect_black_mask.

    Args:
        image_path (Optional[str]): Path to the image file (used for logging and if image_gray is None).
        image_gray (Optional[np.ndarray]): Pre-loaded grayscale image. If None, loads from image_path.
        min_area_ratio (float): Minimum area ratio of black component to total image area.

    Returns:
        bool: True if a black mask artifact is detected, False otherwise.
    """
    from .image_io_operations import load_image # Local import to avoid circular dependency at module level

    img_to_process = image_gray
    path_for_logging = image_path or "<in-memory crop>"
    if img_to_process is None:
        if image_path is None:
            logger.warning("No image path or array provided for black mask detection.")
            return False
        if not os.path.exists(image_path): # Check existence before loading
            logger.warning(f"Image file not found at {image_path} for black mask detection.")
            return False
        img_to_process = load_image(image_path, mode='gray')

    if img_to_process is None:
        logger.warning(f"Cannot load image {path_for_logging} for black mask detection.")
        return False

    try:
        h, w = img_to_process.shape
        total_area = float(h * w) 
        if total_area == 0:
            logger.warning(f"Image {os.path.basename(path_for_logging)} has zero area.")
            return False

        binary_black_pixels_mask = (img_to_process == 0)

        if not np.any(binary_black_pixels_mask):
            logger.debug(f"No black pixels found in {os.path.basename(path_for_logging)}.")
            return False

        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
            binary_black_pixels_mask.astype(np.uint8), connectivity=8
        )

        min_pixel_area_threshold = min_area_ratio * total_area
        logger.debug(f"Black mask check for {os.path.basename(path_for_logging)}: "
                     f"Threshold area = {min_pixel_area_threshold:.0f} pixels.")

        for label_idx in range(1, num_labels):
            left, top, width_comp, height_comp, area_comp = stats[label_idx]
            
            if area_comp > min_pixel_area_threshold:
                touches_top = (top == 0)
                touches_bottom = (top + height_comp == h)
                touches_left = (left == 0)
                touches_right = (left + width_comp == w)

                is_corner_component = (touches_top and touches_left) or \
                                      (touches_top and touches_right) or \
                                      (touches_bottom and touches_left) or \
                                      (touches_bottom and touches_right)
                
                if is_corner_component:
                    logger.debug(f"Component L{label_idx} (area {area_comp}) meets corner criteria. Black mask DETECTED.")
                    return True

        logger.debug(f"No qualifying black mask component found in {os.path.basename(path_for_logging)}. Black mask NOT detected.")
        return False
    except Exception as e:
        logger.exception(f"Error during black mask detection for {path_for_logging}: {e}")
        return False


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    # Test create_circular_mask and apply_mask
    shape = (100, 100)
    center = (50, 50)
    radius = 30
    circ_mask = create_circular_mask(shape, center, radius)
    assert circ_mask is not None and circ_mask.shape == shape, "Circular mask creation failed"
    logger.info("Circular mask creation test passed.")

    dummy_img_for_masking = np.full(shape, 128, dtype=np.uint8)
    masked_by_circ = apply_mask(dummy_img_for_masking, circ_mask)
    assert masked_by_circ is not None, "Apply mask failed"
    logger.info("Apply mask test passed.")

    dummy_gray_img = np.zeros((200, 250), dtype=np.uint8)
    cv2.circle(dummy_gray_img, (125, 100), 70, 150, -1) 
    
    # Test with auto_center_mask = True
    test_mask_config_auto = {'auto_center_mask': True, 'radius_px': 100, 'radius_fraction': 0.8}
    test_system_params_auto = {} 
    result_img_auto, binary_mask_auto = mask_well_area(dummy_gray_img, test_mask_config_auto, test_system_params_auto)
    assert result_img_auto is not None and binary_mask_auto is not None, "mask_well_area (auto) failed"
    logger.info("mask_well_area (auto) test passed. Masked image shape: %s, binary mask shape: %s", result_img_auto.shape, binary_mask_auto.shape)

    # Test with auto_center_mask = False (manual)
    test_mask_config_manual = {'auto_center_mask': False, 'radius_px': 60, 'radius_fraction': 1.0}
    test_system_params_manual = {'image_width_px': 250, 'image_height_px': 200}
    result_img_manual, binary_mask_manual = mask_well_area(dummy_gray_img, test_mask_config_manual, test_system_params_manual)
    assert result_img_manual is not None and binary_mask_manual is not None, "mask_well_area (manual) failed"
    logger.info("mask_well_area (manual) test passed. Masked image shape: %s, binary mask shape: %s", result_img_manual.shape, binary_mask_manual.shape)
    
    # Test detect_black_mask_artifact
    black_corner_img = np.full((100, 100), 128, dtype=np.uint8)
    black_corner_img[0:30, 0:30] = 0 
    temp_img_path = "temp_black_corner_test.png"
    cv2.imwrite(temp_img_path, black_corner_img)

    is_artifact = detect_black_mask_artifact(temp_img_path, min_area_ratio=0.05)
    assert is_artifact is True, "Black mask artifact detection (corner) failed"
    logger.info("Black mask artifact detection (corner) test passed.")
    os.remove(temp_img_path)