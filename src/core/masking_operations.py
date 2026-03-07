# src/core/masking_operations.py
from __future__ import annotations

import os
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from skimage.measure import label as skimage_label, regionprops

try:  # pragma: no cover - import guard depends on environment
    import cv2  # type: ignore
except Exception:  # noqa: BLE001 - propagate as optional dependency
    cv2 = None  # type: ignore


logger = logging.getLogger(__name__)


def _require_cv2(function_name: str) -> None:
    """Raise a helpful error when OpenCV is required but not available."""

    if cv2 is None:  # pragma: no cover - depends on environment
        raise ImportError(
            f"OpenCV is required for {function_name}, but it is not installed in this environment."
        )


def _connected_components_with_stats(binary_mask: np.ndarray):
    """Return connected components stats using OpenCV or a numpy fallback."""

    binary_uint8 = binary_mask.astype(np.uint8)
    if cv2 is not None:
        return cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)

    labeled = skimage_label(binary_mask, connectivity=2)
    props = regionprops(labeled)
    stats = np.zeros((len(props) + 1, 5), dtype=np.int64)
    for idx, prop in enumerate(props, start=1):
        min_row, min_col, max_row, max_col = prop.bbox
        stats[idx] = [
            int(min_col),
            int(min_row),
            int(max_col - min_col),
            int(max_row - min_row),
            int(prop.area),
        ]

    return len(props) + 1, labeled.astype(np.int32), stats, None

def create_circular_mask(image_shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
    """
    Creates a binary circular mask.
    """
    _require_cv2("create_circular_mask")
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies a binary mask to an image.
    """
    _require_cv2("apply_mask")
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

    from .region_props_calculator import find_contours  # Local import to avoid cv2 dependency at module import time

    _require_cv2("mask_well_area")
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
    min_area_ratio: float = 0.05,
    max_black_pixel_value: int = 15,
    min_edge_thickness_fraction: float = 0.2,
    ) -> bool:
    """
    Detects if an image has a large, contiguous area of near-zero intensity
    touching two adjacent borders (i.e., a corner black mask).
    Adapted from wellscanner_util_module.detect_black_mask.

    Args:
        image_path (Optional[str]): Path to the image file (used for logging and if image_gray is None).
        image_gray (Optional[np.ndarray]): Pre-loaded grayscale image. If None, loads from image_path.
        min_area_ratio (float): Minimum area ratio of black component to total image area.
        max_black_pixel_value (int): Upper intensity value (inclusive) that should be
            considered "black" for mask detection.
        min_edge_thickness_fraction (float): Minimum fraction of the image width/height
            that a corner component must span along both edges if it does not meet the
            area threshold. This recovers thin triangular masks that still occlude the
            corner despite a small area ratio.

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
        if img_to_process.ndim != 2:
            logger.warning(
                "Image %s must be grayscale for black mask detection (got shape %s).",
                os.path.basename(path_for_logging),
                img_to_process.shape,
            )
            return False

        h, w = img_to_process.shape
        total_area = float(h * w)
        if total_area == 0:
            logger.warning(f"Image {os.path.basename(path_for_logging)} has zero area.")
            return False

        threshold = int(np.clip(max_black_pixel_value, 0, 255))
        if threshold != max_black_pixel_value:
            logger.warning(
                "max_black_pixel_value %s clipped to %s for %s.",
                max_black_pixel_value,
                threshold,
                os.path.basename(path_for_logging),
            )
        logger.debug(
            "Treating intensities <= %d as black pixels for %s.",
            threshold,
            os.path.basename(path_for_logging),
        )
        binary_black_pixels_mask = (img_to_process <= threshold)

        if not np.any(binary_black_pixels_mask):
            logger.debug(f"No black pixels found in {os.path.basename(path_for_logging)}.")
            return False

        num_labels, labels_map, stats, _ = _connected_components_with_stats(binary_black_pixels_mask)

        min_pixel_area_threshold = min_area_ratio * total_area
        min_edge_fraction = float(np.clip(min_edge_thickness_fraction, 0.0, 1.0))
        logger.debug(f"Black mask check for {os.path.basename(path_for_logging)}: "
                     f"Threshold area = {min_pixel_area_threshold:.0f} pixels.")

        for label_idx in range(1, num_labels):
            left, top, width_comp, height_comp, area_comp = stats[label_idx]

            touches_top = (top == 0)
            touches_bottom = (top + height_comp == h)
            touches_left = (left == 0)
            touches_right = (left + width_comp == w)

            is_corner_component = (touches_top and touches_left) or \
                                  (touches_top and touches_right) or \
                                  (touches_bottom and touches_left) or \
                                  (touches_bottom and touches_right)

            if not is_corner_component:
                continue

            width_fraction = width_comp / float(w) if w else 0.0
            height_fraction = height_comp / float(h) if h else 0.0
            meets_area = area_comp > min_pixel_area_threshold
            meets_edge_thickness = min(width_fraction, height_fraction) >= min_edge_fraction

            if meets_area or meets_edge_thickness:
                reason = "area" if meets_area else "edge thickness"
                logger.debug(
                    "Component L%s (area=%s, width_frac=%.3f, height_frac=%.3f) meets %s criteria. Black mask DETECTED.",
                    label_idx,
                    area_comp,
                    width_fraction,
                    height_fraction,
                    reason,
                )
                return True

        logger.debug(f"No qualifying black mask component found in {os.path.basename(path_for_logging)}. Black mask NOT detected.")
        return False
    except Exception as e:
        logger.exception(f"Error during black mask detection for {path_for_logging}: {e}")
        return False
