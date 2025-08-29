# src/core/region_props_calculator.py
import cv2
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def find_contours(binary_image: np.ndarray, mode: str = "external", method: str = "simple") -> Optional[List[np.ndarray]]:
    """
    Finds contours in a binary image.

    Args:
        binary_image (np.ndarray): Input binary image (expects uint8).
        mode (str): Contour retrieval mode ('external', 'list', 'ccomp', 'tree').
        method (str): Contour approximation method ('none', 'simple', 'tc89_l1', 'tc89_kcos').

    Returns:
        List[np.ndarray]: A list of contours, or None on error.
                         Each contour is an array of (x,y) coordinates.
    """
    if binary_image is None:
        logger.error("Cannot find contours in a None image.")
        return None
    if binary_image.ndim != 2 or binary_image.dtype != np.uint8:
        logger.error(f"Contour finding requires a 2D uint8 binary image. Got shape {binary_image.shape}, dtype {binary_image.dtype}")
        return None

    cv_mode = {
        "external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST,
        "ccomp": cv2.RETR_CCOMP, "tree": cv2.RETR_TREE
    }.get(mode.lower())

    cv_method = {
        "none": cv2.CHAIN_APPROX_NONE, "simple": cv2.CHAIN_APPROX_SIMPLE,
        "tc89_l1": cv2.CHAIN_APPROX_TC89_L1, "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS
    }.get(method.lower())

    if cv_mode is None or cv_method is None:
        logger.error(f"Invalid contour mode ('{mode}') or method ('{method}').")
        return None

    try:
        contours, hierarchy = cv2.findContours(binary_image, cv_mode, cv_method)
        logger.debug(f"Found {len(contours)} contours with mode '{mode}' and method '{method}'.")
        return contours
    except Exception as e:
        logger.exception(f"Error finding contours: {e}")
        return None


def calculate_contour_measurements(contour: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Calculates various particle measurements for a given contour.
    Adapted from wellcounter_imaging_module.calculate_measurements.

    Args:
        contour (np.ndarray): A contour of a detected particle.

    Returns:
        Dict[str, Any]: A dictionary of measurements, or None if contour is invalid (e.g., M00 is 0).
        Keys: 'X', 'Y', 'area', 'perimeter', 'orientation', 'aspect_ratio',
              'solidity', 'eccentricity', 'feret_diameter',
              'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h'.
    """
    if contour is None or len(contour) < 3 : # Need at least 3 points for some calcs
        logger.debug("Contour is None or too small for measurements.")
        return None

    measurements = {}
    try:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            logger.debug("Contour M00 is 0, skipping measurements.")
            return None  # Avoid division by zero

        # Direct measures of the contour
        measurements['X'] = int(M["m10"] / M["m00"])
        measurements['Y'] = int(M["m01"] / M["m00"])
        measurements['area'] = cv2.contourArea(contour)
        measurements['perimeter'] = cv2.arcLength(contour, True)
        measurements['equivalent_diameter'] = np.sqrt(4 * measurements['area'] / np.pi) if measurements['area'] > 0 else 0
        measurements['circularity'] = (4 * np.pi * measurements['area'] / (measurements['perimeter'] ** 2)) if measurements['perimeter'] > 0 else 0
        

        # Fitting an ellipse to get orientation and eccentricity
        try:
            if len(contour) >= 5: # fitEllipse requires at least 5 points
                ellipse = cv2.fitEllipse(contour)
                measurements['orientation'] = ellipse[2] # Angle
                minor_axis = min(ellipse[1]) # ellipse[1] is (minor_axis_len, major_axis_len)
                major_axis = max(ellipse[1])
                if major_axis > 1e-6: # Avoid division by zero or invalid eccentricity
                    ecc_val = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
                    measurements['eccentricity'] = ecc_val if np.isfinite(ecc_val) else np.nan
                else:
                    measurements['eccentricity'] = np.nan
            else:
                measurements['orientation'] = np.nan
                measurements['eccentricity'] = np.nan
        except cv2.error as e:
            logger.warning(f"cv2.error during ellipse fitting (likely insufficient points for contour): {e}. Setting orientation/ecc to NaN.")
            measurements['orientation'] = np.nan
            measurements['eccentricity'] = np.nan

        # Fitting a rotated rectangle
        rot_rect = cv2.minAreaRect(contour)
        (width, height) = rot_rect[1]
        
        if height > 1e-6: # Avoid division by zero  
            measurements['aspect_ratio'] = float(width) / height if width > height else float(height) / width
        else:
            measurements['aspect_ratio'] = np.nan
        measurements['rot_rect_width'] = width
        measurements['rot_rect_height'] = height

        r_angle = rot_rect[2] # Angle in degrees
        # Adjust angle to be in [0, 180], like in an ellipse
        if width < height:
            measurements['rot_rect_angle'] = (r_angle + 90) % 180  
        else:
            measurements['rot_rect_angle'] = r_angle % 180

        measurements['rot_rect_center_x'] = rot_rect[0][0]
        measurements['rot_rect_center_y'] = rot_rect[0][1]
        measurements['rot_rect_area'] = width * height  
       
        # Fitting a bounding box
        x, y, w, h = cv2.boundingRect(contour)
        measurements['bounding_x'] = x
        measurements['bounding_y'] = y
        measurements['bounding_w'] = w
        measurements['bounding_h'] = h
        

        # Fitting a convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        measurements['solidity'] = measurements['area'] / hull_area if hull_area != 0 else np.nan

        # Fitting a minimum enclosing circle
        (center_EC, radius_EC) = cv2.minEnclosingCircle(contour)
        measurements['feret_diameter'] = 2 * radius_EC

        # Ensure all values are finite or NaN, not inf
        for k, v in measurements.items():
            if isinstance(v, (float, int)) and not np.isfinite(v):
                logger.warning(f"Measurement '{k}' is non-finite ({v}). Setting to NaN.")
                measurements[k] = np.nan
        
        return measurements
    except Exception as e:
        logger.exception(f"Error calculating measurements for a contour: {e}")
        return None

def filter_contours_by_area(contours: List[np.ndarray], min_area: float) -> List[np.ndarray]:
    """Filters a list of contours by minimum area."""
    if not contours:
        return []
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            valid_contours.append(cnt)
    logger.debug(f"Filtered contours by area > {min_area}. Kept {len(valid_contours)} out of {len(contours)}.")
    return valid_contours


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    # Create a dummy binary image
    binary_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(binary_img, (20, 20), (50, 50), 255, -1) # A solid rectangle
    cv2.circle(binary_img, (70, 70), 15, 255, -1)      # A solid circle

    # Test find_contours
    contours = find_contours(binary_img, mode="external", method="simple")
    assert contours is not None and len(contours) == 2, f"Contour finding failed or found unexpected number: {len(contours if contours else [])}"
    logger.info(f"find_contours test passed, found {len(contours)} contours.")

    if contours:
        # Test calculate_contour_measurements
        all_measurements = []
        for i, cnt in enumerate(contours):
            measurements = calculate_contour_measurements(cnt)
            assert measurements is not None, f"Measurement calculation failed for contour {i}"
            logger.info(f"Measurements for contour {i}: {measurements}")
            all_measurements.append(measurements)
        
        if all_measurements:
            df = pd.DataFrame(all_measurements)
            print("\nDataFrame of contour measurements:")
            print(df)

        # Test filter_contours_by_area
        # Rectangle area approx 30*30 = 900. Circle area approx pi*15^2 = 706
        filtered_large = filter_contours_by_area(contours, min_area=800)
        assert len(filtered_large) == 1, f"Area filter (large) failed. Expected 1, got {len(filtered_large)}"
        logger.info("Area filter (large) test passed.")

        filtered_small = filter_contours_by_area(contours, min_area=100)
        assert len(filtered_small) == 2, f"Area filter (small) failed. Expected 2, got {len(filtered_small)}"
        logger.info("Area filter (small) test passed.")

        filtered_none = filter_contours_by_area(contours, min_area=1000)
        assert len(filtered_none) == 0, f"Area filter (none) failed. Expected 0, got {len(filtered_none)}"
        logger.info("Area filter (none) test passed.")