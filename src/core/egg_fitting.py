# src/core/egg_fitting.py
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Tuple, Any

logger = logging.getLogger(__name__)

def find_largest_inscribed_circle(mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Finds the largest inscribed circle in a binary mask.

    Args:
        mask (np.ndarray): Binary mask (uint8).

    Returns:
        Optional[Tuple[int, int, int]]: (center_x, center_y, radius), or None if no circle found.
    """
    if mask is None or np.sum(mask) == 0:
        return None

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    radius = int(max_val)
    center_x, center_y = int(max_loc[0]), int(max_loc[1])

    if radius > 0:
        return center_x, center_y, radius
    return None

def fit_two_eggs_in_area(
    egg_area_coords: np.ndarray,
    image_shape: Tuple[int, int]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Fits two largest possible non-overlapping circles (eggs) within the given egg_area_coords.

    Args:
        egg_area_coords (np.ndarray): Nx2 array of (row, col) coordinates defining the egg area.
        image_shape (Tuple[int, int]): Shape (height, width) of the original image for creating masks.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        Dictionaries for egg1 and egg2, each with 'center_x', 'center_y', 'radius', 'area'.
        Returns (None, None) if fitting fails.
    """
    if egg_area_coords is None or egg_area_coords.shape[0] == 0:
        logger.warning("No egg area coordinates provided for fitting eggs.")
        return None, None

    current_egg_mask = np.zeros(image_shape, dtype=np.uint8)
    # Ensure coordinates are within bounds
    valid_rows = np.clip(egg_area_coords[:, 0], 0, image_shape[0] - 1)
    valid_cols = np.clip(egg_area_coords[:, 1], 0, image_shape[1] - 1)
    current_egg_mask[valid_rows, valid_cols] = 255

    egg1_data: Optional[Dict[str, Any]] = None
    egg2_data: Optional[Dict[str, Any]] = None

    # Fit first egg
    egg1_params = find_largest_inscribed_circle(current_egg_mask)
    if egg1_params:
        cx1, cy1, r1 = egg1_params
        egg1_data = {'center_x': cx1, 'center_y': cy1, 'radius': r1, 'area': np.pi * r1**2}
        logger.debug(f"Fitted egg 1: Center=({cx1},{cy1}), Radius={r1}")

        # Create a mutable copy of the mask to remove the first egg
        mask_for_egg2 = current_egg_mask.copy()
        cv2.circle(mask_for_egg2, (cx1, cy1), r1, 0, -1) # Remove first egg area

        # Fit second egg
        egg2_params = find_largest_inscribed_circle(mask_for_egg2)
        if egg2_params:
            cx2, cy2, r2 = egg2_params
            egg2_data = {'center_x': cx2, 'center_y': cy2, 'radius': r2, 'area': np.pi * r2**2}
            logger.debug(f"Fitted egg 2: Center=({cx2},{cy2}), Radius={r2}")
        else:
            logger.warning("Could not fit second egg after removing the first.")
    else:
        logger.warning("Could not fit the first egg.")

    return egg1_data, egg2_data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Test case
    img_h, img_w = 200, 300
    dummy_egg_area_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # Create a large area that could contain two circles
    cv2.ellipse(dummy_egg_area_mask, (100, 100), (60, 30), 0, 0, 360, 255, -1)
    cv2.ellipse(dummy_egg_area_mask, (180, 100), (50, 25), 0, 0, 360, 255, -1) # Overlap a bit

    coords = np.array(np.where(dummy_egg_area_mask > 0)).T # Get (row, col)

    e1, e2 = fit_two_eggs_in_area(coords, (img_h, img_w))

    if e1: print(f"Egg 1: {e1}")
    if e2: print(f"Egg 2: {e2}")

    # Visualize for testing
    vis_img = cv2.cvtColor(dummy_egg_area_mask, cv2.COLOR_GRAY2BGR)
    if e1: cv2.circle(vis_img, (e1['center_x'], e1['center_y']), e1['radius'], (0, 255, 0), 1)
    if e2: cv2.circle(vis_img, (e2['center_x'], e2['center_y']), e2['radius'], (0, 0, 255), 1)

    # cv2.imshow("Egg Fitting Test", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    logger.info("Egg fitting test complete. Check logs/visual output if enabled.")