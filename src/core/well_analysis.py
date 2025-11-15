# src/core/well_analysis.py
import cv2
import numpy as np
import skimage.measure
import skimage.io
import logging
import os 
import shutil
from typing import Optional, Tuple, Dict, Any
from skimage.exposure import rescale_intensity # For a potential adjustment method

from src.utils.common_exceptions import ImageProcessingError, ConfigError, DataValidationError
from src.core.region_props_calculator import find_contours 

logger = logging.getLogger(__name__)

def create_well_mask_black_plate(
    image_gray_uint8: np.ndarray,
    config_well_mask: Dict[str, Any] 
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Creates a binary mask of the well area from an image with a black plate background.
    """
    if image_gray_uint8 is None or image_gray_uint8.ndim != 2:
        logger.error("create_well_mask_black_plate: Invalid input image.")
        return None, None

    low_thresh = config_well_mask.get('low_threshold_value', 5)
    min_diameter = config_well_mask.get('min_expected_diameter_px', 100) 
    max_diameter = config_well_mask.get('max_expected_diameter_px', 4000) 
    min_circularity_thresh = config_well_mask.get('min_circularity', 0.85) 

    _, threshold_img = cv2.threshold(image_gray_uint8, low_thresh, 255, cv2.THRESH_BINARY)

    # --- START: MODIFICATION ---
    # Apply a morphological opening to remove small noise and protrusions from the
    # well's border before finding the contour. This ensures a smooth perimeter
    # for an accurate circularity calculation.
    # 1. Opening: Removes small noise and protrusions from the well's border.
    # 2. Closing: Fills small holes and gaps in the well's border.
    # This combination produces a much smoother contour for accurate circularity calculation.
    kernel_size = 9 # Using a slightly larger kernel for a stronger effect
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_threshold_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Find contours on the cleaned binary image
    contours = find_contours(cleaned_threshold_img, mode="external", method="simple")
    # --- END: MODIFICATION ---

    if not contours:
        logger.warning("create_well_mask_black_plate: No contours found after low thresholding.")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        logger.warning("create_well_mask_black_plate: Largest contour has zero perimeter.")
        return None, None

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity < min_circularity_thresh:
        logger.warning(f"create_well_mask_black_plate: Largest contour circularity {circularity:.3f} is below threshold {min_circularity_thresh}.")
        return None, None

    (x_f, y_f), radius_f = cv2.minEnclosingCircle(largest_contour)
    center_x, center_y, radius_px = int(x_f), int(y_f), int(radius_f)
    diameter_px = 2 * radius_px

    if not (min_diameter <= diameter_px <= max_diameter):
        logger.warning(f"create_well_mask_black_plate: Detected diameter {diameter_px}px is outside expected range [{min_diameter}-{max_diameter}]px.")
        return None, None

    well_mask = np.zeros_like(image_gray_uint8, dtype=np.uint8)
    cv2.drawContours(well_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    well_properties = {
        'center_x': center_x, 'center_y': center_y, 'radius_px': radius_px,
        'diameter_px': diameter_px, 'area_px': area, 'circularity': circularity
    }
    logger.info(f"create_well_mask_black_plate: Well identified. Diameter: {diameter_px}px, Circularity: {circularity:.3f}")
    return well_mask, well_properties

def calculate_well_brightness_peak(
    image_gray_uint8: np.ndarray,
    well_mask_uint8: np.ndarray, 
    min_intensity_for_peak_search: int = 75, 
    percentile_for_proxy: int = 75 
) -> Optional[float]:
    """
    Calculates a proxy for the well background's brightness peak from its histogram.
    """
    if image_gray_uint8 is None or well_mask_uint8 is None:
        logger.error("calculate_well_brightness_peak: Input image or mask is None.")
        return None
    
    well_pixels = image_gray_uint8[well_mask_uint8 == 255]
    if well_pixels.size < 100: 
        logger.warning(f"calculate_well_brightness_peak: Very few pixels ({well_pixels.size}) in well mask. Peak may be unreliable.")
        if well_pixels.size == 0: return None
    
    if percentile_for_proxy < 0 or percentile_for_proxy >= 100:
        percentile_for_proxy = 75 
    
    # Ensure well_pixels is not empty before calling percentile
    if well_pixels.size == 0:
        logger.warning("calculate_well_brightness_peak: well_pixels is empty after masking. Cannot determine peak.")
        return None
        
    lower_bound_intensity = np.percentile(well_pixels, percentile_for_proxy)
    bright_well_pixels = well_pixels[well_pixels >= lower_bound_intensity]

    if bright_well_pixels.size == 0: 
        if well_pixels.size > 0 :
            peak_proxy = np.median(well_pixels)
            logger.debug(f"calculate_well_brightness_peak: Bright percentile empty, used overall median: {peak_proxy:.2f}")
            return float(peak_proxy)
        else: 
            logger.warning("calculate_well_brightness_peak: No bright pixels and no well pixels for fallback. Cannot determine peak.")
            return None

    peak_proxy = np.median(bright_well_pixels)
    logger.debug(f"calculate_well_brightness_peak: Determined peak proxy: {peak_proxy:.2f} (from pixels >= {lower_bound_intensity:.2f})")
    return float(peak_proxy)

def match_well_brightness(
    fg_img_uint8: np.ndarray,
    bg_img_uint8: np.ndarray,
    config_well_analysis: Dict[str, Any], # Contains 'well_identification', 'brightness_peak_calculation', 'brightness_matching'
    # target_avg_well_brightness: Optional[float] = None # Alternative: adjust both to a global target
    # For now, we adjust one to match the other, or both towards a midpoint/target
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusts brightness of FG and/or BG images so their well background brightness is similar.
    It prioritizes making the brighter image dimmer to match the darker one, or adjusting
    the one more deviant from a target (if target_avg_well_brightness was used).
    Currently adjusts the BG to match FG if BG is too different.

    Args:
        fg_img_uint8 (np.ndarray): Foreground image.
        bg_img_uint8 (np.ndarray): Background image.
        config_well_analysis (Dict[str, Any]): Configuration.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted FG and BG images (uint8).
    """
    cfg_well_id = config_well_analysis.get('well_identification')
    cfg_peak_calc = config_well_analysis.get('brightness_peak_calculation')
    cfg_matching = config_well_analysis.get('brightness_matching')

    if not all([cfg_well_id, cfg_peak_calc, cfg_matching]):
        logger.error("match_well_brightness: Missing required configuration sections. Returning original images.")
        return fg_img_uint8, bg_img_uint8

    fg_mask, _ = create_well_mask_black_plate(fg_img_uint8, cfg_well_id)
    bg_mask, _ = create_well_mask_black_plate(bg_img_uint8, cfg_well_id)

    if fg_mask is None or bg_mask is None:
        logger.warning("match_well_brightness: Could not create well masks for FG or BG. Returning original images.")
        return fg_img_uint8, bg_img_uint8

    fg_peak = calculate_well_brightness_peak(fg_img_uint8, fg_mask, 
                                             cfg_peak_calc.get('min_intensity_for_peak_search', 75),
                                             cfg_peak_calc.get('percentile_for_peak_proxy', 75))
    bg_peak = calculate_well_brightness_peak(bg_img_uint8, bg_mask,
                                             cfg_peak_calc.get('min_intensity_for_peak_search', 75),
                                             cfg_peak_calc.get('percentile_for_peak_proxy', 75))

    if fg_peak is None or bg_peak is None:
        logger.warning("match_well_brightness: Could not determine brightness peaks for FG or BG. Returning original images.")
        return fg_img_uint8, bg_img_uint8

    logger.debug(f"match_well_brightness: FG Peak={fg_peak:.2f}, BG Peak={bg_peak:.2f}")

    max_diff_allowed = cfg_matching.get('max_peak_difference_allowed', 15)
    target_offset = cfg_matching.get('target_peak_offset_after_shift', 3) # How much dimmer target should be
    adjustment_method = cfg_matching.get('adjustment_method', 'global_shift')

    adjusted_fg = fg_img_uint8.copy()
    adjusted_bg = bg_img_uint8.copy()
    
    peak_diff = bg_peak - fg_peak

    if abs(peak_diff) > max_diff_allowed:
        logger.info(f"match_well_brightness: Peak difference |{peak_diff:.2f}| > {max_diff_allowed}. Attempting adjustment.")
        if adjustment_method == 'global_shift':
            # Adjust the image whose peak is further from a target (e.g. FG peak, or an overall average if available)
            # For simplicity, if BG is much brighter, dim BG. If FG is much brighter, dim FG.
            # Or, always adjust BG to match FG if that's the policy.
            # Let's try adjusting BG to be closer to FG if it's the outlier.
            
            # Strategy: Adjust BG to be fg_peak - target_offset
            desired_bg_peak = fg_peak - target_offset 
            shift_amount = int(round(bg_peak - desired_bg_peak))

            if shift_amount != 0 : # Only apply if a shift is needed
                logger.info(f"match_well_brightness: Adjusting BG by {-shift_amount} to target peak ~{desired_bg_peak:.2f} (from {bg_peak:.2f})")
                
                # Create a float version for safe arithmetic, then clip and convert back
                bg_float = adjusted_bg.astype(np.float32)
                
                # Apply shift only within the well mask
                bg_float[bg_mask == 255] -= shift_amount
                
                np.clip(bg_float, 0, 255, out=bg_float) # Clip in-place
                adjusted_bg = bg_float.astype(np.uint8)

                # Verification (optional)
                # new_bg_peak = calculate_well_brightness_peak(adjusted_bg, bg_mask, 
                #                                              cfg_peak_calc.get('min_intensity_for_peak_search', 75),
                #                                              cfg_peak_calc.get('percentile_for_peak_proxy', 75))
                # logger.debug(f"match_well_brightness: New BG peak after adjustment: {new_bg_peak if new_bg_peak else 'N/A'}")
            else:
                logger.debug(f"match_well_brightness: Calculated shift is zero. No adjustment to BG needed to match FG target.")

        elif adjustment_method == 'histogram_matching': # Basic intensity rescaling for now
            logger.warning("match_well_brightness: 'histogram_matching' selected, using basic intensity rescaling.")
            # Rescale the image that is "more different" or always BG to FG.
            # Example: Rescale BG to match FG's well pixel range
            fg_well_pixels = fg_img_uint8[fg_mask == 255]
            bg_well_pixels = bg_img_uint8[bg_mask == 255]
            if fg_well_pixels.size > 0 and bg_well_pixels.size > 0:
                p_low_fg, p_high_fg = np.percentile(fg_well_pixels, [2,98])
                p_low_bg, p_high_bg = np.percentile(bg_well_pixels, [2,98])

                if (p_high_bg - p_low_bg) > 1e-3 : # Avoid division by zero if BG well is flat
                    # Create a temporary float image for BG for rescaling
                    temp_bg_float = adjusted_bg.astype(float)
                    # Rescale only the well area of BG
                    bg_well_float_pixels = temp_bg_float[bg_mask == 255]
                    
                    # Apply rescaling
                    rescaled_bg_well_pixels = rescale_intensity(bg_well_float_pixels, 
                                                                in_range=(p_low_bg, p_high_bg), 
                                                                out_range=(p_low_fg, p_high_fg))
                    rescaled_bg_well_pixels = np.clip(rescaled_bg_well_pixels, 0, 255)
                    
                    # Put rescaled pixels back
                    temp_bg_float[bg_mask == 255] = rescaled_bg_well_pixels
                    adjusted_bg = temp_bg_float.astype(np.uint8)
                    logger.info(f"match_well_brightness: Rescaled BG intensity to match FG range.")
                else:
                    logger.warning("match_well_brightness: BG well area has very low contrast, skipping rescaling.")
            else:
                logger.warning("match_well_brightness: Not enough pixels in well masks for histogram matching based rescaling.")
        else:
            logger.warning(f"match_well_brightness: Unknown adjustment method '{adjustment_method}'. No adjustment performed.")
    else:
        logger.debug(f"match_well_brightness: Peak difference |{peak_diff:.2f}| is within allowed {max_diff_allowed}. No adjustment needed.")

    return adjusted_fg, adjusted_bg


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Testing well_analysis.py (including match_well_brightness) ---")

    dummy_dir = "temp_well_analysis_test_images_match"
    os.makedirs(dummy_dir, exist_ok=True)
    
    # Create FG image
    fg_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(fg_img, (100,100), 80, 180, -1) # Well BG = 180
    cv2.circle(fg_img, (80,80), 10, 30, -1)   # Particle
    skimage.io.imsave(os.path.join(dummy_dir, "fg_img.png"), fg_img)

    # Create BG image (brighter)
    bg_img_brighter = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(bg_img_brighter, (100,100), 80, 210, -1) # Well BG = 210
    skimage.io.imsave(os.path.join(dummy_dir, "bg_img_brighter.png"), bg_img_brighter)
    
    # Create BG image (similar)
    bg_img_similar = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(bg_img_similar, (100,100), 80, 183, -1) # Well BG = 183
    skimage.io.imsave(os.path.join(dummy_dir, "bg_img_similar.png"), bg_img_similar)

    test_config_match = {
        'well_identification': {
            'low_threshold_value': 10, 'min_expected_diameter_px': 100,
            'max_expected_diameter_px': 190, 'min_circularity': 0.80
        },
        'brightness_peak_calculation': {
            'min_intensity_for_peak_search': 75, 'percentile_for_peak_proxy': 75
        },
        'brightness_matching': {
            'max_peak_difference_allowed': 10, # BG can be 10 brighter
            'adjustment_method': "global_shift",
            'target_peak_offset_after_shift': 3 # Make BG 3 units dimmer than FG
        }
    }

    logger.info("\n--- Testing match_well_brightness (BG Brighter) ---")
    adj_fg1, adj_bg1 = match_well_brightness(fg_img, bg_img_brighter, test_config_match)
    # For verification, calculate peak of adjusted_bg1
    bg_mask1, _ = create_well_mask_black_plate(adj_bg1, test_config_match['well_identification'])
    if bg_mask1 is not None:
        new_bg_peak1 = calculate_well_brightness_peak(adj_bg1, bg_mask1, 
                                                     test_config_match['brightness_peak_calculation']['min_intensity_for_peak_search'],
                                                     test_config_match['brightness_peak_calculation']['percentile_for_peak_proxy'])
        logger.info(f"Original BG peak ~210, Adjusted BG1 peak: {new_bg_peak1:.2f} (Expected ~177)")
        if new_bg_peak1: assert 170 < new_bg_peak1 < 185

    logger.info("\n--- Testing match_well_brightness (BG Similar) ---")
    adj_fg2, adj_bg2 = match_well_brightness(fg_img, bg_img_similar, test_config_match)
    # Expect no significant change in adj_bg2 from bg_img_similar
    if np.all(adj_bg2 == bg_img_similar):
        logger.info("BG similar, no adjustment made, as expected.")
    else:
        logger.error("BG similar, but adjustment was made unexpectedly.")

    shutil.rmtree(dummy_dir)
