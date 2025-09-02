# src/core/particle_operations.py
import cv2
import numpy as np
import os
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
import shutil
import skimage.util
# skimage.filters is no longer needed directly here for Otsu
# import skimage.filters 

from src.utils.common_exceptions import ImageProcessingError, DataValidationError, ConfigError
from .image_io_operations import save_image, convert_to_grayscale 
# MODIFIED: Added convert_to_grayscale import
# apply_threshold is no longer needed from image_enhancement
# from .image_enhancement import apply_threshold 
from .region_props_calculator import find_contours 
# MODIFIED: Import the new thresholding helper
from src.core.segmentation import generate_binary_mask_from_float_image


logger = logging.getLogger(__name__)

# ADDED: The new rotational cropping function with docstrings and type hints.
def crop_rotated(
    img: np.ndarray, 
    center: Tuple[float, float], 
    size: Tuple[int, int], 
    angle: float, 
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    Crops a rotated rectangular region from an image with optional padding.

    Args:
        img (np.ndarray): The source image (can be color or grayscale).
        center (Tuple[float, float]): The center of the rectangle (cx, cy).
        size (Tuple[int, int]): The size of the rectangle (width, height).
        angle (float): The rotation angle in degrees.
        padding (int): Optional padding to add to each side of the crop.

    Returns:
        Optional[np.ndarray]: The cropped rectangular region, or None on error.
    """
    try:
        padded_size = (int(size[0] + 2 * padding), int(size[1] + 2 * padding))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = img.shape[:2]
        
        # Rotate the entire source image
        img_rot = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
        
        # Extract the axis-aligned crop from the rotated image
        crop = cv2.getRectSubPix(img_rot, padded_size, center)
        return crop
    except Exception as e:
        logger.error(f"Error in crop_rotated: {e}")
        return None

class ParticleCropper:
    """
    Handles cropping of single particles from a larger image based on configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        # MODIFIED: Init logic to handle both modes and their respective parameters.
        self.crop_mode = config.get('mode', 'rotational_rectangle')
        self.square_crop_size = config.get('square_size', 128)
        self.rotational_padding = config.get('rotational_padding', 10)
        self.bbox_keys = config.get('bbox_column_mapping',
                                    {'x': 'bounding_x', 'y': 'bounding_y',
                                     'w': 'bounding_w', 'h': 'bounding_h'})
        
        if self.crop_mode not in ['bounding_box', 'centroid_square', 'rotational_rectangle']:
            raise ConfigError(f"Invalid crop_mode: '{self.crop_mode}'. Must be 'bounding_box', 'centroid_square', or 'rotational_rectangle'.")
        if self.crop_mode == 'centroid_square' and self.square_crop_size <= 0:
            raise ConfigError("square_crop_size must be positive for 'centroid_square' mode.")

    def crop_single_particle_from_image( # MODIFIED: Main logic updated for new mode
        self,
        main_image_color: np.ndarray,
        main_image_gray: np.ndarray,
        particle_data: pd.Series,
        output_dir_for_crop: Optional[str] = None,
        crop_filename_prefix: str = "obj" 
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        particle_id_info = f"(idx={particle_data.name}, X={particle_data.get('X', 'N/A')}, Y={particle_data.get('Y', 'N/A')})"
        img_h, img_w = main_image_gray.shape
        saved_crop_path: Optional[str] = None
        crop_img_color: Optional[np.ndarray] = None
        
        try:
            if 'X' not in particle_data or 'Y' not in particle_data:
                raise DataValidationError(f"'X' and 'Y' centroid coordinates are required in particle_data {particle_id_info}.")
            
            particle_centroid_x, particle_centroid_y = int(particle_data['X']), int(particle_data['Y'])

            if self.crop_mode == 'bounding_box':
                required_bbox_keys = [self.bbox_keys['x'], self.bbox_keys['y'], self.bbox_keys['w'], self.bbox_keys['h']]
                if not all(key in particle_data for key in required_bbox_keys):
                    missing = set(required_bbox_keys) - set(particle_data.keys())
                    raise DataValidationError(f"Missing bounding box keys {missing} for 'bounding_box' mode {particle_id_info}.")
                bx, by, bw, bh = (int(particle_data[k]) for k in required_bbox_keys)
                if bw <= 0 or bh <= 0:
                    logger.debug(f"Skipping crop {particle_id_info}: Invalid bounding box W={bw}, H={bh}.")
                    return None, None
                x1_raw, y1_raw = bx, by; x2_raw, y2_raw = bx + bw, by + bh
                x1_clamped = max(0, x1_raw); y1_clamped = max(0, y1_raw)
                x2_clamped = min(img_w, x2_raw); y2_clamped = min(img_h, y2_raw)
                crop_img_color = main_image_color[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

            elif self.crop_mode == 'centroid_square':
                half_size = self.square_crop_size // 2
                x1_raw = particle_centroid_x - half_size ; y1_raw = particle_centroid_y - half_size
                x2_raw = x1_raw + self.square_crop_size; y2_raw = y1_raw + self.square_crop_size
                x1_clamped = max(0, x1_raw); y1_clamped = max(0, y1_raw)
                x2_clamped = min(img_w, x2_raw); y2_clamped = min(img_h, y2_raw)
                actual_w, actual_h = x2_clamped - x1_clamped, y2_clamped - y1_clamped
                if actual_w < 1 or actual_h < 1 or actual_w < self.square_crop_size or actual_h < self.square_crop_size:
                    logger.debug(f"Skipping crop {particle_id_info}: Clamped crop dimensions ({actual_w}x{actual_h}) "
                                 f"do not meet required size ({self.square_crop_size}x{self.square_crop_size}) for mode '{self.crop_mode}'.")
                    return None, None
                crop_img_color = main_image_color[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

            # ADDED: Logic for the new rotational_rectangle mode.
            elif self.crop_mode == 'rotational_rectangle':
                required_rot_keys = ['rot_rect_width', 'rot_rect_height', 'rot_rect_angle']
                if not all(key in particle_data for key in required_rot_keys):
                    missing = set(required_rot_keys) - set(particle_data.keys())
                    raise DataValidationError(f"Missing rotational rectangle keys {missing} for 'rotational_rectangle' mode {particle_id_info}.")

                cx, cy = float(particle_data['X']), float(particle_data['Y'])
                w, h = float(particle_data['rot_rect_width']), float(particle_data['rot_rect_height'])
                angle = float(particle_data['rot_rect_angle'])

                # Standardize orientation (long side vertical)
                if w > h:
                    w, h = h, w
                    angle += 90
                
                if w <= 0 or h <= 0:
                     logger.debug(f"Skipping crop {particle_id_info}: Invalid rotational rectangle W={w}, H={h}.")
                     return None, None

                # Use the new core function to perform the crop on the color image
                crop_img_color = crop_rotated(main_image_color, (cx, cy), (int(w), int(h)), angle, padding=self.rotational_padding)

            else:
                raise RuntimeError(f"Internal error: Unhandled crop_mode '{self.crop_mode}'.")

            # Common logic for post-cropping steps
            if crop_img_color is None or crop_img_color.size == 0:
                logger.debug(f"Skipping crop {particle_id_info}: Resulting color crop is None or empty.")
                return None, None
            
            crop_img_gray = convert_to_grayscale(crop_img_color)
            if crop_img_gray is None:
                 logger.warning(f"Could not convert color crop to grayscale for {particle_id_info}.")

            if output_dir_for_crop:
                filename = f"{crop_filename_prefix}_X{particle_centroid_x}_Y{particle_centroid_y}.png"
                saved_crop_path = save_image(crop_img_color, output_dir_for_crop, filename)
                if not saved_crop_path:
                    logger.error(f"Failed to save color crop for {particle_id_info} to {output_dir_for_crop}/{filename}.")
                    return crop_img_gray, None 
            
            return crop_img_gray, saved_crop_path
            
        except DataValidationError as dve:
            logger.error(f"Data validation error during cropping {particle_id_info}: {dve}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during particle cropping {particle_id_info}: {e}")
            return None, None

def detect_cropped_particle_on_boundary(
    image_gray_crop_uint8: np.ndarray, 
    image_path_for_log: str,
    illumination_mode_of_crop: str, # 'darkfield' or 'collimated'
    # MODIFIED: Pass threshold_params dict instead of individual threshold args
    threshold_params: Dict[str, Any] 
    ) -> bool:
    """
    Detects if the main particle in a CROPPED image is touching the image boundary.
    Uses thresholding consistent with the main pipeline via a helper.
    """
    base_filename = os.path.basename(image_path_for_log)
    log_prefix = f"BoundaryCheck '{base_filename}' (Illum: {illumination_mode_of_crop}, ThreshMode: {threshold_params.get('thresholding_mode','N/A')}):"

    if image_gray_crop_uint8 is None:
        logger.warning(f"{log_prefix} Cannot detect boundary for None image.")
        return False 
    
    h, w = image_gray_crop_uint8.shape
    if h < 3 or w < 3: 
        logger.warning(f"{log_prefix} Image too small ({h}x{w}) for reliable boundary detection.")
        return False 

    try:
        img_float_for_segment = skimage.util.img_as_float(image_gray_crop_uint8)

        if illumination_mode_of_crop == "collimated":
            logger.debug(f"{log_prefix} Inverting crop for segmentation.")
            img_float_for_segment = 1.0 - img_float_for_segment
        
        # MODIFIED: Call the new helper function for thresholding
        # threshold_params dict already contains mode, otsu_multiplier, fixed_threshold_value
        binary_mask_bool = generate_binary_mask_from_float_image(
            img_float_for_segment, # Pass the processed (e.g., inverted) crop
            threshold_params
        )

        if binary_mask_bool is None:
            logger.error(f"{log_prefix} Failed to generate binary mask for boundary check via helper.")
            return False # Treat as not on boundary or handle as error

        binary_mask_uint8 = binary_mask_bool.astype(np.uint8) * 255
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)

        if num_labels <= 1: 
            logger.debug(f"{log_prefix} No foreground components found. Assuming not cropped by boundary.")
            return False

        areas = stats[1:, cv2.CC_STAT_AREA] 
        if len(areas) == 0: 
             logger.debug(f"{log_prefix} No non-background components with stats.")
             return False
        
        largest_label_idx = np.argmax(areas) + 1 
        
        left = stats[largest_label_idx, cv2.CC_STAT_LEFT]
        top = stats[largest_label_idx, cv2.CC_STAT_TOP]
        width_comp = stats[largest_label_idx, cv2.CC_STAT_WIDTH]
        height_comp = stats[largest_label_idx, cv2.CC_STAT_HEIGHT]
        
        logger.debug(f"{log_prefix} Largest component (L{largest_label_idx}): Area={stats[largest_label_idx, cv2.CC_STAT_AREA]}, BBox=({left},{top},{width_comp},{height_comp})")

        touches_top_border = (top == 0)
        touches_bottom_border = (top + height_comp == h)
        touches_left_border = (left == 0)
        touches_right_border = (left + width_comp == w)

        is_cropped_by_boundary = touches_top_border or touches_bottom_border or \
                                 touches_left_border or touches_right_border
        
        if is_cropped_by_boundary:
            logger.info(f"{log_prefix} Particle touches the crop's boundary. Marked as BOUNDARY_CROPPED.")
        else:
            logger.debug(f"{log_prefix} Particle does not touch the crop's boundary. Marked as NOT BOUNDARY_CROPPED.")
        
        return is_cropped_by_boundary

    except Exception as e:
        logger.exception(f"Error during boundary crop detection for {base_filename}: {e}")
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
    cropper_config_rot = {'mode': 'rotational_rectangle', 'rotational_padding': 15}
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