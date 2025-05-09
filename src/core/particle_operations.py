# src/core/particle_operations.py
import cv2
import numpy as np
import os
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging

from src.utils.common_exceptions import ImageProcessingError, DataValidationError, ConfigError
from .image_io_operations import save_image # For saving cropped particles
from .image_enhancement import apply_threshold # For detect_cropped_particle_on_boundary
from .region_props_calculator import find_contours # For detect_cropped_particle_on_boundary

logger = logging.getLogger(__name__)

class ParticleCropper:
    """
    Handles cropping of single particles from a larger image based on configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the cropper with configuration.
        Config keys expected:
            'mode': 'bounding_box' or 'centroid_square'.
            'square_size': Size for centroid_square mode.
            'bbox_column_mapping': Mappings for bounding_box mode keys (x,y,w,h).
        """
        self.crop_mode = config.get('mode', 'centroid_square')
        self.square_crop_size = config.get('square_size', 128)
        self.bbox_keys = config.get('bbox_column_mapping',
                                    {'x': 'bounding_x', 'y': 'bounding_y',
                                     'w': 'bounding_w', 'h': 'bounding_h'})
        
        if self.crop_mode not in ['bounding_box', 'centroid_square']:
            raise ConfigError(f"Invalid crop_mode: '{self.crop_mode}'. Must be 'bounding_box' or 'centroid_square'.")
        if self.crop_mode == 'centroid_square' and self.square_crop_size <= 0:
            raise ConfigError("square_crop_size must be positive for 'centroid_square' mode.")

    def crop_single_particle_from_image(
        self,
        main_image_color: np.ndarray, # Full original color image
        main_image_gray: np.ndarray,  # Full original grayscale image
        particle_data: pd.Series,     # Row from particles DataFrame
        output_dir_for_crop: Optional[str] = None, # If None, crop is not saved
        crop_filename_prefix: str = "obj"
    ) -> Tuple[Optional[np.ndarray], Optional[str]]: # Returns (grayscale_crop_array, path_to_saved_color_crop_if_any)
        """
        Calculates crop boundaries, extracts crop, optionally saves it,
        and returns the grayscale crop array and its saved path (if any).

        Args:
            main_image_color: The main color image from which to crop.
            main_image_gray: The main grayscale image from which to crop.
            particle_data: A pandas Series containing data for one particle.
                           Must include 'X' and 'Y' for centroid.
                           For 'bounding_box' mode, also needs bbox_keys.
            output_dir_for_crop: Directory to save the color crop. If None, not saved.
            crop_filename_prefix: Prefix for the saved crop image filename.

        Returns:
            A tuple: (cropped_gray_array, path_to_saved_color_crop).
            The path will be None if output_dir_for_crop is None or saving fails.
            The array will be None if cropping fails.
        """
        particle_id_info = f"(idx={particle_data.name}, X={particle_data.get('X', 'N/A')}, Y={particle_data.get('Y', 'N/A')})"
        
        img_h, img_w = main_image_gray.shape
        saved_crop_path: Optional[str] = None

        try:
            if 'X' not in particle_data or 'Y' not in particle_data:
                raise DataValidationError(f"'X' and 'Y' centroid coordinates are required in particle_data {particle_id_info}.")
            cx, cy = int(particle_data['X']), int(particle_data['Y'])

            if self.crop_mode == 'bounding_box':
                required_bbox_keys = [self.bbox_keys['x'], self.bbox_keys['y'], self.bbox_keys['w'], self.bbox_keys['h']]
                if not all(key in particle_data for key in required_bbox_keys):
                    missing = set(required_bbox_keys) - set(particle_data.keys())
                    raise DataValidationError(f"Missing bounding box keys {missing} for 'bounding_box' mode {particle_id_info}.")
                
                bx, by, bw, bh = (int(particle_data[k]) for k in required_bbox_keys)
                if bw <= 0 or bh <= 0:
                    logger.debug(f"Skipping crop {particle_id_info}: Invalid bounding box W={bw}, H={bh}.")
                    return None, None
                x1_raw, y1_raw = bx, by
                x2_raw, y2_raw = bx + bw, by + bh
                required_w, required_h = bw, bh
            
            elif self.crop_mode == 'centroid_square':
                half_size = self.square_crop_size // 2
                x1_raw = cx - half_size
                y1_raw = cy - half_size
                x2_raw = x1_raw + self.square_crop_size
                y2_raw = y1_raw + self.square_crop_size
                required_w, required_h = self.square_crop_size, self.square_crop_size
            
            else: # Should be caught by __init__, but defensive
                raise RuntimeError(f"Internal error: Unhandled crop_mode '{self.crop_mode}'.")

            # Clamp coordinates to image boundaries
            x1_clamped = max(0, x1_raw)
            y1_clamped = max(0, y1_raw)
            x2_clamped = min(img_w, x2_raw)
            y2_clamped = min(img_h, y2_raw)

            actual_w, actual_h = x2_clamped - x1_clamped, y2_clamped - y1_clamped

            # Check if the clamped crop is too small or doesn't meet requirements
            # (e.g., if centroid is too close to edge for a full square crop)
            min_dim_threshold = 1 # Or a more meaningful threshold like self.square_crop_size // 4
            if actual_w < min_dim_threshold or actual_h < min_dim_threshold or \
               (self.crop_mode == 'centroid_square' and (actual_w < required_w or actual_h < required_h)): # For square mode, we often want the full size
                logger.debug(f"Skipping crop {particle_id_info}: Clamped crop dimensions ({actual_w}x{actual_h}) "
                             f"are too small or do not meet required size ({required_w}x{required_h}) for mode '{self.crop_mode}'.")
                return None, None

            crop_img_gray = main_image_gray[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
            
            if crop_img_gray.size == 0:
                logger.debug(f"Skipping crop {particle_id_info}: Resulting grayscale crop is empty.")
                return None, None

            if output_dir_for_crop:
                crop_img_color = main_image_color[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                if crop_img_color.size == 0:
                     logger.warning(f"Color crop for {particle_id_info} is empty, but gray crop was not. Saving gray shape.")
                     # Potentially save the gray crop or skip saving color
                
                filename = f"{crop_filename_prefix}_{cx}_{cy}.png"
                saved_crop_path = save_image(crop_img_color, output_dir_for_crop, filename)
                if not saved_crop_path:
                    logger.error(f"Failed to save color crop for {particle_id_info} to {output_dir_for_crop}/{filename}.")
                    # Decide if this is critical: return None, None or just None for path
                    return crop_img_gray, None 
            
            return crop_img_gray, saved_crop_path

        except DataValidationError as dve:
            logger.error(f"Data validation error during cropping {particle_id_info}: {dve}")
            raise # Re-raise to be caught by calling stage
        except Exception as e:
            logger.exception(f"Unexpected error during particle cropping {particle_id_info}: {e}")
            return None, None # For unexpected errors, return None

def detect_cropped_particle_on_boundary(
    image_gray_crop: np.ndarray, # The cropped image of the particle
    image_path_for_log: str # Path of the cropped image, for logging
    ) -> bool:
    """
    Detects if the main bright particle in a CROPPED image is touching the image boundary.
    Uses Otsu's thresholding and connected components.
    Adapted from wellscanner_util_module.detect_cropped_particle.

    Args:
        image_gray_crop (np.ndarray): Grayscale image of the cropped particle.
        image_path_for_log (str): Path of the (cropped) image, for logging purposes.

    Returns:
        bool: True if the main particle is cropped by the boundary of this image_gray_crop.
    """
    base_filename = os.path.basename(image_path_for_log)
    logger.debug(f"Detecting boundary cropping for: {base_filename}")

    if image_gray_crop is None:
        logger.warning(f"Cannot detect boundary cropping for None image ({base_filename}).")
        return False # Or raise error
    
    h, w = image_gray_crop.shape
    if h < 3 or w < 3: # Arbitrary small size check
        logger.warning(f"Image {base_filename} is too small ({h}x{w}) for reliable boundary crop detection.")
        return False # Consider it not cropped if too small to analyze

    try:
        # Otsu's thresholding
        # Adjust threshold value calculation to prevent issues with nearly blank images
        if np.all(image_gray_crop == image_gray_crop[0,0]): # Uniform image
            logger.debug(f"Image {base_filename} is uniform. Assuming not cropped by boundary.")
            return False

        otsu_thresh_val, binary_mask = cv2.threshold(image_gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug(f"Otsu threshold for {base_filename}: {otsu_thresh_val}")

        # Handle cases where Otsu might give extreme values (e.g., very dark or very bright images)
        # These might indicate no clear foreground/background, so assume not cropped.
        if otsu_thresh_val <= 1 or otsu_thresh_val >= 254:
            logger.warning(f"Otsu threshold is extreme ({otsu_thresh_val}) for {base_filename}. Treating as not cropped by boundary.")
            return False

        # Find connected components in the binary mask
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        if num_labels <= 1: # Only background component
            logger.debug(f"No foreground components found in {base_filename} after Otsu. Assuming not cropped.")
            return False

        # Find the largest non-background component (assumed to be the particle)
        # stats[0] is background. Areas are in stats[:, cv2.CC_STAT_AREA]
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0: # Should be caught by num_labels <= 1, but defensive
             logger.debug(f"No non-background components with stats in {base_filename}.")
             return False
        
        largest_label_idx = np.argmax(areas) + 1 # Add 1 because we sliced off background
        
        # Get bounding box of the largest component
        left = stats[largest_label_idx, cv2.CC_STAT_LEFT]
        top = stats[largest_label_idx, cv2.CC_STAT_TOP]
        width_comp = stats[largest_label_idx, cv2.CC_STAT_WIDTH]
        height_comp = stats[largest_label_idx, cv2.CC_STAT_HEIGHT]
        area_comp = stats[largest_label_idx, cv2.CC_STAT_AREA]

        logger.debug(f"Largest component in {base_filename} (L{largest_label_idx}): "
                     f"Area={area_comp}, BBox=({left},{top},{width_comp},{height_comp})")

        # Check if the bounding box of the largest component touches any of the image_gray_crop's borders
        touches_top_border = (top == 0)
        touches_bottom_border = (top + height_comp == h)
        touches_left_border = (left == 0)
        touches_right_border = (left + width_comp == w)

        is_cropped_by_boundary = touches_top_border or touches_bottom_border or \
                                 touches_left_border or touches_right_border
        
        if is_cropped_by_boundary:
            logger.debug(f"Particle in {base_filename} touches the crop's boundary. Marked as BOUNDARY_CROPPED.")
        else:
            logger.debug(f"Particle in {base_filename} does not touch the crop's boundary. Marked as NOT BOUNDARY_CROPPED.")
        
        return is_cropped_by_boundary

    except Exception as e:
        logger.exception(f"Error during boundary crop detection for {base_filename}: {e}")
        return False # Conservative: assume not cropped if error occurs

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    # Dummy main images
    main_h, main_w = 200, 300
    main_color_img = np.random.randint(0, 256, (main_h, main_w, 3), dtype=np.uint8)
    main_gray_img = cv2.cvtColor(main_color_img, cv2.COLOR_BGR2GRAY)

    # Dummy particle data
    particles = pd.DataFrame([
        {'X': 50, 'Y': 50, 'particle_id': 1, 'bounding_x': 40, 'bounding_y': 40, 'bounding_w': 20, 'bounding_h': 20}, # Center
        {'X': 10, 'Y': 10, 'particle_id': 2}, # Near edge for centroid crop
        {'X': 150, 'Y': 150, 'particle_id': 3, 'bounding_x': 280, 'bounding_y': 180, 'bounding_w': 30, 'bounding_h': 30}, # Near edge for bbox
    ])

    # --- Test ParticleCropper ---
    cropper_config_square = {'mode': 'centroid_square', 'square_size': 64}
    square_cropper = ParticleCropper(cropper_config_square)

    cropper_config_bbox = {
        'mode': 'bounding_box',
        'bbox_column_mapping': {'x': 'bounding_x', 'y': 'bounding_y', 'w': 'bounding_w', 'h': 'bounding_h'}
    }
    bbox_cropper = ParticleCropper(cropper_config_bbox)
    
    temp_crop_output_dir = "temp_particle_crops_test"

    logger.info("\n--- Testing Centroid Square Cropper ---")
    for idx, p_row in particles.iterrows():
        gray_crop, saved_path = square_cropper.crop_single_particle_from_image(
            main_color_img, main_gray_img, p_row, temp_crop_output_dir, f"sq_crop_p{p_row['particle_id']}"
        )
        if gray_crop is not None:
            logger.info(f"Centroid Square Crop for particle {p_row['particle_id']}: shape {gray_crop.shape}, saved to {saved_path}")
            assert gray_crop.shape[0] <= cropper_config_square['square_size']
            assert gray_crop.shape[1] <= cropper_config_square['square_size']
            if saved_path: assert os.path.exists(saved_path)
        else:
            logger.warning(f"Centroid Square Crop FAILED for particle {p_row['particle_id']}")


    logger.info("\n--- Testing Bounding Box Cropper ---")
    for idx, p_row in particles.iterrows():
        gray_crop, saved_path = bbox_cropper.crop_single_particle_from_image(
            main_color_img, main_gray_img, p_row, temp_crop_output_dir, f"bbox_crop_p{p_row['particle_id']}"
        )
        if gray_crop is not None:
            logger.info(f"Bounding Box Crop for particle {p_row['particle_id']}: shape {gray_crop.shape}, saved to {saved_path}")
            assert gray_crop.shape[0] == p_row.get('bounding_h',0) or p_row.get('bounding_x',0)+p_row.get('bounding_w',0) > main_w or p_row.get('bounding_y',0)+p_row.get('bounding_h',0) > main_h # handles edge cases
            if saved_path: assert os.path.exists(saved_path)

        else:
            logger.warning(f"Bounding Box Crop FAILED for particle {p_row['particle_id']}")


    # --- Test detect_cropped_particle_on_boundary ---
    logger.info("\n--- Testing Boundary Crop Detection ---")
    # Create a crop that IS touching boundary
    crop_touching_boundary = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(crop_touching_boundary, (0, 10), (30, 40), 255, -1) # Touches left
    is_touching = detect_cropped_particle_on_boundary(crop_touching_boundary, "test_crop_touching.png")
    assert is_touching is True, "Boundary crop (touching) detection failed"
    logger.info("Boundary crop (touching) detection PASSED.")

    # Create a crop that is NOT touching boundary
    crop_not_touching = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(crop_not_touching, (10, 10), (40, 40), 255, -1) # Centered
    is_not_touching = detect_cropped_particle_on_boundary(crop_not_touching, "test_crop_not_touching.png")
    assert is_not_touching is False, "Boundary crop (not touching) detection failed"
    logger.info("Boundary crop (not touching) detection PASSED.")
    
    # Clean up saved crops
    if os.path.exists(temp_crop_output_dir):
        for f in os.listdir(temp_crop_output_dir):
            os.remove(os.path.join(temp_crop_output_dir, f))
        os.rmdir(temp_crop_output_dir)
        logger.info(f"Cleaned up {temp_crop_output_dir}")