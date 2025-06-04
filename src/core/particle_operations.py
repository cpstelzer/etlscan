# src/core/particle_operations.py
import cv2
import numpy as np
import os
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
import shutil
import skimage.util
import skimage.filters # Added for skimage.filters.threshold_otsu

from src.utils.common_exceptions import ImageProcessingError, DataValidationError, ConfigError
from .image_io_operations import save_image 
from .image_enhancement import apply_threshold 
from .region_props_calculator import find_contours 

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
        main_image_color: np.ndarray,
        main_image_gray: np.ndarray,
        particle_data: pd.Series,
        output_dir_for_crop: Optional[str] = None,
        crop_filename_prefix: str = "obj" 
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        particle_id_info = f"(idx={particle_data.name}, X={particle_data.get('X', 'N/A')}, Y={particle_data.get('Y', 'N/A')})"
        
        img_h, img_w = main_image_gray.shape
        saved_crop_path: Optional[str] = None

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
                x1_raw, y1_raw = bx, by
                x2_raw, y2_raw = bx + bw, by + bh
                required_w, required_h = bw, bh
            
            elif self.crop_mode == 'centroid_square':
                half_size = self.square_crop_size // 2
                x1_raw = particle_centroid_x - half_size 
                y1_raw = particle_centroid_y - half_size
                x2_raw = x1_raw + self.square_crop_size
                y2_raw = y1_raw + self.square_crop_size
                required_w, required_h = self.square_crop_size, self.square_crop_size
            
            else:
                raise RuntimeError(f"Internal error: Unhandled crop_mode '{self.crop_mode}'.")

            x1_clamped = max(0, x1_raw)
            y1_clamped = max(0, y1_raw)
            x2_clamped = min(img_w, x2_raw)
            y2_clamped = min(img_h, y2_raw)

            actual_w, actual_h = x2_clamped - x1_clamped, y2_clamped - y1_clamped

            min_dim_threshold = 1
            if actual_w < min_dim_threshold or actual_h < min_dim_threshold or \
               (self.crop_mode == 'centroid_square' and (actual_w < required_w or actual_h < required_h)):
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
                     logger.warning(f"Color crop for {particle_id_info} is empty, but gray crop was not.")
                
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
    image_gray_crop_uint8: np.ndarray, # Expects uint8 input
    image_path_for_log: str,
    illumination_mode_of_crop: str,
    thresholding_mode: str,
    otsu_threshold_multiplier: float,
    fixed_threshold_value_int: int # 0-255
    ) -> bool:
    """
    Detects if the main particle in a CROPPED image is touching the image boundary.
    Uses thresholding consistent with the main pipeline (e.g., extract_ml_features).
    """
    base_filename = os.path.basename(image_path_for_log)
    log_prefix = f"BoundaryCheck '{base_filename}' (Illum: {illumination_mode_of_crop}, ThreshMode: {thresholding_mode}):"

    if image_gray_crop_uint8 is None:
        logger.warning(f"{log_prefix} Cannot detect boundary for None image.")
        return False 
    
    h, w = image_gray_crop_uint8.shape
    if h < 3 or w < 3: 
        logger.warning(f"{log_prefix} Image too small ({h}x{w}) for reliable boundary detection.")
        return False 

    try:
        # 1. Convert to float [0,1] for consistent processing with skimage
        img_float_for_segment = skimage.util.img_as_float(image_gray_crop_uint8)

        # 2. Invert if collimated (operates on float image)
        if illumination_mode_of_crop == "collimated":
            logger.debug(f"{log_prefix} Inverting crop for segmentation.")
            img_float_for_segment = 1.0 - img_float_for_segment
        
        # 3. (No blurring is applied here, to keep edge precision for boundary check on small crops)

        # 4. Apply threshold based on mode (mimicking extract_ml_features)
        binary_mask_bool: np.ndarray
        if thresholding_mode == 'otsu':
            if img_float_for_segment.min() == img_float_for_segment.max():
                logger.debug(f"{log_prefix} Image is uniform. Assuming not on boundary.")
                return False
            try:
                otsu_thresh_raw_sk = skimage.filters.threshold_otsu(img_float_for_segment) # float 0-1
            except ValueError:
                logger.warning(f"{log_prefix} Skimage Otsu failed (likely uniform/empty). Assuming not on boundary.")
                return False
            
            final_thresh_float = otsu_thresh_raw_sk * otsu_threshold_multiplier
            logger.debug(f"{log_prefix} Otsu: RawSk={otsu_thresh_raw_sk:.4f}, Mult={otsu_threshold_multiplier}, FinalFloatThresh={final_thresh_float:.4f}")
            binary_mask_bool = img_float_for_segment > final_thresh_float

        elif thresholding_mode == 'fixed':
            final_thresh_float = fixed_threshold_value_int / 255.0
            logger.debug(f"{log_prefix} Fixed: ValueInt={fixed_threshold_value_int}, FinalFloatThresh={final_thresh_float:.4f}")
            binary_mask_bool = img_float_for_segment > final_thresh_float
        else:
            logger.error(f"{log_prefix} Invalid thresholding_mode: '{thresholding_mode}'.")
            return False # Or raise error

        binary_mask_uint8 = binary_mask_bool.astype(np.uint8) * 255
        
        # 5. Proceed with connected components on binary_mask_uint8
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)

        if num_labels <= 1: # Only background label
            logger.debug(f"{log_prefix} No foreground components found after thresholding. Assuming not cropped by boundary.")
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

    main_h, main_w = 200, 300
    main_color_img = np.random.randint(0, 256, (main_h, main_w, 3), dtype=np.uint8)
    main_gray_img = cv2.cvtColor(main_color_img, cv2.COLOR_BGR2GRAY)

    particles = pd.DataFrame([
        {'X': 50, 'Y': 50, 'particle_id': 1, 'bounding_x': 40, 'bounding_y': 40, 'bounding_w': 20, 'bounding_h': 20},
        {'X': 10, 'Y': 10, 'particle_id': 2}, 
        {'X': 150, 'Y': 150, 'particle_id': 3, 'bounding_x': 280, 'bounding_y': 180, 'bounding_w': 30, 'bounding_h': 30}, 
    ])

    cropper_config_square = {'mode': 'centroid_square', 'square_size': 64}
    square_cropper = ParticleCropper(cropper_config_square)
    
    temp_crop_output_dir = "temp_particle_crops_test_skimage_thresh" # New test dir name
    os.makedirs(temp_crop_output_dir, exist_ok=True)


    logger.info("\n--- Testing Boundary Crop Detection (Skimage Thresholding Logic) ---")
    
    # Example: Dark particle on bright background (simulating "collimated" original)
    crop_dark_particle_bright_bg_distinct = np.full((50, 50), 250, dtype=np.uint8) # Very bright BG
    cv2.rectangle(crop_dark_particle_bright_bg_distinct, (0, 10), (30, 40), 10, -1) # Very dark particle, touches left
    test_path1_distinct = os.path.join(temp_crop_output_dir, "dark_particle_touching_distinct_sk.png")
    cv2.imwrite(test_path1_distinct, crop_dark_particle_bright_bg_distinct)
    
    is_touching_sk_otsu = detect_cropped_particle_on_boundary(
        crop_dark_particle_bright_bg_distinct, test_path1_distinct, 
        illumination_mode_of_crop="collimated", 
        thresholding_mode="otsu",
        otsu_threshold_multiplier=0.45, # Consistent with typical S3 config
        fixed_threshold_value_int=70   # Not used for Otsu mode
    )
    assert is_touching_sk_otsu is True, "Boundary crop (distinct dark particle, collimated, skimage Otsu logic) detection failed"
    logger.info("Boundary crop (distinct dark particle, collimated, skimage Otsu logic) detection PASSED.")

    # Test with fixed thresholding
    crop_fixed_thresh_touching = np.full((50,50), 100, dtype=np.uint8)
    cv2.rectangle(crop_fixed_thresh_touching, (40,10), (49,40), 200, -1) # Bright particle touches right
    test_path_fixed = os.path.join(temp_crop_output_dir, "fixed_thresh_touching_sk.png")
    cv2.imwrite(test_path_fixed, crop_fixed_thresh_touching)

    is_touching_sk_fixed = detect_cropped_particle_on_boundary(
        crop_fixed_thresh_touching, test_path_fixed,
        illumination_mode_of_crop="darkfield", # Particle is bright
        thresholding_mode="fixed",
        otsu_threshold_multiplier=0.45, # Not used
        fixed_threshold_value_int=150 # Should segment the particle
    )
    assert is_touching_sk_fixed is True, "Boundary crop (bright particle, darkfield, skimage Fixed logic) failed"
    logger.info("Boundary crop (bright particle, darkfield, skimage Fixed logic) PASSED.")

    # Test for not touching
    crop_dark_not_touching_sk = np.full((50,50), 200, dtype=np.uint8)
    cv2.rectangle(crop_dark_not_touching_sk, (10,10), (40,40), 50, -1) # Not touching
    test_path3_sk = os.path.join(temp_crop_output_dir, "dark_particle_not_touching_sk.png")
    cv2.imwrite(test_path3_sk, crop_dark_not_touching_sk)
    is_not_touching_sk = detect_cropped_particle_on_boundary(
        crop_dark_not_touching_sk, test_path3_sk, 
        illumination_mode_of_crop="collimated",
        thresholding_mode="otsu",
        otsu_threshold_multiplier=0.45,
        fixed_threshold_value_int=70
    )
    assert is_not_touching_sk is False, "Boundary crop (dark, not touching, collimated, skimage Otsu) failed"
    logger.info("Boundary crop (dark, not touching, collimated, skimage Otsu) PASSED.")

    if os.path.exists(temp_crop_output_dir):
        shutil.rmtree(temp_crop_output_dir)
        logger.info(f"Cleaned up {temp_crop_output_dir}")