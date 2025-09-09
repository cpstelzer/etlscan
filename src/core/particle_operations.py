# src/core/particle_operations.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import skimage.util
import shutil
import os

from src.utils.common_exceptions import ConfigError, DataValidationError
from src.core.segmentation import generate_binary_mask_from_float_image

logger = logging.getLogger(__name__)

def crop_rotated(
    image: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[int, int],
    angle: float,
    padding: int = 10
) -> np.ndarray:
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

class ParticleCropper:
    def __init__(self, crop_config: Dict[str, Any]):
        self.mode = crop_config.get('mode', 'rotational_rectangle')
        if self.mode not in ['rotational_rectangle', 'centroid_square', 'bounding_box']:
            raise ConfigError(f"Invalid ParticleCropper mode: {self.mode}")
        
        # --- MODIFICATION: Load all new parameters from config ---
        self.rotational_padding = int(crop_config.get('rotational_padding', 10))
        
        self.square_size = int(crop_config.get('square_size', 150))
        self.half_size = self.square_size // 2
        self.centroid_pre_crop_padding = int(crop_config.get('centroid_pre_crop_padding', 20))

        self.bbox_padding = int(crop_config.get('bbox_padding', 15))
        # Keep column mapping for flexibility, though we assume standard names.
        self.bbox_column_mapping = crop_config.get('bbox_column_mapping', {
            'x': 'bounding_x', 'y': 'bounding_y', 'w': 'bounding_w', 'h': 'bounding_h'
        })

    def crop_single_particle_from_image(
        self,
        full_color_image: np.ndarray,
        full_gray_image: np.ndarray,
        particle_data: pd.Series,
        output_dir_for_crop: Optional[str] = None,
        crop_filename_prefix: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        
        # --- MODIFICATION: Refactored logic for each mode ---
        if self.mode == 'centroid_square':
            try:
                cx, cy = int(particle_data['X']), int(particle_data['Y'])
                h_img, w_img = full_gray_image.shape
                
                # Use bounding box from S1 to create a safe, padded pre-crop area
                bb_x = int(particle_data.get('bounding_x', cx - self.half_size))
                bb_y = int(particle_data.get('bounding_y', cy - self.half_size))
                bb_w = int(particle_data.get('bounding_w', self.square_size))
                bb_h = int(particle_data.get('bounding_h', self.square_size))

                # Apply your suggested padding to the bounding box
                pad = self.centroid_pre_crop_padding
                pre_crop_x1 = max(0, bb_x - pad)
                pre_crop_y1 = max(0, bb_y - pad)
                pre_crop_x2 = min(w_img, bb_x + bb_w + pad)
                pre_crop_y2 = min(h_img, bb_y + bb_h + pad)
                
                sub_image_gray = full_gray_image[pre_crop_y1:pre_crop_y2, pre_crop_x1:pre_crop_x2]
                
                # Find the centroid's position *relative to the new sub-image*
                cx_relative = cx - pre_crop_x1
                cy_relative = cy - pre_crop_y1
                
                # Perform the final, precise square crop within this safer sub-image
                h_sub, w_sub = sub_image_gray.shape
                x1 = max(0, cx_relative - self.half_size)
                y1 = max(0, cy_relative - self.half_size)
                x2 = min(w_sub, cx_relative + self.half_size)
                y2 = min(h_sub, cy_relative + self.half_size)
                
                cropped_gray = sub_image_gray[y1:y2, x1:x2]

            except (KeyError, ValueError) as e:
                logger.error(f"Missing/invalid data for centroid_square crop: {e}")
                return None, None
        
        elif self.mode == 'rotational_rectangle':
            try:
                center = (particle_data['X'], particle_data['Y'])
                width = particle_data['rot_rect_width']
                height = particle_data['rot_rect_height']
                angle = particle_data['rot_rect_angle']

                if width > height:
                    width, height = height, width
                    angle += 90
                
                size = (int(width), int(height))
                cropped_gray = crop_rotated(full_gray_image, center, size, angle, self.rotational_padding)

            except (KeyError, ValueError) as e:
                logger.error(f"Missing/invalid data for rotational_rectangle crop: {e}")
                return None, None
        
        elif self.mode == 'bounding_box':
            try:
                x_col, y_col = self.bbox_column_mapping['x'], self.bbox_column_mapping['y']
                w_col, h_col = self.bbox_column_mapping['w'], self.bbox_column_mapping['h']
                x, y = int(particle_data[x_col]), int(particle_data[y_col])
                w, h = int(particle_data[w_col]), int(particle_data[h_col])
                
                # Apply configurable padding to the simple bounding box
                pad = self.bbox_padding
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(full_gray_image.shape[1], x + w + pad)
                y2 = min(full_gray_image.shape[0], y + h + pad)
                
                cropped_gray = full_gray_image[y1:y2, x1:x2]

            except (KeyError, ValueError) as e:
                logger.error(f"Missing/invalid data for bounding_box crop: {e}")
                return None, None
        
        else:
            return None, None
        # --- END OF MODIFICATION ---

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

    if (binary_mask[0, :].any() or binary_mask[-1, :].any() or
        binary_mask[:, 0].any() or binary_mask[:, -1].any()):
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