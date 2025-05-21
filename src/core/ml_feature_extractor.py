# src/core/ml_feature_extractor.py
import skimage.io
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.util
import skimage.draw 
import numpy as np
import os
import logging
from scipy import ndimage as ndi
from typing import Optional, Dict, Any, Tuple, List

from src.utils.common_exceptions import ConfigError # Added for raising config error

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_NAMES = [
    'main_area', 'aspect_ratio', 'solidity', 'extent',
    'posterior_area_ratio', 
    'total_egg_area', 'egg_ecc', 
    'mean_intensity', 'num_components'
]

def extract_ml_features(
    image_path: str,
    params: Dict[str, Any], 
    image_array_gray: Optional[np.ndarray] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[Any]]:
    features: Dict[str, Any] = {key: 0.0 for key in DEFAULT_FEATURE_NAMES} 
    features['posterior_area_ratio'] = 0.5 
    features['egg_ecc'] = 0.0 
    features['total_egg_area'] = 0.0
    features['num_components'] = 0 

    main_props_obj = None
    egg_area_props_obj = None
    
    try:
        if image_array_gray is None:
            image_gray = skimage.io.imread(image_path, as_gray=True)
        else:
            image_gray = image_array_gray.copy() 

        img_gray_input_shape = image_gray.shape 

        if not np.issubdtype(image_gray.dtype, np.floating):
            image_gray = skimage.util.img_as_float(image_gray) # Converts to 0-1 float
        
        if image_gray.max() < 1e-9: 
            logger.debug(f"Image {os.path.basename(image_path)} is empty or near-empty. No features extracted.")
            return None, None, None

        blurred_image = skimage.filters.gaussian(image_gray, sigma=params.get('gaussian_sigma', 1.0))
        
        if blurred_image.min() == blurred_image.max(): 
            logger.debug(f"Image {os.path.basename(image_path)} became blank after blurring. No features.")
            return None, None, None
        
        thresholding_mode = params.get('thresholding_mode', 'otsu') 
        final_threshold_value_float: Optional[float] = None # This will be the 0-1 float threshold

        if thresholding_mode == 'otsu':
            if blurred_image.min() == blurred_image.max(): 
                logger.warning(f"Image {os.path.basename(image_path)} is uniform after blurring. Otsu thresholding cannot be applied robustly. Skipping feature extraction.")
                return None, None, None
            try:
                otsu_thresh_val_raw = skimage.filters.threshold_otsu(blurred_image)
            except ValueError: 
                logger.warning(f"Otsu threshold failed for {os.path.basename(image_path)}. Skipping feature extraction.")
                return None, None, None
            
            otsu_multiplier = params.get('otsu_threshold_multiplier', 0.9)
            final_threshold_value_float = otsu_thresh_val_raw * otsu_multiplier
            logger.debug(f"Using Otsu thresholding for {os.path.basename(image_path)}. Raw Otsu: {otsu_thresh_val_raw:.4f}, Multiplier: {otsu_multiplier}, Final threshold (0-1): {final_threshold_value_float:.4f}")

        elif thresholding_mode == 'fixed':
            fixed_thresh_val_config_int = params.get('fixed_threshold_value')
            if fixed_thresh_val_config_int is None:
                logger.error(f"Thresholding mode is 'fixed' but 'fixed_threshold_value' not provided in params for {os.path.basename(image_path)}. Cannot proceed.")
                raise ConfigError("thresholding_mode is 'fixed' but 'fixed_threshold_value' is missing from feature_extraction_params.")
            
            try:
                fixed_thresh_int = int(fixed_thresh_val_config_int)
            except ValueError:
                logger.error(f"'fixed_threshold_value' ({fixed_thresh_val_config_int}) must be an integer (0-255).")
                raise ConfigError(f"'fixed_threshold_value' ({fixed_thresh_val_config_int}) is not a valid integer.")

            if not (0 <= fixed_thresh_int <= 255):
                logger.warning(f"Fixed threshold value (config) {fixed_thresh_int} for {os.path.basename(image_path)} is outside the 0-255 range. Ensure this is intended.")
            
            final_threshold_value_float = fixed_thresh_int / 255.0 # Convert to 0-1 float
            logger.debug(f"Using fixed thresholding for {os.path.basename(image_path)}. Config value (0-255): {fixed_thresh_int}, Converted threshold (0-1): {final_threshold_value_float:.4f}")
        
        else:
            logger.error(f"Invalid thresholding_mode: '{thresholding_mode}' for {os.path.basename(image_path)}. Supported modes: 'otsu', 'fixed'.")
            raise ConfigError(f"Invalid thresholding_mode: '{thresholding_mode}' in feature_extraction_params.")

        if final_threshold_value_float is None: 
             logger.error(f"Final threshold value (float) not set for {os.path.basename(image_path)}. Aborting feature extraction for this image.")
             return None, None, None
        
        main_binary_mask = blurred_image > final_threshold_value_float
        
        main_cleaned_mask = skimage.morphology.remove_small_objects(
            main_binary_mask, min_size=params.get('min_noise_size', 50)
        )
        
        selem_closing_size = params.get('closing_selem_size', 3)
        if selem_closing_size > 0:
            selem_closing = skimage.morphology.disk(selem_closing_size)
            main_closed_mask = skimage.morphology.binary_closing(main_cleaned_mask, selem_closing)
        else:
            main_closed_mask = main_cleaned_mask
            
        main_filled_mask = ndi.binary_fill_holes(main_closed_mask)

        labeled_main_mask, num_labels = skimage.measure.label(main_filled_mask, return_num=True, connectivity=2)
        features['num_components'] = int(num_labels)

        if num_labels == 0:
            logger.debug(f"No components found in main body mask for {os.path.basename(image_path)}.")
            return features, None, None 
        
        main_props_all = skimage.measure.regionprops(labeled_main_mask, intensity_image=image_gray, cache=False)
        main_props_all = sorted(main_props_all, key=lambda x: x.area, reverse=True)
        main_props_obj = main_props_all[0] 

        features['main_area'] = float(main_props_obj.area)
        minor_axis_len = main_props_obj.minor_axis_length if hasattr(main_props_obj, 'minor_axis_length') else 0.0
        major_axis_len = main_props_obj.major_axis_length if hasattr(main_props_obj, 'major_axis_length') else 0.0
        
        if minor_axis_len > 1e-6:
            features['aspect_ratio'] = major_axis_len / minor_axis_len
        else: 
            features['aspect_ratio'] = 100.0 if major_axis_len > 1e-6 else 1.0
        
        features['solidity'] = getattr(main_props_obj, 'solidity', 0.0)
        features['extent'] = getattr(main_props_obj, 'extent', 0.0)
        features['mean_intensity'] = getattr(main_props_obj, 'mean_intensity', 0.0)
        main_orientation = getattr(main_props_obj, 'orientation', 0.0) 

        main_props_obj._axis_vector_pointing_posterior = np.array([np.cos(main_orientation), np.sin(main_orientation)]) 

        if main_props_obj.area > 0 and hasattr(main_props_obj, 'coords') and hasattr(main_props_obj, 'centroid'):
            coords_main_body = main_props_obj.coords 
            centroid_main_body = np.array(main_props_obj.centroid) 
            
            axis_vector_rc_unit = np.array([np.cos(main_orientation), np.sin(main_orientation)])
            
            relative_coords_main_body = coords_main_body - centroid_main_body
            projections_main_body = np.dot(relative_coords_main_body, axis_vector_rc_unit) 
            
            positive_projection_pixels = np.sum(projections_main_body > 0)

            if main_props_obj.area > 0:
                 features['posterior_area_ratio'] = positive_projection_pixels / main_props_obj.area

            if not (features['posterior_area_ratio'] >= 0.5): 
                main_props_obj._axis_vector_pointing_posterior = -axis_vector_rc_unit
            else:
                main_props_obj._axis_vector_pointing_posterior = axis_vector_rc_unit

        if features['main_area'] < params.get('min_significant_area', 300):
            logger.debug(f"Main object area {features['main_area']} is less than min_significant_area. "
                         f"Skipping egg detection for {os.path.basename(image_path)}.")
            return features, main_props_obj, None

        high_thresh_quantile = params.get('egg_threshold_quantile', 0.92)
        non_zero_pixels = blurred_image[blurred_image > 1e-6] 
        
        if non_zero_pixels.size == 0: high_thresh_val = 1.0 
        else: high_thresh_val = np.quantile(non_zero_pixels, high_thresh_quantile)
            
        egg_binary_mask_initial = blurred_image > high_thresh_val
        
        posterior_pixel_mask = np.zeros(img_gray_input_shape, dtype=bool)
        egg_posterior_fraction = params.get('egg_posterior_pixel_fraction', 0.45)
        if not (0 < egg_posterior_fraction < 1):
            logger.warning(f"egg_posterior_pixel_fraction ({egg_posterior_fraction}) is out of (0,1) range. Defaulting to 0.45.")
            egg_posterior_fraction = 0.45
            
        percentile_to_cut = (1.0 - egg_posterior_fraction) * 100.0 

        if main_props_obj and main_props_obj.area > 0 and hasattr(main_props_obj, 'coords'):
            coords_main_body = main_props_obj.coords
            centroid_main_body = np.array(main_props_obj.centroid)
            
            relative_coords_for_slice = coords_main_body - centroid_main_body
            projections_for_slice = np.dot(relative_coords_for_slice, main_props_obj._axis_vector_pointing_posterior)
            
            if len(projections_for_slice) > 0:
                projection_threshold_val = np.percentile(projections_for_slice, percentile_to_cut) 
                
                posterior_indices = projections_for_slice > projection_threshold_val
                posterior_coords = coords_main_body[posterior_indices]
                
                if posterior_coords.shape[0] > 0:
                    valid_rows = np.clip(posterior_coords[:, 0], 0, img_gray_input_shape[0] - 1)
                    valid_cols = np.clip(posterior_coords[:, 1], 0, img_gray_input_shape[1] - 1)
                    posterior_pixel_mask[valid_rows, valid_cols] = True
                
                main_props_obj._posterior_coords_for_egg_search = posterior_coords 
            else:
                logger.debug(f"No projections calculated for posterior slice for {os.path.basename(image_path)}.")
        else:
            logger.debug(f"Could not define posterior pixel area for {os.path.basename(image_path)} "
                         f"(main_props missing, zero area, or no coords).")

        egg_binary_mask_constrained = egg_binary_mask_initial & posterior_pixel_mask
        
        egg_labeled_mask, egg_num_labels_cand = skimage.measure.label(egg_binary_mask_constrained, return_num=True, connectivity=2)

        if egg_num_labels_cand > 0 and main_props_obj is not None:
            egg_props_candidates = skimage.measure.regionprops(egg_labeled_mask, intensity_image=image_gray, cache=False)
            
            valid_egg_candidates = []
            for p_egg_candidate in egg_props_candidates:
                is_area_ok = (params.get('egg_min_area', 50) <= p_egg_candidate.area <= params.get('egg_max_area', 5000))
                if not is_area_ok: continue

                egg_solidity = getattr(p_egg_candidate, 'solidity', 1.0)
                is_solidity_ok = (params.get('egg_min_solidity', 0.60) < egg_solidity < params.get('egg_max_solidity', 0.99))
                if not is_solidity_ok: continue

                egg_ecc = getattr(p_egg_candidate, 'eccentricity', 0.0)
                is_shape_ok = (egg_ecc > params.get('egg_min_eccentricity', 0.2))
                if not is_shape_ok: continue
                
                egg_orientation = getattr(p_egg_candidate, 'orientation', 0.0) 
                delta_angle_abs = abs(main_orientation - egg_orientation)
                acute_delta_angle = min(delta_angle_abs, np.pi - delta_angle_abs) 
                
                orientation_diff_rad = abs(acute_delta_angle - (np.pi / 2.0)) 
                orientation_tolerance_rad = np.deg2rad(params.get('orientation_diff_tolerance_deg', 60.0))
                
                is_orientation_ok = orientation_diff_rad < orientation_tolerance_rad
                if not is_orientation_ok: continue
                
                valid_egg_candidates.append(p_egg_candidate)

            if valid_egg_candidates:
                egg_area_props_obj = max(valid_egg_candidates, key=lambda p: p.area)
                features['total_egg_area'] = float(egg_area_props_obj.area)
                features['egg_ecc'] = getattr(egg_area_props_obj, 'eccentricity', 0.0)
        
        for key in features:
            if isinstance(features[key], float) and not np.isfinite(features[key]):
                logger.warning(f"Feature '{key}' is non-finite ({features[key]}) for {os.path.basename(image_path)}. Defaulting based on type.")
                if key == 'num_components': features[key] = 0
                elif key == 'egg_ecc': features[key] = 0.0
                elif key == 'posterior_area_ratio': features[key] = 0.5
                else: features[key] = 0.0
            if key == 'num_components': 
                features[key] = int(features[key])

        return features, main_props_obj, egg_area_props_obj

    except ConfigError as ce: 
        logger.error(f"Configuration error during feature extraction for {os.path.basename(image_path)}: {ce}")
        raise 
    except Exception as e:
        logger.exception(f"Error extracting ML features for {os.path.basename(image_path)}: {e}")
        return None, None, None


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)
    
    from src.utils.config_loader import get_config 
    try:
        cfg = get_config()
        ml_params = cfg.get('stage_03_ml_classification', {}).get('feature_extraction_params', {})
        if not ml_params or 'egg_posterior_pixel_fraction' not in ml_params: 
            print("ML feature extraction parameters (or egg_posterior_pixel_fraction) "
                  "not found in config. Using hardcoded defaults for test.")
            ml_params = { 
                'gaussian_sigma': 1.0, 
                'thresholding_mode': 'otsu', 
                'otsu_threshold_multiplier': 0.5, 
                'fixed_threshold_value': 70, # Example for fixed (0-255)
                'closing_selem_size': 3,
                'min_noise_size': 50, 'min_significant_area': 100, 
                'egg_posterior_pixel_fraction': 0.45, 
                'egg_threshold_quantile': 0.90, 
                'egg_min_area': 50, 'egg_max_area': 5000,
                'egg_min_solidity': 0.60, 'egg_max_solidity': 0.99,
                'egg_min_eccentricity': 0.2, 
                'orientation_diff_tolerance_deg': 60
            }
    except FileNotFoundError:
        print("Config file not found. Using hardcoded ML params for test.")
        ml_params = { 
            'gaussian_sigma': 1.0, 
            'thresholding_mode': 'otsu',
            'otsu_threshold_multiplier': 0.5,
            'fixed_threshold_value': 70, # Example for fixed (0-255)
            'closing_selem_size': 3,
            'min_noise_size': 50, 'min_significant_area': 100, 
            'egg_posterior_pixel_fraction': 0.45,
            'egg_threshold_quantile': 0.90,
            'egg_min_area': 50, 'egg_max_area': 5000,
            'egg_min_solidity': 0.60, 'egg_max_solidity': 0.99,
            'egg_min_eccentricity': 0.2,
            'orientation_diff_tolerance_deg': 60
        }

    test_image_v = np.zeros((200, 200), dtype=np.uint8)
    rr_body_v, cc_body_v = skimage.draw.ellipse(100, 100, 70, 25, rotation=np.deg2rad(10)) 
    test_image_v[rr_body_v, cc_body_v] = 150 
    
    rr_heavy, cc_heavy = skimage.draw.ellipse(100 + int(70*0.6*np.cos(np.deg2rad(10))), 
                                              100 + int(70*0.6*np.sin(np.deg2rad(10))), 
                                              15,10, rotation=np.deg2rad(10))
    test_image_v[rr_heavy, cc_heavy] = 160 

    rr_egg_v, cc_egg_v = skimage.draw.ellipse(100 + int(70*0.5*np.cos(np.deg2rad(10))) , 
                                              100 + int(70*0.5*np.sin(np.deg2rad(10))), 
                                              15, 20, rotation=np.deg2rad(10+90))
    test_image_v[rr_egg_v, cc_egg_v] = 220 

    test_image_v_float = skimage.util.img_as_float(test_image_v) 
    temp_img_filename = "temp_ml_feature_test_image_threshold_mode_int.png"
    skimage.io.imsave(temp_img_filename, skimage.util.img_as_ubyte(test_image_v_float)) # Save as ubyte
    logger.info(f"Saved {temp_img_filename} for inspection.")
    
    logger.info("\n--- Testing with Otsu thresholding mode ---")
    ml_params_otsu = ml_params.copy()
    ml_params_otsu['thresholding_mode'] = 'otsu'
    features_otsu, _, _ = extract_ml_features(
        temp_img_filename, ml_params_otsu, image_array_gray=test_image_v_float # Pass float version for processing
    )
    if features_otsu:
        print(f"Otsu mode - Main Area: {features_otsu.get('main_area')}, Egg Area: {features_otsu.get('total_egg_area')}")
    else:
        print("Otsu mode - Feature extraction failed.")

    logger.info("\n--- Testing with Fixed thresholding mode ---")
    ml_params_fixed = ml_params.copy()
    ml_params_fixed['thresholding_mode'] = 'fixed'
    # Ensure fixed_threshold_value is appropriate (0-255)
    fixed_val_to_test_int = ml_params_fixed.get('fixed_threshold_value', 70) 
    print(f"Fixed mode using fixed_threshold_value (0-255 from config): {fixed_val_to_test_int}")

    features_fixed, _, _ = extract_ml_features(
        temp_img_filename, ml_params_fixed, image_array_gray=test_image_v_float # Pass float version
    )
    if features_fixed:
        print(f"Fixed mode - Main Area: {features_fixed.get('main_area')}, Egg Area: {features_fixed.get('total_egg_area')}")
    else:
        print("Fixed mode - Feature extraction failed.")

    logger.info("\n--- Testing with Invalid thresholding mode ---")
    ml_params_invalid = ml_params.copy()
    ml_params_invalid['thresholding_mode'] = 'bad_mode'
    try:
        extract_ml_features(temp_img_filename, ml_params_invalid, image_array_gray=test_image_v_float)
    except ConfigError as e:
        print(f"Correctly caught ConfigError for invalid mode: {e}")
    except Exception as e:
        print(f"Unexpected error for invalid mode test: {e}")

    if os.path.exists(temp_img_filename):
        os.remove(temp_img_filename)