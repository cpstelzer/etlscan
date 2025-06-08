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
# cv2 import is no longer needed here if ndi is removed and segment_main_body does all cv2 work
# import cv2 
# from scipy import ndimage as ndi # Should be removed as fill_holes is now in segmentation.py
from typing import Optional, Dict, Any, Tuple, List
from skimage.measure._regionprops import RegionProperties # For type hinting

from src.utils.common_exceptions import ConfigError
from src.core.segmentation import segment_main_body_from_image_float 

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_NAMES = [
    'main_area', 'aspect_ratio', 'solidity', 'extent',
    'posterior_area_ratio', 
    'total_egg_area', 'egg_ecc', 
    'mean_intensity', 'num_components'
]

def _calculate_and_attach_posterior_attributes(
    props_obj: RegionProperties, 
    base_image_shape: Tuple[int, int], # For posterior_pixel_mask bounds
    egg_posterior_pixel_fraction_config: float
) -> Tuple[RegionProperties, float, Optional[np.ndarray]]:
    """
    Calculates posterior area ratio and identifies posterior coordinates for egg search.
    Attaches '_axis_vector_pointing_posterior' and '_posterior_coords_for_egg_search'
    to the props_obj.

    Args:
        props_obj (RegionProperties): The main properties object of the segmented body.
        base_image_shape (Tuple[int, int]): Shape of the original image (height, width).
        egg_posterior_pixel_fraction_config (float): Fraction of body length to consider posterior.

    Returns:
        Tuple[RegionProperties, float, Optional[np.ndarray]]:
        - props_obj: The input props_obj, now with posterior attributes attached.
        - posterior_area_ratio_calculated: The calculated posterior area ratio.
        - posterior_coords_for_egg_search: Coordinates of the posterior pixel slice.
    """
    posterior_area_ratio = 0.5 # Default
    posterior_coords_for_egg_search_arr: Optional[np.ndarray] = None

    if not (hasattr(props_obj, 'orientation') and 
            hasattr(props_obj, 'area') and props_obj.area > 0 and
            hasattr(props_obj, 'coords') and 
            hasattr(props_obj, 'centroid')):
        logger.debug("Props object missing attributes for posterior calculation.")
        # Attach default posterior vector if orientation is somehow missing but needed by other visualizations
        if not hasattr(props_obj, '_axis_vector_pointing_posterior'):
             props_obj._axis_vector_pointing_posterior = np.array([1.0, 0.0]) # Default (e.g. horizontal right)
        return props_obj, posterior_area_ratio, posterior_coords_for_egg_search_arr

    main_orientation = props_obj.orientation
    axis_vector_rc_unit = np.array([np.cos(main_orientation), np.sin(main_orientation)])
    
    # Determine which side of the major axis is more "posterior" by pixel count
    coords_main_body = props_obj.coords 
    centroid_main_body = np.array(props_obj.centroid) 
    relative_coords_main_body = coords_main_body - centroid_main_body
    projections_main_body = np.dot(relative_coords_main_body, axis_vector_rc_unit) 
    
    positive_projection_pixels = np.sum(projections_main_body > 0)
    if props_obj.area > 0:
         posterior_area_ratio = positive_projection_pixels / props_obj.area
    
    if not (posterior_area_ratio >= 0.5): 
        props_obj._axis_vector_pointing_posterior = -axis_vector_rc_unit
    else:
        props_obj._axis_vector_pointing_posterior = axis_vector_rc_unit

    # Define posterior pixel area for egg search
    if not (0 < egg_posterior_pixel_fraction_config < 1):
        logger.warning(f"egg_posterior_pixel_fraction ({egg_posterior_pixel_fraction_config}) is out of (0,1) range. Defaulting to 0.45 for posterior slice.")
        egg_posterior_pixel_fraction_config = 0.45
        
    percentile_to_cut = (1.0 - egg_posterior_pixel_fraction_config) * 100.0 

    relative_coords_for_slice = coords_main_body - centroid_main_body # Already calculated as relative_coords_main_body
    projections_for_slice = np.dot(relative_coords_for_slice, props_obj._axis_vector_pointing_posterior)
    
    if len(projections_for_slice) > 0:
        projection_threshold_val = np.percentile(projections_for_slice, percentile_to_cut) 
        posterior_indices = projections_for_slice > projection_threshold_val
        posterior_coords_for_egg_search_arr = coords_main_body[posterior_indices]
    else:
        logger.debug("No projections calculated for posterior slice.")

    props_obj._posterior_coords_for_egg_search = posterior_coords_for_egg_search_arr
    return props_obj, posterior_area_ratio, posterior_coords_for_egg_search_arr


def extract_ml_features(
    image_path: str,
    params: Dict[str, Any], 
    image_array_gray: Optional[np.ndarray] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[RegionProperties], Optional[RegionProperties]]:
    features: Dict[str, Any] = {key: 0.0 for key in DEFAULT_FEATURE_NAMES} 
    features['posterior_area_ratio'] = 0.5 
    features['egg_ecc'] = 0.0 
    features['total_egg_area'] = 0.0
    features['num_components'] = 0 

    main_props_obj: Optional[RegionProperties] = None # Explicitly Optional
    egg_area_props_obj: Optional[RegionProperties] = None # Explicitly Optional
    
    try:
        img_float_original: np.ndarray
        if image_array_gray is None:
            img_loaded_gray = skimage.io.imread(image_path, as_gray=True)
            if not np.issubdtype(img_loaded_gray.dtype, np.floating):
                 img_float_original = skimage.util.img_as_float(img_loaded_gray)
            else:
                 img_float_original = img_loaded_gray 
        else:
            if not np.issubdtype(image_array_gray.dtype, np.floating):
                img_float_original = skimage.util.img_as_float(image_array_gray)
            else:
                img_float_original = image_array_gray.copy()

        img_gray_input_shape = img_float_original.shape 
        
        raw_body_mask, filled_body_mask, main_props_obj_from_seg = segment_main_body_from_image_float(
            img_float_original,
            params 
        )

        if main_props_obj_from_seg is None or filled_body_mask is None:
            logger.debug(f"Main body segmentation failed or found no object for {os.path.basename(image_path)}.")
            if filled_body_mask is not None:
                 _, num_labels_temp = skimage.measure.label(filled_body_mask > 0, return_num=True, connectivity=2)
                 features['num_components'] = int(num_labels_temp)
            return features, None, None
        
        main_props_obj = main_props_obj_from_seg # Assign to the variable used throughout this function
        
        _, num_labels_final = skimage.measure.label(filled_body_mask > 0, return_num=True, connectivity=2)
        features['num_components'] = int(num_labels_final)
        
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
        
        # MODIFIED: Call utility to calculate posterior attributes
        egg_posterior_fraction = params.get('egg_posterior_pixel_fraction', 0.45)
        main_props_obj, posterior_area_ratio_calc, posterior_coords_for_search = \
            _calculate_and_attach_posterior_attributes(
                main_props_obj, 
                img_gray_input_shape,
                egg_posterior_fraction
            )
        features['posterior_area_ratio'] = posterior_area_ratio_calc
        # main_props_obj now has _axis_vector_pointing_posterior and _posterior_coords_for_egg_search

        if features['main_area'] < params.get('min_significant_area', 300):
            logger.debug(f"Main object area {features['main_area']} is less than min_significant_area. "
                         f"Skipping egg detection for {os.path.basename(image_path)}.")
            return features, main_props_obj, None # main_props_obj now has posterior vector

        # Egg detection logic
        image_for_egg_detection_float = img_float_original.copy()
        source_illumination_type_egg = params.get('source_image_illumination_type', 'darkfield')
        if source_illumination_type_egg == 'collimated':
            image_for_egg_detection_float = 1.0 - image_for_egg_detection_float

        egg_gaussian_sigma = params.get('gaussian_sigma', 1.0)
        egg_blurred_image = skimage.filters.gaussian(image_for_egg_detection_float, sigma=egg_gaussian_sigma)
        
        actual_egg_quantile = params.get('egg_threshold_quantile', 0.92) 
        non_zero_pixels_in_egg_blurred = egg_blurred_image[egg_blurred_image > 1e-6] 
        if non_zero_pixels_in_egg_blurred.size == 0:
            egg_binary_mask_initial_bool = np.zeros_like(egg_blurred_image, dtype=bool)
        else:
            egg_thresh_val = np.quantile(non_zero_pixels_in_egg_blurred, actual_egg_quantile)
            egg_binary_mask_initial_bool = egg_blurred_image > egg_thresh_val 
            logger.debug(f"Egg detection: quantile={actual_egg_quantile}, thresh_val={egg_thresh_val:.4f} (applied to blurred, potentially inverted, pre-unsharped image)")

        # Use the posterior_coords_for_search obtained from the utility
        posterior_pixel_mask_bool = np.zeros(img_gray_input_shape, dtype=bool)
        if posterior_coords_for_search is not None and posterior_coords_for_search.shape[0] > 0:
            valid_rows = np.clip(posterior_coords_for_search[:, 0].astype(int), 0, img_gray_input_shape[0] - 1)
            valid_cols = np.clip(posterior_coords_for_search[:, 1].astype(int), 0, img_gray_input_shape[1] - 1)
            posterior_pixel_mask_bool[valid_rows, valid_cols] = True
        else:
            logger.debug(f"No posterior coordinates available for egg search constraints for {os.path.basename(image_path)}.")


        egg_binary_mask_constrained_bool = egg_binary_mask_initial_bool & posterior_pixel_mask_bool
        
        egg_labeled_mask, egg_num_labels_cand = skimage.measure.label(egg_binary_mask_constrained_bool, return_num=True, connectivity=2)

        if egg_num_labels_cand > 0 and main_props_obj is not None:
            egg_props_candidates = skimage.measure.regionprops(egg_labeled_mask, intensity_image=image_for_egg_detection_float, cache=False)
            
            valid_egg_candidates = []
            main_orientation_for_egg_check = getattr(main_props_obj, 'orientation', 0.0) # Use orientation from unified props
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
                delta_angle_abs = abs(main_orientation_for_egg_check - egg_orientation) # Use main_orientation from unified props
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
        
        for key in features: # Type cleanup
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
    
    from src.utils.config_loader import get_unified_segmentation_params, get_config
    
    try:
        # Get the full unified params for testing
        main_cfg_test = get_config() # Ensure config is loaded
        ml_params_for_test = get_unified_segmentation_params(main_cfg_test)
        
        # Add S3-specific egg params to ml_params_for_test if they are not already there
        # (get_unified_segmentation_params already includes everything from feature_extraction_params)
        s3_feat_cfg = main_cfg_test.get('stage_03_ml_classification', {}).get('feature_extraction_params',{})
        for k, v in s3_feat_cfg.items(): # Ensure all S3 specific params are there
            ml_params_for_test.setdefault(k, v)


    except FileNotFoundError:
        print("Config file not found for ml_feature_extractor test. Using hardcoded ML params.")
        # Define a fallback ml_params_for_test if config loading fails
        ml_params_for_test = { 
            'source_image_illumination_type': 'darkfield', 'thresholding_mode': 'otsu', 
            'otsu_threshold_multiplier': 0.5, 'fixed_threshold_value': 70,
            'enable_unsharp_masking': False, 'unsharp_mask_sigma': 1.0, 'unsharp_mask_amount': 1.0,
            'min_noise_size': 50, 'closing_selem_size': 3, 
            'gaussian_sigma': 1.0, 'min_significant_area': 100, 
            'egg_posterior_pixel_fraction': 0.45, 'egg_threshold_quantile': 0.90, 
            'egg_min_area': 50, 'egg_max_area': 5000, 'egg_min_solidity': 0.60, 
            'egg_max_solidity': 0.99, 'egg_min_eccentricity': 0.2, 
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
    temp_img_filename = "temp_ml_feature_test_posterior_util.png"
    skimage.io.imsave(temp_img_filename, skimage.util.img_as_ubyte(test_image_v_float))
    
    logger.info(f"\n--- Testing ML feature extraction (with posterior util) ---")
    logger.info(f"Runtime params for extract_ml_features: {ml_params_for_test}")
    
    features_test, main_props_out, _ = extract_ml_features(
        temp_img_filename, ml_params_for_test, image_array_gray=test_image_v_float 
    )
    if features_test:
        print(f"  Main Area: {features_test.get('main_area')}, Egg Area: {features_test.get('total_egg_area')}")
        print(f"  Posterior Area Ratio: {features_test.get('posterior_area_ratio')}")
        assert features_test.get('total_egg_area', 0) > 0, "Expected egg detection."
        if main_props_out:
            assert hasattr(main_props_out, '_axis_vector_pointing_posterior'), "main_props_out missing _axis_vector_pointing_posterior"
            assert hasattr(main_props_out, '_posterior_coords_for_egg_search'), "main_props_out missing _posterior_coords_for_egg_search"
            logger.info(f"  Posterior vector: {main_props_out._axis_vector_pointing_posterior}")
            if main_props_out._posterior_coords_for_egg_search is not None:
                logger.info(f"  Num posterior coords for search: {main_props_out._posterior_coords_for_egg_search.shape[0]}")
    else:
        print(f"  Feature extraction failed.")

    if os.path.exists(temp_img_filename):
        os.remove(temp_img_filename)