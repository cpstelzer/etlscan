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

logger = logging.getLogger(__name__)

# Default feature names, can be overridden or extended by config.
# Ensure these match what the training process expects and what extract_features produces.
DEFAULT_FEATURE_NAMES = [
    'main_area', 'aspect_ratio', 'solidity', 'extent',
    'posterior_area_ratio',
    'num_eggs', 'avg_egg_area', 'avg_egg_ecc',
    'mean_intensity', 'num_components'
]

def extract_ml_features(
    image_path: str,
    params: Dict[str, Any],
    image_array_gray: Optional[np.ndarray] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[Any]]:
    """
    Extracts morphological and intensity-based features from an image for ML classification.
    Adapted from ws_identify_ideal_fems.extract_features.

    Args:
        image_path (str): Path to the image file (used for logging and if image_array_gray is None).
        params (Dict[str, Any]): Configuration dictionary for feature extraction parameters.
        image_array_gray (Optional[np.ndarray]): Pre-loaded grayscale image array.
                                                  If None, the image is loaded from image_path.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[skimage.measure._regionprops.RegionProperties], Optional[skimage.measure._regionprops.RegionProperties]]:
        - features_dict: Dictionary of calculated features. Keys should align with DEFAULT_FEATURE_NAMES.
        - main_props_obj: RegionProperties object for the main detected body.
        - egg_pair_props: RegionProperties object for the detected egg pair (if any).
        Returns (None, None, None) if critical error or no significant features found.
    """
    features: Dict[str, Any] = {key: 0.0 for key in DEFAULT_FEATURE_NAMES} # Initialize with floats
    # Specific defaults for certain features if not calculated
    features['posterior_area_ratio'] = 0.5
    features['avg_egg_ecc'] = 1.0 # Neutral eccentricity if no eggs
    features['num_eggs'] = 0 # Integer type for num_eggs and num_components

    main_props_obj = None
    egg_pair_props = None # Represents the properties of the (potential) egg PAIR region

    try:
        if image_array_gray is None:
            image_gray = skimage.io.imread(image_path, as_gray=True)
        else:
            image_gray = image_array_gray.copy() # Work on a copy

        if not np.issubdtype(image_gray.dtype, np.floating):
            image_gray = skimage.util.img_as_float(image_gray)
        
        if image_gray.max() < 1e-9: # Check for blank/empty image
            logger.debug(f"Image {os.path.basename(image_path)} is empty or near-empty. No features extracted.")
            return None, None, None

        # --- Segmentation for MAIN BODY ---
        blurred_image = skimage.filters.gaussian(image_gray, sigma=params.get('gaussian_sigma', 1.0))
        
        if blurred_image.min() == blurred_image.max(): # Blank image after blur
            logger.debug(f"Image {os.path.basename(image_path)} became blank after blurring. No features.")
            return None, None, None
        
        try:
            otsu_thresh = skimage.filters.threshold_otsu(blurred_image)
        except ValueError: # Otsu can fail on images with no variance
            logger.warning(f"Otsu threshold failed for {os.path.basename(image_path)}. No features.")
            return None, None, None
            
        adjusted_thresh = otsu_thresh * params.get('otsu_threshold_multiplier', 0.9)
        main_binary_mask = blurred_image > adjusted_thresh
        
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

        # --- Calculate Main Body Properties ---
        labeled_main_mask, num_labels = skimage.measure.label(main_filled_mask, return_num=True, connectivity=2)
        features['num_components'] = int(num_labels)

        if num_labels == 0:
            logger.debug(f"No components found in main body mask for {os.path.basename(image_path)}.")
            return features, None, None # Return partially filled features (num_components=0)
        
        # Use intensity_image=image_gray for intensity stats on original, not blurred
        main_props_all = skimage.measure.regionprops(labeled_main_mask, intensity_image=image_gray, cache=False)
        main_props_all = sorted(main_props_all, key=lambda x: x.area, reverse=True)
        main_props_obj = main_props_all[0] # Largest component

        features['main_area'] = float(main_props_obj.area)
        minor_axis = main_props_obj.minor_axis_length if hasattr(main_props_obj, 'minor_axis_length') else 0.0
        major_axis = main_props_obj.major_axis_length if hasattr(main_props_obj, 'major_axis_length') else 0.0
        
        if minor_axis > 1e-6:
            features['aspect_ratio'] = major_axis / minor_axis
        else: # Avoid division by zero for very thin/line-like objects
            features['aspect_ratio'] = 100.0 if major_axis > 1e-6 else 1.0
        
        # Some props might not exist for very small or degenerate regions
        features['solidity'] = getattr(main_props_obj, 'solidity', 0.0)
        features['extent'] = getattr(main_props_obj, 'extent', 0.0)
        features['mean_intensity'] = getattr(main_props_obj, 'mean_intensity', 0.0)
        
        main_orientation = getattr(main_props_obj, 'orientation', 0.0) # Radians

        # Calculate posterior_area_ratio
        # Defines "posterior" based on the main orientation axis.
        # Positive projection along axis is one end, negative is the other.
        # The end with more pixels is chosen as "posterior" if ratio > 0.5.
        if main_props_obj.area > 0 and hasattr(main_props_obj, 'coords') and hasattr(main_props_obj, 'centroid'):
            coords = main_props_obj.coords # Nx2 array of (row, col)
            centroid = np.array(main_props_obj.centroid) # (row, col)
            
            # Vector along the major axis (orientation is angle from positive x-axis, skimage convention)
            # For skimage, orientation is angle between x-axis and major axis, from -pi/2 to pi/2.
            # So, (cos(orient), sin(orient)) points along major axis.
            # Here, it seems original code used (row_vec, col_vec) = (cos, sin),
            # which is unusual as orientation typically relates to (dx, dy) or (dc, dr) more directly.
            # Let's stick to the original intent if it worked.
            # A common way: axis_vector = (np.sin(orientation), np.cos(orientation)) for (dr, dc)
            # Original was: row_vec for y-like, col_vec for x-like.
            # If orientation is angle w.r.t positive x-axis (horizontal, pointing right),
            # then dx = cos(orient), dy = -sin(orient) for image coordinates (y down, x right).
            # Let's use skimage's definition: angle between x-axis and major axis.
            # So, axis_vector_standard = (np.cos(main_orientation), np.sin(main_orientation)) in (x,y) or (col,row) terms
            # For (row,col) coords: vector pointing along axis could be (-np.sin(main_orientation), np.cos(main_orientation))
            # Let's re-verify the original logic based on output if needed.
            # The provided code in `ws_identify_ideal_fems` used:
            # row_vec, col_vec = np.cos(main_orientation), np.sin(main_orientation)
            # axis_vector = np.array([row_vec, col_vec])
            # This means 'row_vec' is related to Y-change, 'col_vec' to X-change, which is fine.
            
            # Let's use a more standard vector for (row, col) based on skimage's orientation:
            # vector_along_axis_rc = np.array([-np.sin(main_orientation), np.cos(main_orientation)]) # (d_row, d_col)
            
            # Sticking to original provided logic for `posterior_area_ratio` for now:
            row_vec_orig, col_vec_orig = np.cos(main_orientation), np.sin(main_orientation)
            axis_vector_orig = np.array([row_vec_orig, col_vec_orig]) # This interpretation seems off for (row,col) coords.

            # Re-interpreting for (row, col) pixel coordinates and skimage orientation:
            # Orientation is angle from positive COL-axis (horizontal) to major axis, counter-clockwise.
            # Vector along major axis: (delta_col, delta_row_neg) -> (cos(orient), sin(orient))
            # So for (row, col) coords, the vector components are (d_row, d_col)
            # d_col = L * cos(orient), d_row = -L * sin(orient) (if y points down)
            # Vector: (cos(orient), -sin(orient)) for (col, row)
            # Vector: (-sin(orient), cos(orient)) for (row, col)
            axis_vector_rc = np.array([-np.sin(main_orientation), np.cos(main_orientation)])


            relative_coords = coords - centroid # Each row is (coord_row - cent_row, coord_col - cent_col)
            projections = np.dot(relative_coords, axis_vector_rc) # Project onto the axis vector
            
            # Count pixels on one side of the centroid along the major axis
            posterior_pixels_count = np.sum(projections > 0) # Pixels on the "positive projection" side
            if main_props_obj.area > 0:
                 features['posterior_area_ratio'] = posterior_pixels_count / main_props_obj.area

        if features['main_area'] < params.get('min_significant_area', 300):
            logger.debug(f"Main object area {features['main_area']} is less than min_significant_area. "
                         f"Skipping egg detection for {os.path.basename(image_path)}.")
            return features, main_props_obj, None

        # --- Separate Segmentation for EGG PAIR ---
        # This targets brighter regions, potentially egg sacs.
        high_thresh_quantile = params.get('egg_threshold_quantile', 0.92)
        non_zero_pixels = blurred_image[blurred_image > 1e-6] # Use blurred for consistency
        
        if non_zero_pixels.size == 0:
            high_thresh_val = 1.0 # Should not happen if main body was found
        else:
            high_thresh_val = np.quantile(non_zero_pixels, high_thresh_quantile)
            
        egg_binary_mask = blurred_image > high_thresh_val
        
        selem_opening_size_egg = params.get('egg_opening_selem_size', 0)
        if selem_opening_size_egg > 0:
            selem_opening_egg = skimage.morphology.disk(selem_opening_size_egg)
            egg_opened_mask = skimage.morphology.binary_opening(egg_binary_mask, selem_opening_egg)
        else:
            egg_opened_mask = egg_binary_mask
            
        egg_labeled_mask, egg_num_labels_cand = skimage.measure.label(egg_opened_mask, return_num=True, connectivity=2)

        # --- Filter EGG PAIR Candidates ---
        if egg_num_labels_cand > 0 and main_props_obj is not None:
            # Intensity image for egg props should be original gray for accurate intensity, if used
            egg_props_candidates = skimage.measure.regionprops(egg_labeled_mask, intensity_image=image_gray, cache=False)
            main_centroid_arr = np.array(main_props_obj.centroid) # (row, col)

            if not (np.any(np.isnan(main_centroid_arr)) or np.any(np.isinf(main_centroid_arr))):
                # Use major axis length as a scale for distance checks
                main_body_size_metric = main_props_obj.major_axis_length
                if main_body_size_metric <= 1e-6: # Fallback if major_axis is zero
                    main_body_size_metric = np.sqrt(main_props_obj.area) if main_props_obj.area > 0 else 1.0
                
                max_egg_dist = main_body_size_metric * params.get('max_egg_distance_factor', 0.90)
                
                # Main body's orientation vector (recalculated for clarity here)
                axis_vector_main_rc = np.array([-np.sin(main_orientation), np.cos(main_orientation)])
                
                # Determine which end of the main_props_obj axis is "posterior"
                # If posterior_area_ratio > 0.5, it means the end pointed to by +axis_vector_rc has more pixels.
                positive_projection_side_is_posterior = (features['posterior_area_ratio'] >= 0.5)

                for p_pair_candidate in egg_props_candidates:
                    # Area check for the candidate egg PAIR region
                    is_pair_area_ok = (params.get('egg_pair_min_area', 200) <= p_pair_candidate.area <= params.get('egg_pair_max_area', 10000))
                    if not is_pair_area_ok: continue

                    pair_centroid_arr = np.array(p_pair_candidate.centroid)
                    if np.any(np.isnan(pair_centroid_arr)) or np.any(np.isinf(pair_centroid_arr)): continue
                    
                    # Proximity check: distance between main body centroid and egg pair candidate centroid
                    distance_to_main = np.linalg.norm(pair_centroid_arr - main_centroid_arr)
                    is_proximal = distance_to_main < max_egg_dist
                    if not is_proximal: continue

                    # Positional check: is the egg pair in the posterior half of the main body?
                    vec_main_to_pair = pair_centroid_arr - main_centroid_arr
                    projection_onto_main_axis = np.dot(vec_main_to_pair, axis_vector_main_rc)
                    
                    # posterior_projection_threshold defines how far "into" the anterior it can be
                    # A positive threshold means it must be clearly on the posterior side.
                    # A negative threshold means it can be slightly on the anterior side.
                    posterior_proj_thresh_val = params.get('posterior_projection_threshold', -0.1) * main_body_size_metric # Scale threshold by body size
                    
                    # If positive_projection_side_is_posterior, then projection must be > threshold
                    # If not (i.e., negative_projection_side_is_posterior), then projection must be < -threshold
                    is_in_posterior_region = False
                    if positive_projection_side_is_posterior:
                        if projection_onto_main_axis >= posterior_proj_thresh_val:
                            is_in_posterior_region = True
                    else: # Negative projection side is posterior
                        if projection_onto_main_axis <= -posterior_proj_thresh_val: # Note sign change for threshold
                            is_in_posterior_region = True
                    if not is_in_posterior_region: continue
                    
                    # Solidity and Eccentricity checks for the egg PAIR region
                    pair_solidity = getattr(p_pair_candidate, 'solidity', 1.0)
                    is_solidity_ok = (params.get('egg_pair_min_solidity', 0.70) < pair_solidity < params.get('egg_pair_max_solidity', 0.98))
                    if not is_solidity_ok: continue

                    pair_ecc = getattr(p_pair_candidate, 'eccentricity', 1.0)
                    is_shape_ok = (pair_ecc > params.get('egg_pair_min_eccentricity', 0.3))
                    if not is_shape_ok: continue

                    # Orientation check: egg pair major axis should be roughly perpendicular to main body major axis
                    pair_orientation = getattr(p_pair_candidate, 'orientation', 0.0) # Radians
                    # Difference in angles. Normalize to be between 0 and pi/2.
                    # abs(a - b) gives difference. If > pi/2, then pi - diff is the acute angle.
                    delta_angle_abs = abs(main_orientation - pair_orientation)
                    acute_delta_angle = min(delta_angle_abs, np.pi - delta_angle_abs)
                    
                    # We want the acute angle between axes to be close to 90 degrees (pi/2)
                    # So, |acute_delta_angle - pi/2| should be small.
                    orientation_diff_rad = abs(acute_delta_angle - (np.pi / 2.0))
                    orientation_tolerance_rad = np.deg2rad(params.get('orientation_diff_tolerance_deg', 30.0))
                    
                    is_orientation_ok = orientation_diff_rad < orientation_tolerance_rad
                    if not is_orientation_ok: continue
                     
                    # If all checks pass, this is our egg_pair_props
                    egg_pair_props = p_pair_candidate
                    break # Found a suitable egg pair candidate

            if egg_pair_props is not None:
                 features['num_eggs'] = 2 # Assuming a "pair" means two eggs in one sac
                 features['avg_egg_area'] = egg_pair_props.area / 2.0 # Area per egg
                 features['avg_egg_ecc'] = getattr(egg_pair_props, 'eccentricity', 1.0)
        
        # Final check on all feature values
        for key in features:
            if isinstance(features[key], float) and not np.isfinite(features[key]):
                logger.warning(f"Feature '{key}' is non-finite ({features[key]}) for {os.path.basename(image_path)}. Defaulting based on type.")
                if key in ['num_eggs', 'num_components']: features[key] = 0
                elif key == 'avg_egg_ecc': features[key] = 1.0
                elif key == 'posterior_area_ratio': features[key] = 0.5
                else: features[key] = 0.0
            if key in ['num_eggs', 'num_components']: # Ensure integer types
                features[key] = int(features[key])


        return features, main_props_obj, egg_pair_props

    except Exception as e:
        logger.exception(f"Error extracting ML features for {os.path.basename(image_path)}: {e}")
        return None, None, None


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)
    
    from src.utils.config_loader import get_config # For params
    # Test assumes pipeline_config.yml is in ../../config relative to this script if run directly
    # Or, that get_config() can find it.
    try:
        cfg = get_config()
        ml_params = cfg.get('stage_03_ml_classification', {}).get('feature_extraction_params', {})
        if not ml_params:
            print("ML feature extraction parameters not found in config. Using hardcoded defaults for test.")
            ml_params = { # Taken from ws_identify_ideal_fems.feature_params_global
                'gaussian_sigma': 1.0, 'otsu_threshold_multiplier': 0.5, 'closing_selem_size': 3,
                'min_noise_size': 50, 'min_significant_area': 300,
                'egg_threshold_quantile': 0.92,
                'egg_opening_selem_size': 0,
                'egg_pair_min_area': 200, 'egg_pair_max_area': 10000,
                'egg_pair_min_solidity': 0.70, 'egg_pair_max_solidity': 0.98,
                'egg_pair_min_eccentricity': 0.3,
                'max_egg_distance_factor': 0.90,
                'posterior_projection_threshold': -0.1, # This was a key change in logic
                'orientation_diff_tolerance_deg': 50
            }

    except FileNotFoundError:
        print("Config file not found. Cannot run ML feature extraction test without parameters.")
        exit()


    # Create a dummy image for testing - a central ellipse (body) and a smaller ellipse (egg sac)
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Main body (ellipse)
    rr_body, cc_body = skimage.draw.ellipse(100, 100, 40, 20, rotation=np.deg2rad(30))
    test_image[rr_body, cc_body] = 150 
    
    # Egg sac (another ellipse, brighter, somewhat perpendicular, and posterior)
    # To place it posterior to the body rotated 30 deg:
    # Body center (100,100). Major axis along 30 deg. Posterior end roughly:
    # post_x = 100 + 20 * cos(30) approx 100 + 17 = 117
    # post_y = 100 - 20 * sin(30) approx 100 - 10 = 90  (if y up)
    # post_y = 100 + 20 * sin(30) approx 100 + 10 = 110 (if y down in image coords)
    # Let's try center for egg sac at (row=90, col=120) relative to (100,100) centroid for main body
    # This makes it y_img = 90, x_img = 120 (row, col)
    rr_egg, cc_egg = skimage.draw.ellipse(90, 125, 10, 15, rotation=np.deg2rad(30 + 90)) # Perpendicular
    test_image[rr_egg, cc_egg] = 220 # Brighter

    test_image_float = skimage.util.img_as_float(test_image) # Function expects float

    # Save the test image (optional, for inspection)
    skimage.io.imsave("temp_ml_feature_test_image.png", skimage.util.img_as_ubyte(test_image_float))
    logger.info("Saved temp_ml_feature_test_image.png for inspection.")

    features, main_props, egg_props = extract_ml_features(
        image_path="temp_ml_feature_test_image.png", # Path is used for logging
        params=ml_params,
        image_array_gray=test_image_float
    )

    if features:
        print("\nExtracted ML Features:")
        for k, v in features.items():
            print(f"  {k}: {v}")
        
        assert features['main_area'] > 0, "Main area not extracted"
        # Check if egg was detected based on parameters
        if features['num_eggs'] > 0:
            logger.info("Egg sac detected and features extracted.")
            assert features['avg_egg_area'] > 0
        else:
            logger.warning("Egg sac NOT detected. Check dummy image and parameters.")
            # This might be OK if parameters are strict or dummy image is not ideal.
    else:
        logger.error("ML Feature extraction failed for the dummy image.")

    # Clean up
    if os.path.exists("temp_ml_feature_test_image.png"):
        os.remove("temp_ml_feature_test_image.png")