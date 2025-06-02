# src/core/detailed_morphology_calculator.py
import cv2
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.filters # For Gaussian blur
import skimage.util # For img_as_float
from scipy import ndimage as ndi # For binary_fill_holes
import logging
from typing import Dict, Any, Optional, Tuple
# skimage.draw no longer needed here
from skimage.measure._regionprops import RegionProperties

from src.utils.common_exceptions import ConfigError # Added for raising config error

logger = logging.getLogger(__name__)

def reconstruct_main_body_masks(
    image_gray_original: np.ndarray,
    s3_feature_extraction_params: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
    """
    Reconstructs the raw (unfilled) and filled main body masks using Stage 3 parameters.
    """
    try:
        if not np.issubdtype(image_gray_original.dtype, np.floating):
            image_gray = skimage.util.img_as_float(image_gray_original) # Converts to 0-1 float
        else:
            image_gray = image_gray_original.copy()

        blurred_image = skimage.filters.gaussian(
            image_gray, sigma=s3_feature_extraction_params.get('gaussian_sigma', 1.0)
        )

        if blurred_image.min() == blurred_image.max():
            logger.warning("Image became blank after blurring during mask reconstruction.")
            return None, None, None

        thresholding_mode = s3_feature_extraction_params.get('thresholding_mode', 'otsu')
        final_threshold_value_float: Optional[float] = None # This will be the 0-1 float threshold

        if thresholding_mode == 'otsu':
            if blurred_image.min() == blurred_image.max():
                logger.warning("Image is uniform after blurring. Otsu thresholding cannot be applied robustly in mask reconstruction.")
                return None, None, None
            try:
                otsu_thresh_val_raw = skimage.filters.threshold_otsu(blurred_image)
            except ValueError:
                logger.warning("Otsu threshold failed during mask reconstruction.")
                return None, None, None

            otsu_multiplier = s3_feature_extraction_params.get('otsu_threshold_multiplier', 0.9)
            final_threshold_value_float = otsu_thresh_val_raw * otsu_multiplier
            logger.debug(f"MaskRecon: Using Otsu thresholding. Raw: {otsu_thresh_val_raw:.4f}, Mult: {otsu_multiplier}, Final (0-1): {final_threshold_value_float:.4f}")

        elif thresholding_mode == 'fixed':
            fixed_thresh_val_config_int = s3_feature_extraction_params.get('fixed_threshold_value')
            if fixed_thresh_val_config_int is None:
                logger.error("MaskRecon: Thresholding mode is 'fixed' but 'fixed_threshold_value' not provided in s3_feature_extraction_params.")
                raise ConfigError("thresholding_mode is 'fixed' but 'fixed_threshold_value' is missing from s3_feature_extraction_params for mask reconstruction.")

            try:
                fixed_thresh_int = int(fixed_thresh_val_config_int)
            except ValueError:
                logger.error(f"MaskRecon: 'fixed_threshold_value' ({fixed_thresh_val_config_int}) must be an integer (0-255).")
                raise ConfigError(f"MaskRecon: 'fixed_threshold_value' ({fixed_thresh_val_config_int}) is not a valid integer.")

            if not (0 <= fixed_thresh_int <= 255):
                logger.warning(f"MaskRecon: Fixed threshold (config) {fixed_thresh_int} outside 0-255. Ensure intended.")

            final_threshold_value_float = fixed_thresh_int / 255.0 # Convert to 0-1 float
            logger.debug(f"MaskRecon: Using fixed thresholding. Config (0-255): {fixed_thresh_int}, Converted (0-1): {final_threshold_value_float:.4f}")

        else:
            logger.error(f"MaskRecon: Invalid thresholding_mode: '{thresholding_mode}'. Supported: 'otsu', 'fixed'.")
            raise ConfigError(f"Invalid thresholding_mode: '{thresholding_mode}' in s3_feature_extraction_params for mask reconstruction.")

        if final_threshold_value_float is None:
             logger.error("MaskRecon: Final threshold value (float) not set. Aborting mask reconstruction.")
             return None, None, None

        main_binary_mask_initial = blurred_image > final_threshold_value_float

        main_cleaned_mask = skimage.morphology.remove_small_objects(
            main_binary_mask_initial, min_size=s3_feature_extraction_params.get('min_noise_size', 50)
        )

        selem_closing_size = s3_feature_extraction_params.get('closing_selem_size', 3)
        if selem_closing_size > 0:
            selem_closing = skimage.morphology.disk(selem_closing_size)
            raw_body_mask_for_props = skimage.morphology.binary_closing(main_cleaned_mask, selem_closing)
        else:
            raw_body_mask_for_props = main_cleaned_mask

        filled_body_mask = ndi.binary_fill_holes(raw_body_mask_for_props)

        labeled_filled_mask = skimage.measure.label(filled_body_mask, connectivity=2)
        props_filled_all = skimage.measure.regionprops(labeled_filled_mask, intensity_image=image_gray, cache=False)

        if not props_filled_all:
            logger.warning("No regions found in the reconstructed filled body mask.")
            return raw_body_mask_for_props.astype(np.uint8) * 255, \
                   filled_body_mask.astype(np.uint8) * 255, \
                   None

        main_props_filled = sorted(props_filled_all, key=lambda x: x.area, reverse=True)[0]

        return raw_body_mask_for_props.astype(np.uint8) * 255, \
               filled_body_mask.astype(np.uint8) * 255, \
               main_props_filled

    except ConfigError as ce:
        logger.error(f"Configuration error during mask reconstruction: {ce}")
        raise
    except Exception as e:
        logger.exception(f"Error reconstructing main body masks: {e}")
        return None, None, None


def calculate_detailed_body_morphology(
    raw_body_mask: Optional[np.ndarray],      # uint8, 0 or 255
    filled_body_mask: Optional[np.ndarray],   # uint8, 0 or 255
    intensity_image_for_props: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    keys_to_init = [
        'main_area_raw_s4', 'solidity_s4', 'compactness_rect_s4', 'extent_raw_s4',
        'eccentricity_raw_s4', 'equivalent_diameter_raw_s4',
        'major_axis_length_raw_s4', 'minor_axis_length_raw_s4',
        'main_area_filled_s4', 'porosity_s4',
        'hu_moment1_s4', # Only Hu moment 1
        'moment_of_inertia_xx_s4', 'moment_of_inertia_yy_s4', 'moment_of_inertia_xy_s4',
        'type_factor_s4', 'heywood_circularity_factor_s4', 'elongation_factor_s4',
        'compactness_factor_ni_s4' # NI specific compactness
    ]
    for k_init in keys_to_init: features[k_init] = np.nan


    if raw_body_mask is None or filled_body_mask is None:
        logger.warning("Raw or filled body mask is None. Cannot calculate detailed morphology.")
        return features

    raw_body_mask_bool = raw_body_mask > 0
    # Calculate regionprops on the raw mask to get most features
    # Pass intensity_image for intensity-based props, though most NI are geometric.
    props_raw_list = skimage.measure.regionprops(
        skimage.measure.label(raw_body_mask_bool),
        intensity_image=intensity_image_for_props, # Using original gray image for intensities
        cache=False
    )
    if props_raw_list:
        props_raw = props_raw_list[0] # Assuming largest component
        features['main_area_raw_s4'] = float(props_raw.area)
        features['solidity_s4'] = float(props_raw.solidity) if hasattr(props_raw, 'solidity') else np.nan
        
        # Compactness_rect_s4 (uses oriented bounding box)
        if hasattr(props_raw, 'area_oriented_bbox'):
             oriented_bbox_area = props_raw.area_oriented_bbox
        elif hasattr(props_raw, 'area_bbox'): # Fallback to axis-aligned if oriented not available
            oriented_bbox_area = props_raw.area_bbox
        else:
            oriented_bbox_area = 0.0

        if oriented_bbox_area > 1e-6:
            features['compactness_rect_s4'] = float(props_raw.area / oriented_bbox_area)
        else:
            features['compactness_rect_s4'] = np.nan

        # Extent (uses axis-aligned bounding box, also NI Compactness Factor)
        features['extent_raw_s4'] = float(props_raw.extent) if hasattr(props_raw, 'extent') else np.nan
        features['compactness_factor_ni_s4'] = features['extent_raw_s4'] # NI Compactness Factor

        features['eccentricity_raw_s4'] = float(props_raw.eccentricity) if hasattr(props_raw, 'eccentricity') else np.nan
        features['equivalent_diameter_raw_s4'] = float(props_raw.equivalent_diameter) if hasattr(props_raw, 'equivalent_diameter') else np.nan
        features['major_axis_length_raw_s4'] = float(props_raw.major_axis_length) if hasattr(props_raw, 'major_axis_length') else np.nan
        features['minor_axis_length_raw_s4'] = float(props_raw.minor_axis_length) if hasattr(props_raw, 'minor_axis_length') else np.nan

        # Moments of Inertia from regionprops central moments (mu_pq)
        # mu20 = sum(sum(I(x,y)*(x-xc)^2)), mu02 = sum(sum(I(x,y)*(y-yc)^2)), mu11 = sum(sum(I(x,y)*(x-xc)*(y-yc)))
        # For binary mask, I(x,y)=1 for pixels in object.
        # skimage's moments_central are already weighted by pixel values if intensity_image is provided.
        # For binary shapes (like a mask), these are geometric moments.
        # Ixx in NI context is moment about x-axis passing through centroid, so mu20 is correct.
        # Iyy is moment about y-axis passing through centroid, so mu02 is correct.
        if hasattr(props_raw, 'moments_central'):
            # moments_central[row, col] where row is power of r (y-yc) and col is power of c (x-xc)
            # So, mu20 is moments_central[0,2] (power of (y-yc)^0 * (x-xc)^2) -> this is Iyy by physics convention if x is horizontal
            # And, mu02 is moments_central[2,0] (power of (y-yc)^2 * (x-xc)^0) -> this is Ixx by physics convention if y is vertical
            # Check skimage docs: "Weighted central moments..."
            # If we consider standard image coordinates (row=y, col=x):
            # I_yy (around y-axis, resistance to rotation in x-z plane) = sum(x^2 dm) -> mu20 if (x-xc) is c
            # I_xx (around x-axis, resistance to rotation in y-z plane) = sum(y^2 dm) -> mu02 if (y-yc) is r
            # OpenCV moments: mu20 (for x), mu02 (for y), mu11 (for xy)
            # Let's assume:
            # features['moment_of_inertia_xx_s4'] = float(props_raw.moments_central[2,0]) # sum (y-yc)^2  (Ixx)
            # features['moment_of_inertia_yy_s4'] = float(props_raw.moments_central[0,2]) # sum (x-xc)^2  (Iyy)
            # features['moment_of_inertia_xy_s4'] = float(props_raw.moments_central[1,1]) # sum (x-xc)(y-yc) (Ixy)

            # To be consistent with OpenCV's mu20, mu02, mu11 from cv2.moments():
            # cv2.moments() on a binary image gives purely geometric moments.
            # mu20 = sum over (x_i - x_c)^2
            # mu02 = sum over (y_i - y_c)^2
            # mu11 = sum over (x_i - x_c)(y_i - y_c)
            # skimage regionprops.moments_central[p,q] refers to (row-rc)^p * (col-cc)^q
            # So:
            # Ixx (NI) ~ mu02 (skimage row-power 2, col-power 0) = sum (r-rc)^2
            # Iyy (NI) ~ mu20 (skimage row-power 0, col-power 2) = sum (c-cc)^2
            # Ixy (NI) ~ mu11 (skimage row-power 1, col-power 1) = sum (r-rc)(c-cc)
            # This seems to be the correct mapping given (row, col) image coordinates.
            Ixx = float(props_raw.moments_central[2,0]) # around horizontal axis through centroid
            Iyy = float(props_raw.moments_central[0,2]) # around vertical axis through centroid
            Ixy = float(props_raw.moments_central[1,1])
            features['moment_of_inertia_xx_s4'] = Ixx
            features['moment_of_inertia_yy_s4'] = Iyy
            features['moment_of_inertia_xy_s4'] = Ixy

            # Type Factor
            if Ixx > 1e-9 and Iyy > 1e-9 : # Avoid sqrt of zero or negative (if Ixx or Iyy very small)
                denominator_tf = 4 * np.pi * np.sqrt(Ixx * Iyy)
                if denominator_tf > 1e-9:
                    features['type_factor_s4'] = (props_raw.area**2) / denominator_tf
                else:
                    features['type_factor_s4'] = np.nan
            else:
                features['type_factor_s4'] = np.nan

        # Heywood Circularity Factor
        if hasattr(props_raw, 'perimeter') and props_raw.perimeter > 1e-9 and props_raw.area > 1e-9:
            denominator_hcf = 2 * np.sqrt(np.pi * props_raw.area)
            if denominator_hcf > 1e-9:
                features['heywood_circularity_factor_s4'] = props_raw.perimeter / denominator_hcf
            else:
                features['heywood_circularity_factor_s4'] = np.nan
        else:
            features['heywood_circularity_factor_s4'] = np.nan

        # Elongation Factor: Max Feret Diameter / minor_axis_length
        if hasattr(props_raw, 'feret_diameter_max') and \
           hasattr(props_raw, 'minor_axis_length') and props_raw.minor_axis_length > 1e-9:
            features['elongation_factor_s4'] = props_raw.feret_diameter_max / props_raw.minor_axis_length
        else:
            features['elongation_factor_s4'] = np.nan

    else:
        logger.warning("No regionprops found for raw_body_mask.")

    # Hu Moment 1
    moments_raw_cv = cv2.moments(raw_body_mask.astype(np.uint8)) # raw_body_mask is 0 or 255
    hu_moments_cv = cv2.HuMoments(moments_raw_cv).flatten()
    if len(hu_moments_cv) >= 1:
        val_hu1 = float(hu_moments_cv[0])
        features['hu_moment1_s4'] = val_hu1 if np.isfinite(val_hu1) else np.nan
    else:
        features['hu_moment1_s4'] = np.nan

    filled_area = np.sum(filled_body_mask > 0)
    features['main_area_filled_s4'] = float(filled_area)

    raw_area_val_for_porosity = features.get('main_area_raw_s4')
    if isinstance(raw_area_val_for_porosity, (int, float)) and \
       np.isfinite(raw_area_val_for_porosity) and filled_area > 1e-6:
         features['porosity_s4'] = 1.0 - (raw_area_val_for_porosity / filled_area)
    else:
        features['porosity_s4'] = np.nan

    for key, value in features.items():
        if isinstance(value, (np.float32, np.float64)):
            features[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
             features[key] = int(value)
        elif isinstance(value, float) and not np.isfinite(value):
            features[key] = None # Store as Python None for JSON/CSV compatibility if it's NaN/inf

    return features


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_img_gray = np.zeros((200, 200), dtype=np.uint8)
    # Main body with intensity 150
    cv2.ellipse(test_img_gray, (100,100), (80,40), angle=30, startAngle=0, endAngle=360, color=150, thickness=-1)
    # Hole inside
    cv2.ellipse(test_img_gray, (90,90), (15,10), angle=30, startAngle=0, endAngle=360, color=0, thickness=-1)

    # Test with Otsu mode
    dummy_s3_params_otsu = {
        'gaussian_sigma': 1.0,
        'thresholding_mode': 'otsu',
        'otsu_threshold_multiplier': 0.4,
        'min_noise_size': 10,
        'closing_selem_size': 0
    }
    print("\n--- Testing with Otsu Thresholding Mode for Mask Reconstruction ---")
    raw_mask_o, filled_mask_o, _ = reconstruct_main_body_masks(test_img_gray.copy(), dummy_s3_params_otsu)
    if raw_mask_o is not None:
        print(f"Otsu - Reconstructed Raw mask area: {np.sum(raw_mask_o > 0)}")
        # Pass test_img_gray as intensity_image_for_props for props calculation
        detailed_features_o = calculate_detailed_body_morphology(raw_mask_o, filled_mask_o, test_img_gray)
        print("Otsu - Detailed Body Morphology Features:")
        for k, v in detailed_features_o.items(): print(f"  {k}: {v}")
    else:
        print("Otsu - Mask reconstruction failed.")

    # Test with Fixed mode (0-255 integer from config)
    dummy_s3_params_fixed = {
        'gaussian_sigma': 1.0,
        'thresholding_mode': 'fixed',
        'fixed_threshold_value': 80, # Integer 0-255 from config
        'min_noise_size': 10,
        'closing_selem_size': 0
    }
    print("\n--- Testing with Fixed Thresholding Mode for Mask Reconstruction ---")
    raw_mask_f, filled_mask_f, _ = reconstruct_main_body_masks(test_img_gray.copy(), dummy_s3_params_fixed)
    if raw_mask_f is not None:
        print(f"Fixed - Reconstructed Raw mask area: {np.sum(raw_mask_f > 0)}")
        detailed_features_f = calculate_detailed_body_morphology(raw_mask_f, filled_mask_f, test_img_gray)
        print("Fixed - Detailed Body Morphology Features:")
        for k, v in detailed_features_f.items(): print(f"  {k}: {v}")
    else:
        print("Fixed - Mask reconstruction failed.")

    dummy_s3_params_invalid = {
        'thresholding_mode': 'invalid_mode'
    }
    print("\n--- Testing with Invalid Thresholding Mode ---")
    try:
        reconstruct_main_body_masks(test_img_gray.copy(), dummy_s3_params_invalid)
    except ConfigError as e:
        print(f"Correctly caught ConfigError for invalid mode: {e}")
    except Exception as e:
        print(f"Unexpected error during invalid mode test: {e}")

    # Test with a simple rectangle to verify NI moments and factors
    print("\n--- Testing with a simple rectangle (100x50) for NI features ---")
    rect_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(rect_mask, (50, 75), (150, 125), 255, -1) # 100 wide, 50 high
    # Area = 100*50 = 5000, Perimeter = 2*(100+50) = 300

    # Create a dummy filled mask (same as raw for a solid rectangle)
    rect_filled_mask = rect_mask.copy()
    
    # Intensity image can be the mask itself for geometric props or None
    rect_features = calculate_detailed_body_morphology(rect_mask, rect_filled_mask, rect_mask)
    print("Rectangle - Detailed Body Morphology Features:")
    for k, v in rect_features.items():
        if k in ['main_area_raw_s4', 'moment_of_inertia_xx_s4', 'moment_of_inertia_yy_s4',
                 'type_factor_s4', 'heywood_circularity_factor_s4', 'elongation_factor_s4',
                 'compactness_factor_ni_s4', 'hu_moment1_s4']:
            print(f"  {k}: {v}")
    # Expected for a 100x50 rectangle centered at (100,100) (mask coords 50-150, 75-125)
    # Area = 5000
    # Ixx for rect width w, height h, about centroid = M * h^2 / 12. If M=Area, then Area * h^2 / 12
    # Iyy for rect width w, height h, about centroid = M * w^2 / 12. If M=Area, then Area * w^2 / 12
    # Here, skimage's moments_central might be slightly different due to pixel discretiztion.
    # h_rect = 50, w_rect = 100
    # Expected Ixx ~ 5000 * 50^2 / 12 = 1041666.67
    # Expected Iyy ~ 5000 * 100^2 / 12 = 4166666.67
    # (Note: skimage's (row,col) convention, so Ixx from `moments_central[2,0]` is for rows (height),
    # Iyy from `moments_central[0,2]` is for columns (width))

    # Heywood: P / (2*sqrt(pi*A)) = 300 / (2 * sqrt(pi*5000)) = 300 / (2 * sqrt(15707.96)) = 300 / (2 * 125.33) = 300 / 250.66 = 1.196
    # Compactness_factor_ni (Extent) = Area / (BoundingBoxArea) = 5000 / (100*50) = 1.0