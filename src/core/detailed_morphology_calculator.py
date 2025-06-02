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

from src.utils.common_exceptions import ConfigError 

logger = logging.getLogger(__name__)

def reconstruct_main_body_masks(
    image_gray_original: np.ndarray, # Expected to be float [0,1]
    s3_feature_extraction_params: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
    """
    Reconstructs the raw (unfilled) and filled main body masks using Stage 3 parameters.
    Handles illumination mode for consistent processing.
    """
    try:
        if not np.issubdtype(image_gray_original.dtype, np.floating):
            image_gray_proc = skimage.util.img_as_float(image_gray_original) 
        else:
            image_gray_proc = image_gray_original.copy()

        # Apply illumination adjustment based on S3 params
        s3_illumination_mode = s3_feature_extraction_params.get('illumination', 'darkfield')
        if s3_illumination_mode == 'collimated':
            logger.debug("MaskRecon: Illumination '{s3_illumination_mode}'. Inverting image.")
            image_gray_proc = 1.0 - image_gray_proc # Invert float image

        # image_gray_for_props should be the same as image_gray_proc for regionprops intensity calculations
        # because regionprops should see the image as it was when the mask was formed.
        image_gray_for_props = image_gray_proc 

        blurred_image = skimage.filters.gaussian(
            image_gray_proc, sigma=s3_feature_extraction_params.get('gaussian_sigma', 1.0)
        )

        if blurred_image.min() == blurred_image.max():
            logger.warning("Image became blank after blurring during mask reconstruction.")
            return None, None, None

        thresholding_mode = s3_feature_extraction_params.get('thresholding_mode', 'otsu')
        final_threshold_value_float: Optional[float] = None 

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

            final_threshold_value_float = fixed_thresh_int / 255.0 
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
        # Use the (potentially inverted) image_gray_for_props for intensity calculations by regionprops
        props_filled_all = skimage.measure.regionprops(labeled_filled_mask, intensity_image=image_gray_for_props, cache=False)

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
    intensity_image_for_props: Optional[np.ndarray] = None # float [0,1], ALREADY INVERTED if collimated
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    keys_to_init = [
        'main_area_raw_s4', 'solidity_s4', 'compactness_rect_s4', 'extent_raw_s4',
        'eccentricity_raw_s4', 'equivalent_diameter_raw_s4',
        'major_axis_length_raw_s4', 'minor_axis_length_raw_s4',
        'main_area_filled_s4', 'porosity_s4',
        'hu_moment1_s4', 
        'moment_of_inertia_xx_s4', 'moment_of_inertia_yy_s4', 'moment_of_inertia_xy_s4',
        'type_factor_s4', 'heywood_circularity_factor_s4', 'elongation_factor_s4',
        'compactness_factor_ni_s4' 
    ]
    for k_init in keys_to_init: features[k_init] = np.nan


    if raw_body_mask is None or filled_body_mask is None:
        logger.warning("Raw or filled body mask is None. Cannot calculate detailed morphology.")
        return features

    raw_body_mask_bool = raw_body_mask > 0
    
    # intensity_image_for_props here should be the image state (e.g., potentially inverted if collimated)
    # that corresponds to how the masks were generated.
    props_raw_list = skimage.measure.regionprops(
        skimage.measure.label(raw_body_mask_bool),
        intensity_image=intensity_image_for_props, 
        cache=False
    )
    if props_raw_list:
        props_raw = props_raw_list[0] 
        features['main_area_raw_s4'] = float(props_raw.area)
        features['solidity_s4'] = float(props_raw.solidity) if hasattr(props_raw, 'solidity') else np.nan
        
        if hasattr(props_raw, 'area_oriented_bbox'):
             oriented_bbox_area = props_raw.area_oriented_bbox
        elif hasattr(props_raw, 'area_bbox'): 
            oriented_bbox_area = props_raw.area_bbox
        else:
            oriented_bbox_area = 0.0

        if oriented_bbox_area > 1e-6:
            features['compactness_rect_s4'] = float(props_raw.area / oriented_bbox_area)
        else:
            features['compactness_rect_s4'] = np.nan

        features['extent_raw_s4'] = float(props_raw.extent) if hasattr(props_raw, 'extent') else np.nan
        features['compactness_factor_ni_s4'] = features['extent_raw_s4'] 

        features['eccentricity_raw_s4'] = float(props_raw.eccentricity) if hasattr(props_raw, 'eccentricity') else np.nan
        features['equivalent_diameter_raw_s4'] = float(props_raw.equivalent_diameter) if hasattr(props_raw, 'equivalent_diameter') else np.nan
        features['major_axis_length_raw_s4'] = float(props_raw.major_axis_length) if hasattr(props_raw, 'major_axis_length') else np.nan
        features['minor_axis_length_raw_s4'] = float(props_raw.minor_axis_length) if hasattr(props_raw, 'minor_axis_length') else np.nan

        if hasattr(props_raw, 'moments_central'):
            Ixx = float(props_raw.moments_central[2,0]) 
            Iyy = float(props_raw.moments_central[0,2]) 
            Ixy = float(props_raw.moments_central[1,1])
            features['moment_of_inertia_xx_s4'] = Ixx
            features['moment_of_inertia_yy_s4'] = Iyy
            features['moment_of_inertia_xy_s4'] = Ixy

            if Ixx > 1e-9 and Iyy > 1e-9 : 
                denominator_tf = 4 * np.pi * np.sqrt(Ixx * Iyy)
                if denominator_tf > 1e-9:
                    features['type_factor_s4'] = (props_raw.area**2) / denominator_tf
                else:
                    features['type_factor_s4'] = np.nan
            else:
                features['type_factor_s4'] = np.nan

        if hasattr(props_raw, 'perimeter') and props_raw.perimeter > 1e-9 and props_raw.area > 1e-9:
            denominator_hcf = 2 * np.sqrt(np.pi * props_raw.area)
            if denominator_hcf > 1e-9:
                features['heywood_circularity_factor_s4'] = props_raw.perimeter / denominator_hcf
            else:
                features['heywood_circularity_factor_s4'] = np.nan
        else:
            features['heywood_circularity_factor_s4'] = np.nan

        if hasattr(props_raw, 'feret_diameter_max') and \
           hasattr(props_raw, 'minor_axis_length') and props_raw.minor_axis_length > 1e-9:
            features['elongation_factor_s4'] = props_raw.feret_diameter_max / props_raw.minor_axis_length
        else:
            features['elongation_factor_s4'] = np.nan

    else:
        logger.warning("No regionprops found for raw_body_mask.")

    moments_raw_cv = cv2.moments(raw_body_mask.astype(np.uint8)) 
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
            features[key] = None 

    return features


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_img_gray_uint8 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(test_img_gray_uint8, (100,100), (80,40), angle=30, startAngle=0, endAngle=360, color=150, thickness=-1)
    cv2.ellipse(test_img_gray_uint8, (90,90), (15,10), angle=30, startAngle=0, endAngle=360, color=0, thickness=-1)
    test_img_gray_float = skimage.util.img_as_float(test_img_gray_uint8) # For passing to reconstruct

    # Test with Otsu mode (darkfield)
    dummy_s3_params_otsu_dark = {
        'illumination': 'darkfield',
        'gaussian_sigma': 1.0,
        'thresholding_mode': 'otsu',
        'otsu_threshold_multiplier': 0.4,
        'min_noise_size': 10,
        'closing_selem_size': 0
    }
    print("\n--- Testing Mask Reconstruction (Otsu, Darkfield) ---")
    raw_mask_od, filled_mask_od, _ = reconstruct_main_body_masks(test_img_gray_float.copy(), dummy_s3_params_otsu_dark)
    if raw_mask_od is not None:
        print(f"Otsu (darkfield) - Reconstructed Raw mask area: {np.sum(raw_mask_od > 0)}")
        # For morphology, pass the image as it was used for mask creation (i.e., darkfield means original float image)
        detailed_features_od = calculate_detailed_body_morphology(raw_mask_od, filled_mask_od, test_img_gray_float)
        print("Otsu (darkfield) - Detailed Body Morphology Features:")
        for k, v in detailed_features_od.items(): print(f"  {k}: {v}")
    else:
        print("Otsu (darkfield) - Mask reconstruction failed.")

    # Test with Otsu mode (collimated)
    dummy_s3_params_otsu_collim = {
        'illumination': 'collimated', # Image will be inverted
        'gaussian_sigma': 1.0,
        'thresholding_mode': 'otsu',
        'otsu_threshold_multiplier': 0.4, # Otsu should adapt to inverted image
        'min_noise_size': 10,
        'closing_selem_size': 0
    }
    print("\n--- Testing Mask Reconstruction (Otsu, Collimated) ---")
    raw_mask_oc, filled_mask_oc, _ = reconstruct_main_body_masks(test_img_gray_float.copy(), dummy_s3_params_otsu_collim)
    if raw_mask_oc is not None:
        print(f"Otsu (collimated) - Reconstructed Raw mask area: {np.sum(raw_mask_oc > 0)}")
        # For morphology, pass the INVERTED image if collimated
        test_img_gray_float_inverted = 1.0 - test_img_gray_float 
        detailed_features_oc = calculate_detailed_body_morphology(raw_mask_oc, filled_mask_oc, test_img_gray_float_inverted)
        print("Otsu (collimated) - Detailed Body Morphology Features:")
        for k, v in detailed_features_oc.items(): print(f"  {k}: {v}")
    else:
        print("Otsu (collimated) - Mask reconstruction failed.")


    dummy_s3_params_invalid = {
        'thresholding_mode': 'invalid_mode'
    }
    print("\n--- Testing with Invalid Thresholding Mode ---")
    try:
        reconstruct_main_body_masks(test_img_gray_float.copy(), dummy_s3_params_invalid)
    except ConfigError as e:
        print(f"Correctly caught ConfigError for invalid mode: {e}")
    except Exception as e:
        print(f"Unexpected error during invalid mode test: {e}")

    print("\n--- Testing with a simple rectangle (100x50) for NI features ---")
    rect_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(rect_mask, (50, 75), (150, 125), 255, -1) 
    rect_filled_mask = rect_mask.copy()
    # For geometric props from a binary mask, intensity_image can be the mask itself (converted to float) or None
    rect_mask_float_for_props = skimage.util.img_as_float(rect_mask)
    rect_features = calculate_detailed_body_morphology(rect_mask, rect_filled_mask, rect_mask_float_for_props)
    print("Rectangle - Detailed Body Morphology Features:")
    for k, v in rect_features.items():
        if k in ['main_area_raw_s4', 'moment_of_inertia_xx_s4', 'moment_of_inertia_yy_s4',
                 'type_factor_s4', 'heywood_circularity_factor_s4', 'elongation_factor_s4',
                 'compactness_factor_ni_s4', 'hu_moment1_s4']:
            print(f"  {k}: {v}")