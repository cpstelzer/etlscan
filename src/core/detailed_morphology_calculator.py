# src/core/detailed_morphology_calculator.py
import cv2
import numpy as np
import skimage.measure
import skimage.morphology # Still needed for regionprops related mask operations if any
import skimage.filters 
import skimage.util 
from scipy import ndimage as ndi # Should be removed if fill_holes not directly used
import logging
from typing import Dict, Any, Optional, Tuple
from skimage.measure._regionprops import RegionProperties

from src.utils.common_exceptions import ConfigError
from src.core.segmentation import segment_main_body_from_image_float # MODIFIED: Import new function

logger = logging.getLogger(__name__)

def reconstruct_main_body_masks(
    image_gray_original_float: np.ndarray, # Expected to be float [0,1], UNPROCESSED (not yet inverted)
    s3_feature_extraction_params: Dict[str, Any] # This will now contain ALL params for segment_main_body_from_image_float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
    """
    Reconstructs the raw (unfilled) and filled main body masks using the shared segmentation function.
    The s3_feature_extraction_params must contain all necessary parameters for
    segment_main_body_from_image_float (illumination, thresholding, unsharp masking, etc.).
    """
    try:
        # The s3_feature_extraction_params are directly passed as segmentation_params
        # to the new shared function. It handles float conversion, inversion,
        # unsharp masking, thresholding, and post-processing.
        
        logger.debug(f"DetailedMorpho: Calling shared segment_main_body with params: {s3_feature_extraction_params}")

        raw_body_mask_uint8, filled_body_mask_uint8, main_props_object = \
            segment_main_body_from_image_float(
                image_gray_original_float,
                s3_feature_extraction_params # Pass the comprehensive S3 params
            )

        if raw_body_mask_uint8 is None or filled_body_mask_uint8 is None:
            logger.warning("DetailedMorpho: Shared segmentation function returned None for masks.")
            # main_props_object might also be None here
            return None, None, main_props_object # Return what we got

        if main_props_object is None:
            logger.warning("DetailedMorpho: Shared segmentation found no main properties object, though masks might exist.")
            # This can happen if the largest object is too small as per min_significant_area,
            # or if no objects were found at all.

        return raw_body_mask_uint8, filled_body_mask_uint8, main_props_object

    except ConfigError as ce:
        logger.error(f"Configuration error during mask reconstruction via shared function: {ce}")
        raise
    except Exception as e:
        logger.exception(f"Error reconstructing main body masks via shared function: {e}")
        return None, None, None


def calculate_detailed_body_morphology(
    raw_body_mask: Optional[np.ndarray],      # uint8 mask
    filled_body_mask: Optional[np.ndarray],   # uint8 mask
    intensity_image_for_props: Optional[np.ndarray] = None # float [0,1], ALREADY INVERTED by caller if source was collimated
                                                          # AND PRE-ENHANCEMENT if props need original intensities
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

    # Ensure masks are boolean for skimage.measure.regionprops if they are not already
    # The masks from segment_main_body_from_image_float are uint8, so convert.
    raw_body_mask_bool = raw_body_mask > 0 
    
    # Note: The intensity_image_for_props passed here should be the one
    # that segment_main_body_from_image_float used for its props calculation
    # (i.e., potentially inverted, but *before* unsharp masking if unsharp masking was applied).
    # This ensures consistency if any intensity-based props are re-calculated or are crucial.
    # The RegionProperties object returned by segment_main_body already has props calculated
    # with the correct intensity image. This function primarily re-calculates or derives
    # additional morphology metrics from the *masks*.
    
    props_raw_list = skimage.measure.regionprops(
        skimage.measure.label(raw_body_mask_bool), # Label the raw mask to get its specific props
        intensity_image=intensity_image_for_props, # This image should be inverted if collimated, pre-unsharp
        cache=False
    )

    if props_raw_list:
        # Assuming the raw_body_mask_bool (after being filtered by main object in segment_main_body)
        # should predominantly contain the main object of interest.
        # If multiple regions still exist in raw_mask, take the largest.
        props_raw = sorted(props_raw_list, key=lambda p: p.area, reverse=True)[0]
        
        features['main_area_raw_s4'] = float(props_raw.area)
        features['solidity_s4'] = float(props_raw.solidity) if hasattr(props_raw, 'solidity') else np.nan
        
        # Use area_oriented_bbox if available, else area_bbox for compactness_rect_s4
        if hasattr(props_raw, 'area_oriented_bbox'):
             oriented_bbox_area = props_raw.area_oriented_bbox
        elif hasattr(props_raw, 'bbox_area'): # skimage < 0.19 uses bbox_area
            oriented_bbox_area = props_raw.bbox_area
        elif hasattr(props_raw, 'area_bbox'): # skimage >= 0.19 might use area_bbox
            oriented_bbox_area = props_raw.area_bbox
        else: # Fallback if specific bbox area attributes are missing
            minr, minc, maxr, maxc = props_raw.bbox
            oriented_bbox_area = (maxr - minr) * (maxc - minc)

        if oriented_bbox_area > 1e-6:
            features['compactness_rect_s4'] = float(props_raw.area / oriented_bbox_area)
        else:
            features['compactness_rect_s4'] = np.nan
            
        features['extent_raw_s4'] = float(props_raw.extent) if hasattr(props_raw, 'extent') else np.nan
        features['compactness_factor_ni_s4'] = features['extent_raw_s4'] # Alias for consistency if needed
        
        features['eccentricity_raw_s4'] = float(props_raw.eccentricity) if hasattr(props_raw, 'eccentricity') else np.nan
        features['equivalent_diameter_raw_s4'] = float(props_raw.equivalent_diameter_approx) if hasattr(props_raw, 'equivalent_diameter_approx') else (float(props_raw.equivalent_diameter) if hasattr(props_raw, 'equivalent_diameter') else np.nan)
        features['major_axis_length_raw_s4'] = float(props_raw.major_axis_length) if hasattr(props_raw, 'major_axis_length') else np.nan
        features['minor_axis_length_raw_s4'] = float(props_raw.minor_axis_length) if hasattr(props_raw, 'minor_axis_length') else np.nan
        
        if hasattr(props_raw, 'moments_central'):
            # Ensure moments are finite, some can be very small leading to issues
            mc = props_raw.moments_central
            Ixx = float(mc[2,0]) if np.isfinite(mc[2,0]) else 0.0
            Iyy = float(mc[0,2]) if np.isfinite(mc[0,2]) else 0.0
            Ixy = float(mc[1,1]) if np.isfinite(mc[1,1]) else 0.0
            
            features['moment_of_inertia_xx_s4'] = Ixx
            features['moment_of_inertia_yy_s4'] = Iyy
            features['moment_of_inertia_xy_s4'] = Ixy # Note: this is mu_11, not Ixy for rotation

            if Ixx > 1e-9 and Iyy > 1e-9 : 
                denominator_tf = 4 * np.pi * np.sqrt(Ixx * Iyy)
                if denominator_tf > 1e-9:
                    features['type_factor_s4'] = (props_raw.area**2) / denominator_tf
                else: features['type_factor_s4'] = np.nan
            else: features['type_factor_s4'] = np.nan
        
        if hasattr(props_raw, 'perimeter') and props_raw.perimeter > 1e-9 and props_raw.area > 1e-9:
            # perimeter_crofton is often more robust if available
            perimeter_val = props_raw.perimeter_crofton if hasattr(props_raw, 'perimeter_crofton') else props_raw.perimeter
            denominator_hcf = 2 * np.sqrt(np.pi * props_raw.area)
            if denominator_hcf > 1e-9:
                features['heywood_circularity_factor_s4'] = perimeter_val / denominator_hcf
            else: features['heywood_circularity_factor_s4'] = np.nan
        else: features['heywood_circularity_factor_s4'] = np.nan
        
        # Elongation factor: Feret diameter / min_axis_length or equivalent
        # Using major_axis_length / minor_axis_length if feret_diameter_max is not directly available or problematic
        if features['minor_axis_length_raw_s4'] > 1e-9 and features['major_axis_length_raw_s4'] > 1e-9 :
            features['elongation_factor_s4'] = features['major_axis_length_raw_s4'] / features['minor_axis_length_raw_s4']
        elif hasattr(props_raw, 'feret_diameter_max') and features['minor_axis_length_raw_s4'] > 1e-9:
             features['elongation_factor_s4'] = props_raw.feret_diameter_max / features['minor_axis_length_raw_s4']
        else: features['elongation_factor_s4'] = np.nan
    else:
        logger.warning("No regionprops found for raw_body_mask in detailed morphology calculation.")

    # Hu moments from raw mask (as before)
    moments_raw_cv = cv2.moments(raw_body_mask.astype(np.uint8)) # raw_body_mask is already uint8
    hu_moments_cv = cv2.HuMoments(moments_raw_cv).flatten()
    if len(hu_moments_cv) >= 1:
        val_hu1 = float(hu_moments_cv[0])
        features['hu_moment1_s4'] = val_hu1 if np.isfinite(val_hu1) else np.nan
    else: features['hu_moment1_s4'] = np.nan

    # Filled area and Porosity (as before)
    filled_area = np.sum(filled_body_mask > 0) # filled_body_mask is uint8
    features['main_area_filled_s4'] = float(filled_area)
    
    raw_area_val_for_porosity = features.get('main_area_raw_s4')
    if isinstance(raw_area_val_for_porosity, (int, float)) and \
       np.isfinite(raw_area_val_for_porosity) and filled_area > 1e-6:
         features['porosity_s4'] = 1.0 - (raw_area_val_for_porosity / filled_area)
    else: features['porosity_s4'] = np.nan

    # Final type cleanup
    for key, value in features.items():
        if isinstance(value, (np.float32, np.float64)): features[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)): features[key] = int(value)
        elif isinstance(value, float) and not np.isfinite(value): features[key] = None # Use None for JSON/CSV friendliness
    return features


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_img_gray_uint8 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(test_img_gray_uint8, (100,100), (80,40), angle=30, startAngle=0, endAngle=360, color=150, thickness=-1)
    cv2.ellipse(test_img_gray_uint8, (90,90), (15,10), angle=30, startAngle=0, endAngle=360, color=0, thickness=-1) # A hole
    test_img_gray_float_orig = skimage.util.img_as_float(test_img_gray_uint8) 

    # Test with source_image_illumination_type = 'darkfield'
    # These params will be passed to segment_main_body_from_image_float
    dummy_s3_params_darkfield = {
        'source_image_illumination_type': 'darkfield',
        'thresholding_mode': 'otsu', 
        'otsu_threshold_multiplier': 0.4, 
        'fixed_threshold_value': 70, 
        'enable_unsharp_masking': False, # Test without unsharp
        'unsharp_mask_sigma': 1.0,
        'unsharp_mask_amount': 1.0,
        'min_noise_size': 10,
        'closing_selem_size': 0 
    }
    print("\n--- Testing Mask Reconstruction (Darkfield Source, No Unsharp, via Shared Seg) ---")
    raw_mask_df, filled_mask_df, props_df_recalc = reconstruct_main_body_masks(test_img_gray_float_orig.copy(), dummy_s3_params_darkfield)
    
    if raw_mask_df is not None and filled_mask_df is not None:
        print(f"Darkfield source - Reconstructed Raw mask area: {np.sum(raw_mask_df > 0)}")
        # For detailed morphology, intensity image should be what props were based on in segment_main_body
        # which is image_gray_float_orig (since darkfield, no inversion) and pre-unsharp.
        detailed_features_df = calculate_detailed_body_morphology(raw_mask_df, filled_mask_df, test_img_gray_float_orig)
        print("Darkfield source - Detailed Body Morphology Features:")
        for k, v in detailed_features_df.items(): print(f"  {k}: {v}")
        if props_df_recalc: print(f"  Area from main_props_object (S4 recalc): {props_df_recalc.area}")
    else:
        print("Darkfield source - Mask reconstruction failed.")

    # Test with source_image_illumination_type = 'collimated' and unsharp masking
    dummy_s3_params_collimated_unsharp = {
        'source_image_illumination_type': 'collimated', 
        'thresholding_mode': 'otsu', 
        'otsu_threshold_multiplier': 0.4, 
        'fixed_threshold_value': 70,
        'enable_unsharp_masking': True, 
        'unsharp_mask_sigma': 0.5,
        'unsharp_mask_amount': 1.2,
        'min_noise_size': 10,
        'closing_selem_size': 0
    }
    print("\n--- Testing Mask Reconstruction (Collimated Source, Unsharp Enabled, via Shared Seg) ---")
    raw_mask_coll_us, filled_mask_coll_us, props_coll_us_recalc = reconstruct_main_body_masks(test_img_gray_float_orig.copy(), dummy_s3_params_collimated_unsharp)
    
    if raw_mask_coll_us is not None and filled_mask_coll_us is not None:
        print(f"Collimated source (unsharp) - Reconstructed Raw mask area: {np.sum(raw_mask_coll_us > 0)}")
        # Intensity image for detailed_morphology_calculator: original image inverted, but PRE-unsharp
        test_img_gray_float_inverted_pre_unsharp = 1.0 - test_img_gray_float_orig 
        detailed_features_coll_us = calculate_detailed_body_morphology(raw_mask_coll_us, filled_mask_coll_us, test_img_gray_float_inverted_pre_unsharp)
        print("Collimated source (unsharp) - Detailed Body Morphology Features:")
        for k, v in detailed_features_coll_us.items(): print(f"  {k}: {v}")
        if props_coll_us_recalc: print(f"  Area from main_props_object (S4 recalc, unsharp): {props_coll_us_recalc.area}")

    else:
        print("Collimated source (unsharp) - Mask reconstruction failed.")