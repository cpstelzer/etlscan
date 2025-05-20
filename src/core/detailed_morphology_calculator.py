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

logger = logging.getLogger(__name__)

def reconstruct_main_body_masks(
    image_gray_original: np.ndarray,
    s3_feature_extraction_params: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
    """
    Reconstructs the raw (unfilled) and filled main body masks using Stage 3 parameters.
    (Identical to previous version)
    """
    try:
        if not np.issubdtype(image_gray_original.dtype, np.floating):
            image_gray = skimage.util.img_as_float(image_gray_original)
        else:
            image_gray = image_gray_original.copy()

        blurred_image = skimage.filters.gaussian(
            image_gray, sigma=s3_feature_extraction_params.get('gaussian_sigma', 1.0)
        )

        if blurred_image.min() == blurred_image.max():
            logger.warning("Image became blank after blurring during mask reconstruction.")
            return None, None, None

        try:
            otsu_thresh = skimage.filters.threshold_otsu(blurred_image)
        except ValueError:
            logger.warning("Otsu threshold failed during mask reconstruction.")
            return None, None, None

        adjusted_thresh = otsu_thresh * s3_feature_extraction_params.get('otsu_threshold_multiplier', 0.9)
        main_binary_mask_initial = blurred_image > adjusted_thresh

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

    except Exception as e:
        logger.exception(f"Error reconstructing main body masks: {e}")
        return None, None, None

# estimate_ideal_body_mask_and_area and its helpers are REMOVED

def calculate_detailed_body_morphology(
    raw_body_mask: Optional[np.ndarray],      # uint8, 0 or 255
    filled_body_mask: Optional[np.ndarray],   # uint8, 0 or 255
    intensity_image_for_props: Optional[np.ndarray] = None
    # ideal_body_mask parameter removed
) -> Dict[str, Any]:
    """
    Calculates detailed morphological features for the main body.
    `main_area_ideal_s4` is removed.
    (Identical to previous version where ideal_body_mask was removed as a parameter)
    """
    features: Dict[str, Any] = {}

    keys_to_init = ['main_area_raw_s4', 'solidity_s4', 'compactness_rect_s4', 'extent_raw_s4',
                    'eccentricity_raw_s4', 'equivalent_diameter_raw_s4', 
                    'major_axis_length_raw_s4', 'minor_axis_length_raw_s4',
                    'main_area_filled_s4', 'porosity_s4'] + \
                   [f'hu_moment{i+1}_s4' for i in range(7)]
    for k_init in keys_to_init: features[k_init] = np.nan


    if raw_body_mask is None or filled_body_mask is None:
        logger.warning("Raw or filled body mask is None. Cannot calculate detailed morphology.")
        return features

    raw_body_mask_bool = raw_body_mask > 0
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
        features['eccentricity_raw_s4'] = float(props_raw.eccentricity) if hasattr(props_raw, 'eccentricity') else np.nan
        features['equivalent_diameter_raw_s4'] = float(props_raw.equivalent_diameter) if hasattr(props_raw, 'equivalent_diameter') else np.nan
        features['major_axis_length_raw_s4'] = float(props_raw.major_axis_length) if hasattr(props_raw, 'major_axis_length') else np.nan
        features['minor_axis_length_raw_s4'] = float(props_raw.minor_axis_length) if hasattr(props_raw, 'minor_axis_length') else np.nan
    else:
        logger.warning("No regionprops found for raw_body_mask.")

    moments_raw = cv2.moments(raw_body_mask.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments_raw).flatten()
    for i in range(7):
        val = float(hu_moments[i])
        features[f'hu_moment{i+1}_s4'] = val if np.isfinite(val) else np.nan
    
    filled_area = np.sum(filled_body_mask > 0)
    features['main_area_filled_s4'] = float(filled_area)
    
    raw_area_val_for_porosity = features.get('main_area_raw_s4')
    if isinstance(raw_area_val_for_porosity, (int, float)) and np.isfinite(raw_area_val_for_porosity) and filled_area > 1e-6:
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
    test_img_gray = np.zeros((200, 200), dtype=np.uint8) 
    cv2.ellipse(test_img_gray, (100,100), (80,40), angle=30, startAngle=0, endAngle=360, color=150, thickness=-1)
    cv2.ellipse(test_img_gray, (90,90), (15,10), angle=30, startAngle=0, endAngle=360, color=0, thickness=-1)
    
    dummy_s3_params = {
        'gaussian_sigma': 1.0,
        'otsu_threshold_multiplier': 0.4, 
        'min_noise_size': 10,
        'closing_selem_size': 0
    }

    raw_mask, filled_mask, props_filled = reconstruct_main_body_masks(test_img_gray, dummy_s3_params)

    if raw_mask is not None and filled_mask is not None and props_filled is not None:
        logger.info(f"Reconstructed Raw mask area: {np.sum(raw_mask > 0)}")
        logger.info(f"Reconstructed Filled mask area: {np.sum(filled_mask > 0)}")

        # ideal_body_mask is no longer calculated or passed here
        detailed_features = calculate_detailed_body_morphology(raw_mask, filled_mask, test_img_gray)
        print("\nDetailed Body Morphology Features (no ideal, no egg specific):")
        for k, v in detailed_features.items():
            print(f"  {k}: {v}")
        assert 'main_area_ideal_s4' not in detailed_features 
    else:
        logger.error("Mask reconstruction failed for test image in __main__.")