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
import skimage.draw # For polygon drawing
from skimage.measure._regionprops import RegionProperties # MODIFIED: Import RegionProperties directly

logger = logging.getLogger(__name__)

def reconstruct_main_body_masks(
    image_gray_original: np.ndarray,
    s3_feature_extraction_params: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]: # MODIFIED: Use the imported RegionProperties
    """
    Reconstructs the raw (unfilled) and filled main body masks using Stage 3 parameters.

    Args:
        image_gray_original (np.ndarray): The original grayscale image of the particle.
        s3_feature_extraction_params (Dict[str, Any]): Feature extraction parameters
                                                       from Stage 3 configuration.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], RegionProperties]:
            - raw_body_mask: The main body mask before hole filling.
            - filled_body_mask: The main body mask after hole filling.
            - main_props_filled: RegionProperties object for the filled_body_mask.
            Returns (None, None, None) on failure.
    """
    try:
        if not np.issubdtype(image_gray_original.dtype, np.floating):
            image_gray = skimage.util.img_as_float(image_gray_original)
        else:
            image_gray = image_gray_original.copy()

        # --- Replicate Segmentation from Stage 3 ml_feature_extractor ---
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


def estimate_ideal_body_mask_and_area(
    filled_body_mask_input: Optional[np.ndarray], # uint8 mask, 0 or 255
    egg_area_coords: Optional[np.ndarray], # Nx2 array of (row, col) from S3 egg props
    image_shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], float]:
    """
    Estimates an "ideal" body mask by subtracting the egg area from the filled body mask
    and taking the largest remaining connected component. This does not "reconstruct"
    a smooth posterior but rather removes the egg occlusion.

    Args:
        filled_body_mask_input (np.ndarray): The filled binary mask of the main body.
        egg_area_coords (np.ndarray): Coordinates of the egg area from Stage 3.
        image_shape (Tuple[int, int]): Shape (height, width) of the original image.

    Returns:
        Tuple[Optional[np.ndarray], float]:
            - ideal_body_mask_s4 (uint8)
            - main_area_ideal_s4 (float)
            Returns (None, 0.0) on failure or if inputs are unsuitable.
    """
    if filled_body_mask_input is None or filled_body_mask_input.sum() == 0:
        logger.warning("estimate_ideal_body_mask: filled_body_mask is empty or None.")
        return None, 0.0

    filled_body_mask = filled_body_mask_input.copy() # Work on a copy

    egg_mask = np.zeros(image_shape, dtype=np.uint8)
    if egg_area_coords is not None and egg_area_coords.shape[0] > 0:
        valid_rows = np.clip(egg_area_coords[:, 0], 0, image_shape[0] - 1).astype(int)
        valid_cols = np.clip(egg_area_coords[:, 1], 0, image_shape[1] - 1).astype(int)
        egg_mask[valid_rows, valid_cols] = 255
    
    # Dilate the egg mask slightly to ensure good subtraction from the body contour
    # The kernel size and iterations can be tuned.
    dilation_kernel_size = 5 # Experiment with this
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_egg_mask = cv2.dilate(egg_mask, kernel, iterations=1)
    else:
        dilated_egg_mask = egg_mask

    # Subtract the dilated egg mask from the filled body mask
    body_without_eggs_mask = cv2.bitwise_and(
        filled_body_mask,
        cv2.bitwise_not(dilated_egg_mask)
    )

    if body_without_eggs_mask.sum() == 0:
        logger.warning("estimate_ideal_body_mask: Body mask became empty after egg subtraction. "
                       "Falling back to original filled_body_mask.")
        return filled_body_mask, float(np.sum(filled_body_mask > 0))

    # Find connected components and take the largest one
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        body_without_eggs_mask, connectivity=8
    )

    if num_labels <= 1: # Only background component
        logger.warning("estimate_ideal_body_mask: No foreground components after egg subtraction. "
                       "Falling back to original filled_body_mask.")
        return filled_body_mask, float(np.sum(filled_body_mask > 0))

    # Find the largest component (excluding background at index 0)
    largest_label_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    ideal_body_mask_s4 = np.zeros(image_shape, dtype=np.uint8)
    ideal_body_mask_s4[labels_map == largest_label_idx] = 255
    
    main_area_ideal_s4 = float(stats[largest_label_idx, cv2.CC_STAT_AREA])
    
    logger.debug(f"estimate_ideal_body_mask (subtraction method): Calculated ideal area {main_area_ideal_s4}")
    return ideal_body_mask_s4, main_area_ideal_s4

def calculate_detailed_body_morphology(
    raw_body_mask: np.ndarray,      # uint8, 0 or 255
    filled_body_mask: np.ndarray,   # uint8, 0 or 255
    ideal_body_mask: Optional[np.ndarray], # uint8, 0 or 255, new!
    intensity_image_for_props: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates detailed morphological features for the main body.
    Now includes main_area_ideal_s4.
    """
    features: Dict[str, Any] = {}

    keys_to_init = ['main_area_raw_s4', 'solidity_s4', 'compactness_rect_s4', 'extent_raw_s4',
                    'eccentricity_raw_s4', 'equivalent_diameter_raw_s4', 
                    'major_axis_length_raw_s4', 'minor_axis_length_raw_s4',
                    'main_area_filled_s4', 'porosity_s4', 'main_area_ideal_s4'] + \
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

        if oriented_bbox_area > 1e-6: # Avoid division by zero
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
        # NaNs already initialized

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
        
    if ideal_body_mask is not None:
        features['main_area_ideal_s4'] = float(np.sum(ideal_body_mask > 0))
    else:
        features['main_area_ideal_s4'] = np.nan
        
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
    
    egg_center_y, egg_center_x = 100 + int(60*np.sin(np.deg2rad(30))), 100 + int(60*np.cos(np.deg2rad(30))) 
    rr_egg, cc_egg = skimage.draw.ellipse(egg_center_y, egg_center_x, 30, 25, rotation=np.deg2rad(30+90), shape=test_img_gray.shape)
    simulated_egg_coords = np.array(list(zip(rr_egg, cc_egg)))

    dummy_s3_params = {
        'gaussian_sigma': 1.0,
        'otsu_threshold_multiplier': 0.4, 
        'min_noise_size': 10,
        'closing_selem_size': 0
    }

    raw_mask, filled_mask, props_filled = reconstruct_main_body_masks(test_img_gray, dummy_s3_params)

    if raw_mask is not None and filled_mask is not None:
        logger.info(f"Reconstructed Raw mask area: {np.sum(raw_mask > 0)}")
        logger.info(f"Reconstructed Filled mask area: {np.sum(filled_mask > 0)}")

        ideal_mask, ideal_area = estimate_ideal_body_mask_and_area(
            filled_mask, simulated_egg_coords, test_img_gray.shape
        )

        if ideal_mask is not None:
            logger.info(f"Estimated Ideal Body Area: {ideal_area}")
            
            # Visualization test
            vis_debug = cv2.cvtColor(test_img_gray.copy(), cv2.COLOR_GRAY2BGR)
            vis_debug_egg_overlay = np.zeros_like(vis_debug)

            if simulated_egg_coords is not None and simulated_egg_coords.shape[0] > 0:
                egg_mask_vis = np.zeros(test_img_gray.shape, dtype=np.uint8)
                egg_mask_vis[simulated_egg_coords[:,0], simulated_egg_coords[:,1]] = 255
                vis_debug_egg_overlay[egg_mask_vis > 0] = [0,100,0] # Dark green for egg area

            if filled_mask is not None:
                contours_fm, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_debug, contours_fm, -1, (0,255,0), 1) 
            if ideal_mask is not None:
                contours_im, _ = cv2.findContours(ideal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_debug, contours_im, -1, (255,0,0), 1)
            
            # combined_vis = cv2.addWeighted(vis_debug, 1, vis_debug_egg_overlay, 0.4, 0)
            # cv2.imshow("Ideal Body Mask Debug", combined_vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        detailed_features = calculate_detailed_body_morphology(raw_mask, filled_mask, ideal_mask, test_img_gray)
        print("\nDetailed Body Morphology Features (with ideal):")
        for k, v in detailed_features.items():
            print(f"  {k}: {v}")
        assert 'main_area_ideal_s4' in detailed_features
        if ideal_mask is not None:
            if ideal_area > 0 and detailed_features.get('main_area_filled_s4', 0) > 0 and simulated_egg_coords is not None:
                 print(f"Area check: Filled={detailed_features['main_area_filled_s4']}, Ideal={ideal_area}")
    else:
        logger.error("Mask reconstruction failed for test image in __main__.")