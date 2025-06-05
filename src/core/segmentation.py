# src/core/segmentation.py
import skimage.filters
import skimage.morphology
import skimage.util
import numpy as np
from scipy import ndimage as ndi
import logging
from typing import Dict, Any, Optional, Tuple
from skimage.measure._regionprops import RegionProperties # If returning props

from src.utils.common_exceptions import ConfigError

logger = logging.getLogger(__name__)

def segment_main_body_from_image_float(
    image_gray_float_input: np.ndarray, # Expected to be original float [0,1], NOT YET INVERTED
    segmentation_params: Dict[str, Any] # Combined params: illum, thresh, unsharp masking
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
    """
    Segments the main body from a float grayscale image using shared logic.
    Handles illumination inversion, optional unsharp masking, and thresholding.

    Args:
        image_gray_float_input (np.ndarray): Original input image as float [0,1].
        segmentation_params (Dict[str, Any]): Dictionary containing all necessary parameters:
            - 'source_image_illumination_type': 'darkfield' or 'collimated'
            - 'thresholding_mode': 'otsu' or 'fixed'
            - 'otsu_threshold_multiplier': float
            - 'fixed_threshold_value': int (0-255)
            - 'enable_unsharp_masking': bool (optional, defaults to False)
            - 'unsharp_mask_sigma': float (optional, for Gaussian blur in unsharp mask)
            - 'unsharp_mask_amount': float (optional, strength of sharpening)
            - 'min_noise_size': int (for remove_small_objects)
            - 'closing_selem_size': int (for binary_closing disk radius)
            - 'min_significant_area': int (used to filter for the main object if multiple found, though typically we take largest)

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
        - raw_body_mask_uint8: The binary mask after cleaning and closing, before filling holes (uint8).
        - filled_body_mask_uint8: The binary mask after filling holes (uint8).
        - main_props_object: RegionProperties of the largest filled object.
        Returns (None, None, None) on critical error.
    """
    if not np.issubdtype(image_gray_float_input.dtype, np.floating):
        logger.warning("segment_main_body received non-float image. Converting.")
        image_proc_float = skimage.util.img_as_float(image_gray_float_input)
    else:
        image_proc_float = image_gray_float_input.copy()

    # 1. Handle Illumination
    source_illumination_type = segmentation_params.get('source_image_illumination_type', 'darkfield')
    if source_illumination_type not in ['darkfield', 'collimated']:
        logger.warning(f"Invalid source_image_illumination_type '{source_illumination_type}'. Defaulting to 'darkfield'.")
        source_illumination_type = 'darkfield'

    image_for_segmentation_float = image_proc_float.copy()
    if source_illumination_type == 'collimated':
        logger.debug(f"SegmentMainBody: Source Illumination '{source_illumination_type}'. Inverting image.")
        image_for_segmentation_float = 1.0 - image_for_segmentation_float
    
    # image_for_props will be this potentially inverted image
    image_intensity_for_props = image_for_segmentation_float.copy()

    if image_for_segmentation_float.max() < 1e-9: # Check after potential inversion
        logger.debug("SegmentMainBody: Image is empty or near-empty after potential inversion. No segmentation.")
        return None, None, None

    # 2. Optional Unsharp Masking
    if segmentation_params.get('enable_unsharp_masking', False):
        sigma = segmentation_params.get('unsharp_mask_sigma', 1.0)
        amount = segmentation_params.get('unsharp_mask_amount', 1.0)
        if sigma > 0 and amount > 0: # Only apply if params are meaningful
            logger.debug(f"SegmentMainBody: Applying Unsharp Masking (sigma={sigma}, amount={amount}).")
            blurred_for_unsharp = skimage.filters.gaussian(image_for_segmentation_float, sigma=sigma, preserve_range=True, channel_axis=None)
            image_for_segmentation_float = image_for_segmentation_float + amount * (image_for_segmentation_float - blurred_for_unsharp)
            image_for_segmentation_float = np.clip(image_for_segmentation_float, 0, 1)
        else:
            logger.debug("SegmentMainBody: Unsharp masking enabled but sigma or amount is zero/negative. Skipping.")


    # 3. Thresholding (No pre-segmentation blur for main body here)
    image_to_threshold_float = image_for_segmentation_float # Use the (potentially unsharp-masked) image

    if image_to_threshold_float.min() == image_to_threshold_float.max():
        logger.debug("SegmentMainBody: Image became blank or uniform before thresholding. No segmentation.")
        return None, None, None

    thresholding_mode = segmentation_params.get('thresholding_mode', 'otsu')
    final_threshold_value_float: Optional[float] = None

    if thresholding_mode == 'otsu':
        try:
            otsu_thresh_val_raw_sk = skimage.filters.threshold_otsu(image_to_threshold_float)
        except ValueError:
            logger.warning("SegmentMainBody: Skimage Otsu threshold failed. Skipping segmentation.")
            return None, None, None
        otsu_multiplier = segmentation_params.get('otsu_threshold_multiplier', 0.45)
        final_threshold_value_float = otsu_thresh_val_raw_sk * otsu_multiplier
    elif thresholding_mode == 'fixed':
        fixed_thresh_val_config_int = segmentation_params.get('fixed_threshold_value')
        if fixed_thresh_val_config_int is None:
            raise ConfigError("SegmentMainBody: thresholding_mode is 'fixed' but 'fixed_threshold_value' is missing.")
        try:
            fixed_thresh_int = int(fixed_thresh_val_config_int)
        except ValueError:
            raise ConfigError(f"SegmentMainBody: 'fixed_threshold_value' ({fixed_thresh_val_config_int}) is not a valid int.")
        final_threshold_value_float = fixed_thresh_int / 255.0
    else:
        raise ConfigError(f"SegmentMainBody: Invalid thresholding_mode: '{thresholding_mode}'.")

    if final_threshold_value_float is None:
         logger.error("SegmentMainBody: Final threshold value (float) not set.")
         return None, None, None

    main_binary_mask_initial_bool = image_to_threshold_float > final_threshold_value_float

    # 4. Post-processing of mask
    main_cleaned_mask_bool = skimage.morphology.remove_small_objects(
        main_binary_mask_initial_bool, min_size=segmentation_params.get('min_noise_size', 50)
    )

    selem_closing_size = segmentation_params.get('closing_selem_size', 3)
    if selem_closing_size > 0:
        selem_closing = skimage.morphology.disk(selem_closing_size) # Use disk as it's common
        raw_body_mask_bool = skimage.morphology.binary_closing(main_cleaned_mask_bool, selem_closing)
    else:
        raw_body_mask_bool = main_cleaned_mask_bool

    filled_body_mask_bool = ndi.binary_fill_holes(raw_body_mask_bool)

    # 5. Get RegionProps for the largest filled object
    labeled_filled_mask, num_labels = skimage.measure.label(filled_body_mask_bool, return_num=True, connectivity=2)
    
    main_props_object = None
    if num_labels > 0:
        # Use image_intensity_for_props (which is the potentially inverted, but pre-enhancement image)
        # for consistent intensity measurements if needed by props.
        props_filled_all = skimage.measure.regionprops(
            labeled_filled_mask,
            intensity_image=image_intensity_for_props, # Important: Use the image before unsharp masking for intensity props
            cache=False
        )
        if props_filled_all:
            # Sort by area and take the largest if multiple objects are found
            # (though ideally, for single particle crops, there's only one significant object)
            main_props_object = sorted(props_filled_all, key=lambda x: x.area, reverse=True)[0]
            
            # Filter out the main object from the raw_body_mask_bool and filled_body_mask_bool
            # to ensure only the main object is represented in the returned masks.
            # This is important if remove_small_objects or closing didn't perfectly isolate one object.
            temp_main_mask = np.zeros_like(labeled_filled_mask, dtype=bool)
            temp_main_mask[labeled_filled_mask == main_props_object.label] = True
            
            raw_body_mask_bool_main_only = raw_body_mask_bool & temp_main_mask
            filled_body_mask_bool_main_only = filled_body_mask_bool & temp_main_mask # Should be redundant if filled from raw
    else:
        logger.debug("SegmentMainBody: No regions found after filling holes.")
        raw_body_mask_bool_main_only = raw_body_mask_bool # Return the (empty) raw mask
        filled_body_mask_bool_main_only = filled_body_mask_bool # Return the (empty) filled mask

    raw_mask_uint8 = raw_body_mask_bool_main_only.astype(np.uint8) * 255
    filled_mask_uint8 = filled_body_mask_bool_main_only.astype(np.uint8) * 255
    
    return raw_mask_uint8, filled_mask_uint8, main_props_object

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Create a dummy image (e.g., dark particle on bright background for collimated test)
    test_image_orig_float = np.ones((100, 100), dtype=float) * 0.8 # Bright background
    rr, cc = skimage.draw.ellipse(50, 50, 20, 30, rotation=np.deg2rad(15))
    test_image_orig_float[rr, cc] = 0.2 # Dark particle
    test_image_orig_float[40:45, 40:45] = 0.1 # A smaller, darker spot

    test_params_collimated_unsharp = {
        'source_image_illumination_type': 'collimated',
        'thresholding_mode': 'otsu',
        'otsu_threshold_multiplier': 0.5,
        'enable_unsharp_masking': True,
        'unsharp_mask_sigma': 1.0,
        'unsharp_mask_amount': 1.5,
        'min_noise_size': 10,
        'closing_selem_size': 2,
    }

    logger.info("--- Testing segment_main_body (collimated, unsharp) ---")
    raw_m, filled_m, props = segment_main_body_from_image_float(test_image_orig_float.copy(), test_params_collimated_unsharp)

    if raw_m is not None and filled_m is not None:
        logger.info(f"Raw mask sum: {np.sum(raw_m > 0)}, Filled mask sum: {np.sum(filled_m > 0)}")
        if props:
            logger.info(f"Main props area: {props.area}")
            # For visual check:
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            # axes[0].imshow(test_image_orig_float, cmap='gray'); axes[0].set_title("Original")
            # axes[1].imshow(raw_m, cmap='gray'); axes[1].set_title("Raw Mask")
            # axes[2].imshow(filled_m, cmap='gray'); axes[2].set_title("Filled Mask")
            # if props.coords is not None:
            #     axes[2].plot(props.coords[:, 1], props.coords[:, 0], '.r', markersize=2)
            # plt.show()
        else:
            logger.info("No main props returned.")
    else:
        logger.error("Segmentation failed.")

    test_params_darkfield_no_unsharp = {
        'source_image_illumination_type': 'darkfield',
        'thresholding_mode': 'fixed',
        'fixed_threshold_value': int(0.4 * 255), # e.g. 102
        'enable_unsharp_masking': False,
        'min_noise_size': 10,
        'closing_selem_size': 0,
    }
    # Create a bright particle on dark background for darkfield test
    test_image_darkfield_float = np.ones((100, 100), dtype=float) * 0.1 # Dark background
    rr_df, cc_df = skimage.draw.ellipse(50, 50, 20, 30, rotation=np.deg2rad(15))
    test_image_darkfield_float[rr_df, cc_df] = 0.7 # Bright particle

    logger.info("--- Testing segment_main_body (darkfield, no unsharp, fixed thresh) ---")
    raw_m_df, filled_m_df, props_df = segment_main_body_from_image_float(test_image_darkfield_float.copy(), test_params_darkfield_no_unsharp)
    if raw_m_df is not None and filled_m_df is not None:
        logger.info(f"Darkfield - Raw mask sum: {np.sum(raw_m_df > 0)}, Filled mask sum: {np.sum(filled_m_df > 0)}")
        if props_df: logger.info(f"Darkfield - Main props area: {props_df.area}")