# src/core/segmentation.py
import skimage.filters
import skimage.morphology
import skimage.util
import numpy as np
from scipy import ndimage as ndi
import logging
from typing import Dict, Any, Optional, Tuple
from skimage.measure._regionprops import RegionProperties 

from src.utils.common_exceptions import ConfigError

logger = logging.getLogger(__name__)

def generate_binary_mask_from_float_image(
    image_processed_float: np.ndarray, # Expected to be float [0,1], already inverted and enhanced if applicable
    threshold_params: Dict[str, Any]
) -> Optional[np.ndarray]: # Returns a boolean mask
    """
    Generates a binary boolean mask from a processed float image using specified thresholding parameters.

    Args:
        image_processed_float (np.ndarray): Float image [0,1] ready for thresholding.
                                            (Assumed to be already inverted and enhanced if applicable).
        threshold_params (Dict[str, Any]): Dictionary containing:
            - 'thresholding_mode': 'otsu' or 'fixed'
            - 'otsu_threshold_multiplier': float
            - 'fixed_threshold_value': int (0-255)

    Returns:
        Optional[np.ndarray]: Boolean binary mask, or None on error.
    """
    if image_processed_float.min() == image_processed_float.max():
        logger.debug("generate_binary_mask: Image is uniform. Returning empty mask.")
        return np.zeros_like(image_processed_float, dtype=bool)

    thresholding_mode = threshold_params["thresholding_mode"]
    final_threshold_value_float: Optional[float] = None

    if thresholding_mode == 'otsu':
        try:
            otsu_thresh_val_raw_sk = skimage.filters.threshold_otsu(image_processed_float)
        except ValueError:
            logger.warning("generate_binary_mask: Skimage Otsu threshold failed. Returning empty mask.")
            return np.zeros_like(image_processed_float, dtype=bool)
        otsu_multiplier = threshold_params["otsu_threshold_multiplier"]
        final_threshold_value_float = otsu_thresh_val_raw_sk * otsu_multiplier
        logger.debug(f"generate_binary_mask (Otsu): raw={otsu_thresh_val_raw_sk:.4f}, mult={otsu_multiplier}, final={final_threshold_value_float:.4f}")
    elif thresholding_mode == 'fixed':
        fixed_thresh_val_config_int = threshold_params.get('fixed_threshold_value')
        if fixed_thresh_val_config_int is None:
            raise ConfigError("generate_binary_mask: thresholding_mode is 'fixed' but 'fixed_threshold_value' is missing.")
        try:
            fixed_thresh_int = int(fixed_thresh_val_config_int)
        except ValueError:
            raise ConfigError(f"generate_binary_mask: 'fixed_threshold_value' ({fixed_thresh_val_config_int}) is not a valid int.")
        final_threshold_value_float = fixed_thresh_int / 255.0
        logger.debug(f"generate_binary_mask (Fixed): val_int={fixed_thresh_int}, final={final_threshold_value_float:.4f}")
    else:
        raise ConfigError(f"generate_binary_mask: Invalid thresholding_mode: '{thresholding_mode}'.")

    if final_threshold_value_float is None: # Should be caught by logic above
         logger.error("generate_binary_mask: Final threshold value (float) not set.")
         return None
    
    return image_processed_float > final_threshold_value_float


def segment_main_body_from_image_float(
    image_gray_float_input: np.ndarray, 
    segmentation_params: Dict[str, Any] 
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[RegionProperties]]:
    """
    Segments the main body from a float grayscale image using shared logic.
    Handles illumination inversion, optional unsharp masking, and thresholding.
    MODIFIED: Now uses generate_binary_mask_from_float_image helper.
    """
    if not np.issubdtype(image_gray_float_input.dtype, np.floating):
        logger.warning("segment_main_body received non-float image. Converting.")
        image_proc_float = skimage.util.img_as_float(image_gray_float_input)
    else:
        image_proc_float = image_gray_float_input.copy()

    source_illumination_type = segmentation_params["source_image_illumination_type"]
    if source_illumination_type not in ['darkfield', 'collimated']:
        raise ConfigError(
            f"Invalid source_image_illumination_type '{source_illumination_type}'. "
            "Expected 'darkfield' or 'collimated'."
        )

    image_for_segmentation_float = image_proc_float.copy()
    if source_illumination_type == 'collimated':
        logger.debug(f"SegmentMainBody: Source Illumination '{source_illumination_type}'. Inverting image.")
        image_for_segmentation_float = 1.0 - image_for_segmentation_float
    
    image_intensity_for_props = image_for_segmentation_float.copy() # This is post-inversion, pre-enhancement

    if image_for_segmentation_float.max() < 1e-9: 
        logger.debug("SegmentMainBody: Image is empty or near-empty after potential inversion. No segmentation.")
        return None, None, None

    if segmentation_params["enable_unsharp_masking"]:
        sigma = segmentation_params["unsharp_mask_sigma"]
        amount = segmentation_params["unsharp_mask_amount"]
        if sigma > 0 and amount > 0:
            logger.debug(f"SegmentMainBody: Applying Unsharp Masking (sigma={sigma}, amount={amount}).")
            blurred_for_unsharp = skimage.filters.gaussian(image_for_segmentation_float, sigma=sigma, preserve_range=True, channel_axis=None)
            image_for_segmentation_float = image_for_segmentation_float + amount * (image_for_segmentation_float - blurred_for_unsharp)
            image_for_segmentation_float = np.clip(image_for_segmentation_float, 0, 1)
        else:
            logger.debug("SegmentMainBody: Unsharp masking enabled but sigma or amount is zero/negative. Skipping.")

    # MODIFIED: Call the helper for thresholding
    # The segmentation_params already contain thresholding_mode, otsu_threshold_multiplier, fixed_threshold_value
    main_binary_mask_initial_bool = generate_binary_mask_from_float_image(
        image_for_segmentation_float, # This is post-inversion, post-unsharp
        segmentation_params # Pass all params, helper will pick what it needs
    )

    if main_binary_mask_initial_bool is None:
        logger.error("SegmentMainBody: Failed to generate binary mask from helper.")
        return None, None, None

    main_cleaned_mask_bool = skimage.morphology.remove_small_objects(
        main_binary_mask_initial_bool, min_size=segmentation_params["min_noise_size"]
    )

    selem_closing_size = segmentation_params["closing_selem_size"]
    if selem_closing_size > 0:
        selem_closing = skimage.morphology.disk(selem_closing_size)
        raw_body_mask_bool = skimage.morphology.binary_closing(main_cleaned_mask_bool, selem_closing)
    else:
        raw_body_mask_bool = main_cleaned_mask_bool

    filled_body_mask_bool = ndi.binary_fill_holes(raw_body_mask_bool)
    
    labeled_filled_mask, num_labels = skimage.measure.label(filled_body_mask_bool, return_num=True, connectivity=2)
    
    main_props_object = None
    raw_body_mask_bool_main_only = raw_body_mask_bool
    filled_body_mask_bool_main_only = filled_body_mask_bool

    if num_labels > 0:
        props_filled_all = skimage.measure.regionprops(
            labeled_filled_mask,
            intensity_image=image_intensity_for_props, 
            cache=False
        )
        if props_filled_all:
            main_props_object = sorted(props_filled_all, key=lambda x: x.area, reverse=True)[0]
            
            temp_main_mask = np.zeros_like(labeled_filled_mask, dtype=bool)
            temp_main_mask[labeled_filled_mask == main_props_object.label] = True
            
            raw_body_mask_bool_main_only = raw_body_mask_bool & temp_main_mask
            filled_body_mask_bool_main_only = filled_body_mask_bool & temp_main_mask
    else:
        logger.debug("SegmentMainBody: No regions found after filling holes.")

    raw_mask_uint8 = raw_body_mask_bool_main_only.astype(np.uint8) * 255
    filled_mask_uint8 = filled_body_mask_bool_main_only.astype(np.uint8) * 255
    
    return raw_mask_uint8, filled_mask_uint8, main_props_object