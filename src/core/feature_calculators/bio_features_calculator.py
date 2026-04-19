# src/core/feature_calculators/bio_features_calculator.py
import logging
import os
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import skimage.filters
import skimage.io
import skimage.measure
import skimage.util
from skimage.measure._regionprops import RegionProperties

from src.utils.common_exceptions import ConfigError
from src.core.segmentation import segment_main_body_from_image_float

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_NAMES = [
    'main_area', 'aspect_ratio', 'solidity', 'extent',
    'posterior_area_ratio',
    'total_egg_area', 'egg_ecc',
    'mean_intensity', 'num_components'
]


def calculate_posterior_attributes(
    props_obj: RegionProperties,
    base_image_shape: Tuple[int, int],
    egg_posterior_pixel_fraction_config: float
) -> Tuple[RegionProperties, float, Optional[np.ndarray]]:
    """
    Calculate posterior area ratio and identify posterior coordinates for egg search.

    Attaches ``_axis_vector_pointing_posterior`` and
    ``_posterior_coords_for_egg_search`` to *props_obj* as dynamic attributes
    so downstream visualisation can access them without re-computing.

    Args:
        props_obj: RegionProperties of the main segmented body.
        base_image_shape: (height, width) of the source image.
        egg_posterior_pixel_fraction_config: Fraction of body length to treat
            as the posterior region (must be in (0, 1)).

    Returns:
        Tuple of (annotated props_obj, posterior_area_ratio, posterior_coords).
    """
    posterior_area_ratio = 0.5
    posterior_coords_for_egg_search_arr: Optional[np.ndarray] = None

    if not (hasattr(props_obj, 'orientation') and
            hasattr(props_obj, 'area') and props_obj.area > 0 and
            hasattr(props_obj, 'coords') and
            hasattr(props_obj, 'centroid')):
        logger.debug("Props object missing attributes for posterior calculation.")
        if not hasattr(props_obj, '_axis_vector_pointing_posterior'):
            props_obj._axis_vector_pointing_posterior = np.array([1.0, 0.0])
        return props_obj, posterior_area_ratio, posterior_coords_for_egg_search_arr

    main_orientation = props_obj.orientation
    axis_vector_rc_unit = np.array([np.cos(main_orientation), np.sin(main_orientation)])

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

    if not (0 < egg_posterior_pixel_fraction_config < 1):
        logger.warning(
            f"egg_posterior_pixel_fraction ({egg_posterior_pixel_fraction_config}) "
            "is out of (0,1) range. Defaulting to 0.45."
        )
        egg_posterior_pixel_fraction_config = 0.45

    percentile_to_cut = (1.0 - egg_posterior_pixel_fraction_config) * 100.0
    projections_for_slice = np.dot(
        coords_main_body - centroid_main_body,
        props_obj._axis_vector_pointing_posterior,
    )

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
    """Segment a particle and compute ML-oriented morphology and intensity features."""
    features: Dict[str, Any] = {key: 0.0 for key in DEFAULT_FEATURE_NAMES}
    features['posterior_area_ratio'] = 0.5
    features['egg_ecc'] = 0.0
    features['total_egg_area'] = 0.0
    features['num_components'] = 0

    main_props_obj: Optional[RegionProperties] = None
    egg_area_props_obj: Optional[RegionProperties] = None

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
            logger.debug(f"Main body segmentation failed for {os.path.basename(image_path)}.")
            if filled_body_mask is not None:
                _, num_labels_temp = skimage.measure.label(filled_body_mask > 0, return_num=True, connectivity=2)
                features['num_components'] = int(num_labels_temp)
            return features, None, None

        main_props_obj = main_props_obj_from_seg

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

        egg_posterior_fraction = params['egg_posterior_pixel_fraction']
        main_props_obj, posterior_area_ratio_calc, posterior_coords_for_search = \
            calculate_posterior_attributes(
                main_props_obj,
                img_gray_input_shape,
                egg_posterior_fraction
            )
        features['posterior_area_ratio'] = posterior_area_ratio_calc

        if features['main_area'] < params['min_significant_area']:
            logger.debug(
                f"Main object area {features['main_area']} < min_significant_area. "
                f"Skipping egg detection for {os.path.basename(image_path)}."
            )
            return features, main_props_obj, None

        image_for_egg_detection_float = img_float_original.copy()
        if params['source_image_illumination_type'] == 'collimated':
            image_for_egg_detection_float = 1.0 - image_for_egg_detection_float

        egg_blurred_image = skimage.filters.gaussian(
            image_for_egg_detection_float, sigma=params['gaussian_sigma']
        )

        non_zero_pixels = egg_blurred_image[egg_blurred_image > 1e-6]
        if non_zero_pixels.size == 0:
            egg_initial_mask = np.zeros_like(egg_blurred_image, dtype=bool)
        else:
            egg_thresh_val = np.quantile(non_zero_pixels, params['egg_threshold_quantile'])
            egg_initial_mask = egg_blurred_image > egg_thresh_val

        posterior_pixel_mask = np.zeros(img_gray_input_shape, dtype=bool)
        if posterior_coords_for_search is not None and posterior_coords_for_search.shape[0] > 0:
            valid_rows = np.clip(posterior_coords_for_search[:, 0].astype(int), 0, img_gray_input_shape[0] - 1)
            valid_cols = np.clip(posterior_coords_for_search[:, 1].astype(int), 0, img_gray_input_shape[1] - 1)
            posterior_pixel_mask[valid_rows, valid_cols] = True

        egg_constrained_mask = egg_initial_mask & posterior_pixel_mask
        egg_labeled_mask, egg_num_labels_cand = skimage.measure.label(
            egg_constrained_mask, return_num=True, connectivity=2
        )

        if egg_num_labels_cand > 0 and main_props_obj is not None:
            egg_props_candidates = skimage.measure.regionprops(
                egg_labeled_mask,
                intensity_image=image_for_egg_detection_float,
                cache=False,
            )

            valid_egg_candidates = []
            main_orientation_for_egg_check = getattr(main_props_obj, 'orientation', 0.0)
            for p_egg_candidate in egg_props_candidates:
                if not (params['egg_min_area'] <= p_egg_candidate.area <= params['egg_max_area']):
                    continue
                egg_solidity = getattr(p_egg_candidate, 'solidity', 1.0)
                if not (params['egg_min_solidity'] < egg_solidity < params['egg_max_solidity']):
                    continue
                egg_ecc = getattr(p_egg_candidate, 'eccentricity', 0.0)
                if not (egg_ecc > params['egg_min_eccentricity']):
                    continue
                egg_orientation = getattr(p_egg_candidate, 'orientation', 0.0)
                delta_angle_abs = abs(main_orientation_for_egg_check - egg_orientation)
                acute_delta_angle = min(delta_angle_abs, np.pi - delta_angle_abs)
                orientation_diff_rad = abs(acute_delta_angle - (np.pi / 2.0))
                orientation_tolerance_rad = np.deg2rad(params['orientation_diff_tolerance_deg'])
                if orientation_diff_rad < orientation_tolerance_rad:
                    valid_egg_candidates.append(p_egg_candidate)

            if valid_egg_candidates:
                egg_area_props_obj = max(valid_egg_candidates, key=lambda p: p.area)
                features['total_egg_area'] = float(egg_area_props_obj.area)
                features['egg_ecc'] = getattr(egg_area_props_obj, 'eccentricity', 0.0)

        for key in features:
            if isinstance(features[key], float) and not np.isfinite(features[key]):
                logger.warning(
                    f"Feature '{key}' is non-finite ({features[key]}) "
                    f"for {os.path.basename(image_path)}. Defaulting."
                )
                if key == 'num_components':
                    features[key] = 0
                elif key == 'egg_ecc':
                    features[key] = 0.0
                elif key == 'posterior_area_ratio':
                    features[key] = 0.5
                else:
                    features[key] = 0.0
            if key == 'num_components':
                features[key] = int(features[key])

        return features, main_props_obj, egg_area_props_obj

    except ConfigError as ce:
        logger.error(f"Configuration error during feature extraction for {os.path.basename(image_path)}: {ce}")
        raise
    except Exception as e:
        logger.exception(f"Error extracting ML features for {os.path.basename(image_path)}: {e}")
        return None, None, None
