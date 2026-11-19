# src/core/feature_calculators/bio_features_calculator.py
import logging
from typing import Optional, Tuple

import numpy as np
from skimage.measure._regionprops import RegionProperties

from src.utils.common_exceptions import ConfigError

logger = logging.getLogger(__name__)


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
        raise ConfigError(
            "stage_04_biological_analysis.egg_detection.egg_posterior_pixel_fraction must be in (0, 1), "
            f"got {egg_posterior_pixel_fraction_config!r}"
        )

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
