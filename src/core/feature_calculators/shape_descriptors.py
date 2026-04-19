"""Texture descriptor calculations for particle crops."""

import numpy as np
from typing import Dict, Any, List
import logging
from skimage.feature import local_binary_pattern

from src.utils.constants import LBP_DEFAULT_RADII, LBP_DEFAULT_POINTS

logger = logging.getLogger(__name__)

def compute_texture_descriptors(
    gray_image: np.ndarray,
    mask: np.ndarray,
    lbp_radii: List[int],
    lbp_points: int,
) -> Dict[str, Any]:
    """
    Compute LBP texture features for a masked region.
    """
    features = {}
    mask_region = mask > 0
    if not np.any(mask_region):
        logger.warning("Mask is empty, cannot compute texture descriptors.")
        return features

    # --- LBP features ---
    lbp_features_list = []
    for r in lbp_radii:
        try:
            lbp = local_binary_pattern(gray_image, P=lbp_points, R=r, method='uniform')
            hist, _ = np.histogram(
                lbp[mask_region],
                bins=np.arange(lbp_points + 3),
                range=(0, lbp_points + 2),
                density=True
            )
            lbp_features_list.append(hist)
        except Exception as e:
            logger.error(f"Failed to compute LBP for radius {r}: {e}")
            lbp_features_list.append(np.full(lbp_points + 2, np.nan))
    
    flat_lbp_features = np.concatenate(lbp_features_list)
    for i, val in enumerate(flat_lbp_features):
        features[f'lbp_{i}'] = val

    return features

def calculate_shape_descriptors(
    image_crop_gray: np.ndarray,
    filled_mask: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrator to calculate all advanced shape and texture descriptors
    using parameters from the configuration.
    """
    all_shape_features = {}

    # --- Get parameters from config with safe defaults ---
    shape_config = config.get('shape_descriptors', {})
    lbp_radii_to_use = shape_config.get('lbp_radii', LBP_DEFAULT_RADII)
    lbp_points_to_use = shape_config.get('lbp_points', LBP_DEFAULT_POINTS)

    texture_features = compute_texture_descriptors(
        image_crop_gray,
        filled_mask,
        lbp_radii=lbp_radii_to_use,
        lbp_points=lbp_points_to_use
    )
    all_shape_features.update(texture_features)

    return all_shape_features