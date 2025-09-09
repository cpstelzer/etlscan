# src/core/feature_calculators/shape_descriptors.py
import cv2
import numpy as np
from typing import Dict, Any, List
import logging
from skimage.feature import local_binary_pattern
from pyefd import elliptic_fourier_descriptors

logger = logging.getLogger(__name__)

def compute_efds(
    contour: np.ndarray,
    harmonics_list: List[int]
) -> Dict[int, np.ndarray]:
    """
    Compute normalized elliptic Fourier descriptors for a given contour.
    """
    if contour is None or len(contour) < 20: # EFD needs sufficient points
        logger.warning("Contour has too few points for EFD calculation.")
        return {}
        
    coeffs = {}
    for order in harmonics_list:
        try:
            coeffs[order] = elliptic_fourier_descriptors(
                np.squeeze(contour), order=order, normalize=True
            )
        except Exception as e:
            logger.error(f"Failed to compute EFD for order {order}: {e}")
            coeffs[order] = np.full((order, 4), np.nan)
            
    return coeffs

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
    mask_bool = mask > 0
    if not np.any(mask_bool):
        logger.warning("Mask is empty, cannot compute texture descriptors.")
        return features

    # --- LBP features ---
    lbp_features_list = []
    for r in lbp_radii:
        try:
            lbp = local_binary_pattern(gray_image, P=lbp_points, R=r, method='uniform')
            hist, _ = np.histogram(
                lbp[mask_bool],
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
    
    # --- MODIFIED: Get parameters from config with safe defaults ---
    shape_config = config.get('shape_descriptors', {})
    efd_harmonics_to_calc = shape_config.get('efd_harmonics', [10])
    lbp_radii_to_use = shape_config.get('lbp_radii', [1, 3])
    lbp_points_to_use = shape_config.get('lbp_points', 8)

    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 1. Elliptic Fourier Descriptors
        efd_coeffs_dict = compute_efds(largest_contour, harmonics_list=efd_harmonics_to_calc)
        for order, coeffs in efd_coeffs_dict.items():
            if coeffs is not None:
                flat_coeffs = coeffs.flatten()
                for i, c in enumerate(flat_coeffs):
                    all_shape_features[f'efd_o{order}_{i}'] = c
    else:
        logger.warning("No contour found in filled mask for shape descriptor calculation.")

    # 2. Texture Descriptors
    texture_features = compute_texture_descriptors(
        image_crop_gray,
        filled_mask,
        lbp_radii=lbp_radii_to_use,
        lbp_points=lbp_points_to_use
    )
    all_shape_features.update(texture_features)

    return all_shape_features