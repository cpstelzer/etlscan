# src/core/feature_calculators/morphology_calculator.py
import cv2
import numpy as np
import skimage.measure
import logging
from typing import Dict, Any, Optional

# IMPORTANT: The function 'calculate_detailed_body_morphology' should be
# moved from 'src/core/detailed_morphology_calculator.py' into this new file.
# The contents of that function remain unchanged.

# For brevity, I will not repeat the entire function here. Please perform the move.
# After moving, 'src/core/detailed_morphology_calculator.py' can be deleted.

# --- PASTE the entire 'calculate_detailed_body_morphology' function here ---
# Example start of the function:
# def calculate_detailed_body_morphology(
#     raw_body_mask: Optional[np.ndarray],
#     ...
# ) -> Dict[str, Any]:
#     features: Dict[str, Any] = {}
#     ... (rest of the function)

# For the script to be complete, I will paste it here.
logger = logging.getLogger(__name__)

def calculate_detailed_body_morphology(
    raw_body_mask: Optional[np.ndarray],      # uint8 mask
    filled_body_mask: Optional[np.ndarray],   # uint8 mask
    intensity_image_for_props: Optional[np.ndarray] = None
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

    props_raw_list = skimage.measure.regionprops(
        skimage.measure.label(raw_body_mask_bool),
        intensity_image=intensity_image_for_props,
        cache=False
    )

    if props_raw_list:
        props_raw = sorted(props_raw_list, key=lambda p: p.area, reverse=True)[0]
        
        features['main_area_raw_s4'] = float(props_raw.area)
        features['solidity_s4'] = float(props_raw.solidity) if hasattr(props_raw, 'solidity') else np.nan
        
        if hasattr(props_raw, 'area_oriented_bbox'):
             oriented_bbox_area = props_raw.area_oriented_bbox
        elif hasattr(props_raw, 'bbox_area'):
            oriented_bbox_area = props_raw.bbox_area
        elif hasattr(props_raw, 'area_bbox'):
            oriented_bbox_area = props_raw.area_bbox
        else:
            minr, minc, maxr, maxc = props_raw.bbox
            oriented_bbox_area = (maxr - minr) * (maxc - minc)

        if oriented_bbox_area > 1e-6:
            features['compactness_rect_s4'] = float(props_raw.area / oriented_bbox_area)
        else:
            features['compactness_rect_s4'] = np.nan
            
        features['extent_raw_s4'] = float(props_raw.extent) if hasattr(props_raw, 'extent') else np.nan
        features['compactness_factor_ni_s4'] = features['extent_raw_s4']
        
        features['eccentricity_raw_s4'] = float(props_raw.eccentricity) if hasattr(props_raw, 'eccentricity') else np.nan
        features['equivalent_diameter_raw_s4'] = float(props_raw.equivalent_diameter_approx) if hasattr(props_raw, 'equivalent_diameter_approx') else (float(props_raw.equivalent_diameter) if hasattr(props_raw, 'equivalent_diameter') else np.nan)
        features['major_axis_length_raw_s4'] = float(props_raw.major_axis_length) if hasattr(props_raw, 'major_axis_length') else np.nan
        features['minor_axis_length_raw_s4'] = float(props_raw.minor_axis_length) if hasattr(props_raw, 'minor_axis_length') else np.nan
        
        if hasattr(props_raw, 'moments_central'):
            mc = props_raw.moments_central
            Ixx = float(mc[2,0]) if np.isfinite(mc[2,0]) else 0.0
            Iyy = float(mc[0,2]) if np.isfinite(mc[0,2]) else 0.0
            Ixy = float(mc[1,1]) if np.isfinite(mc[1,1]) else 0.0
            
            features['moment_of_inertia_xx_s4'] = Ixx
            features['moment_of_inertia_yy_s4'] = Iyy
            features['moment_of_inertia_xy_s4'] = Ixy

            if Ixx > 1e-9 and Iyy > 1e-9 :
                denominator_tf = 4 * np.pi * np.sqrt(Ixx * Iyy)
                if denominator_tf > 1e-9:
                    features['type_factor_s4'] = (props_raw.area**2) / denominator_tf
                else: features['type_factor_s4'] = np.nan
            else: features['type_factor_s4'] = np.nan
        
        if hasattr(props_raw, 'perimeter') and props_raw.perimeter > 1e-9 and props_raw.area > 1e-9:
            perimeter_val = props_raw.perimeter_crofton if hasattr(props_raw, 'perimeter_crofton') else props_raw.perimeter
            denominator_hcf = 2 * np.sqrt(np.pi * props_raw.area)
            if denominator_hcf > 1e-9:
                features['heywood_circularity_factor_s4'] = perimeter_val / denominator_hcf
            else: features['heywood_circularity_factor_s4'] = np.nan
        else: features['heywood_circularity_factor_s4'] = np.nan
        
        if features['minor_axis_length_raw_s4'] > 1e-9 and features['major_axis_length_raw_s4'] > 1e-9 :
            features['elongation_factor_s4'] = features['major_axis_length_raw_s4'] / features['minor_axis_length_raw_s4']
        elif hasattr(props_raw, 'feret_diameter_max') and features['minor_axis_length_raw_s4'] > 1e-9:
             features['elongation_factor_s4'] = props_raw.feret_diameter_max / features['minor_axis_length_raw_s4']
        else: features['elongation_factor_s4'] = np.nan
    else:
        logger.warning("No regionprops found for raw_body_mask in detailed morphology calculation.")

    moments_raw_cv = cv2.moments(raw_body_mask.astype(np.uint8))
    hu_moments_cv = cv2.HuMoments(moments_raw_cv).flatten()
    if len(hu_moments_cv) >= 1:
        val_hu1 = float(hu_moments_cv[0])
        features['hu_moment1_s4'] = val_hu1 if np.isfinite(val_hu1) else np.nan
    else: features['hu_moment1_s4'] = np.nan

    filled_area = np.sum(filled_body_mask > 0)
    features['main_area_filled_s4'] = float(filled_area)
    
    raw_area_val_for_porosity = features.get('main_area_raw_s4')
    if isinstance(raw_area_val_for_porosity, (int, float)) and \
       np.isfinite(raw_area_val_for_porosity) and filled_area > 1e-6:
         features['porosity_s4'] = 1.0 - (raw_area_val_for_porosity / filled_area)
    else: features['porosity_s4'] = np.nan

    for key, value in features.items():
        if isinstance(value, (np.float32, np.float64)): features[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)): features[key] = int(value)
        elif isinstance(value, float) and not np.isfinite(value): features[key] = None
    return features