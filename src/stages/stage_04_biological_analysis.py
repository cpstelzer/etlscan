# src/stages/stage_04_biological_analysis.py
import os
import pandas as pd
import numpy as np
import logging
import skimage.util
import skimage.filters
import skimage.measure
from typing import Dict, Any, Optional, Tuple

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv
)
# --- MODIFIED: Import visualization tools ---
from src.utils.config_loader import get_unified_segmentation_params
from src.visualization.feature_visualization import visualize_ml_features
from src.utils.common_exceptions import ProcessingError
from src.core.image_io_operations import load_image
from src.core.segmentation import segment_main_body_from_image_float
from src.core.ml_feature_extractor import _calculate_and_attach_posterior_attributes

logger = logging.getLogger(__name__)

# --- MODIFIED: Function signature and return value ---
def perform_egg_analysis(
    image_crop_gray: np.ndarray,
    particle_row_data: pd.Series,
    s4_config: Dict[str, Any],
    system_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[skimage.measure._regionprops.RegionProperties], Optional[skimage.measure._regionprops.RegionProperties]]:
    """
    Performs detailed egg detection and returns both the numerical features
    and the regionprops objects needed for visualization.
    """
    egg_features = {
        's4_total_egg_area': 0.0,
        's4_egg_eccentricity': 0.0,
        's4_egg_count': 0
    }
    
    crop_float = skimage.util.img_as_float(image_crop_gray)
    _, _, main_props = segment_main_body_from_image_float(crop_float, system_config)

    if main_props is None:
        logger.warning(f"Could not re-segment particle in crop {particle_row_data.get('crop_filename')}. Skipping egg analysis.")
        return egg_features, None, None

    egg_config = s4_config.get('egg_detection', {})
    
    egg_posterior_fraction = egg_config.get('egg_posterior_pixel_fraction', 0.45)
    # Get main_props with posterior attributes attached for visualization
    main_props_with_posterior, _, posterior_coords = _calculate_and_attach_posterior_attributes(
        main_props, crop_float.shape, egg_posterior_fraction
    )

    if posterior_coords is None or posterior_coords.shape[0] == 0:
        logger.debug("No posterior coordinates found for egg search.")
        return egg_features, main_props_with_posterior, None

    # Egg thresholding and filtering logic
    image_for_egg_detection = crop_float.copy()
    if system_config.get('source_image_illumination_type') == 'collimated':
        image_for_egg_detection = 1.0 - image_for_egg_detection

    egg_blurred = skimage.filters.gaussian(image_for_egg_detection, sigma=egg_config.get('gaussian_sigma', 1.0))
    
    quantile = egg_config.get('egg_threshold_quantile', 0.92)
    non_zero_pixels = egg_blurred[egg_blurred > 1e-6]
    if non_zero_pixels.size == 0:
        return egg_features, main_props_with_posterior, None

    egg_thresh_val = np.quantile(non_zero_pixels, quantile)
    egg_binary_mask = egg_blurred > egg_thresh_val

    posterior_pixel_mask = np.zeros(crop_float.shape, dtype=bool)
    valid_rows = np.clip(posterior_coords[:, 0].astype(int), 0, crop_float.shape[0] - 1)
    valid_cols = np.clip(posterior_coords[:, 1].astype(int), 0, crop_float.shape[1] - 1)
    posterior_pixel_mask[valid_rows, valid_cols] = True
    
    egg_mask_constrained = egg_binary_mask & posterior_pixel_mask
    egg_labeled_mask = skimage.measure.label(egg_mask_constrained, connectivity=2)
    
    egg_props_candidates = skimage.measure.regionprops(egg_labeled_mask, intensity_image=image_for_egg_detection)

    valid_eggs = []
    for p_egg in egg_props_candidates:
        if (egg_config.get('egg_min_area', 50) <= p_egg.area <= egg_config.get('egg_max_area', 5000) and
            egg_config.get('egg_min_solidity', 0.6) < p_egg.solidity < egg_config.get('egg_max_solidity', 0.99)):
            valid_eggs.append(p_egg)

    best_egg_props = None
    if valid_eggs:
        egg_features['s4_egg_count'] = len(valid_eggs)
        best_egg_props = max(valid_eggs, key=lambda p: p.area)
        egg_features['s4_total_egg_area'] = sum(p.area for p in valid_eggs)
        egg_features['s4_egg_eccentricity'] = best_egg_props.eccentricity

    return egg_features, main_props_with_posterior, best_egg_props


def run_stage_04_biological_analysis(
    input_s3_csv_path: str,
    stage3_crops_dir: str,
    output_dir_stage4: str,
    sample_id: str
) -> Optional[str]:
    """
    Performs final biological analysis and generates visualizations for each
    curated particle.
    """
    logger.info(f"--- Stage 4: Biological Analysis for Sample ID: {sample_id} ---")
    
    config = get_config()
    s4_config = config.get('stage_04_biological_analysis', {})
    system_config = config.get('system', {})
    
    # --- MODIFIED: Get unified params for visualization ---
    vis_params = get_unified_segmentation_params(config)

    s3_good_particles_df = load_dataframe_from_csv(input_s3_csv_path)
    if s3_good_particles_df is None or s3_good_particles_df.empty:
        logger.warning(f"No 'good' particles loaded from Stage 3 CSV: {input_s3_csv_path}. Aborting Stage 4.")
        return None

    # --- MODIFIED: Create directory for visualizations ---
    output_visualizations_dir = os.path.join(output_dir_stage4, "visualizations")
    os.makedirs(output_visualizations_dir, exist_ok=True)

    analysis_results = []

    crop_column = 'meta__crop_filename' if 'meta__crop_filename' in s3_good_particles_df.columns else 'crop_filename'

    for index, particle_row in s3_good_particles_df.iterrows():
        crop_filename = particle_row.get(crop_column)
        if not crop_filename:
            logger.warning(f"Row {index} is missing 'crop_filename'. Skipping.")
            continue
            
        crop_path = os.path.join(stage3_crops_dir, crop_filename)
        if not os.path.exists(crop_path):
            logger.warning(f"Crop file not found: {crop_path}. Skipping.")
            continue
            
        image_crop = load_image(crop_path, mode='gray')
        if image_crop is None:
            logger.warning(f"Failed to load crop image: {crop_path}. Skipping.")
            continue
            
        # --- MODIFIED: Unpack new return values ---
        biological_features, main_props_for_vis, best_egg_for_vis = perform_egg_analysis(
            image_crop, particle_row, s4_config, system_config
        )
        
        final_particle_data = particle_row.to_dict()
        final_particle_data.update(biological_features)
        analysis_results.append(final_particle_data)

        # --- MODIFIED: Add visualization call ---
        try:
            vis_filename = f"{os.path.splitext(crop_filename)[0]}_s4_analysis_vis.png"
            vis_output_path = os.path.join(output_visualizations_dir, vis_filename)
            visualize_ml_features(
                image_path_or_array=crop_path,
                main_props=main_props_for_vis,
                egg_area_props=best_egg_for_vis,
                output_visualization_path=vis_output_path,
                ml_feature_params=vis_params
            )
        except Exception as e:
            logger.error(f"Failed to generate Stage 4 visualization for {crop_filename}: {e}")


    if not analysis_results:
        logger.warning(f"No particles were successfully analyzed in Stage 4 for sample {sample_id}.")
        return None

    final_analysis_df = pd.DataFrame(analysis_results)
    
    output_filename = f"{sample_id}_stage4_final_analysis.csv"
    saved_csv_path = save_dataframe_to_csv(final_analysis_df, output_dir_stage4, output_filename)

    logger.info(f"--- Stage 4: Biological Analysis for {sample_id} COMPLETED ---")
    return saved_csv_path