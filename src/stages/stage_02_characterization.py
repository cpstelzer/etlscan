# src/stages/stage_02_characterization.py
import os
import cv2
import pandas as pd
import numpy as np
import logging
import skimage.util
from typing import Optional

from src.utils import (
    get_config, load_dataframe_from_csv, save_dataframe_to_csv
)
from src.utils.path_helpers import extract_frame_from_filename
from src.utils.common_exceptions import ProcessingError, ConfigError, DataValidationError
from src.core.image_io_operations import load_image, convert_to_grayscale
from src.core.particle_operations import ParticleCropper
from src.core.segmentation import segment_main_body_from_image_float
from src.core.feature_calculators.quality_calculator import calculate_quality_features
from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors

logger = logging.getLogger(__name__)

def run_stage_02_characterization(
    input_s1_csv_path: str,
    image_dir_path: str,
    output_dir_stage2: str,
    sample_id: str
) -> Optional[str]:
    """
    Performs detailed characterization for every particle candidate from Stage 1
    using a modular feature calculator architecture.
    """
    logger.info(f"--- Stage 2: Universal Characterization for Sample ID: {sample_id} ---")
    
    config = get_config()
    s2_config = config.get('stage_02_characterization', {})
    system_config = config.get('system', {})
    
    s1_particles_df = load_dataframe_from_csv(input_s1_csv_path)
    if s1_particles_df is None or s1_particles_df.empty:
        logger.warning(f"No particles loaded from Stage 1 CSV: {input_s1_csv_path}. Aborting Stage 2.")
        return None

    cropping_config = s2_config.get('cropping', {})
    if not cropping_config:
        logger.warning("Cropping configuration not found in 'stage_02_characterization'. Using defaults.")
        cropping_config = {'mode': 'rotational_rectangle', 'rotational_padding': 10}

    save_all_particle_crops = bool(cropping_config.get('save_all_particle_crops', True))
    all_crops_dir = None
    if save_all_particle_crops:
        all_crops_dir = os.path.join(output_dir_stage2, "stage2_all_particle_crops")
        os.makedirs(all_crops_dir, exist_ok=True)
        
    try:
        cropper = ParticleCropper(cropping_config)
        logger.info(f"Initialized ParticleCropper with mode: '{cropping_config.get('mode')}'")
    except ConfigError as e:
        logger.error(f"Failed to initialize ParticleCropper: {e}")
        return None

    all_particle_features = []

    s1_particles_df['s1_original_index'] = s1_particles_df.index
    
    for image_filename, group in s1_particles_df.groupby('source_image_filename'):
        logger.info(f"Characterizing {len(group)} particles from image: {image_filename}")
        full_image_path = os.path.join(image_dir_path, image_filename)
        
        main_img_color = load_image(full_image_path, mode='color')
        main_img_gray = convert_to_grayscale(main_img_color)

        if main_img_color is None or main_img_gray is None:
            logger.warning(f"Failed to load image {full_image_path}. Skipping {len(group)} particles.")
            continue
            
        for index, s1_particle_row in group.iterrows():
            original_index = s1_particle_row['s1_original_index']
            particle_features = {'s1_original_index': original_index}
            
            frame_num = extract_frame_from_filename(image_filename)
            if frame_num is None:
                logger.warning(f"Could not parse frame from {image_filename} for particle index {original_index}. Skipping.")
                continue

            raw_track_id = s1_particle_row.get('track_id')
            if pd.isna(raw_track_id):
                track_id_str = f"untracked{original_index}"
            else:
                track_id_str = f"track{int(raw_track_id)}"

            crop_filename_prefix = f"{sample_id}_{track_id_str}_frame{frame_num}"
            
            crop_gray, saved_crop_path = cropper.crop_single_particle_from_image(
                main_img_color, main_img_gray, s1_particle_row,
                output_dir_for_crop=all_crops_dir if save_all_particle_crops else None,
                crop_filename_prefix=crop_filename_prefix
            )

            if crop_gray is None:
                logger.warning(f"Failed to crop particle at index {original_index}. Skipping.")
                continue

            particle_features['crop_filename'] = (
                os.path.basename(saved_crop_path) if saved_crop_path else None
            )
            
            crop_float_orig = skimage.util.img_as_float(crop_gray)
            raw_mask, filled_mask, _ = segment_main_body_from_image_float(crop_float_orig, system_config)

            if filled_mask is None or raw_mask is None:
                logger.warning(f"High-precision segmentation failed for crop {particle_features['crop_filename']}. Skipping feature calculation.")
                continue

            quality_features = calculate_quality_features(crop_gray, saved_crop_path, s2_config)
            particle_features.update(quality_features)

            intensity_img_for_props = crop_float_orig.copy()
            if system_config.get('source_image_illumination_type') == 'collimated':
                intensity_img_for_props = 1.0 - intensity_img_for_props
            morphology_features = calculate_detailed_body_morphology(raw_mask, filled_mask, intensity_img_for_props)
            renamed_morphology = {k.replace('_s4', ''): v for k, v in morphology_features.items()}
            particle_features.update(renamed_morphology)

            # --- THIS IS THE CORRECTED LINE ---
            shape_features = calculate_shape_descriptors(crop_gray, filled_mask, s2_config)
            particle_features.update(shape_features)

            all_particle_features.append(particle_features)

    if not all_particle_features:
        logger.warning(f"No particles were successfully characterized for sample {sample_id}.")
        return None

    features_df = pd.DataFrame(all_particle_features)
    s1_particles_df_indexed = s1_particles_df.copy()
    
    comprehensive_df = pd.merge(
        s1_particles_df_indexed, features_df, on='s1_original_index', how='right'
    )
    comprehensive_df.drop(columns=['s1_original_index'], inplace=True, errors='ignore')

    output_filename = f"{sample_id}_stage2_all_particle_features.csv"
    saved_csv_path = save_dataframe_to_csv(comprehensive_df, output_dir_stage2, output_filename)

    logger.info(f"--- Stage 2: Characterization for {sample_id} COMPLETED ---")
    return saved_csv_path