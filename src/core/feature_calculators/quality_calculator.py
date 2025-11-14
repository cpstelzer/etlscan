# src/core/feature_calculators/quality_calculator.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.core.sharpness_assessment import calculate_sharpness_metrics
from src.core.masking_operations import detect_black_mask_artifact

logger = logging.getLogger(__name__)

def calculate_quality_features(
    image_crop_gray: np.ndarray,
    saved_crop_path: Optional[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrator to calculate all quality-related features for a particle crop.

    Args:
        image_crop_gray (np.ndarray): The grayscale cropped image of the particle.
        saved_crop_path (Optional[str]): Full path to the saved crop image if it exists.
        config (Dict[str, Any]): The configuration dictionary for Stage 2.

    Returns:
        Dict[str, Any]: A dictionary containing all calculated quality features.
    """
    features = {}

    # 1. Sharpness Metrics
    sharpness_scores = calculate_sharpness_metrics(image_crop_gray)
    features.update(sharpness_scores)

    # 2. Black Mask Artifact Detection
    black_mask_config = config.get('black_mask_check', {})
    # If a crop was not written to disk (e.g., save_all_particle_crops is False)
    # we have to rely on the in-memory grayscale crop for the black mask check.
    # Otherwise, allow the detector to read from disk to avoid redundant work.
    image_gray_for_artifact = image_crop_gray if saved_crop_path is None else None
    has_artifact = detect_black_mask_artifact(
        image_path=saved_crop_path,
        image_gray=image_gray_for_artifact,
        min_area_ratio=black_mask_config.get('min_area_ratio', 0.05)
    )
    features['has_black_mask_artifact'] = has_artifact

    # Note: Boundary detection will be handled in Stage 3 filtering based on this flag,
    # but we could add a preliminary check here if needed in the future.
    
    return features