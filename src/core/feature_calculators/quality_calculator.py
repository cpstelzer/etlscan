# src/core/feature_calculators/quality_calculator.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple

from src.core.sharpness_assessment import calculate_sharpness_metrics
from src.core.masking_operations import detect_black_mask_artifact
from src.utils.common_exceptions import DataValidationError
from src.utils.config_loader import require_config_sections

logger = logging.getLogger(__name__)

BLACK_MASK_FLAG_COLUMN = 'has_black_mask_artifact'
CROP_BOUNDARY_FLAG_COLUMN = 'is_on_crop_boundary'


_BLACK_MASK_CHECK_KEYS = ("min_area_ratio", "max_black_pixel_value", "min_edge_thickness_fraction")


def _resolve_black_mask_check_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``quality_features.black_mask_check`` of the Stage-2 config (ConfigError if incomplete)."""
    require_config_sections(
        config, {"quality_features.black_mask_check": _BLACK_MASK_CHECK_KEYS}, "stage_02_characterization"
    )
    return config["quality_features"]["black_mask_check"]


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
    black_mask_config = _resolve_black_mask_check_config(config)
    # If a crop was not written to disk (e.g., save_all_particle_crops is False)
    # we have to rely on the in-memory grayscale crop for the black mask check.
    # Otherwise, allow the detector to read from disk to avoid redundant work.
    image_gray_for_artifact = image_crop_gray if saved_crop_path is None else None
    has_artifact = detect_black_mask_artifact(
        image_path=saved_crop_path,
        image_gray=image_gray_for_artifact,
        min_area_ratio=float(black_mask_config['min_area_ratio']),
        max_black_pixel_value=int(black_mask_config['max_black_pixel_value']),
        min_edge_thickness_fraction=float(black_mask_config['min_edge_thickness_fraction']),
    )
    features['has_black_mask_artifact'] = has_artifact

    # Note: Boundary detection will be handled in Stage 3 filtering based on this flag,
    # but we could add a preliminary check here if needed in the future.
    
    return features


def _drop_flagged_particles(particles_df: pd.DataFrame, flag_column: str) -> Tuple[pd.DataFrame, int]:
    """Return the particles without ``flag_column`` set, and how many were dropped."""
    if flag_column not in particles_df.columns:
        raise DataValidationError(
            f"Quality-flag filter needs the Stage-2 column {flag_column!r}, which is missing. "
            "Re-run Stage 2 or disable the filter in stage_03_filtering_and_classification.quality_flags_filter."
        )
    flagged_mask = particles_df[flag_column].fillna(False).astype(bool)
    return particles_df.loc[~flagged_mask], int(flagged_mask.sum())


def apply_quality_flag_filters(
    particles_df: pd.DataFrame,
    filter_black_mask: bool,
    filter_on_boundary: bool,
) -> Tuple[pd.DataFrame, int, int]:
    """Drop particles that carry an enabled Stage-2 quality flag.

    The single implementation of ``stage_03_filtering_and_classification.quality_flags_filter``,
    used by batch Stage 3 and realtime Stage 3. The black-mask filter runs first, so the
    boundary count covers only particles that passed it.

    Returns:
        (kept particles with a fresh index, removed for a black-mask artifact,
        removed for touching the crop boundary).

    Raises:
        DataValidationError: an enabled filter's flag column is missing.
    """
    kept_df = particles_df
    removed_black_mask = 0
    removed_on_boundary = 0
    if particles_df.empty:
        return kept_df.reset_index(drop=True), removed_black_mask, removed_on_boundary
    if filter_black_mask:
        kept_df, removed_black_mask = _drop_flagged_particles(kept_df, BLACK_MASK_FLAG_COLUMN)
    if filter_on_boundary:
        kept_df, removed_on_boundary = _drop_flagged_particles(kept_df, CROP_BOUNDARY_FLAG_COLUMN)
    return kept_df.reset_index(drop=True), removed_black_mask, removed_on_boundary
