# src/core/sharpness_assessment.py
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Define the standard sharpness metrics that can be calculated.
# This helps in standardizing output dictionaries.
AVAILABLE_SHARPNESS_METRICS = [
    'variance_laplacian',
    'tenengrad_variance',
    'energy_laplacian',
    'energy_gradient'
]

def calculate_sharpness_metrics(image_gray: np.ndarray) -> Dict[str, float]:
    """
    Calculates various sharpness metrics for a grayscale image.
    Adapted from wellscanner_util_module.calculate_sharpness.

    Args:
        image_gray (np.ndarray): Input grayscale image.

    Returns:
        Dict[str, float]: A dictionary with sharpness scores.
                          Keys are from AVAILABLE_SHARPNESS_METRICS.
                          Values are np.nan if calculation fails for a metric.
    """
    scores: Dict[str, float] = {key: np.nan for key in AVAILABLE_SHARPNESS_METRICS}

    if image_gray is None:
        logger.warning("Cannot calculate sharpness for a None image.")
        return scores
    if not isinstance(image_gray, np.ndarray) or image_gray.ndim != 2:
        logger.error("Sharpness calculation requires a 2D grayscale NumPy array.")
        return scores
    if image_gray.size == 0 or image_gray.shape[0] < 3 or image_gray.shape[1] < 3:
        logger.warning(f"Image too small for sharpness calculation (shape: {image_gray.shape}). Min 3x3 required.")
        return scores

    try:
        # Variance of Laplacian
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        scores['variance_laplacian'] = laplacian.var()
        scores['energy_laplacian'] = np.sum(np.square(laplacian))

        # Tenengrad (Sobel-based)
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude_squared = np.square(sobel_x) + np.square(sobel_y)
        scores['energy_gradient'] = np.sum(gradient_magnitude_squared)
        
        gradient_magnitude = np.sqrt(gradient_magnitude_squared)
        scores['tenengrad_variance'] = gradient_magnitude.var()

        # Sanity check for NaN/inf values post-calculation
        for k, v in scores.items():
            if not np.isfinite(v):
                logger.warning(f"Sharpness metric '{k}' resulted in a non-finite value ({v}). Setting to NaN.")
                scores[k] = np.nan
        
        logger.debug("Sharpness metrics calculated.")
        return scores

    except Exception as e:
        logger.exception(f"Error during sharpness calculation: {e}")
        # Return dict with NaNs if any exception occurs
        return {key: np.nan for key in AVAILABLE_SHARPNESS_METRICS}


def normalize_and_combine_sharpness(
    df: pd.DataFrame,
    metrics_to_normalize: List[str]
    ) -> pd.DataFrame:
    """
    Calculates normalized ([0,1]) scores for specified sharpness metrics
    and a 'combined_sharpness' score (mean of normalized scores).
    Adds '_norm' columns and 'combined_sharpness' to the DataFrame.
    Adapted from wellscanner_util_module.calculate_combined_sharpness.

    Args:
        df (pd.DataFrame): DataFrame containing raw sharpness scores.
        metrics_to_normalize (List[str]): List of column names for sharpness metrics to normalize.

    Returns:
        pd.DataFrame: DataFrame with added normalized columns and 'combined_sharpness'.
    """
    logger.info("Calculating combined sharpness...")
    if df.empty:
        logger.info("Input DataFrame is empty, skipping combined sharpness calculation.")
        # Add empty columns if they are expected by downstream processes
        for metric in metrics_to_normalize:
            df[metric + '_norm'] = np.nan
        df['combined_sharpness'] = np.nan
        return df

    df_out = df.copy()
    normalized_metric_cols = []

    for metric in metrics_to_normalize:
        norm_col_name = metric + '_norm'
        if metric not in df_out.columns:
            logger.warning(f"Metric '{metric}' not found in DataFrame. Skipping normalization for it.")
            df_out[norm_col_name] = np.nan
            continue
        
        metric_series = df_out[metric]
        if metric_series.isnull().all() or metric_series.nunique() < 2:
            logger.warning(f"Metric '{metric}' has all NaNs or fewer than 2 unique values. Normalization will result in NaNs.")
            df_out[norm_col_name] = np.nan
            continue
        
        try:
            min_val = metric_series.min()
            max_val = metric_series.max()

            if max_val <= min_val: # Avoid division by zero if all values are the same
                # If all values are same, normalized could be 0, 0.5, or 1 depending on convention.
                # Setting to 1.0 here if non-NaN, else NaN. Or based on problem.
                # Original code set to 1.0.
                df_out[norm_col_name] = 1.0 if not metric_series.isnull().all() else np.nan
            else:
                df_out[norm_col_name] = (metric_series - min_val) / (max_val - min_val)
            
            normalized_metric_cols.append(norm_col_name)
            logger.debug(f"Normalized '{metric}' into '{norm_col_name}'.")
        except Exception as e:
            logger.exception(f"Error normalizing metric '{metric}': {e}")
            df_out[norm_col_name] = np.nan # Set to NaN on error

    if normalized_metric_cols:
        try:
            logger.debug(f"Calculating 'combined_sharpness' from: {normalized_metric_cols}")
            # Calculate mean, skipping NaNs. If all are NaN for a row, result is NaN.
            df_out['combined_sharpness'] = df_out[normalized_metric_cols].mean(axis=1, skipna=True)
            
            # Ensure 'combined_sharpness' is NaN if all underlying normalized scores were NaN for that row
            all_norm_cols_are_nan_mask = df_out[normalized_metric_cols].isnull().all(axis=1)
            df_out.loc[all_norm_cols_are_nan_mask, 'combined_sharpness'] = np.nan
            
            logger.info("Calculated 'combined_sharpness'.")
        except Exception as e:
            logger.exception(f"Error calculating 'combined_sharpness': {e}")
            df_out['combined_sharpness'] = np.nan
    else:
        logger.warning("No metrics were successfully normalized. 'combined_sharpness' will be NaN.")
        df_out['combined_sharpness'] = np.nan
            
    return df_out
