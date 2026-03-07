# src/utils/data_io.py
import pandas as pd
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def save_dataframe_to_csv(df: pd.DataFrame, output_path: str, filename: str, float_format: str = '%.4f') -> Optional[str]:
    """
    Saves a Pandas DataFrame to a CSV file in the specified output_path.

    Args:
        df: The DataFrame to save.
        output_path: The directory where the CSV should be saved.
        filename: The name of the CSV file (e.g., "results.csv").
        float_format: Format string for floating point numbers.

    Returns:
        The full path to the saved CSV file, or None if saving failed.
    """
    if df is None:
        logger.warning(f"DataFrame is None. Cannot save to {filename} in {output_path}.")
        return None

    os.makedirs(output_path, exist_ok=True)
    full_csv_path = os.path.join(output_path, filename)

    try:
        df.to_csv(full_csv_path, index=False, float_format=float_format)
        logger.info(f"DataFrame ({len(df)} rows) saved successfully to: {full_csv_path}")
        return full_csv_path
    except Exception as e:
        logger.exception(f"Failed to save DataFrame to CSV at {full_csv_path}: {e}")
        return None

def load_dataframe_from_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a Pandas DataFrame from a CSV file.

    Args:
        csv_path: The full path to the CSV file.

    Returns:
        A pandas DataFrame, or None if loading failed or file not found.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"DataFrame ({len(df)} rows) loaded successfully from: {csv_path}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load DataFrame from CSV at {csv_path}: {e}")
        return None
