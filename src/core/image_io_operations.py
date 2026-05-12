# src/core/image_io_operations.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from PIL import Image

try:  # pragma: no cover - depends on environment
    import cv2  # type: ignore
except Exception:  # noqa: BLE001 - keep optional dependency for headless envs
    cv2 = None  # type: ignore

from src.utils.common_exceptions import ImageProcessingError
from src.utils.config_loader import load_config_once

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlatFieldCorrectionAssets:
    """Holds sample-level flat-field calibration images for repeated use."""
    flat_float: np.ndarray
    dark_float: np.ndarray


def _get_flat_field_candidate_dirs(sample_dir_path: str, subdir: str) -> list[str]:
    """Return candidate calibration directories ordered from most to least specific."""
    sample_dir = os.path.normpath(sample_dir_path)
    parent_dir = os.path.dirname(sample_dir)
    return [
        os.path.join(sample_dir, subdir),
        os.path.join(parent_dir, subdir),
    ]


def load_flat_field_assets(sample_dir_path: str, mode: str = 'gray') -> FlatFieldCorrectionAssets:
    """
    Load flat-field calibration images once for a sample directory.

    The flat image captures the uneven illumination pattern in the microscope view,
    while the dark image captures baseline sensor signal. We reuse both for every
    frame from the same sample to keep correction consistent and efficient.
    """
    ffc_cfg = load_config_once().get('flat_field_calibration', {})
    subdir = ffc_cfg.get('subdir', '1_flat_field')
    flat_filename = ffc_cfg.get('flat_filename', 'flat.png')
    dark_filename = ffc_cfg.get('dark_filename', 'dark.png')

    candidate_dirs = _get_flat_field_candidate_dirs(sample_dir_path, subdir)
    attempted_dirs: list[str] = []

    for calibration_dir in candidate_dirs:
        attempted_dirs.append(calibration_dir)
        flat_path = os.path.join(calibration_dir, flat_filename)
        dark_path = os.path.join(calibration_dir, dark_filename)

        if not os.path.exists(flat_path) or not os.path.exists(dark_path):
            logger.debug(
                "Skipping flat-field directory %s because one or both calibration files are missing.",
                calibration_dir,
            )
            continue

        flat_img = load_image(flat_path, mode=mode)
        dark_img = load_image(dark_path, mode=mode)
        if flat_img is None or dark_img is None:
            logger.warning(
                "Found flat-field files in %s but failed to load one or both images.",
                calibration_dir,
            )
            continue

        if flat_img.shape != dark_img.shape:
            raise ImageProcessingError(
                "Flat-field image and dark-frame image must have the same shape. "
                f"Got flat={flat_img.shape}, dark={dark_img.shape}."
            )

        return FlatFieldCorrectionAssets(
            flat_float=flat_img.astype(np.float32),
            dark_float=dark_img.astype(np.float32),
        )

    raise ImageProcessingError(
        "Failed to load flat-field calibration images. "
        f"Checked directories: {', '.join(attempted_dirs)}. "
        f"Expected files: {flat_filename} and {dark_filename}."
    )


def apply_flat_field_correction(
    raw_image: np.ndarray,
    ffc_assets: FlatFieldCorrectionAssets,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Apply pixel-wise flat-field correction and return a normalized float image.

    Formula: Icorr = (I - D) / (F - D)
    - I: raw frame
    - F: flat-field frame
    - D: dark frame

    The corrected result is normalized to [0, 1] so downstream segmentation sees
    a stable intensity range independent of exposure level.
    """
    raw_float = raw_image.astype(np.float32)
    flat_float = ffc_assets.flat_float
    dark_float = ffc_assets.dark_float

    if raw_float.shape != flat_float.shape or raw_float.shape != dark_float.shape:
        raise ImageProcessingError(
            "Shape mismatch for flat-field correction. "
            f"raw={raw_float.shape}, flat={flat_float.shape}, dark={dark_float.shape}."
        )

    denominator = flat_float - dark_float
    denominator = np.where(denominator <= 0, epsilon, denominator)
    corrected = (raw_float - dark_float) / denominator

    corrected_min = float(np.min(corrected))
    corrected_max = float(np.max(corrected))
    range_value = corrected_max - corrected_min

    if range_value <= 0:
        # If the corrected frame is constant, keep a valid [0, 1] image of zeros.
        normalized = np.zeros_like(corrected, dtype=np.float32)
    else:
        normalized = (corrected - corrected_min) / range_value

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)

def _load_image_with_pil(image_path: str, mode: str) -> Optional[np.ndarray]:
    """
    Load an image using PIL instead of OpenCV.

    For non-experts: PIL (Python Imaging Library) is an alternative image loader
    that reads image files using a different code path than OpenCV. This matters
    when OpenCV cannot load a specific file due to memory allocation errors or
    decoder issues. The PIL-based loader decodes the same image data and then
    converts it into a NumPy array in the same grayscale/BGR formats the rest
    of the pipeline expects, so downstream processing can continue normally.
    """
    try:
        pil_img = Image.open(image_path)
        if mode == 'gray':
            img = np.array(pil_img.convert('L'))
        elif mode == 'color':
            rgb = np.array(pil_img.convert('RGB'))
            img = rgb[:, :, ::-1]  # Convert RGB -> BGR to match OpenCV convention
        else:
            logger.error(f"Invalid image loading mode: {mode}. Must be 'color' or 'gray'.")
            return None
        logger.debug(f"Successfully loaded image: {image_path} in {mode} mode via PIL fallback.")
        return img
    except Exception as e:
        logger.exception(f"Exception during PIL image loading for {image_path}: {e}")
        return None

def load_image(image_path: str, mode: str = 'color') -> Optional[np.ndarray]:
    """
    Loads an image from the specified path.

    Args:
        image_path (str): Path to the image file.
        mode (str): 'color' to load as BGR, 'gray' to load as grayscale.

    Returns:
        np.ndarray: The loaded image as a NumPy array, or None if loading fails.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
    try:
        if cv2 is not None:
            if mode == 'gray':
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            elif mode == 'color':
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                logger.error(f"Invalid image loading mode: {mode}. Must be 'color' or 'gray'.")
                return None
            if img is None:
                logger.warning(
                    "Failed to load image (cv2.imread returned None). "
                    "Attempting PIL fallback: %s",
                    image_path,
                )
                return _load_image_with_pil(image_path, mode)
            logger.debug(f"Successfully loaded image: {image_path} in {mode} mode via OpenCV.")
            return img

        # PIL fallback for headless environments
        return _load_image_with_pil(image_path, mode)
    except Exception as e:
        logger.exception(f"Exception during image loading for {image_path}: {e}")
        if cv2 is not None:
            logger.warning(f"Attempting PIL fallback after OpenCV load exception: {image_path}")
            return _load_image_with_pil(image_path, mode)
        return None

def save_image(image_array: np.ndarray, output_path: str, filename: str) -> Optional[str]:
    """
    Saves a NumPy image array to a file.

    Args:
        image_array (np.ndarray): The image to save.
        output_path (str): The directory where the image should be saved.
        filename (str): The name of the image file (e.g., "image.png").

    Returns:
        The full path to the saved image file, or None if saving failed.
    """
    if image_array is None:
        logger.warning(f"Image array is None. Cannot save {filename} to {output_path}.")
        return None

    os.makedirs(output_path, exist_ok=True)
    full_image_path = os.path.join(output_path, filename)

    try:
        if cv2 is not None:
            success = cv2.imwrite(full_image_path, image_array)
            if not success:
                logger.error(f"Failed to save image (cv2.imwrite returned False): {full_image_path}")
                return None
            logger.debug(f"Image saved successfully to: {full_image_path} via OpenCV.")
            return full_image_path

        pil_img_array = image_array
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            pil_img_array = image_array[:, :, ::-1]  # Convert BGR -> RGB
        pil_img = Image.fromarray(pil_img_array)
        pil_img.save(full_image_path)
        logger.debug(f"Image saved successfully to: {full_image_path} via PIL fallback.")
        return full_image_path
    except Exception as e:
        logger.exception(f"Exception during image saving to {full_image_path}: {e}")
        return None

def convert_to_grayscale(image_color: np.ndarray) -> Optional[np.ndarray]:
    """
    Converts a color image (BGR) to grayscale.
    """
    if image_color is None:
        logger.warning("Cannot convert None image to grayscale.")
        return None
    if image_color.ndim == 2:
        logger.debug("Image is already grayscale.")
        return image_color
    if image_color.ndim == 3 and image_color.shape[2] == 3:
        try:
            if cv2 is not None:
                image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            else:
                coeffs = np.array([0.114, 0.587, 0.299], dtype=np.float32)
                image_gray = np.tensordot(image_color, coeffs, axes=([2], [0]))
                image_gray = image_gray.astype(np.uint8)
            logger.debug("Image converted to grayscale.")
            return image_gray
        except Exception as e:
            logger.exception(f"Error converting image to grayscale: {e}")
            return None
    else:
        logger.error(f"Unsupported image format for grayscale conversion. Shape: {image_color.shape}")
        return None


