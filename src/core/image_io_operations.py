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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlatFieldCorrectionAssets:
    """Holds sample-level flat-field calibration images for repeated use."""
    flat_float: np.ndarray
    dark_float: np.ndarray


def load_flat_field_assets(sample_dir_path: str, mode: str = 'gray') -> FlatFieldCorrectionAssets:
    """
    Load flat-field calibration images once for a sample directory.

    The flat image captures the uneven illumination pattern in the microscope view,
    while the dark image captures baseline sensor signal. We reuse both for every
    frame from the same sample to keep correction consistent and efficient.
    """
    calibration_dir = os.path.join(sample_dir_path, '1_flat_field')
    flat_path = os.path.join(calibration_dir, 'flat.png')
    dark_path = os.path.join(calibration_dir, 'dark.png')

    flat_img = load_image(flat_path, mode=mode)
    dark_img = load_image(dark_path, mode=mode)
    if flat_img is None or dark_img is None:
        raise ImageProcessingError(
            f"Failed to load flat-field calibration images from {calibration_dir}. "
            "Expected files: flat.png and dark.png."
        )

    if flat_img.shape != dark_img.shape:
        raise ImageProcessingError(
            "Flat-field image and dark-frame image must have the same shape. "
            f"Got flat={flat_img.shape}, dark={dark_img.shape}."
        )

    return FlatFieldCorrectionAssets(
        flat_float=flat_img.astype(np.float32),
        dark_float=dark_img.astype(np.float32),
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
            logger.warning("Attempting PIL fallback after OpenCV load exception: %s", image_path)
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
                gray_img = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            else:
                coeffs = np.array([0.114, 0.587, 0.299], dtype=np.float32)
                gray_img = np.tensordot(image_color, coeffs, axes=([2], [0]))
                gray_img = gray_img.astype(np.uint8)
            logger.debug("Image converted to grayscale.")
            return gray_img
        except Exception as e:
            logger.exception(f"Error converting image to grayscale: {e}")
            return None
    else:
        logger.error(f"Unsupported image format for grayscale conversion. Shape: {image_color.shape}")
        return None


if __name__ == '__main__':
    # Setup basic logging for testing this module directly
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    # Create a dummy color image for testing
    dummy_color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_color_image[:, :50] = [255, 0, 0]  # Blue half
    dummy_color_image[:, 50:] = [0, 0, 255]  # Red half
    temp_dir = "temp_image_io_test"

    # Test saving
    saved_file = save_image(dummy_color_image, temp_dir, "dummy_color.png")
    if saved_file:
        # Test loading color
        loaded_color = load_image(saved_file, mode='color')
        assert loaded_color is not None and loaded_color.shape == dummy_color_image.shape, "Color load failed"
        logger.info("Color load test passed.")

        # Test loading gray
        loaded_gray_direct = load_image(saved_file, mode='gray')
        assert loaded_gray_direct is not None and loaded_gray_direct.ndim == 2, "Gray load failed"
        logger.info("Gray load test passed.")

        # Test conversion to gray
        converted_gray = convert_to_grayscale(loaded_color)
        assert converted_gray is not None and converted_gray.ndim == 2, "Grayscale conversion failed"
        logger.info("Grayscale conversion test passed.")
        save_image(converted_gray, temp_dir, "dummy_gray_converted.png")


        # Clean up
        try:
            os.remove(os.path.join(temp_dir, "dummy_color.png"))
            os.remove(os.path.join(temp_dir, "dummy_gray_converted.png"))
            os.rmdir(temp_dir)
            logger.info("Cleaned up test image files and directory.")
        except OSError as e:
            logger.error(f"Error during cleanup: {e}")
    else:
        logger.error("Initial save failed, cannot run further tests.")

    # Test loading non-existent file
    non_existent_img = load_image("non_existent.png")
    assert non_existent_img is None, "Non-existent image load test failed"
    logger.info("Non-existent image load test passed.")
