# src/core/image_io_operations.py
import cv2
import numpy as np
import os
import logging
from typing import Optional, Tuple

from src.utils.common_exceptions import ImageProcessingError

logger = logging.getLogger(__name__)

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
        if mode == 'gray':
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif mode == 'color':
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            logger.error(f"Invalid image loading mode: {mode}. Must be 'color' or 'gray'.")
            return None

        if img is None:
            logger.error(f"Failed to load image (cv2.imread returned None): {image_path}")
            return None
        logger.debug(f"Successfully loaded image: {image_path} in {mode} mode.")
        return img
    except Exception as e:
        logger.exception(f"Exception during image loading for {image_path}: {e}")
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
        success = cv2.imwrite(full_image_path, image_array)
        if not success:
            logger.error(f"Failed to save image (cv2.imwrite returned False): {full_image_path}")
            return None
        logger.debug(f"Image saved successfully to: {full_image_path}")
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
            gray_img = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
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