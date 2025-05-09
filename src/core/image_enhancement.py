# src/core/image_enhancement.py
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from src.utils.common_exceptions import ImageProcessingError

logger = logging.getLogger(__name__)

def subtract_images(image_a: np.ndarray, image_b: np.ndarray) -> Optional[np.ndarray]:
    """
    Subtracts image_b from image_a (element-wise).
    Result is clipped to [0, 255] and converted to uint8.
    Assumes images are grayscale.
    """
    if image_a is None or image_b is None:
        logger.error("One or both images for subtraction are None.")
        return None
    if image_a.shape != image_b.shape:
        logger.error(f"Images must have the same dimensions for subtraction. "
                     f"Got {image_a.shape} and {image_b.shape}")
        return None
    if image_a.ndim != 2 or image_b.ndim != 2:
        logger.warning("Subtracting images that are not grayscale. Ensure this is intended.")

    try:
        subtracted = cv2.subtract(image_a, image_b)
        # cv2.subtract handles clipping to 0 automatically for uint8.
        # If inputs were float, manual clipping would be needed before converting to uint8.
        # subtracted = np.clip(subtracted, 0, 255).astype(np.uint8) # Redundant if inputs are uint8
        logger.debug("Image subtraction successful.")
        return subtracted
    except Exception as e:
        logger.exception(f"Error during image subtraction: {e}")
        return None

def apply_threshold(image_gray: np.ndarray, threshold_value: int, threshold_type: str = "binary") -> Optional[np.ndarray]:
    """
    Applies thresholding to a grayscale image.

    Args:
        image_gray (np.ndarray): Input grayscale image.
        threshold_value (int): Threshold value.
        threshold_type (str): 'binary', 'binary_inv', 'otsu', 'otsu_binary'.

    Returns:
        np.ndarray: Thresholded binary image, or None on error.
    """
    if image_gray is None:
        logger.error("Cannot apply threshold to a None image.")
        return None
    if image_gray.ndim != 2:
        logger.error("Thresholding requires a grayscale image (2D).")
        return None

    try:
        if threshold_type == "binary":
            _, thresh_img = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == "binary_inv":
            _, thresh_img = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        elif threshold_type == "otsu": # Deprecated, use otsu_binary for clarity
             logger.warning("'otsu' threshold type is ambiguous. Use 'otsu_binary'. Falling back to THRESH_OTSU.")
             _, thresh_img = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == "otsu_binary":
            _, thresh_img = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            logger.error(f"Unsupported threshold type: {threshold_type}")
            return None
        logger.debug(f"Thresholding applied with type '{threshold_type}'.")
        return thresh_img
    except Exception as e:
        logger.exception(f"Error applying threshold: {e}")
        return None

def apply_blur(image: np.ndarray, kernel_size: int, blur_type: str = "median") -> Optional[np.ndarray]:
    """
    Applies blurring to an image.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        kernel_size (int): Size of the kernel (must be odd for median/gaussian).
        blur_type (str): 'median', 'gaussian'.

    Returns:
        np.ndarray: Blurred image, or None on error.
    """
    if image is None:
        logger.error("Cannot apply blur to a None image.")
        return None
    if not (isinstance(kernel_size, int) and kernel_size > 0 and kernel_size % 2 == 1):
        logger.error(f"Kernel size must be a positive odd integer. Got {kernel_size}.")
        return None

    try:
        if blur_type == "median":
            blurred_img = cv2.medianBlur(image, kernel_size)
        elif blur_type == "gaussian":
            blurred_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            logger.error(f"Unsupported blur type: {blur_type}")
            return None
        logger.debug(f"Blur applied with type '{blur_type}' and kernel size {kernel_size}.")
        return blurred_img
    except Exception as e:
        logger.exception(f"Error applying blur: {e}")
        return None

def apply_morphological_operation(image: np.ndarray, kernel_size: int, operation: str, shape: str = 'rect') -> Optional[np.ndarray]:
    """
    Applies a morphological operation (e.g., Erode, Dilate, Open, Close).
    Assumes a binary image or an image where the operation makes sense (e.g., grayscale).

    Args:
        image (np.ndarray): Input binary or grayscale image.
        kernel_size (int): Size of the structuring element.
        operation (str): 'erode', 'dilate', 'open', 'close'.
        shape (str): Shape of the kernel: 'rect', 'ellipse', 'cross'.

    Returns:
        np.ndarray: Image after morphological operation, or None on error.
    """
    if image is None:
        logger.error("Cannot apply morphological operation to a None image.")
        return None
    if not (isinstance(kernel_size, int) and kernel_size > 0):
        logger.error(f"Kernel size must be a positive integer. Got {kernel_size}.")
        return None

    try:
        if shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            logger.error(f"Unsupported kernel shape: {shape}")
            return None

        if operation == 'erode':
            result_img = cv2.erode(image, kernel, iterations=1)
        elif operation == 'dilate':
            result_img = cv2.dilate(image, kernel, iterations=1)
        elif operation == 'open':
            result_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            result_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            logger.error(f"Unsupported morphological operation: {operation}")
            return None
        
        logger.debug(f"Morphological operation '{operation}' with kernel size {kernel_size} and shape '{shape}' applied.")
        return result_img
    except Exception as e:
        logger.exception(f"Error applying morphological operation '{operation}': {e}")
        return None


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    # Create dummy images
    img_a = np.zeros((50, 50), dtype=np.uint8)
    img_a[10:30, 10:30] = 150 # A square
    img_b = np.zeros((50, 50), dtype=np.uint8)
    img_b[15:25, 15:25] = 50  # A smaller square inside

    # Test subtraction
    subtracted = subtract_images(img_a, img_b)
    assert subtracted is not None, "Subtraction failed"
    logger.info("Subtraction test passed.")
    # cv2.imwrite("temp_subtracted.png", subtracted) # Optional: save to view

    # Test thresholding
    thresh_binary = apply_threshold(img_a, 100, "binary")
    assert thresh_binary is not None, "Binary threshold failed"
    logger.info("Binary threshold test passed.")
    # cv2.imwrite("temp_thresh_binary.png", thresh_binary)

    thresh_otsu = apply_threshold(img_a, 0, "otsu_binary") # threshold_value is ignored for Otsu
    assert thresh_otsu is not None, "Otsu threshold failed"
    logger.info("Otsu threshold test passed.")
    # cv2.imwrite("temp_thresh_otsu.png", thresh_otsu)

    # Test blur
    blurred_median = apply_blur(img_a, 3, "median")
    assert blurred_median is not None, "Median blur failed"
    logger.info("Median blur test passed.")
    # cv2.imwrite("temp_blurred_median.png", blurred_median)

    # Test morphological operation (using the binary thresholded image)
    if thresh_binary is not None:
        eroded_img = apply_morphological_operation(thresh_binary, 3, "erode", "rect")
        assert eroded_img is not None, "Erosion failed"
        logger.info("Erosion test passed.")
        # cv2.imwrite("temp_eroded.png", eroded_img)

        dilated_img = apply_morphological_operation(thresh_binary, 3, "dilate", "rect")
        assert dilated_img is not None, "Dilation failed"
        logger.info("Dilation test passed.")
        # cv2.imwrite("temp_dilated.png", dilated_img)