# src/utils/common_exceptions.py

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class ProcessingError(RuntimeError):
    """Custom exception for errors during generic processing stages."""
    pass

class ImageProcessingError(ProcessingError):
    """Custom exception for errors specific to image processing operations."""
    pass

class MLModelError(RuntimeError):
    """Custom exception for errors related to ML model loading, training, or prediction."""
    pass

class DataValidationError(ValueError):
    """Custom exception for data validation errors (e.g., missing columns, incorrect types)."""
    pass