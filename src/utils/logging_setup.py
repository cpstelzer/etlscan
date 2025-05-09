# src/utils/logging_setup.py
import logging
from .config_loader import get_config

def setup_logging() -> logging.Logger:
    """
    Sets up global logging configuration.
    """
    config = get_config()
    log_config = config.get('logging', {})
    
    level_str = log_config.get('level', 'INFO').upper()
    log_level = getattr(logging, level_str, logging.INFO)
    
    log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
    
    # Example: Silence overly verbose libraries if needed
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger() # Return root logger or a specific one

if __name__ == '__main__':
    # This basic setup is for testing the logging_setup.py itself.
    # In the main application, `setup_logging()` would be called once.
    # To test, you might need to ensure a dummy config is loadable or mock get_config()
    # For a quick test, assuming config can be loaded:
    try:
        logger = setup_logging()
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")
    except Exception as e:
        print(f"Error during logging setup test: {e}")
        print("Ensure 'config/pipeline_config.yml' is accessible or mock get_config().")