# src/utils/logging_setup.py
import logging
import os # Added for path joining
from .config_loader import get_config # get_project_base_dir removed from here
from .path_helpers import get_project_base_dir # Added import for get_project_base_dir

def setup_logging() -> logging.Logger:
    """
    Sets up global logging configuration.
    Logs can be directed to a file specified in the config.
    """
    config = get_config()
    log_config = config.get('logging', {})
    
    level_str = log_config.get('level', 'INFO').upper()
    log_level = getattr(logging, level_str, logging.INFO)
    
    log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    log_file_config_path = log_config.get('log_file') # Can be 'null' or path string
    
    basicConfig_kwargs = {
        'level': log_level,
        'format': log_format,
        'datefmt': date_format
    }

    if log_file_config_path and log_file_config_path.lower() != 'null':
        # Make log file path relative to project base dir if not absolute
        if not os.path.isabs(log_file_config_path):
            project_base = get_project_base_dir() # This now comes from path_helpers
            log_file_full_path = os.path.join(project_base, log_file_config_path)
        else:
            log_file_full_path = log_file_config_path
        
        os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True) # Ensure log directory exists
        basicConfig_kwargs['filename'] = log_file_full_path
        basicConfig_kwargs['filemode'] = 'a' # Append mode is default, but explicit
        print(f"Logging to file: {log_file_full_path}")
    else:
        print("Logging to console.")


    logging.basicConfig(**basicConfig_kwargs)
    
    # Example: Silence overly verbose libraries if needed
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # logging.getLogger("PIL").setLevel(logging.WARNING)

    # After basicConfig, get the root logger to inform it's set up.
    # basicConfig configures the root logger.
    root_logger = logging.getLogger()
    if log_file_config_path and log_file_config_path.lower() != 'null':
        root_logger.info(f"Logging initialized. Log file: {basicConfig_kwargs.get('filename')}")
    else:
        root_logger.info("Logging initialized (console output).")
        
    return root_logger

if __name__ == '__main__':
    # This basic setup is for testing the logging_setup.py itself.
    # In the main application, `setup_logging()` would be called once.
    # To test, you might need to ensure a dummy config is loadable or mock get_config()
    # For a quick test, assuming config can be loaded:
    try:
        # To test file logging, ensure 'log_file' is set in your test config.
        # e.g., in pipeline_config.yml:
        # logging:
        #   level: "DEBUG"
        #   log_file: "test_run.log"
        
        logger = setup_logging() # This will print where it's logging
        
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")

        # Explicitly flush handlers to ensure file write for testing
        for handler in logging.root.handlers:
            handler.flush()
        
        print("Test messages logged. Check console or configured log file.")

    except Exception as e:
        print(f"Error during logging setup test: {e}")
        print("Ensure 'config/pipeline_config.yml' is accessible or mock get_config().")
        print("Make sure 'project_base_dir' is correctly set in the config.")