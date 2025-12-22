# src/utils/logging_setup.py
import logging
import os 
import sys  # Import sys for StreamHandler
from typing import Dict, Optional
from .config_loader import get_config 
from .path_helpers import get_project_base_dir 

def setup_logging(config: Optional[Dict[str, object]] = None) -> logging.Logger:
    """
    Input: optional config dict. Output: configured root logger with file/console handlers.
    Sets up global logging configuration.
    Logs can be directed to a file specified in the config.
    Ensures INFO level (and above) messages also appear on console if file logging is enabled.
    """
    config = config or get_config()
    log_config = config.get('logging', {})
    
    level_str = log_config.get('level', 'INFO').upper()
    # Ensure the basicConfig level is the most verbose one needed for any handler
    # If console needs INFO and file needs DEBUG, basicConfig should be DEBUG.
    # Then, individual handlers can filter further.
    # For simplicity, let's assume file_log_level is the same or more verbose than console_log_level for now.
    # The root logger level will be set by basicConfig.
    log_level = getattr(logging, level_str, logging.INFO)
    
    log_format_str = log_config.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    date_format_str = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    formatter = logging.Formatter(log_format_str, date_format_str)

    log_file_config_path = log_config.get('log_file') 
    
    # Get the root logger
    # If basicConfig has already been called elsewhere (e.g. by an imported library), 
    # this might behave unexpectedly. It's best if setup_logging is called once, early.
    # We clear existing handlers to ensure a clean setup.
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # print("Note: Root logger already had handlers. Clearing them for re-configuration.")
        for handler in root_logger.handlers[:]: # Iterate over a copy
            root_logger.removeHandler(handler)
            handler.close() # Close the handler before removing

    root_logger.setLevel(log_level) # Set the overall minimum level for the root logger

    # Configure File Handler (if specified)
    file_handler_active = False
    if log_file_config_path and log_file_config_path.lower() != 'null':
        if not os.path.isabs(log_file_config_path):
            project_base = get_project_base_dir(config) 
            log_file_full_path = os.path.join(project_base, log_file_config_path)
        else:
            log_file_full_path = log_file_config_path
        
        os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_full_path, mode='a')
        file_handler.setFormatter(formatter)
        # File handler can have its own level, potentially different from root logger's overall level
        # For now, let it inherit the root logger's level (set by log_level)
        # Or, set it explicitly: file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        file_handler_active = True
        print(f"Detailed logging to file: {log_file_full_path} (Level: {level_str})") # This print will always go to console
    # else:
        # This print is redundant if console handler is always added.
        # print("Detailed logging to console only.")


    # Configure Console Handler
    # Add a console handler regardless of whether a file handler is active,
    # but its level can be controlled.
    console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for INFO, sys.stderr for ERROR/WARNING
    console_handler.setFormatter(formatter)
    
    # Set console handler level. Typically INFO or WARNING for progress.
    # If file logging is active, we might want console to be less verbose.
    # If no file logging, console should adhere to the main 'level_str'.
    if file_handler_active:
        # If logging to file, console can be INFO to show progress, 
        # while file captures everything from log_level (e.g. DEBUG).
        console_log_level_str = log_config.get('console_level', 'INFO').upper() # New config option
        console_log_level = getattr(logging, console_log_level_str, logging.INFO)
        console_handler.setLevel(console_log_level)
        if console_log_level > log_level: # If console is less verbose than root
            print(f"Console logging progress at level: {console_log_level_str}")
    else: # No file logging, console is the primary output
        console_handler.setLevel(log_level) # Console gets the main configured level
        print(f"Logging to console at level: {level_str}")

    root_logger.addHandler(console_handler)
    
    # Example: Silence overly verbose libraries if needed (after our handlers are set)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Log initialization message through the configured system
    if file_handler_active:
        root_logger.info(f"Logging system initialized. File output: {log_file_full_path}, Console output active.")
    else:
        root_logger.info("Logging system initialized (console output only).")
        
    return root_logger

if __name__ == '__main__':
    # To test, you might need to ensure a dummy config is loadable or mock get_config()
    # For a quick test, assuming config can be loaded:
    try:
        # To test file logging, ensure 'log_file' is set in your test config.
        # e.g., in pipeline_config.yml:
        # logging:
        #   level: "DEBUG"
        #   log_file: "test_run.log"
        #   console_level: "INFO" # Test this new option
        
        logger = setup_logging() 
        
        logger.debug("This is a debug message. (Should go to file if DEBUG, not console if console is INFO)")
        logger.info("This is an info message. (Should go to both if console is INFO or more verbose)")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")

        # Create a child logger to test naming
        child_logger = logging.getLogger("my_module.child")
        child_logger.info("Info message from child logger.")
        child_logger.debug("Debug message from child logger.")


        # Explicitly flush handlers to ensure file write for testing
        for handler in logging.root.handlers:
            handler.flush()
        
        print("\nTest messages logged. Check console and configured log file.")
        print("If file logging is active with DEBUG level, 'debug message' should be in file.")
        print("If console_level is INFO, 'debug message' should NOT be on console.")

    except Exception as e:
        print(f"Error during logging setup test: {e}")
        print("Ensure 'config/pipeline_config.yml' is accessible or mock get_config().")
