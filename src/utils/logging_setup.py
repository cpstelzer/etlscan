# src/utils/logging_setup.py
import logging
import os
import sys
from typing import Dict

from .path_helpers import get_project_base_dir


def setup_logging(config: Dict[str, object]) -> logging.Logger:
    """
    Input: config dict. Output: configured root logger with file/console handlers.
    Sets up global logging configuration.
    Logs can be directed to a file specified in the config.
    Ensures INFO level (and above) messages also appear on console if file logging is enabled.
    """
    log_config = config["logging"]

    level_str = log_config["level"].upper()
    # Ensure the basicConfig level is the most verbose one needed for any handler
    # If console needs INFO and file needs DEBUG, basicConfig should be DEBUG.
    # Then, individual handlers can filter further.
    # For simplicity, let's assume file_log_level is the same or more verbose than console_log_level for now.
    # The root logger level will be set by basicConfig.
    log_level = getattr(logging, level_str, logging.INFO)
    
    log_format_str = log_config["format"]
    console_format_str = log_config.get("console_format", log_format_str)
    date_format_str = log_config["date_format"]

    formatter = logging.Formatter(log_format_str, date_format_str)
    console_formatter = logging.Formatter(console_format_str, date_format_str)

    log_file_config_path = log_config["log_file"]
    
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
        log_file_full_path_resolved = log_file_full_path
    else:
        log_file_full_path_resolved = None


    # Configure Console Handler
    # Add a console handler regardless of whether a file handler is active,
    # but its level can be controlled.
    console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for INFO, sys.stderr for ERROR/WARNING
    console_handler.setFormatter(console_formatter)
    
    # Set console handler level. Typically INFO or WARNING for progress.
    # If file logging is active, we might want console to be less verbose.
    # If no file logging, console should adhere to the main 'level_str'.
    if file_handler_active:
        # If logging to file, console can be INFO to show progress, 
        # while file captures everything from log_level (e.g. DEBUG).
        console_log_level_str = log_config["console_level"].upper()
        console_log_level = getattr(logging, console_log_level_str, logging.INFO)
        console_handler.setLevel(console_log_level)
    else: # No file logging, console is the primary output
        console_handler.setLevel(log_level) # Console gets the main configured level
        console_log_level_str = level_str

    root_logger.addHandler(console_handler)
    
    # Example: Silence overly verbose libraries if needed (after our handlers are set)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Log initialization message through the configured system
    if file_handler_active:
        root_logger.info(
            "Logging system initialized. File output: %s (level=%s), console level=%s.",
            log_file_full_path_resolved,
            level_str,
            console_log_level_str,
        )
    else:
        root_logger.info(
            "Logging system initialized (console only). Console level=%s.", console_log_level_str
        )
        
    return root_logger

