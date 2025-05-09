# src/utils/path_helpers.py
import os
import re
from typing import Optional, Tuple
from .config_loader import get_config

def get_project_base_dir() -> str:
    """Gets the project base directory from config."""
    config = get_config()
    return config.get('project_base_dir', '.') # Default to current dir if not set

def get_sample_dir(sample_subdir_name: Optional[str] = None) -> str:
    """Constructs the full path to a sample directory."""
    config = get_config()
    base_dir = get_project_base_dir()
    if sample_subdir_name is None:
        sample_subdir_name = config.get('sample_subdir_name', 'default_sample')
    return os.path.join(base_dir, 'data', 'input', sample_subdir_name)

def get_output_dir_for_sample(sample_subdir_name: Optional[str] = None) -> str:
    """Constructs the base output path for a given sample."""
    config = get_config()
    base_dir = get_project_base_dir()
    if sample_subdir_name is None:
        sample_subdir_name = config.get('sample_subdir_name', 'default_sample')
    output_path = os.path.join(base_dir, 'data', 'output', sample_subdir_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def get_models_dir() -> str:
    """Constructs the full path to the models directory."""
    base_dir = get_project_base_dir()
    models_path = os.path.join(base_dir, 'models')
    os.makedirs(models_path, exist_ok=True)
    return models_path
    
def get_training_data_dir() -> str:
    """Constructs the full path to the training data directory."""
    base_dir = get_project_base_dir()
    training_data_path = os.path.join(base_dir, 'data', 'training_data')
    os.makedirs(training_data_path, exist_ok=True)
    return training_data_path

def parse_date_plate_well(directory_path_or_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses date, plate, and well number from a directory path or name
    like YYYYMMDD_plateX_wellY.
    """
    dir_name = os.path.basename(os.path.normpath(directory_path_or_name))
    match = re.match(r'(\d{8})_plate(\d+)_well(\d+)', dir_name, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2), match.group(3)  # date, plate, well
    # logger.warning(f"Could not parse date/plate/well from directory name: {dir_name}") # Use logger from calling module
    return None, None, None

def extract_frame_from_filename(filename: str) -> Optional[int]:
    """Extracts the frame number (like 1 from _f00001_) from a filename."""
    match = re.search(r'_f(\d+)_', filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            # logger.warning(f"Could not convert extracted frame to int in {filename}") # Use logger from calling module
            return None
    return None

if __name__ == '__main__':
    # Ensure config is loaded for testing
    from .config_loader import load_config
    try:
        load_config('../../config/pipeline_config.yml') # Adjust path if running directly
        print(f"Project Base Dir: {get_project_base_dir()}")
        print(f"Sample Dir (default): {get_sample_dir()}")
        print(f"Sample Dir (specific): {get_sample_dir('my_sample_01')}")
        print(f"Output Dir for Sample (default): {get_output_dir_for_sample()}")
        print(f"Models Dir: {get_models_dir()}")
        print(f"Training Data Dir: {get_training_data_dir()}")

        date, plate, well = parse_date_plate_well("20230101_plate1_well5_extra_info")
        print(f"Parsed: Date={date}, Plate={plate}, Well={well}")
        date, plate, well = parse_date_plate_well("/path/to/20230101_plate1_well5")
        print(f"Parsed: Date={date}, Plate={plate}, Well={well}")
        frame = extract_frame_from_filename("20230101_plate1_well1_f00023_dpt-0.2_fd0.0mm.png")
        print(f"Parsed Frame: {frame}")

    except Exception as e:
        print(f"Error during path_helpers test: {e}")
        print("Ensure 'config/pipeline_config.yml' is accessible or mock get_config().")