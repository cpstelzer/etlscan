# src/utils/path_helpers.py
import os
import re
import logging # Import logging
from typing import Optional, Tuple, Union, List
from .config_loader import get_config

# Initialize logger for this module.
# It will be configured by setup_logging() when the application starts.
logger = logging.getLogger(__name__)

def get_project_base_dir() -> str:
    """Gets the project base directory from config."""
    config = get_config()
    return config.get('project_base_dir', '.') # Default to current dir if not set

def get_sample_dir(sample_subdir_name: Optional[str] = None) -> str:
    """
    Constructs the full path to a sample directory within data/input.
    If sample_subdir_name is 'MyBatch/SampleA', it resolves to project_base_dir/data/input/MyBatch/SampleA.
    If sample_subdir_name is None, it uses the 'sample_subdir_name' from config.
    """
    config = get_config()
    base_dir = get_project_base_dir()
    input_data_root = os.path.join(base_dir, 'data', 'input')

    if sample_subdir_name is None:
        sample_subdir_name_resolved = config.get('sample_subdir_name', 'default_sample')
    else:
        sample_subdir_name_resolved = sample_subdir_name
    
    return os.path.join(input_data_root, sample_subdir_name_resolved)


def get_output_dir_for_sample(sample_subdir_name: Optional[str] = None, batch_name: Optional[str] = None) -> str:
    """
    Constructs the base output path for a given sample.
    If batch_name is provided, output is project_base_dir/data/output/batch_name/sample_subdir_name.
    Otherwise, output is project_base_dir/data/output/sample_subdir_name.
    """
    config = get_config()
    base_dir = get_project_base_dir()
    output_data_root = os.path.join(base_dir, 'data', 'output')

    if sample_subdir_name is None:
        sample_subdir_name_resolved = config.get('sample_subdir_name', 'default_sample')
    else:
        sample_subdir_name_resolved = sample_subdir_name

    if batch_name:
        # Ensure batch_name doesn't inadvertently create nested "output/output" if it's already part of path
        if os.path.basename(output_data_root) == batch_name: # Unlikely but defensive
            output_path_base = output_data_root 
        else:
            output_path_base = os.path.join(output_data_root, batch_name)
        output_path = os.path.join(output_path_base, sample_subdir_name_resolved)
    else:
        output_path = os.path.join(output_data_root, sample_subdir_name_resolved)
    
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
    # logger.warning(f"Could not parse date/plate/well from directory name: {dir_name}") # Too verbose for listing
    return None, None, None

def get_valid_sample_subdirectories(batch_input_dir: str) -> List[str]:
    """
    Lists valid sample subdirectories from a given batch input directory.
    Valid subdirectories are those whose names match the 'YYYYMMDD_plateX_wellY' format.
    """
    valid_sample_dirs = []
    if not os.path.isdir(batch_input_dir):
        logger.error(f"Batch input directory not found: {batch_input_dir}")
        return valid_sample_dirs
    
    for item_name in sorted(os.listdir(batch_input_dir)):
        item_path = os.path.join(batch_input_dir, item_name)
        if os.path.isdir(item_path):
            date, plate, well = parse_date_plate_well(item_name)
            if date and plate and well:
                valid_sample_dirs.append(item_name)
            else:
                logger.debug(f"Skipping '{item_name}' in batch directory: does not match sample name format.")
    
    if not valid_sample_dirs:
        logger.warning(f"No valid sample subdirectories found in: {batch_input_dir}")
    else:
        logger.info(f"Found {len(valid_sample_dirs)} valid sample subdirectories in '{batch_input_dir}'.")
    return valid_sample_dirs


def extract_frame_from_filename(filename: str) -> Optional[int]:
    """Extracts the frame number (like 1 from _f00001_) from a filename."""
    match = re.search(r'_f(\d+)_', filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert extracted frame to int in {filename}")
            return None
    return None

def parse_sharp_crop_filename(filename: str) -> Tuple[Optional[str], Optional[Union[int, str]], Optional[int], Optional[int]]:
    """
    Parses a sharp crop filename generated by ParticleCropper in Stage 2.
    Expected format examples:
      YYYYMMDD_plateX_wellY_F<frame>_sharp_X<coord_x>_Y<coord_y>.png
      YYYYMMDD_plateX_wellY_FUnkF_sharp_X<coord_x>_Y<coord_y>.png
      intermediate_crop_idx<some_id>_X<coord_x>_Y<coord_y>.png (if used for intermediate crops)

    Returns (full_prefix_before_X, frame_num_or_str, x_coord, y_coord)
    The frame_num_or_str will be an int if numeric, or "UnkF" (or other string) if not.
    The full_prefix_before_X includes all parts of the filename before "_X<coord_x>..."
    """
    match = re.search(
        r"^(?P<prefix_part>.*?)_X(?P<x_coord>\d+)_Y(?P<y_coord>\d+)\.png$",
        filename,
        re.IGNORECASE
    )

    if match:
        try:
            full_prefix = match.group('prefix_part')
            x = int(match.group("x_coord"))
            y = int(match.group("y_coord"))

            frame_match_in_prefix = re.search(r"_F(?P<frame_val>\d+|UnkF)", full_prefix, re.IGNORECASE)
            frame_val: Optional[Union[int, str]] = None
            if frame_match_in_prefix:
                frame_str = frame_match_in_prefix.group("frame_val")
                if frame_str.upper() == "UNKF":
                    frame_val = "UnkF"
                else:
                    frame_val = int(frame_str)
            else:
                logger.debug(f"Frame pattern (_F<val>) not found in prefix '{full_prefix}' of filename '{filename}'.")

            return full_prefix, frame_val, x, y
        except ValueError:
            logger.error(f"ValueError parsing coordinates or frame from filename: {filename} using regex for _X_Y suffix.")
            return None, None, None, None
    else:
        logger.debug(f"Filename '{filename}' did not match general pattern ending in _X<digits>_Y<digits>.png.")
    return None, None, None, None


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    from .config_loader import load_config 
    try:
        config_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_config.yml')
        if not os.path.exists(config_file_path):
            config_file_path = os.path.join('config', 'pipeline_config.yml')
        
        load_config(config_file_path) 
        logger.info(f"Config loaded for path_helpers __main__ from: {config_file_path}")
        
        logger.info(f"Project Base Dir: {get_project_base_dir()}")
        
        # Test get_sample_dir
        logger.info(f"Sample Dir (default config): {get_sample_dir()}")
        logger.info(f"Sample Dir (specific): {get_sample_dir('my_sample_01')}")
        logger.info(f"Sample Dir (nested specific): {get_sample_dir(os.path.join('MyBatch', 'my_sample_01'))}")


        # Test get_output_dir_for_sample
        logger.info(f"Output Dir for Sample (default config, no batch): {get_output_dir_for_sample()}")
        logger.info(f"Output Dir for Sample (specific sample, no batch): {get_output_dir_for_sample('my_sample_01')}")
        logger.info(f"Output Dir for Sample (specific sample, with batch): {get_output_dir_for_sample('my_sample_01', batch_name='MyBatchOutput')}")
        logger.info(f"Output Dir for Sample (default config sample, with batch): {get_output_dir_for_sample(batch_name='MyBatchOutput')}")


        logger.info(f"Models Dir: {get_models_dir()}")
        logger.info(f"Training Data Dir: {get_training_data_dir()}")

        date, plate, well = parse_date_plate_well("20230101_plate1_well5_extra_info")
        logger.info(f"Parsed: Date={date}, Plate={plate}, Well={well}")
        date, plate, well = parse_date_plate_well("/path/to/20230101_plate1_well5")
        logger.info(f"Parsed: Date={date}, Plate={plate}, Well={well}")
        
        frame = extract_frame_from_filename("20230101_plate1_well1_f00023_dpt-0.2_fd0.0mm.png")
        logger.info(f"Parsed Original Image Frame: {frame}")
        
        logger.info("\n--- Testing parse_sharp_crop_filename ---")
        test_filenames = [
            "20250410_plate4_well4_F00008_sharp_X3208_Y1564.png",
            "intermediate_crop_idx55_X123_Y456.png"
        ]
        for fn in test_filenames:
            prefix, frm, x, y = parse_sharp_crop_filename(fn)
            logger.info(f"File: '{fn}' -> Prefix='{prefix}', Frame='{frm}', X='{x}', Y='{y}'")

        logger.info("\n--- Testing get_valid_sample_subdirectories ---")
        # Create dummy dirs for testing
        test_batch_dir_path = os.path.join(get_project_base_dir(), "temp_test_batch_input")
        os.makedirs(test_batch_dir_path, exist_ok=True)
        valid_sample_names = ["20230101_plate1_well1", "20230102_plate2_well3"]
        invalid_names = ["not_a_sample", "20230103_plate3_well"] # missing part
        for name in valid_sample_names + invalid_names:
            os.makedirs(os.path.join(test_batch_dir_path, name), exist_ok=True)
        
        # Create a file to ensure it's ignored
        with open(os.path.join(test_batch_dir_path, "a_file.txt"), "w") as f: f.write("test")

        found_samples = get_valid_sample_subdirectories(test_batch_dir_path)
        logger.info(f"Valid samples found in '{test_batch_dir_path}': {found_samples}")
        assert sorted(found_samples) == sorted(valid_sample_names)

        # Clean up dummy dirs
        for name in valid_sample_names + invalid_names:
            os.rmdir(os.path.join(test_batch_dir_path, name))
        os.remove(os.path.join(test_batch_dir_path, "a_file.txt"))
        os.rmdir(test_batch_dir_path)
        logger.info(f"Cleaned up {test_batch_dir_path}")

    except Exception as e:
        logger.exception(f"Error during path_helpers test: {e}")
        logger.error("Ensure 'config/pipeline_config.yml' is accessible or mock get_config().")