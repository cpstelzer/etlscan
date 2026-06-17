# src/utils/path_helpers.py
import os
import re
import logging
from typing import Any, Dict, Optional, Tuple, Union, List

logger = logging.getLogger(__name__)


def get_project_base_dir(config: Dict[str, Any]) -> str:
    """Input: config dict. Output: project root path without creating files.

    Reads ``paths.pipeline_base`` and falls back to the legacy top-level
    ``project_base_dir`` key for configs that have not been migrated.
    """
    paths_cfg = config.get("paths", {})
    configured_base_dir = paths_cfg.get("pipeline_base", config.get("project_base_dir"))
    if configured_base_dir is None:
        raise KeyError("config must define paths.pipeline_base (or legacy project_base_dir)")
    expanded_base_dir = os.path.expanduser(os.path.expandvars(str(configured_base_dir)))
    return expanded_base_dir


def get_sample_dir(config: Dict[str, Any], sample_subdir_name: Optional[str] = None) -> str:
    """
    Constructs the full path to a sample directory within data/input.
    If sample_subdir_name is 'MyBatch/SampleA', it resolves to project_base_dir/data/input/MyBatch/SampleA.
    If sample_subdir_name is None, it uses the 'sample_subdir_name' from config.
    """
    base_dir = get_project_base_dir(config)
    input_data_root = os.path.join(base_dir, 'data', 'input')

    if sample_subdir_name is None:
        sample_subdir_name_resolved = config.get("sample_subdir_name", "")
    else:
        sample_subdir_name_resolved = sample_subdir_name

    return os.path.join(input_data_root, sample_subdir_name_resolved)


def get_output_dir_for_sample(
    config: Dict[str, Any],
    sample_subdir_name: Optional[str] = None,
    batch_name: Optional[str] = None,
) -> str:
    """
    Constructs the base output path for a given sample.
    If batch_name is provided, output is project_base_dir/data/output/batch_name/sample_subdir_name.
    Otherwise, output is project_base_dir/data/output/sample_subdir_name.
    """
    base_dir = get_project_base_dir(config)
    output_data_root = os.path.join(base_dir, 'data', 'output')

    if sample_subdir_name is None:
        sample_subdir_name_resolved = config.get("sample_subdir_name", "")
    else:
        sample_subdir_name_resolved = sample_subdir_name

    if batch_name:
        if os.path.basename(output_data_root) == batch_name:
            output_path_base = output_data_root
        else:
            output_path_base = os.path.join(output_data_root, batch_name)
        output_path = os.path.join(output_path_base, sample_subdir_name_resolved)
    else:
        output_path = os.path.join(output_data_root, sample_subdir_name_resolved)

    os.makedirs(output_path, exist_ok=True)
    return output_path


def get_models_dir(config: Dict[str, Any]) -> str:
    """Input: config dict. Output: models directory path (ensures existence)."""
    base_dir = get_project_base_dir(config)
    models_path = os.path.join(base_dir, 'models')
    os.makedirs(models_path, exist_ok=True)
    return models_path


def get_training_data_dir(config: Dict[str, Any]) -> str:
    """Input: config dict. Output: training data directory path (ensures existence)."""
    base_dir = get_project_base_dir(config)
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
        return match.group(1), match.group(2), match.group(3)
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
