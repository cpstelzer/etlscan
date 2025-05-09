# src/utils/config_loader.py
import yaml
import os
from typing import Dict, Any

_config: Dict[str, Any] = {}

def load_config(config_path: str = "config/pipeline_config.yml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    """
    global _config
    if not _config: # Load only once
        if not os.path.exists(config_path):
            # Try to find it relative to this script's parent's parent directory if not found directly
            # This helps if scripts are run from different locations.
            # project_root / src / utils / config_loader.py -> project_root
            abs_path = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(abs_path))
            project_root = os.path.dirname(src_dir)
            alt_config_path = os.path.join(project_root, config_path)

            if os.path.exists(alt_config_path):
                config_path_to_load = alt_config_path
            else:
                raise FileNotFoundError(
                    f"Configuration file not found at '{config_path}' or '{alt_config_path}'. "
                    "Ensure it exists or provide the correct path."
                )
        else:
            config_path_to_load = config_path

        try:
            with open(config_path_to_load, "r") as f:
                _config = yaml.safe_load(f)
            # Replace 'null' strings from YAML with Python None for specific nested keys
            _replace_null_with_none(_config)
            print(f"Configuration loaded from: {config_path_to_load}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file: {e}")
            raise
        except FileNotFoundError as e: # Should be caught above, but as safeguard
            print(str(e))
            raise
    return _config

def _replace_null_with_none(d: Dict[str, Any]):
    """Recursively replaces 'null' string values with Python None in a dictionary."""
    for k, v in d.items():
        if isinstance(v, dict):
            _replace_null_with_none(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _replace_null_with_none(item)
                elif item == 'null':
                    v[i] = None
        elif v == 'null':
            d[k] = None


def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration. Loads it if not already loaded.
    """
    if not _config:
        load_config()
    return _config

if __name__ == "__main__":
    # Example of how to use it
    try:
        config = load_config("../config/pipeline_config.yml") # Adjust path for direct run
        print("Config loaded successfully.")
        # print(config)
        print(f"Log level: {config.get('logging', {}).get('level')}")
        print(f"RF max_depth options: {config.get('stage_03_ml_classification', {}).get('training_params', {}).get('rf_param_grid', {}).get('max_depth')}")
    except FileNotFoundError:
        print("Run this from the 'utils' directory or adjust path for testing.")
    except Exception as e:
        print(f"An error occurred: {e}")