# src/utils/config_loader.py
import copy
import functools
import os
from typing import Any, Dict

import yaml


def _resolve_config_path(config_path: str) -> str:
    """Resolve a config path relative to the project root when needed."""
    if os.path.exists(config_path):
        return config_path

    abs_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(abs_path))
    project_root = os.path.dirname(src_dir)
    alt_config_path = os.path.join(project_root, config_path)
    if os.path.exists(alt_config_path):
        return alt_config_path

    raise FileNotFoundError(
        f"Configuration file not found at '{config_path}' or '{alt_config_path}'. "
        "Ensure it exists or provide the correct path."
    )


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict where values in overlay override values in base recursively."""
    merged = copy.deepcopy(base)
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            merged[key] = _deep_merge(base_value, overlay_value)
        else:
            merged[key] = copy.deepcopy(overlay_value)
    return merged


def _apply_hardware_profile(config: Dict[str, Any]) -> None:
    """Deep-merge the active hardware profile into config['acquisition'] in place."""
    profile_name = config["active_hardware_profile"]
    profiles = config["hardware_profiles"]
    if profile_name not in profiles:
        raise KeyError(
            f"active_hardware_profile '{profile_name}' not found in hardware_profiles; "
            f"available: {sorted(profiles.keys())}"
        )
    profile = profiles[profile_name]
    acquisition = config.get("acquisition", {})
    config["acquisition"] = _deep_merge(acquisition, profile)


@functools.lru_cache(maxsize=None)
def load_config_once(config_path: str = "config/config.yml") -> Dict[str, Any]:
    """Load the merged YAML configuration file once, apply hardware profile, return dict."""
    resolved_path = _resolve_config_path(config_path)
    try:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            loaded_config: Dict[str, Any] = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML configuration file: {exc}")
        raise

    _replace_null_with_none(loaded_config)
    _apply_hardware_profile(loaded_config)
    return loaded_config


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


def get_unified_segmentation_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a segmentation parameter dictionary from system + stage_04 egg keys.

    All required keys must be present under config['system']; missing keys raise KeyError.
    """
    system_cfg = config["system"]
    global_thresh_cfg = system_cfg["thresholding"]
    system_segmentation = system_cfg["segmentation"]

    unified_params: Dict[str, Any] = dict(system_segmentation)
    unified_params["source_image_illumination_type"] = system_cfg["source_image_illumination_type"]
    unified_params["thresholding_mode"] = global_thresh_cfg["thresholding_mode"]
    unified_params["otsu_threshold_multiplier"] = global_thresh_cfg["otsu_threshold_multiplier"]
    unified_params["fixed_threshold_value"] = global_thresh_cfg["fixed_threshold_value"]

    # Merge egg detection parameters so the extractor receives them in a single dict.
    egg_detection_params = config["stage_04_biological_analysis"]["egg_detection"]
    for key, value in egg_detection_params.items():
        unified_params[key] = value

    return unified_params
