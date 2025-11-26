# src/utils/config_loader.py
import yaml
import os
from typing import Dict, Any, Optional 

_config: Dict[str, Any] = {}

def load_config(config_path: str = "config/pipeline_config.yml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    """
    global _config
    if not _config: # Load only once
        if not os.path.exists(config_path):
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
            _replace_null_with_none(_config)
            print(f"Configuration loaded from: {config_path_to_load}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file: {e}")
            raise
        except FileNotFoundError as e: 
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

def get_unified_segmentation_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Assemble a segmentation parameter dictionary by combining system defaults with
    any stage-specific overrides. The resulting mapping is used by both the
    segmentation helpers and Stage 04 visualizations.
    """
    if config is None:
        config = get_config()

    system_cfg = config.get('system', {})
    global_thresh_cfg = system_cfg.get('thresholding', {})

    # Prefer the current Stage 03 configuration, but fall back to legacy keys if needed
    s3_config = config.get('stage_03_filtering_and_classification', {})
    if not s3_config:
        s3_config = config.get('stage_03_ml_classification', {})

    s3_feat_params_overall = s3_config.get('feature_extraction_params', {})
    s3_main_body_seg_params = s3_feat_params_overall.get('main_body_segmentation', {})
    s3_egg_detection_params = s3_feat_params_overall.get('egg_detection', {})

    # Start with system-level segmentation defaults when provided
    system_segmentation_defaults = system_cfg.get('segmentation', {})
    unified_params = system_segmentation_defaults.copy()

    # Apply any Stage 03 overrides from the ML feature extraction config
    unified_params.update(s3_main_body_seg_params)

    # Override/add with global system settings
    unified_params['source_image_illumination_type'] = system_cfg.get('source_image_illumination_type', 'darkfield')
    unified_params['thresholding_mode'] = global_thresh_cfg.get('mode', 'otsu')
    unified_params['otsu_threshold_multiplier'] = global_thresh_cfg.get('otsu_threshold_multiplier', 0.45)
    unified_params['fixed_threshold_value'] = global_thresh_cfg.get('fixed_threshold_value', 70)

    # Ensure core main body segmentation params have defaults if somehow missing
    unified_params.setdefault('enable_unsharp_masking', False)
    unified_params.setdefault('unsharp_mask_sigma', 1.0)
    unified_params.setdefault('unsharp_mask_amount', 1.0)
    unified_params.setdefault('min_noise_size', 50)
    unified_params.setdefault('closing_selem_size', 1)
    unified_params.setdefault('min_significant_area', 300)  # Used by downstream ML feature extraction

    # Add egg detection parameters to this dictionary as well,
    # because extract_ml_features expects them in the single `params` dict it receives.
    # segment_main_body_from_image_float will ignore these egg-specific keys.
    if s3_egg_detection_params:  # Check if the dict exists
        for key, value in s3_egg_detection_params.items():
            unified_params[key] = value  # Add or overwrite if somehow already present

    # Ensure egg params also have defaults if not in config (for robustness, though config should provide them)
    unified_params.setdefault('gaussian_sigma', 1.0)  # For egg blur
    unified_params.setdefault('egg_posterior_pixel_fraction', 0.45)
    unified_params.setdefault('egg_threshold_quantile', 0.92)
    # ... add defaults for other egg params if desired for extreme robustness

    return unified_params


if __name__ == "__main__":
    try:
        config_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_config.yml')
        if not os.path.exists(config_file_path): 
            config_file_path = os.path.join('..', 'config', 'pipeline_config.yml') 
        if not os.path.exists(config_file_path):
             config_file_path = "config/pipeline_config.yml"

        main_config = load_config(config_file_path) 
        print("Config loaded successfully.")
        print(f"Log level: {main_config.get('logging', {}).get('level')}")
        
        rf_max_depth = main_config.get('stage_03_ml_classification', {})\
                               .get('training_params', {})\
                               .get('rf_param_grid', {})\
                               .get('max_depth')
        print(f"RF max_depth options: {rf_max_depth}")

        print("\n--- Testing get_unified_segmentation_params (with new config structure) ---")
        seg_params = get_unified_segmentation_params(main_config)
        print("Unified Segmentation Parameters (should include main_body_seg and egg_detection):")
        for key, value in seg_params.items():
            print(f"  {key}: {value}")
        
        assert 'enable_unsharp_masking' in seg_params
        assert 'min_noise_size' in seg_params
        assert 'egg_threshold_quantile' in seg_params 
        assert 'gaussian_sigma' in seg_params # From egg_detection params

    except FileNotFoundError:
        print("Run this from the 'utils' directory or project root, or adjust path for testing.")
        print(f"Attempted to load config from: {config_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")