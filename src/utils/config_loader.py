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


def _link_acquisition_output_root(config: Dict[str, Any]) -> None:
    """Inject paths.raw_data_root as acquisition.paths.output_folder_base.

    Keeps raw_data_root the single source of truth for where acquisition writes
    captured frames, while the acquisition module still reads it from its own
    subtree (which is all it receives).
    """
    raw_data_root = config.get("paths", {}).get("raw_data_root")
    if raw_data_root is None:
        return
    acquisition = config.setdefault("acquisition", {})
    acq_paths = acquisition.setdefault("paths", {})
    acq_paths["output_folder_base"] = raw_data_root


def _assemble_system_view(config: Dict[str, Any]) -> None:
    """Assemble the internal `system` view from the `imaging` + `segmentation` sections.

    The config file keeps imaging geometry and segmentation/thresholding in two
    clearly separated sections, but a lot of pipeline code consumes a single
    combined `config['system']` dict (image dims + source_image_illumination_type
    + thresholding + segmentation tuning).  This reconstructs that view so the
    split is purely a config-file concern with no behavior change.
    """
    imaging = config.get("imaging")
    segmentation = config.get("segmentation")
    if imaging is None or segmentation is None:
        return
    seg = dict(segmentation)
    thresholding = seg.pop("thresholding", {})
    # subtracted_foreground is the shared FG/BG gate for stage 1 (batch + realtime). Lift it
    # to a top-level system key (like thresholding) so it stays out of the segmentation-tuning
    # view consumed by get_unified_segmentation_params (stage 2/4).
    subtracted_foreground = seg.pop("subtracted_foreground", {})
    config["system"] = {
        **imaging,
        "thresholding": thresholding,
        "subtracted_foreground": subtracted_foreground,
        "segmentation": seg,
    }


def _link_imaging_dimensions_from_roi(config: Dict[str, Any]) -> None:
    """Derive imaging.image_width_px/height_px from the active hardware profile's ROI.

    The saved frame size equals the camera ROI, which is per-hardware-profile. Keeping a
    separately hard-coded imaging dimension lets it silently drift from the real frames
    (it did: 4504 vs the actual 4512 for S5VPJ6415). ROI is the single source of truth, so
    this must run after _apply_hardware_profile and before _assemble_system_view.
    """
    roi = config.get("acquisition", {}).get("roi", {})
    imaging = config.setdefault("imaging", {})
    if roi.get("width") is not None:
        imaging["image_width_px"] = int(roi["width"])
    if roi.get("height") is not None:
        imaging["image_height_px"] = int(roi["height"])


def _link_stage_controller(config: Dict[str, Any]) -> None:
    """Inject the dedicated stage_hardware_profile into acquisition.stage_controller.

    The Zaber XY stage is independent of the ETL-lens hardware_profiles, so its
    settings live in a single top-level `stage_hardware_profile` block. Injecting
    it here keeps `acquisition.stage_controller` as the single read location for
    all consumers (workflows, zaber_move_binary) without duplicating values.
    """
    stage_profile = config.get("stage_hardware_profile")
    if stage_profile is None:
        return
    acquisition = config.setdefault("acquisition", {})
    acquisition["stage_controller"] = copy.deepcopy(stage_profile)


def _apply_hardware_profile(config: Dict[str, Any]) -> None:
    """Deep-merge the active hardware profile into config['acquisition'] in place."""
    profile_name = config.get("active_hardware_profile")
    if profile_name is None:
        return
    profiles = config.get("hardware_profiles", {})
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
    # Order matters: resolve the active hardware profile (ROI) first, derive the imaging
    # dimensions from it, then assemble the system view so it picks up the derived dims.
    _apply_hardware_profile(loaded_config)
    _link_imaging_dimensions_from_roi(loaded_config)
    _assemble_system_view(loaded_config)
    _link_stage_controller(loaded_config)
    _link_acquisition_output_root(loaded_config)
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
    """Assemble the shared segmentation parameter dictionary from the system view.

    Returns thresholding + segmentation tuning + source illumination only.  Stage-4
    egg-detection parameters are intentionally NOT merged here: only stage 4 needs
    them, and it merges ``stage_04_biological_analysis.egg_detection`` itself.

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

    return unified_params
