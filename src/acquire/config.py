from __future__ import annotations

from src.utils.config_loader import require_config_sections


def normalize_plate_type(plate_type) -> str:
    """Normalise a caller-supplied plate-type value to a canonical string key.

    Accepts integers, mixed-case strings, and common hyphenated/underscored
    variants (e.g. ``"24-well"``, ``"96_WELL"``).  Returns ``"24"`` or
    ``"96"``.  Raises ``ValueError`` for any unrecognised value.
    """
    if isinstance(plate_type, int):
        plate_type = str(plate_type)
    plate_type = str(plate_type).strip().lower()
    mapping = {
        "24": "24", "24-well": "24", "24well": "24", "24_well": "24",
        "96": "96", "96-well": "96", "96well": "96", "96_well": "96",
    }
    if plate_type not in mapping:
        raise ValueError(f"Unsupported plate type '{plate_type}'. Use 24 or 96.")
    return mapping[plate_type]


def get_plate_settings(acq_conf) -> tuple:
    """Resolve the active plate type and its coordinate settings.

    Reads the ``plate`` sub-tree from *acq_conf*: the run-level ``type`` selection
    and the (hardware-profile-merged) ``definitions`` for that plate.

    Args:
        acq_conf: The acquisition config dict.

    Returns:
        ``(plate_type, settings)`` where *plate_type* is ``"24"`` or ``"96"``
        and *settings* is a dict containing the keys ``start_x``, ``start_y``,
        ``spacing``, ``origin_x``, ``origin_y``.

    Raises:
        ConfigError: ``acquisition.plate.type`` or ``acquisition.plate.definitions`` is missing.
        ValueError: If required keys are absent for the selected plate (e.g. the active
            hardware profile defines no origin for it).
    """
    require_config_sections(acq_conf, {'plate': ('type', 'definitions')}, 'acquisition')
    plate_cfg = acq_conf['plate']
    plate_type = normalize_plate_type(plate_cfg['type'])
    merged = dict(plate_cfg['definitions'].get(plate_type, {}))
    required = ["start_x", "start_y", "spacing", "origin_x", "origin_y"]
    missing = [key for key in required if merged.get(key) is None]
    if missing:
        raise ValueError(f"Missing plate settings for {plate_type}-well plate: {', '.join(missing)}")
    return plate_type, merged
