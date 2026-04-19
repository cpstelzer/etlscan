from __future__ import annotations


def normalize_plate_type(plate_type) -> str:
    """Normalise a caller-supplied plate-type value to a canonical string key.

    Accepts integers, mixed-case strings, and common hyphenated/underscored
    variants (e.g. ``"24-well"``, ``"96_WELL"``).  Returns ``"24"`` or
    ``"96"``.  Raises ``ValueError`` for any unrecognised value.  When
    *plate_type* is ``None``, defaults to ``"24"``.
    """if plate_type is None:
        return "24"
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


def get_plate_settings(acq_conf, default_plate_24, default_plate_96) -> tuple:
    """Resolve the active plate type and its merged coordinate settings.

    Reads the ``plate`` sub-tree from *acq_conf*, determines whether a 24-well
    or 96-well plate is in use, and deep-merges the caller-supplied defaults
    with any per-run overrides found in the config.

    Args:
        acq_conf: The acquisition config dict (may be ``None`` or empty).
        default_plate_24: Fallback coordinate settings for a 24-well plate.
        default_plate_96: Fallback coordinate settings for a 96-well plate.

    Returns:
        ``(plate_type, settings)`` where *plate_type* is ``"24"`` or ``"96"``
        and *settings* is a dict containing at minimum the keys ``start_x``,
        ``start_y``, ``spacing``, ``origin_x``, ``origin_y``.

    Raises:
        ValueError: If required keys are absent after merging.
    """
    plate_cfg = acq_conf.get("plate", {}) if acq_conf else {}
    plate_type = normalize_plate_type(plate_cfg.get("type", "24"))
    definitions = plate_cfg.get("definitions", {})
    defaults = default_plate_24 if plate_type == "24" else default_plate_96
    plate_def = definitions.get(plate_type, {})
    merged = {**defaults, **plate_def}
    required = ["start_x", "start_y", "spacing", "origin_x", "origin_y"]
    missing = [key for key in required if merged.get(key) is None]
    if missing:
        raise ValueError(f"Missing plate settings for {plate_type}-well plate: {', '.join(missing)}")
    return plate_type, merged
