def build_crop_filename(
    prefix: str,
    particle_label: str,
    frame_num: int,
    x: int,
    y: int,
    *,
    ext: str = "png",
    frame_padding: int = 0,
) -> str:
    """Build a deterministic filename for a particle crop image."""
    frame_str = str(frame_num) if frame_padding == 0 else f"{frame_num:0{frame_padding}d}"
    return f"{prefix}_{particle_label}_frame{frame_str}_X{x}_Y{y}.{ext}"
