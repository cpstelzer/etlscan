# src/utils/constants.py
"""
Project-wide named constants.

All magic numbers and hardcoded string literals live here so that
every tuneable value has one authoritative location.  Import only
what you need; group imports by category for clarity.
"""

# ---------------------------------------------------------------------------
# Flat-field calibration
# ---------------------------------------------------------------------------
FLAT_FIELD_SUBDIR = "1_flat_field"
FLAT_FIELD_FLAT_FILENAME = "flat.png"
FLAT_FIELD_DARK_FILENAME = "dark.png"

# ---------------------------------------------------------------------------
# Well masking
# ---------------------------------------------------------------------------
# Fraction of config radius_px used as the final mask radius.
# Matches the original wellcounter_imaging_module default.
WELL_MASK_DEFAULT_RADIUS_FRACTION = 0.67

# ---------------------------------------------------------------------------
# Well brightness peak calculation
# ---------------------------------------------------------------------------
# Minimum number of pixels inside the well mask required for a reliable
# brightness estimate.
BRIGHTNESS_MIN_WELL_PIXELS = 100

# Default lower-bound intensity used when filtering well pixels before
# computing the brightness peak proxy.
BRIGHTNESS_MIN_INTENSITY_DEFAULT = 75

# Percentile used to select the bright pixel population for the peak proxy.
BRIGHTNESS_PERCENTILE_DEFAULT = 75

# ---------------------------------------------------------------------------
# LBP (Local Binary Pattern) texture descriptor defaults
# ---------------------------------------------------------------------------
LBP_DEFAULT_RADII = [1, 3]
LBP_DEFAULT_POINTS = 8

# ---------------------------------------------------------------------------
# Morphological operation kernel sizes
# ---------------------------------------------------------------------------
# Small kernel for noise removal / fine detail.
MORPH_KERNEL_SIZE_SMALL = 3

# Large kernel for well-boundary smoothing (open + close in well_analysis).
MORPH_KERNEL_SIZE_LARGE = 9

# ---------------------------------------------------------------------------
# Zaber ASR stage hardware defaults
# (overridden at runtime via config or explicit constructor arguments)
# ---------------------------------------------------------------------------
ZABER_SERIAL_PORT = "COM7"
ZABER_DEVICE_X_NUMBER = 1
ZABER_DEVICE_Y_NUMBER = 2

# Microstep size in micrometres for both axes.
ZABER_MICROSTEP_SIZE_UM = 0.15625

# Acceptable deviation (in microsteps) when deciding whether an axis move
# is necessary.
ZABER_POSITION_TOLERANCE_MICROSTEPS = 10
