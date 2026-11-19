# src/core/track_columns.py
"""Column names of the per-frame track tables that Stage 1 writes and Stage 2 extends.

Shared by focus-peak detection, depth reconstruction and the track plots.
"""

FOCUS_Z_COLUMN = 'z_mm'
FOCUS_Z_RECONSTRUCTED_COLUMN = 'z_reconstructed'
FOCUS_X_PIXEL_COLUMN = 'X'
FOCUS_Y_PIXEL_COLUMN = 'Y'
FOCUS_X_MM_COLUMN = 'x_mm'
FOCUS_Y_MM_COLUMN = 'y_mm'
