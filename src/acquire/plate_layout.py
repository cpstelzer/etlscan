"""Well-plate coordinate generation for 24-well and 96-well formats.

Given a plate settings dict (produced by :func:`src.acquire.config.get_plate_settings`),
functions in this module compute the physical XY stage coordinates (in mm) for
every well and return them as a ``pandas.DataFrame`` with columns:

- ``wellname``  – row-column label, e.g. ``"A1"``, ``"B3"``
- ``wellnr``    – sequential 1-based well index
- ``X``, ``Y``  – stage coordinates in mm (origin-adjusted, clamped to ≥ 0)
- ``negX``, ``negY`` – flags set when a coordinate was clamped to zero
"""
from __future__ import annotations

import pandas as pd


def _calculate(rows_labels, cols_labels, plate_settings) -> pd.DataFrame:
    """Build a well-coordinate DataFrame for any rectangular plate layout.

    Args:
        rows_labels: Sequence of row identifiers (e.g. ``['A', 'B', 'C', 'D']``).
        cols_labels: Sequence of column identifiers (e.g. ``range(1, 7)``).
        plate_settings: Dict with keys ``start_x``, ``start_y``, ``spacing``,
            ``origin_x``, ``origin_y`` (all in mm).

    Returns:
        DataFrame with one row per well and columns
        ``wellname``, ``wellnr``, ``X``, ``Y``, ``negX``, ``negY``.
    """
    wells_data = []
    well_no_counter = 1
    start_x = plate_settings["start_x"]
    start_y = plate_settings["start_y"]
    spacing = plate_settings["spacing"]
    origin_x = plate_settings["origin_x"]
    origin_y = plate_settings["origin_y"]
    for i, _ in enumerate(rows_labels):
        computed_x = start_x + i * spacing
        for j, _ in enumerate(cols_labels):
            computed_y = start_y + j * spacing
            new_x = max(0, computed_x - origin_x)
            new_y = max(0, computed_y - origin_y)
            wells_data.append({
                'wellname': f"{rows_labels[i]}{cols_labels[j]}",
                'wellnr': well_no_counter,
                'X': new_x,
                'Y': new_y,
                'negX': new_x == 0,
                'negY': new_y == 0,
            })
            well_no_counter += 1
    df = pd.DataFrame(wells_data)
    print("Calculated well coordinates:")
    print(df)
    return df


def calculate_coords_24wellplate(plate_settings) -> pd.DataFrame:
    """Return stage coordinates for all 24 wells (rows A–D, columns 1–6).

    Args:
        plate_settings: Dict produced by
            :func:`src.acquire.config.get_plate_settings`.

    Returns:
        DataFrame with 24 rows; see :func:`_calculate` for column details.
    """
    return _calculate(['A', 'B', 'C', 'D'], range(1, 7), plate_settings)


def calculate_coords_96wellplate(plate_settings) -> pd.DataFrame:
    """Return stage coordinates for all 96 wells (rows A–H, columns 1–12).

    Args:
        plate_settings: Dict produced by
            :func:`src.acquire.config.get_plate_settings`.

    Returns:
        DataFrame with 96 rows; see :func:`_calculate` for column details.
    """
    return _calculate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], range(1, 13), plate_settings)
