from __future__ import annotations

import pandas as pd


def _calculate(rows_labels, cols_labels, plate_settings):
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


def calculate_coords_24wellplate(plate_settings):
    return _calculate(['A', 'B', 'C', 'D'], range(1, 7), plate_settings)


def calculate_coords_96wellplate(plate_settings):
    return _calculate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], range(1, 13), plate_settings)
