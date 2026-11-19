# -*- coding: utf-8 -*-
"""
CLI entry point for moving the Zaber ASR stage through a CSV list of coordinates.

Delegates all hardware interaction to StageController. Run directly:
    python -m src.acquire.zaber_move_binary

Requires wellscanner_coords.csv in the working directory with columns:
    wellname, wellnr, X (mm), Y (mm)
"""

import csv
import sys

from src.acquire.stage_controller import StageController
from src.utils.config_loader import load_config_once, require_config_keys

CSV_FILENAME = "wellscanner_coords.csv"


def main():
    acq_conf = require_config_keys(load_config_once()['acquisition'], ('stage_controller',), 'acquisition')
    controller = StageController.from_config(acq_conf['stage_controller'])

    port, device_x, device_y = controller.initialize()

    last_x, last_y = None, None

    print(f"\nLoading coordinates from {CSV_FILENAME}...")
    try:
        with open(CSV_FILENAME, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    x_coord = float(row["X"])
                    y_coord = float(row["Y"])
                except (KeyError, ValueError) as exc:
                    print(f"Skipping row {row} due to conversion error: {exc}")
                    continue

                print(f"\nProcessing well {row.get('wellname')} (wellnr {row.get('wellnr')})")
                port.flush()
                last_x, last_y = controller.move_to(x_coord, y_coord, last_x, last_y, device_x, device_y, port)
    finally:
        print("\nRe-homing devices...")
        controller.home_device(device_x, "X", port)
        controller.home_device(device_y, "Y", port)
        print("Sequence complete.")


if __name__ == "__main__":
    sys.exit(main())
