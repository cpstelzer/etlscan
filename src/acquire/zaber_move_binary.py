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
from src.utils.config_loader import load_config_once

CSV_FILENAME = "wellscanner_coords.csv"


def main():
    sc_cfg = load_config_once().get('acquisition', {}).get('stage_controller', {})
    controller = StageController(
        serial_port=sc_cfg.get('serial_port', 'COM4'),
        device_x_number=sc_cfg.get('device_x_number', 1),
        device_y_number=sc_cfg.get('device_y_number', 2),
        device_x_microstep_size=sc_cfg.get('microstep_size_um', 0.15625),
        device_y_microstep_size=sc_cfg.get('microstep_size_um', 0.15625),
        position_tolerance=sc_cfg.get('position_tolerance_microsteps', 10),
    )

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
