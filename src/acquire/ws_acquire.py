#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim for multi-well acquisition workflow."""

from src.acquire.workflows import run_multiwell_workflow

DEFAULT_PLATE_24 = {
    "start_x": 15.74,
    "start_y": 18.88,
    "spacing": 18.0,
    "origin_x": 7.89,
    "origin_y": 15.21,
}

DEFAULT_PLATE_96 = {
    "start_x": 11.3,
    "start_y": 14.5,
    "spacing": 9.0,
    "origin_x": 17.096,
    "origin_y": 14.981,
}

CSV_FILENAME = "wellscanner_coords.csv"
SERIAL_PORT = "COM4"
DEVICE_X_NUMBER = 1
DEVICE_Y_NUMBER = 2
DEVICE_X_MICROSTEP_SIZE_MICROMETERS = 0.15625
DEVICE_Y_MICROSTEP_SIZE_MICROMETERS = 0.15625
POSITION_TOLERANCE = 10
ACQUISITION_CONFIG_PATH = "ws_config.yaml"


def main():
    run_multiwell_workflow(
        acquisition_config_path=ACQUISITION_CONFIG_PATH,
        csv_filename=CSV_FILENAME,
        default_plate_24=DEFAULT_PLATE_24,
        default_plate_96=DEFAULT_PLATE_96,
        serial_port=SERIAL_PORT,
        device_x_number=DEVICE_X_NUMBER,
        device_y_number=DEVICE_Y_NUMBER,
        device_x_microstep_size=DEVICE_X_MICROSTEP_SIZE_MICROMETERS,
        device_y_microstep_size=DEVICE_Y_MICROSTEP_SIZE_MICROMETERS,
        position_tolerance=POSITION_TOLERANCE,
    )


if __name__ == "__main__":
    main()
