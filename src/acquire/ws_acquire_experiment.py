#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim for experiment-sweep acquisition workflow."""

from src.acquire.workflows import run_experiment_sweep_workflow

DEFAULT_PLATE_24 = {
    "start_x": 15.74,
    "start_y": 18.88,
    "spacing": 18.0,
    "origin_x": 6.25,
    "origin_y": 11.78,
}

DEFAULT_PLATE_96 = {
    "start_x": 15.74,
    "start_y": 18.88,
    "spacing": 9.0,
    "origin_x": 6.25,
    "origin_y": 11.78,
}

CSV_FILENAME = "wellscanner_coords.csv"
SERIAL_PORT = "COM7"
DEVICE_X_NUMBER = 1
DEVICE_Y_NUMBER = 2
DEVICE_X_MICROSTEP_SIZE_MICROMETERS = 0.15625
DEVICE_Y_MICROSTEP_SIZE_MICROMETERS = 0.15625
POSITION_TOLERANCE = 10
ACQUISITION_CONFIG_PATH = "ws_config.yaml"
SCAN_STEPS_UM_OPTIONS = [100, 200, 300, 400, 500, 600, 700]
FOCUSING_PATTERN_OPTIONS = ['zigzag', 'sawtooth']


def main():
    run_experiment_sweep_workflow(
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
        scan_steps_um_options=SCAN_STEPS_UM_OPTIONS,
        focusing_pattern_options=FOCUSING_PATTERN_OPTIONS,
    )


if __name__ == "__main__":
    main()
