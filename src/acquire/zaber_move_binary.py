# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:58:17 2025

This script moves the Zaber ASR stage via commands of the zaber.serial package.

This is a preliminary version for testing purposes only. It has been verified
to work on 31.03.2025 in the 'wellscan24_env' environment.

This program requires a csv-file (coords_greiner24well.csv) as an input, which
stores the coordinates the table is supposed to move to

@author: Stelzer Lab
"""

import time
import csv
from zaber.serial import BinarySerial, BinaryDevice, BinaryCommand
from zaber.serial.unexpectedreplyerror import UnexpectedReplyError
from math import isclose

# Serial port settings
SERIAL_PORT = "COM7"
# After renumbering, assume device_x becomes 1 and device_y becomes 2.
DEVICE_X_NUMBER = 1
DEVICE_Y_NUMBER = 2

# Conversion factors (in micrometers)
DEVICE_X_MICROSTEP_SIZE_MICROMETERS = 0.15625
DEVICE_Y_MICROSTEP_SIZE_MICROMETERS = 0.15625

# Tolerance (in microsteps) for considering an axis "at target"
POSITION_TOLERANCE = 10

# Open the serial port with an extended timeout.
port = BinarySerial(SERIAL_PORT, baud=115200, timeout=60)

# --- Renumber devices ---
print("Renumbering devices...")
renumber_cmd = BinaryCommand(0, 2)
try:
    port.write(renumber_cmd)
except UnexpectedReplyError as e:
    print("Ignored UnexpectedReplyError during renumbering:", e)
time.sleep(3)   # Allow renumbering to complete.
port.flush()

# Instantiate devices with their new unique addresses.
device_x = BinaryDevice(port, DEVICE_X_NUMBER)
device_y = BinaryDevice(port, DEVICE_Y_NUMBER)

# --- Helper: Homing a device with delay and flush ---
def home_device(device, device_name):
    print(f"Homing device {device_name}...")
    try:
        device.home()
    except UnexpectedReplyError as e:
        print(f"Ignored UnexpectedReplyError during homing of device {device_name}: {e}")
    time.sleep(2)  # Wait to ensure the home command has completed.
    port.flush()

print("Initial homing sequence:")
home_device(device_x, "X")
home_device(device_y, "Y")

# We'll track the last commanded positions (in microsteps). Initialize to None.
last_x = None
last_y = None

# --- Helper: Move an axis only if needed, ignoring unexpected reply errors ---
def move_axis_if_needed(device, target_steps, last_value, axis_name):
    print(f"Axis {axis_name}: last_value = {last_value}, target_steps = {target_steps}")
    if last_value is None or not isclose(last_value, target_steps, abs_tol=POSITION_TOLERANCE):
        print(f"Moving {axis_name}-axis to {target_steps} steps...")
        port.flush()
        try:
            device.move_abs(target_steps)
        except UnexpectedReplyError as e:
            print(f"Ignored UnexpectedReplyError while moving {axis_name}-axis: {e}")
        time.sleep(1)  # Allow time for the move to complete.
        port.flush()
        # Assume the target is reached.
        new_val = target_steps
        print(f"{axis_name}-axis assumed at {new_val} steps")
        return new_val
    else:
        print(f"{axis_name}-axis already at {target_steps} steps; skipping move and flushing port.")
        port.flush()
        return last_value

# --- Helper: Move both axes to target coordinates (in mm) ---
def move_to(x_mm, y_mm, last_x, last_y):
    # Convert mm to microsteps.
    x_target = int(x_mm * 1000 / DEVICE_X_MICROSTEP_SIZE_MICROMETERS)
    y_target = int(y_mm * 1000 / DEVICE_Y_MICROSTEP_SIZE_MICROMETERS)
    print(f"\nMoving to X = {x_mm} mm ({x_target} steps), Y = {y_mm} mm ({y_target} steps)")
    new_last_x = move_axis_if_needed(device_x, x_target, last_x, "X")
    time.sleep(0.5)
    new_last_y = move_axis_if_needed(device_y, y_target, last_y, "Y")
    time.sleep(2)
    port.flush()
    return new_last_x, new_last_y

# --- Process CSV file and move to each coordinate ---
CSV_FILENAME = "wellscanner_coords.csv"
print(f"\nLoading coordinates from {CSV_FILENAME}...")
with open(CSV_FILENAME, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            x_coord = float(row['X'])
            y_coord = float(row['Y'])
        except ValueError as ve:
            print(f"Skipping row {row} due to conversion error: {ve}")
            continue

        print(f"\nProcessing well {row['wellname']} (wellnr {row['wellnr']})")
        port.flush()
        last_x, last_y = move_to(x_coord, y_coord, last_x, last_y)

# --- Final re-homing sequence ---
print("\nRe-homing devices...")
home_device(device_x, "X")
home_device(device_y, "Y")
print("Test sequence complete.")
