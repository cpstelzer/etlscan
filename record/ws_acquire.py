#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Cannonical script for aquiring images from wellplates
using well selection and stage control of the Zaber XY scanning table.

Step 1: Provides a GUI for user-guided selection of wells on a 24-well plate.
        Computes the coordinates based on given dimensions and saves the selected wells to a CSV file.
Step 2: After a user prompt, loads the CSV file and moves the XY linear stage to each coordinate,
        triggering image acquisition at each well.

This version intelligently detects camera sensor type (Mono/Color) by checking its
capabilities, and configures the acquisition pipeline accordingly based on user input
and config settings.

Author: Stelzer Lab
Date: 31 March 2025
"""

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import time
import csv
from math import isclose
import os
import sys
import shutil
from datetime import datetime
import traceback

# Zaber serial imports for stage control
from zaber.serial import BinarySerial, BinaryDevice, BinaryCommand
from zaber.serial.unexpectedreplyerror import UnexpectedReplyError
# Pylon for camera detection
from pypylon import pylon

# Import module for image acquisition
from wellscanner_acquisition_module import run_acquisition, load_config


# ========================
# Configuration Parameters
# ========================

# Plate parameters for coordinate calculation (Azenta 24-well plate defaults)
DEFAULT_PLATE_24 = {
    "start_x": 15.74,   # X coordinate for well A1 (in mm)
    "start_y": 18.88,   # Y coordinate for well A1 (in mm)
    "spacing": 18.0,    # Uniform spacing (in mm)
    "origin_x": 7.89,   # Origin offset X (in mm) Update 10.11.25
    "origin_y": 15.21,  # Origin offset Y (in mm)
}

# Plate parameters for coordinate calculation (ibidi 96-well plate defaults)
DEFAULT_PLATE_96 = {
    "start_x": 11.3,   # X coordinate for well A1 (in mm) - update as needed
    "start_y": 14.5,   # Y coordinate for well A1 (in mm) - update as needed
    "spacing": 9.0,     # Uniform spacing (in mm)
    "origin_x": 17.096,   # Origin offset X (in mm) - update as needed
    "origin_y": 14.981,  # Origin offset Y (in mm) - update as needed
}

CSV_FILENAME = "wellscanner_coords.csv"

# Serial port settings for Zaber stage
SERIAL_PORT = "COM4"
DEVICE_X_NUMBER = 1
DEVICE_Y_NUMBER = 2
DEVICE_X_MICROSTEP_SIZE_MICROMETERS = 0.15625
DEVICE_Y_MICROSTEP_SIZE_MICROMETERS = 0.15625
POSITION_TOLERANCE = 10
ACQUISITION_CONFIG_PATH = "ws_config.yaml" # default: ws_config.yaml

def initialize_experiment_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "date", "time", "plate_id", "well_id", "well_name",
                "start_dpt", "exposure_time_us", "final_acquisition_fps",
                "any_frames_skipped", "acquisition_status", "run_folder"
            ])
        print(f"Initialized experiment summary log: {log_path}")

def append_to_experiment_log(log_path, plate_id, well_id, well_name, start_dpt, acq_results, run_folder):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    try:
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                date_str, time_str, plate_id, well_id, well_name,
                f"{start_dpt:.2f}",
                acq_results.get('actual_exposure_us', 'N/A'),
                f"{acq_results.get('final_fps', 0.0):.2f}",
                acq_results.get('any_frames_skipped', True),
                acq_results.get('status', 'failed'),
                run_folder
            ])
    except Exception as e:
        print(f"Error writing to experiment summary log {log_path}: {e}")

def normalize_plate_type(plate_type):
    if plate_type is None:
        return "24"
    if isinstance(plate_type, int):
        plate_type = str(plate_type)
    plate_type = str(plate_type).strip().lower()
    mapping = {
        "24": "24", "24-well": "24", "24well": "24", "24_well": "24",
        "96": "96", "96-well": "96", "96well": "96", "96_well": "96",
    }
    if plate_type not in mapping:
        raise ValueError(f"Unsupported plate type '{plate_type}'. Use 24 or 96.")
    return mapping[plate_type]

def get_plate_settings(acq_conf):
    plate_cfg = acq_conf.get("plate", {}) if acq_conf else {}
    plate_type = normalize_plate_type(plate_cfg.get("type", "24"))
    definitions = plate_cfg.get("definitions", {})
    defaults = DEFAULT_PLATE_24 if plate_type == "24" else DEFAULT_PLATE_96
    plate_def = definitions.get(plate_type, {})
    merged = {**defaults, **plate_def}
    required = ["start_x", "start_y", "spacing", "origin_x", "origin_y"]
    missing = [key for key in required if merged.get(key) is None]
    if missing:
        raise ValueError(f"Missing plate settings for {plate_type}-well plate: {', '.join(missing)}")
    return plate_type, merged

def calculate_coords_24wellplate(plate_settings):
    rows_labels = ['A', 'B', 'C', 'D']
    cols_labels = range(1, 7)
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
                'X': new_x, 'Y': new_y,
                'negX': new_x == 0, 'negY': new_y == 0
            })
            well_no_counter += 1
    df = pd.DataFrame(wells_data)
    print("Calculated well coordinates:")
    print(df)
    return df

def calculate_coords_96wellplate(plate_settings):
    rows_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols_labels = range(1, 13)
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
                'X': new_x, 'Y': new_y,
                'negX': new_x == 0, 'negY': new_y == 0
            })
            well_no_counter += 1
    df = pd.DataFrame(wells_data)
    print("Calculated well coordinates:")
    print(df)
    return df

def well_selection_gui(plate_type, plate_settings):
    if plate_type == "24":
        df_plate = calculate_coords_24wellplate(plate_settings)
        gui_rows = ['A', 'B', 'C', 'D']
        gui_cols = range(1, 7)
        title = "24 Well Plate Selector"
    else:
        df_plate = calculate_coords_96wellplate(plate_settings)
        gui_rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        gui_cols = range(1, 13)
        title = "96 Well Plate Selector"
    selected_wells = set()
    buttons = {}
    def toggle_selection(well_name_arg):
        if well_name_arg in selected_wells:
            selected_wells.remove(well_name_arg)
            buttons[well_name_arg].config(bg="SystemButtonFace")
        else:
            selected_wells.add(well_name_arg)
            buttons[well_name_arg].config(bg="yellow")
    def save_selection():
        if not selected_wells:
            messagebox.showinfo("No Selection", "Please select at least one well.")
            return
        selected_df = df_plate[df_plate['wellname'].isin(selected_wells)].sort_values(by='wellnr')
        selected_df.to_csv(CSV_FILENAME, index=False)
        messagebox.showinfo("Selected Well Coordinates", selected_df.to_string(index=False))
        root.destroy()
    root = tk.Tk()
    root.title(title)
    for i, r_label in enumerate(gui_rows):
        for j, c_label in enumerate(gui_cols):
            well_name_key = f"{r_label}{c_label}"
            btn = tk.Button(root, text=well_name_key, width=8, height=4,
                            command=lambda wn_cb=well_name_key: toggle_selection(wn_cb))
            btn.grid(row=i, column=j, padx=5, pady=5)
            buttons[well_name_key] = btn
    enter_btn = tk.Button(root, text="Enter", command=save_selection)
    enter_btn.grid(row=len(gui_rows), column=0, columnspan=len(gui_cols), pady=10)
    root.mainloop()

def select_wells(acq_conf):
    print("Starting well selection GUI...")
    try:
        plate_type, plate_settings = get_plate_settings(acq_conf)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False
    well_selection_gui(plate_type, plate_settings)
    print(f"Selected wells (if any) saved to '{CSV_FILENAME}'.")
    return True

def initialize_zaber_stage():
    print(f"Initializing Zaber stage on port {SERIAL_PORT}...")
    port = None
    try:
        port = BinarySerial(SERIAL_PORT, baud=115200, timeout=60)
        print("  Port opened.")
        print("  Renumbering devices...")
        renumber_cmd = BinaryCommand(0, 2)
        try:
            port.write(renumber_cmd)
        except UnexpectedReplyError as e:
            print(f"  Ignored UnexpectedReplyError during renumbering: {e}")
        except Exception as e_renumber:
            print(f"  An error occurred during renumbering: {e_renumber}")
        time.sleep(3)
        port.flush()
        print("  Renumbering attempt finished.")
        device_x = BinaryDevice(port, DEVICE_X_NUMBER)
        print(f"  X-axis device (ID {DEVICE_X_NUMBER}) instantiated.")
        device_y = BinaryDevice(port, DEVICE_Y_NUMBER)
        print(f"  Y-axis device (ID {DEVICE_Y_NUMBER}) instantiated.")
        print("  Performing initial homing sequence...")
        zaber_home_device(device_x, "X", port)
        zaber_home_device(device_y, "Y", port)
        print("Stage initialization and initial homing complete.")
        return port, device_x, device_y
    except Exception as e:
        print(f"ERROR during Zaber stage initialization: {e}")
        if port and hasattr(port, 'is_open') and port.is_open:
            try:
                port.close()
                print("  Port closed due to initialization error.")
            except Exception as e_close:
                print(f"  Error closing port during initialization cleanup: {e_close}")
        raise

def zaber_home_device(device, device_name_str, port_obj):
    if not device:
        print(f"Device {device_name_str} is None, skipping homing.")
        return
    if not port_obj:
        print(f"Port object for {device_name_str} is None, skipping homing.")
        return
    print(f"  Homing Zaber device {device_name_str} (Device {device.number})...")
    try:
        device.home()
        print(f"    Homing command sent to device {device_name_str}.")
    except UnexpectedReplyError as e:
        print(f"    Ignored UnexpectedReplyError during homing of {device_name_str}: {e}")
    except Exception as e_home:
        print(f"    An error occurred during homing of {device_name_str}: {e_home}")
    time.sleep(2.5)
    port_obj.flush()
    print(f"  Finished homing attempt for Zaber device {device_name_str}.")

def zaber_move_axis_if_needed(device, target_steps, last_value, axis_name, port_obj):
    if last_value is None or not isclose(last_value, target_steps, abs_tol=POSITION_TOLERANCE):
        print(f"  Moving Zaber {axis_name}-axis (Device {device.number}) to {target_steps} microsteps...")
        port_obj.flush()
        try:
            device.move_abs(target_steps)
        except UnexpectedReplyError as e:
            print(f"    Ignored UnexpectedReplyError while moving {axis_name}-axis: {e}")
        except Exception as e_move:
            print(f"    An error occurred while moving {axis_name}-axis: {e_move}")
        time.sleep(1.5)
        port_obj.flush()
        print(f"  Zaber {axis_name}-axis move command sent. Assumed at {target_steps} microsteps.")
        return target_steps
    else:
        print(f"  Zaber {axis_name}-axis already at {target_steps} microsteps (within tolerance); skipping move.")
        port_obj.flush()
        return last_value

def zaber_move_to(x_mm, y_mm, last_x_steps, last_y_steps, dev_x, dev_y, port_obj):
    x_target_s = int(x_mm * 1000 / DEVICE_X_MICROSTEP_SIZE_MICROMETERS)
    y_target_s = int(y_mm * 1000 / DEVICE_Y_MICROSTEP_SIZE_MICROMETERS)
    print(f"\nMoving Zaber stage to X = {x_mm:.2f} mm ({x_target_s} steps), Y = {y_mm:.2f} mm ({y_target_s} steps)")
    
    new_lx = zaber_move_axis_if_needed(dev_x, x_target_s, last_x_steps, "X", port_obj)
    time.sleep(0.2)
    new_ly = zaber_move_axis_if_needed(dev_y, y_target_s, last_y_steps, "Y", port_obj)
    
    print("  Waiting for Zaber stage to settle at position...")
    time.sleep(2)
    port_obj.flush()
    return new_lx, new_ly

def check_camera_is_color():
    """
    Checks if the first Pylon camera is a color camera by querying its
    list of available pixel formats. This is more reliable than checking
    the current format, which might be left over from a previous session.
    """
    camera = None
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            print("CAMERA CHECK: No Pylon camera found.")
            return False
            
        camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        camera.Open()
        
        # --- CORRECTED LOGIC: CHECK CAPABILITIES, NOT CURRENT STATE ---
        available_formats = camera.PixelFormat.Symbolics
        is_color = any(fmt.startswith("Bayer") for fmt in available_formats)
        
        print(f"CAMERA CHECK: Found '{camera.GetDeviceInfo().GetModelName()}'. Current format: '{camera.PixelFormat.GetValue()}'.")
        print(f"CAMERA CHECK: Sensor is {'COLOR (Bayer capable)' if is_color else 'MONOCHROME'}.")
        
        camera.Close()
        return is_color
        
    except Exception as e:
        print(f"CAMERA CHECK: Error while checking camera sensor type: {e}")
        if camera and camera.IsOpen():
            camera.Close()
        return False

def main():
    print("\nLoading acquisition configuration...")
    acq_conf = None
    try:
        acq_conf = load_config(ACQUISITION_CONFIG_PATH)
        acq_conf['loaded_from_path'] = os.path.abspath(ACQUISITION_CONFIG_PATH)
    except SystemExit:
        return
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}. Exiting.")
        return

    print("Step 1: Well Selection")
    if not select_wells(acq_conf):
        return
    if not os.path.exists(CSV_FILENAME):
        print(f"Well selection CSV ('{CSV_FILENAME}') not found. Exiting.")
        return
    try:
        with open(CSV_FILENAME, 'r') as f:
            if len(f.readlines()) <= 1:
                print(f"'{CSV_FILENAME}' is empty or contains no well data. Exiting.")
                if os.path.exists(CSV_FILENAME): os.remove(CSV_FILENAME)
                return
    except Exception as e:
        print(f"Error reading '{CSV_FILENAME}': {e}. Exiting.")
        return

    print("\nStep 2: Determining Acquisition Mode")
    process_images_in_color = False
    is_color_camera = check_camera_is_color()
    color_mode_config = acq_conf.get('camera', {}).get('color_mode', 'Ask').capitalize()

    if color_mode_config == 'Color':
        if not is_color_camera:
            print("WARNING: Config requests 'Color' mode, but a monochrome camera was detected. Forcing monochrome.")
        else:
            print("Configured for 'Color' mode.")
            process_images_in_color = True
    elif color_mode_config == 'Mono':
        print("Configured for 'Mono' mode.")
        process_images_in_color = False
    else: # 'Ask' mode
        if is_color_camera:
            while True:
                response = input("Color camera detected. Save images in color? (y/n): ").lower().strip()
                if response == 'y':
                    process_images_in_color = True
                    break
                elif response == 'n':
                    process_images_in_color = False
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        else:
            print("Monochrome camera detected. Proceeding with monochrome acquisition.")
            process_images_in_color = False
    
    print(f"FINAL DECISION: Image acquisition will be processed in {'COLOR' if process_images_in_color else 'MONOCHROME'}.")

    summary_log_fn = acq_conf.get('paths', {}).get('experiment_summary_log_filename', 'experiment_summary_log.csv')
    base_out_dir = acq_conf.get('paths', {}).get('output_folder_base', '.')
    os.makedirs(base_out_dir, exist_ok=True)
    exp_summary_log_pth = os.path.join(base_out_dir, summary_log_fn)
    initialize_experiment_log(exp_summary_log_pth)

    while True:
        try: plate_num = int(input("Enter the plate number (integer): "))
        except ValueError: print("Invalid input. Plate number must be an integer.")
        else: break
    while True:
        try:
            start_dpt_val = round(float(input("Enter dioptries at 0.0mm (-2.00 to +3.00): ")), 2)
            if not (-2.00 <= start_dpt_val <= 3.00): raise ValueError("Out of range.")
        except ValueError as e: print(f"Invalid input: {e}. Range -2.00 to 3.00.")
        else: break

    print(f"Accepted Plate No: {plate_num}, Start DPT: {start_dpt_val:.2f}\n---------------------------\n")

    # --- Pre-run check for existing directories ---
    print("Step 3: Checking for existing output directories...")
    try:
        with open(CSV_FILENAME, newline='') as csv_f:
            all_wells_data_for_check = list(csv.DictReader(csv_f))
        
        date_str = datetime.now().strftime("%Y%m%d")
        
        existing_dirs = []
        for well_row in all_wells_data_for_check:
            try:
                well_num = int(well_row['wellnr'])
                dir_to_check = os.path.join(base_out_dir, f"{date_str}_plate{plate_num}_well{well_num}")
                if os.path.isdir(dir_to_check):
                    existing_dirs.append(dir_to_check)
            except (ValueError, KeyError):
                continue
                
        if existing_dirs:
            print("\n--- WARNING: Existing Data Found ---")
            print("The following output directories for today's date and plate number already exist:")
            for d in existing_dirs:
                print(f"  - {d}")
            print("\nChoose an option:")
            print("  1) PROCEED: Delete the contents of these directories and start the new acquisition.")
            print("  2) EXIT: Stop the program without making any changes.")
            
            while True:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice == '1':
                    print("\nProceeding. Deleting contents of existing directories...")
                    try:
                        for d in existing_dirs:
                            shutil.rmtree(d)
                            print(f"  - Cleared: {d}")
                        print("All targeted directories have been cleared.")
                    except Exception as e:
                        print(f"\nFATAL ERROR: Could not delete directories: {e}")
                        print("Please check file permissions and close any programs that might be using these folders.")
                        sys.exit(1)
                    break
                elif choice == '2':
                    print("Exiting program as requested. No measurements will be performed.")
                    sys.exit(0)
                else:
                    print("Invalid input. Please enter 1 or 2.")
        else:
            print("No existing directories found for this plate and date. Proceeding.")
    
    except FileNotFoundError:
        print(f"ERROR: Cannot perform directory check because '{CSV_FILENAME}' was not found.")
        sys.exit(1)


    input("Ensure:\n1) Plate orientation (A1 lower right)\n2) Lid\n3) Light source\n4) Devices connected\n(Enter to start)\n")
 
    zab_port, zab_device_x, zab_device_y = None, None, None
    initialization_successful = False
    
    try:
        print("\nStep 4: Stage Initialization and Main Processing")
        zab_port, zab_device_x, zab_device_y = initialize_zaber_stage()
        initialization_successful = True
        
        current_x_pos, current_y_pos = None, None
        
        print(f"\nLoading coordinates from '{CSV_FILENAME}'...")
        with open(CSV_FILENAME, newline='') as csv_f:
            well_reader = csv.DictReader(csv_f)
            all_wells_data = list(well_reader)
            
            for idx, well_row in enumerate(all_wells_data):
                wellname = well_row.get('wellname', 'N/A')
                print(f"\n--- Processing well {idx+1}/{len(all_wells_data)}: {wellname} (WellNr: {well_row.get('wellnr', 'N/A')}) ---")
                try:
                    x_mm, y_mm = float(well_row['X']), float(well_row['Y'])
                    well_num = int(well_row['wellnr'])
                except (ValueError, KeyError) as err:
                    print(f"Skipping row {well_row} due to error: {err}")
                    run_fldr = f"plate{plate_num}_well{well_row.get('wellnr', 'err')}_data_error"
                    append_to_experiment_log(exp_summary_log_pth, plate_num, well_row.get('wellnr', -1), wellname, start_dpt_val,
                                             {'status': 'data_error', 'any_frames_skipped': True}, run_fldr)
                    continue

                if zab_port: zab_port.flush()
                current_x_pos, current_y_pos = zaber_move_to(x_mm, y_mm, current_x_pos, current_y_pos, zab_device_x, zab_device_y, zab_port)
                
                print(f"Starting acquisition for Plate {plate_num}, Well {well_num}...")
                date_fldr_str = datetime.now().strftime("%Y%m%d")
                run_fldr_name = os.path.join(acq_conf['paths']['output_folder_base'],
                                           f"{date_fldr_str}_plate{plate_num}_well{well_num}")
                
                well_acq_results = run_acquisition(plate_num, well_num, start_dpt_val, acq_conf, process_as_color=process_images_in_color)
                
                append_to_experiment_log(exp_summary_log_pth, plate_num, well_num, wellname,
                                         start_dpt_val, well_acq_results, run_fldr_name)
                if well_acq_results.get('status') == 'success':
                    print(f"Acquisition for {wellname} successful. Results: {well_acq_results}")
                else:
                    print(f"Acquisition for {wellname} FAILED. Results: {well_acq_results}")
                
                if idx < len(all_wells_data) - 1:
                    print("Waiting before next well...")
                    time.sleep(5)
            
            print("\nAll wells processed.")

    except FileNotFoundError: 
        print(f"ERROR: CSV file '{CSV_FILENAME}' not found.")
    except UnexpectedReplyError as zab_err_main: 
        print(f"Zaber Stage Communication Error during main processing: {zab_err_main}.")
    except Exception as gen_err_main:
        print(f"An error occurred: {gen_err_main}")
        traceback.print_exc()
    
    if initialization_successful:
        print("\n--- Attempting Final Homing Sequence ---")
        if zab_port and hasattr(zab_port, 'is_open') and zab_port.is_open:
            zaber_home_device(zab_device_x, "X", zab_port)
            zaber_home_device(zab_device_y, "Y", zab_port)
            print("Final homing attempts finished.")
        elif zab_port :
             print("Final Homing: Zaber port was initialized but is not open. Cannot home.")
        else:
            print("Final Homing: Zaber port was not properly initialized. Cannot home.")

    print("\n--- Entering Final Resource Cleanup ---")
    if zab_port:
        print(f"Zaber port object exists. Checking if open for closure...")
        try:
            if hasattr(zab_port, 'is_open'):
                if zab_port.is_open:
                    print("Closing Zaber serial port.")
                    zab_port.close()
                    print("Zaber port closed successfully.")
                else:
                    print("Zaber port was found but is already closed.")
            else:
                print("Zaber port object does not have 'is_open' property, cannot determine state for closure.")
        except Exception as e_close_check:
            print(f"Error during port state check or closure: {e_close_check}")
    else:
        print("Zaber port was not initialized (is None). No port to close.")
            
    print("\nWell scanning sequence complete.")
       
if __name__ == "__main__":
    main()
