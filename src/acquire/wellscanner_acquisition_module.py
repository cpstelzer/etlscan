# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:11:54 2025 - Revised [Current Date]

Acquisition Module (New Version)

This module provides the `run_acquisition` function to perform image acquisition
using a liquid lens and camera system with configurable ROI and settings loaded
from an external YAML file.

It connects to devices, sets ROI, configures camera (adjusting FPS if needed),
calculates focus pattern, runs the acquisition loop (logging detailed metadata
to a CSV file), and handles cleanup.

The `run_acquisition` function now returns a dictionary with key acquisition
results (final_fps, actual_exposure_us, any_frames_skipped) for summary logging.

Revision includes logic to handle both monochrome and color (Bayer) sensors.
Image processing is dynamically adjusted for debayering if color acquisition is enabled,
ensuring minimal performance impact. It now intelligently selects the output file
format: PNG for smaller, lossless monochrome files and the user-configured format
(e.g., BMP) for high-speed color capture.
"""

import sys
import time
import os
import traceback
import csv
import yaml # For loading config
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import math # For FPS rounding

import cv2
import numpy as np
from pypylon import pylon
import optoICC

from .roi import reset_roi_to_full, set_centered_roi

# --- Global Mappings ---
# Maps Pylon Bayer formats to OpenCV debayering codes
BAYER_TO_CV2_CODE = {
    'BayerBG8': cv2.COLOR_BAYER_BG2BGR,
    'BayerRG8': cv2.COLOR_BAYER_RG2BGR,
    'BayerGB8': cv2.COLOR_BAYER_GB2BGR,
    'BayerGR8': cv2.COLOR_BAYER_GR2BGR,
    # Add 16-bit formats if needed, though 8-bit is faster
    'BayerBG12': cv2.COLOR_BAYER_BG2BGR,
    'BayerRG12': cv2.COLOR_BAYER_RG2BGR,
    'BayerGB12': cv2.COLOR_BAYER_GB2BGR,
    'BayerGR12': cv2.COLOR_BAYER_GR2BGR,
}

# --- Configuration Loading ---
DEFAULT_CONFIG_PATH = "config/config.yml"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load the unified config and return the acquisition subtree (hardware profile merged).

    Restructures `plate_definitions` into `plate.definitions` so `get_plate_settings`
    keeps working on the returned dict, and resolves the cv2/pylon string constants.
    """
    from src.utils.config_loader import load_config_once

    print(f"Loading configuration from: {config_path}")
    try:
        unified_config = load_config_once(config_path)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading configuration: {e}")
        sys.exit(1)

    acq = unified_config["acquisition"]

    plate_cfg = acq.get("plate", {})
    plate_definitions = acq.get("plate_definitions", {})
    plate_cfg = {**plate_cfg, "definitions": plate_definitions}
    acq["plate"] = plate_cfg

    watermark_cfg = acq.get("watermark")
    if watermark_cfg and "font" in watermark_cfg:
        watermark_cfg["font"] = getattr(cv2, watermark_cfg["font"], cv2.FONT_HERSHEY_SIMPLEX)
    if watermark_cfg and "color_grayscale" in watermark_cfg and "color_bgr" not in watermark_cfg:
        c = watermark_cfg["color_grayscale"]
        watermark_cfg["color_bgr"] = [c, c, c]

    camera_cfg = acq.get("camera")
    if camera_cfg and "grab_strategy" in camera_cfg:
        camera_cfg["grab_strategy"] = getattr(
            pylon, f"GrabStrategy_{camera_cfg['grab_strategy']}", pylon.GrabStrategy_OneByOne
        )

    return acq

# --- Acquisition Function ---
def acquire_and_save_frames(camera, lens, full_focus_pattern,
                            actual_frame_width, actual_frame_height, watermark_base_x, watermark_first_line_y,
                            run_folder_path, log_file_path, current_date_str,
                            PLATE_ID, WELL_ID, START_DPT, config,
                            final_fps, actual_exposure_us,
                            process_as_color, debayer_code):
    print(f"\n--- Starting Frame Acquisition & Saving ---")
    print(f"Outputting to folder: {run_folder_path}")
    print(f"Logging metadata to: {log_file_path}")
    
    if process_as_color:
        output_image_format = config['output']['image_format']
        print(f"Color Mode: Saving frames as '.{output_image_format}' (from config for max speed).")
    else:
        output_image_format = 'png'
        print(f"Monochrome Mode: Saving frames as '.{output_image_format}' (lossless, smaller files).")

    print(f"Acquiring with frame dimensions: {actual_frame_width}x{actual_frame_height}")
    print(f"Processing as color: {process_as_color}")
    if process_as_color: print(f"Using debayer code: {debayer_code}")
    print(f"Using final acquisition FPS: {final_fps:.2f}")
    print(f"Using actual exposure time: {actual_exposure_us} us")

    lens_settle_time_sec = config['focusing']['lens_settle_time_sec']
    large_jump_settle_sec = config['focusing'].get('large_jump_settle_sec', 0.040)
    large_jump_threshold_dpt = config['focusing'].get('large_jump_threshold_dpt', -3.0)
    scan_ratio = config['focusing']['scan_ratio']
    watermark_font = config['watermark']['font']
    watermark_font_scale = config['watermark']['font_scale']
    watermark_color_bgr = tuple(config['watermark']['color_bgr'])
    watermark_color_mono = (watermark_color_bgr[0],)
    watermark_thickness = config['watermark']['thickness']
    grab_strategy = config['camera']['grab_strategy']
    retrieve_timeout = int(1000 / final_fps + config['camera']['retrieve_timeout_margin_ms']) if final_fps > 0 else 2000

    total_frames_to_record = len(full_focus_pattern)
    if total_frames_to_record <= 0:
        print("Warning: Focus pattern is empty. No frames will be acquired.")
        return False

    print(f"Attempting to acquire {total_frames_to_record} frames based on pre-calculated focus pattern...")
    log_fieldnames = [
        "timestamp", "frame_count", "pylon_frame_id", "skipped_frame_warning",
        "diopter", "focusing_distance_mm", "lens_set_time_ms", "lens_settle_wait_ms",
        "frame_grab_time_ms", "image_processing_time_ms", "image_save_submit_time_ms",
        "total_loop_iteration_time_ms", "output_filename"
    ]
    log_data = []
    any_frames_skipped_in_well = False

    camera.StartGrabbing(grab_strategy)
    print(f"Camera grabbing started with strategy: {grab_strategy.name if hasattr(grab_strategy, 'name') else grab_strategy}")

    last_pylon_frame_id = None
    frames_processed_count = 0
    start_acquisition_time = time.perf_counter()
    previous_diopter = START_DPT 

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {}
        for frame_count in range(total_frames_to_record):
            loop_iter_start_time = time.perf_counter()
            log_entry = {}
            try:
                current_diopter = full_focus_pattern[frame_count]
                focusing_distance_mm = (current_diopter - START_DPT) * scan_ratio
                log_entry["diopter"] = f"{current_diopter:.3f}"
                log_entry["focusing_distance_mm"] = f"{focusing_distance_mm:.2f}"
                
                diopter_change = current_diopter - previous_diopter
                is_large_jump = diopter_change < large_jump_threshold_dpt

                lens_set_start_time = time.perf_counter()
                try:
                    lens.StaticInput.SetAsInput()
                    lens.StaticInput.SetFocalPower(current_diopter)
                except Exception as e:
                    print(f"Frame {frame_count}: Error setting focal power to {current_diopter:.3f}: {e}")
                lens_set_end_time = time.perf_counter()
                log_entry["lens_set_time_ms"] = (lens_set_end_time - lens_set_start_time) * 1000

                settle_time_this_frame = large_jump_settle_sec if is_large_jump else lens_settle_time_sec
                time.sleep(settle_time_this_frame)
                log_entry["lens_settle_wait_ms"] = (time.perf_counter() - lens_set_end_time) * 1000

                if is_large_jump:
                    try:
                        sacrificial_grab = camera.RetrieveResult(retrieve_timeout, pylon.TimeoutHandling_ThrowException)
                        sacrificial_grab.Release()
                    except pylon.TimeoutException:
                        print(f"  Warning: Timeout during sacrificial grab for frame {frame_count}.")
                    except Exception as e_sac:
                        print(f"  Warning: Error during sacrificial grab for frame {frame_count}: {e_sac}")
                
                grab_start_time = time.perf_counter()
                grabResult = None
                grab_success = False
                try:
                    grabResult = camera.RetrieveResult(retrieve_timeout, pylon.TimeoutHandling_ThrowException)
                    grab_success = grabResult.GrabSucceeded()
                    if not grab_success:
                        print(f"Frame {frame_count}: Error grabbing: {grabResult.GetErrorCode()} - {grabResult.GetErrorDescription()}")
                except pylon.TimeoutException:
                     print(f"Frame {frame_count}: Timeout waiting for frame retrieval (>{retrieve_timeout}ms).")
                     any_frames_skipped_in_well = True
                except pylon.GenericException as e:
                    print(f"Frame {frame_count}: Pylon exception during RetrieveResult: {e}")
                grab_end_time = time.perf_counter()
                log_entry["frame_grab_time_ms"] = (grab_end_time - grab_start_time) * 1000
                
                previous_diopter = current_diopter

                if grab_success and grabResult:
                    proc_start_time = time.perf_counter()
                    current_pylon_frame_id = grabResult.GetBlockID() if hasattr(grabResult, "GetBlockID") else None
                    log_entry["pylon_frame_id"] = current_pylon_frame_id if current_pylon_frame_id is not None else "N/A"
                    skipped_warning = False
                    if current_pylon_frame_id is not None and last_pylon_frame_id is not None:
                        # --- MODIFIED SKIP LOGIC ---
                        # Expect a jump of 2 after an intentional sacrificial grab, otherwise expect a jump of 1
                        expected_increment = 2 if is_large_jump else 1
                        expected_next_id = (last_pylon_frame_id + expected_increment)
                        
                        block_id_bits = 16 
                        if hasattr(grabResult, 'GetBlockID') and grabResult.GetBlockID() is not None:
                            try: block_id_bits = grabResult.PayloadType.GetValue()
                            except: pass 
                        max_block_id = (1 << block_id_bits) -1
                        is_wrap_around = (current_pylon_frame_id == 0 and last_pylon_frame_id > (max_block_id - 1000) )
                        
                        if current_pylon_frame_id != expected_next_id and not is_wrap_around:
                            print(f"Warning: Frame skip detected! Last PylonID: {last_pylon_frame_id}, Current: {current_pylon_frame_id}, Expected Increment: {expected_increment}")
                            skipped_warning = True
                            any_frames_skipped_in_well = True
                    log_entry["skipped_frame_warning"] = skipped_warning
                    last_pylon_frame_id = current_pylon_frame_id

                    raw_frame = grabResult.Array
                    
                    if raw_frame.dtype != np.uint8:
                        if raw_frame.dtype in [np.uint16, np.uint12, np.uint10]:
                            shift = max(0, raw_frame.itemsize * 8 - 8)
                            frame_8bit = (raw_frame >> shift).astype(np.uint8)
                        else:
                            print(f"Warning: Unsupported frame dtype {raw_frame.dtype}. Attempting direct cast to uint8.")
                            try: frame_8bit = raw_frame.astype(np.uint8)
                            except Exception as conv_e:
                                print(f"Error converting frame dtype: {conv_e}. Skipping frame {frame_count}.")
                                grabResult.Release()
                                continue
                    else:
                        frame_8bit = raw_frame

                    if process_as_color and debayer_code is not None:
                        processed_frame = cv2.cvtColor(frame_8bit, debayer_code)
                        watermark_color = watermark_color_bgr
                    else:
                        processed_frame = frame_8bit
                        watermark_color = watermark_color_mono

                    watermark_text = (f"Dpt: {current_diopter:.3f}, FD: {focusing_distance_mm:.2f} mm, "
                                      f"Frame: {frame_count+1}/{total_frames_to_record}")
                    if not processed_frame.flags.writeable: frame_display = processed_frame.copy()
                    else: frame_display = processed_frame
                    
                    watermark_lines = []
                    for part in watermark_text.split(','):
                        watermark_lines.extend(part.strip().split('\n'))
                    watermark_lines = [line.strip() for line in watermark_lines if line.strip()]

                    line_height_approx = cv2.getTextSize("Test", watermark_font, watermark_font_scale, watermark_thickness)[0][1]
                    line_spacing_factor = 1.5 
                    line_height = int(line_height_approx * line_spacing_factor)

                    for i, line in enumerate(watermark_lines):
                        cv2.putText(frame_display, line,
                                    (watermark_base_x, watermark_first_line_y + i * line_height),
                                    watermark_font, watermark_font_scale,
                                    watermark_color, watermark_thickness, cv2.LINE_AA)

                    proc_end_time = time.perf_counter()
                    log_entry["image_processing_time_ms"] = (proc_end_time - proc_start_time) * 1000

                    img_filename = (f"{current_date_str}_plate{PLATE_ID}_well{WELL_ID}_"
                                    f"f{frame_count:05d}_dpt{current_diopter:.3f}_fd{focusing_distance_mm:.2f}mm."
                                    f"{output_image_format}")
                    output_path = os.path.join(run_folder_path, img_filename)
                    log_entry["output_filename"] = output_path

                    save_submit_start_time = time.perf_counter()
                    future = executor.submit(cv2.imwrite, output_path, frame_display)
                    futures[future] = output_path
                    save_submit_end_time = time.perf_counter()
                    log_entry["image_save_submit_time_ms"] = (save_submit_end_time - save_submit_start_time) * 1000
                    frames_processed_count += 1

                    if (frame_count + 1) % int(max(1, final_fps * 2)) == 0:
                        elapsed_total = time.perf_counter() - start_acquisition_time
                        current_avg_fps = frames_processed_count / elapsed_total if elapsed_total > 0 else 0
                        print(f"  Processed frame {frames_processed_count}/{total_frames_to_record}. Current Avg FPS: {current_avg_fps:.2f}")

                elif not grab_success and grabResult is None:
                    any_frames_skipped_in_well = True

                if grabResult:
                    grabResult.Release()
            except Exception as loop_e:
                print(f"ERROR in main loop for frame {frame_count}: {type(loop_e).__name__} - {loop_e}")
                traceback.print_exc(file=sys.stdout)
                log_entry["output_filename"] = f"ERROR_IN_LOOP: {loop_e}"
                any_frames_skipped_in_well = True
                if grabResult and grabResult.IsReadable():
                    try: grabResult.Release()
                    except Exception as release_err: print(f"Error releasing grabResult after loop error: {release_err}")

            loop_iter_end_time = time.perf_counter()
            log_entry["timestamp"] = datetime.now().isoformat()
            log_entry["frame_count"] = frame_count
            log_entry["total_loop_iteration_time_ms"] = (loop_iter_end_time - loop_iter_start_time) * 1000
            for field in log_fieldnames:
                if field not in log_entry: log_entry[field] = "Error/NA"
            log_data.append(log_entry)

            iter_duration = loop_iter_end_time - loop_iter_start_time
            frame_interval = 1.0 / final_fps if final_fps > 0 else float('inf')
            if iter_duration > frame_interval:
                print(f"Warning: Loop {frame_count} took {iter_duration*1000:.2f} ms > frame interval ({frame_interval*1000:.2f} ms).")

        print("\nFinished acquisition loop.")
        print(f"Total frames processed in loop: {frames_processed_count}")
        print("Waiting for all frame saving tasks to complete...")
        saved_count = 0
        error_count = 0
        for future in as_completed(futures):
            output_path = futures[future]
            try:
                future.result()
                saved_count += 1
            except Exception as e:
                print(f"ERROR saving frame {output_path}: {e}")
                error_count += 1
        print(f"Successfully saved {saved_count} frames.")
        if error_count > 0: print(f"Encountered {error_count} errors during saving.")

    print(f"Writing {len(log_data)} log entries to {log_file_path}...")
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, mode="w", newline="", encoding="utf-8") as csvfile:
            csvfile.write("# === Acquisition Run Metadata ===\n")
            csvfile.write(f"# Timestamp (Start): {datetime.fromtimestamp(start_acquisition_time).isoformat()}\n")
            csvfile.write(f"# Plate ID: {PLATE_ID}\n")
            csvfile.write(f"# Well ID: {WELL_ID}\n")
            csvfile.write(f"# Start Diopter (dpt): {START_DPT:.3f}\n")
            csvfile.write(f"# Output Folder: {run_folder_path}\n")
            csvfile.write(f"# Date String: {current_date_str}\n")
            csvfile.write("# --- Configuration ---\n")
            csvfile.write(f"# Config File Path: {config.get('loaded_from_path', 'N/A')}\n")
            csvfile.write(f"# Processed as Color: {process_as_color}\n")
            csvfile.write(f"# Target Duration (s): {config['camera']['video_duration_sec']}\n")
            csvfile.write(f"# Configured FPS: {config['camera']['fps']:.2f}\n")
            csvfile.write(f"# Final Acquisition FPS: {final_fps:.2f}\n")
            csvfile.write(f"# Actual Exposure (us): {actual_exposure_us}\n")
            csvfile.write(f"# Actual ROI (WxH): {actual_frame_width}x{actual_frame_height}\n")
            csvfile.write(f"# Pixel Format: {camera.PixelFormat.GetValue()}\n")
            csvfile.write(f"# Image Format: .{output_image_format}\n")
            csvfile.write(f"# Lens Controller Port: {config['focusing']['lens_controller_port']}\n")
            csvfile.write(f"# Focusing Pattern: {config['focusing'].get('pattern', 'zigzag')}\n")
            csvfile.write(f"# Lens Settle Time (s): {lens_settle_time_sec}\n")
            csvfile.write(f"# Scan Height (mm): {config['focusing']['scan_height_mm']}\n")
            csvfile.write(f"# Scan Step (um): {config['focusing']['scan_step_um']}\n")
            csvfile.write(f"# Scan Ratio (mm/dpt): {scan_ratio}\n")
            csvfile.write("# === Frame Data ===\n")
            writer = csv.DictWriter(csvfile, fieldnames=log_fieldnames)
            writer.writeheader()
            writer.writerows(log_data)
        print("Log file written successfully.")
    except IOError as e:
        print(f"ERROR writing log file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during log writing: {e}")

    if camera and camera.IsGrabbing():
        print("Stopping camera grabbing (end of acquire_and_save_frames).")
        try: camera.StopGrabbing()
        except Exception as e: print(f"Error stopping camera grabbing: {e}")
    return any_frames_skipped_in_well

# --- Main Acquisition Entry Point ---
def run_acquisition(PLATE_ID, WELL_ID, START_DPT, config, process_as_color=False):
    print(f"\n=== Running Acquisition for Plate {PLATE_ID}, Well {WELL_ID}, Start DPT {START_DPT:.3f} ===")
    print(f"Color processing requested: {process_as_color}")
    acquisition_results = {
        'final_fps': 0.0, 'actual_exposure_us': 0,
        'any_frames_skipped': True, 'status': 'failed'
    }
    current_date_str = date.today().strftime("%Y%m%d")
    run_folder_name = f"{current_date_str}_plate{PLATE_ID}_well{WELL_ID}"
    run_folder_path = os.path.join(config['paths']['output_folder_base'], run_folder_name)
    log_file_path = os.path.join(run_folder_path, config['paths']['log_filename'])

    try:
        os.makedirs(run_folder_path, exist_ok=True)
        print(f"Ensured output folder exists: {run_folder_path}")
    except OSError as e:
        print(f"ERROR: Cannot create output folder {run_folder_path}: {e}")
        return acquisition_results

    lens, camera, ecc1c = None, None, None
    final_fps, actual_exposure_us = 0.0, 0
    actual_frame_width, actual_frame_height = 0, 0
    
    try:
        print("Connecting to liquid lens controller...")
        try:
            ecc1c = optoICC.connectEcc(port=config['focusing']['lens_controller_port'])
            lens = ecc1c.Lens
            if lens is None: raise ConnectionError("Failed to get valid Lens object from ECC-1C.")
            print(f"Connected to ECC-1C on {config['focusing']['lens_controller_port']}.")
        except optoICC.NoEccError as e:
            print(f"ERROR OptoICC: {e}. Check lens controller connection and port '{config['focusing']['lens_controller_port']}'.")
            raise
        except Exception as e:
            print(f"ERROR connecting to liquid lens controller: {e}")
            raise

        print("Calculating focusing pattern...")
        focus_mode = config['focusing'].get('pattern', 'zigzag')
        scan_height_mm = config['focusing']['scan_height_mm']
        scan_step_um = config['focusing']['scan_step_um']
        scan_ratio = config['focusing']['scan_ratio']
        if abs(scan_ratio) < 1e-9: raise ValueError("Scan ratio cannot be zero.")
        end_dpt = START_DPT + scan_height_mm / scan_ratio
        
        step_dpt = (scan_step_um / 1000.0) / scan_ratio
        if end_dpt < START_DPT: step_dpt = -abs(step_dpt)
        else: step_dpt = abs(step_dpt)
            
        if abs(step_dpt) < 1e-9:
             raise ValueError("Calculated diopter step is zero, cannot generate a pattern.")

        diopter_values_up = np.arange(START_DPT, end_dpt + (step_dpt / 2.0), step_dpt)
        
        if diopter_values_up.size == 0: diopter_values_up = np.array([START_DPT])
        diopter_values_up = diopter_values_up.round(3).tolist()

        single_cycle_pattern = []
        if focus_mode.lower() == 'sawtooth':
            print("  Using 'sawtooth' focusing pattern.")
            single_cycle_pattern = diopter_values_up
            print(f"  Calculated {len(diopter_values_up)} steps for one upward ramp.")
        else:
            if focus_mode.lower() != 'zigzag':
                print(f"  Warning: Unknown focus pattern '{focus_mode}'. Defaulting to 'zigzag'.")
            print("  Using 'zigzag' focusing pattern.")
            diopter_values_down = []
            if len(diopter_values_up) > 1:
                diopter_values_down = diopter_values_up[-2:0:-1]
            single_cycle_pattern = diopter_values_up + diopter_values_down
            print(f"  Calculated {len(diopter_values_up)} steps upward and {len(diopter_values_down)} steps downward.")

        if not single_cycle_pattern: single_cycle_pattern = [START_DPT]
        
        print(f"\n--- Pre-focusing lens to starting position: {START_DPT:.3f} dpt ---")
        try:
            lens.StaticInput.SetAsInput()
            lens.StaticInput.SetFocalPower(START_DPT)
            settle_wait = 1.0 
            print(f"Waiting {settle_wait}s for lens to settle...")
            time.sleep(settle_wait)
            print("Lens pre-focused and settled.")
        except Exception as e:
            print(f"ERROR during lens pre-focus: {e}")
            raise


        print("\nConnecting to camera...")
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if not devices: raise ConnectionError("No Pylon compatible camera found.")
            camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            camera.Open()
            print(f"Connected to camera: {camera.GetDeviceInfo().GetModelName()}")
        except (pylon.GenericException, ConnectionError) as e:
            print(f"ERROR connecting to Pylon camera: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error connecting to camera: {e}")
            raise

        print("Configuring camera settings (Pixel Format, Exposure, Buffers)...")

        current_format_str = camera.PixelFormat.GetValue()
        is_currently_bayer = current_format_str.startswith("Bayer")

        if process_as_color:
            if not is_currently_bayer:
                print(f"  Requesting COLOR, but camera is in MONO format ('{current_format_str}').")
                print("  Searching for a compatible Bayer format from config preferences...")
                format_set = False
                for fmt_pref in config['camera']['pixel_format_preference']:
                    if fmt_pref.startswith("Bayer") and fmt_pref in camera.PixelFormat.Symbolics:
                        try:
                            camera.PixelFormat.SetValue(fmt_pref)
                            print(f"  Successfully set Pixel Format to: {fmt_pref}")
                            format_set = True
                            break
                        except Exception as e:
                            print(f"  Warning: Could not set format to {fmt_pref}: {e}")
                if not format_set:
                    print("ERROR: Failed to set a Bayer format on the camera. Cannot proceed with color acquisition.")
                    raise ValueError("Could not configure camera for Bayer pixel format.")
            else:
                print(f"  Camera is already in a compatible Bayer format: {current_format_str}")
        else:
            if is_currently_bayer:
                print(f"  Requesting MONO, but camera is in COLOR format ('{current_format_str}').")
                print("  Searching for a compatible Mono format from config preferences...")
                format_set = False
                for fmt_pref in config['camera']['pixel_format_preference']:
                    if fmt_pref.startswith("Mono") and fmt_pref in camera.PixelFormat.Symbolics:
                        try:
                            camera.PixelFormat.SetValue(fmt_pref)
                            print(f"  Successfully set Pixel Format to: {fmt_pref}")
                            format_set = True
                            break
                        except Exception as e:
                            print(f"  Warning: Could not set format to {fmt_pref}: {e}")
                if not format_set:
                    print("  Warning: Could not switch camera to a Mono format. Will proceed with current Bayer format.")
            else:
                print(f"  Camera is already in a compatible Mono format: {current_format_str}")

        final_pixel_format = camera.PixelFormat.GetValue()
        print(f"  Final active Pixel Format is: {final_pixel_format}")

        debayer_code = None
        final_format_is_bayer = final_pixel_format.startswith("Bayer")

        if process_as_color:
            if final_format_is_bayer:
                debayer_code = BAYER_TO_CV2_CODE.get(final_pixel_format)
                if debayer_code is None:
                    raise ValueError(f"Unsupported Bayer format '{final_pixel_format}' for color conversion. Check BAYER_TO_CV2_CODE map.")
                print(f"  Will use debayering for '{final_pixel_format}'.")
            else:
                print("WARNING: Color processing was requested, but the final camera format is not Bayer. Images will be monochrome.")
                process_as_color = False

        try:
            exposure_to_set = config['camera']['exposure_us']
            if hasattr(camera.ExposureTime, 'Min') and hasattr(camera.ExposureTime, 'Max'):
                min_exp, max_exp = camera.ExposureTime.Min, camera.ExposureTime.Max
                if exposure_to_set < min_exp :
                    print(f"  Warning: Requested exposure {exposure_to_set}us < min {min_exp}us. Clamping.")
                    exposure_to_set = min_exp
                if exposure_to_set > max_exp :
                    print(f"  Warning: Requested exposure {exposure_to_set}us > max {max_exp}us. Clamping.")
                    exposure_to_set = max_exp
            camera.ExposureTime.SetValue(exposure_to_set)
            actual_exposure_us = camera.ExposureTime.GetValue()
            print(f"  Exposure Time set to: {actual_exposure_us} us (Requested: {config['camera']['exposure_us']} us)")
        except Exception as e:
            actual_exposure_us = camera.ExposureTime.GetValue() if camera.ExposureTime.IsReadable() else config['camera']['exposure_us']
            print(f"Warning: Could not set exposure time: {e}. Using current/config: {actual_exposure_us} us")

        try:
            buffer_count = max(config['camera']['max_num_buffer_min'], int(config['camera']['fps']))
            camera.MaxNumBuffer = buffer_count
            print(f"  MaxNumBuffer set to: {camera.MaxNumBuffer.GetValue()}")
        except Exception as e:
            print(f"Warning: Could not set MaxNumBuffer: {e}. Using default: {camera.MaxNumBuffer.GetValue() if camera.MaxNumBuffer.IsReadable() else 'N/A'}")

        try:
            actual_frame_width, actual_frame_height = set_centered_roi(
                camera, config['roi']['width'], config['roi']['height']
            )
            print(f"ROI set. Actual acquisition dimensions: {actual_frame_width}x{actual_frame_height}")
        except Exception as e:
            print(f"ERROR: Failed to set camera ROI. Error: {e}")
            if camera and camera.IsOpen():
                actual_frame_width = camera.Width.GetValue()
                actual_frame_height = camera.Height.GetValue()
                print(f"Using current camera dimensions after ROI failure: {actual_frame_width}x{actual_frame_height}")
            raise

        print("Configuring camera frame rate...")
        desired_fps = config['camera']['fps']
        max_possible_fps = None
        try:
            camera.AcquisitionFrameRateEnable.SetValue(True)
            print(f"  Attempting to set desired FPS: {desired_fps:.2f}")

            try:
                if hasattr(camera, "AcquisitionFrameRate") and camera.AcquisitionFrameRate.HasMax:
                    max_possible_fps = camera.AcquisitionFrameRate.Max
                    print(f"  Camera's AcquisitionFrameRate.Max: {max_possible_fps:.2f} FPS")
                elif hasattr(camera, "ResultingFrameRate") and camera.ResultingFrameRate.IsReadable():
                    max_possible_fps_resulting = camera.ResultingFrameRate.Value
                    print(f"  Camera's current ResultingFrameRate: {max_possible_fps_resulting:.2f} FPS")
                    if max_possible_fps is None or max_possible_fps_resulting < max_possible_fps :
                         max_possible_fps = max_possible_fps_resulting
                else:
                    print(f"  Warning: Cannot directly query max FPS. Will attempt to set desired FPS ({desired_fps:.2f}) and read back.")
                    max_possible_fps = desired_fps
            except Exception as e_fps_query:
                print(f"  Warning: Error querying max FPS capabilities: {e_fps_query}. Proceeding cautiously.")
                max_possible_fps = desired_fps

            if max_possible_fps is not None and desired_fps > max_possible_fps:
                final_fps_to_set = math.floor(max_possible_fps)
                print(f"Warning: Desired FPS ({desired_fps:.2f}) exceeds reported/estimated max possible ({max_possible_fps:.2f}).")
                print(f"  Adjusting FPS to {final_fps_to_set:.2f} FPS.")
                if final_fps_to_set <=0:
                    print("Error: Calculated FPS to set is zero or negative. Defaulting to 5.0 FPS.")
                    final_fps_to_set = 5.0
                camera.AcquisitionFrameRate.SetValue(final_fps_to_set)
            else:
                camera.AcquisitionFrameRate.SetValue(desired_fps)
            
            final_fps = camera.AcquisitionFrameRate.GetValue()
            print(f"  Final Acquisition Frame Rate set to: {final_fps:.2f} FPS")

            if abs(final_fps - desired_fps) > 0.1 and (max_possible_fps is None or desired_fps <= max_possible_fps):
                 print(f"  Note: Final FPS ({final_fps:.2f}) differs slightly from desired ({desired_fps:.2f}) due to camera constraints.")

        except Exception as e:
            final_fps = camera.AcquisitionFrameRate.GetValue() if camera.AcquisitionFrameRate.IsReadable() else 0
            print(f"Warning: Could not configure frame rate: {e}. Using current value: {final_fps:.2f} FPS")
            if camera.AcquisitionFrameRateEnable.IsReadable() and not camera.AcquisitionFrameRateEnable.GetValue():
                 print("  Note: Frame rate control might be disabled on camera.")
        
        if final_fps <= 0:
            print("ERROR: Final FPS is zero or negative. Cannot proceed.")
            raise ValueError("Final FPS is not positive.")

        total_frames_to_record = int(config['camera']['video_duration_sec'] * final_fps)

        watermark_base_x = actual_frame_width - config['watermark']['offset_horizontal']
        watermark_first_line_y = actual_frame_height - config['watermark']['offset_vertical']
        watermark_base_x = max(10, watermark_base_x)
        watermark_first_line_y = max(10, min(actual_frame_height - 30, watermark_first_line_y))
        
        num_steps_in_cycle = len(single_cycle_pattern)
        print(f"  Total steps in one focus cycle: {num_steps_in_cycle}")
        if num_steps_in_cycle == 0: raise ValueError("Focusing pattern cycle is empty.")

        full_focus_pattern = []
        if total_frames_to_record > 0:
            num_repeats = (total_frames_to_record // num_steps_in_cycle) + 1
            full_focus_pattern = (single_cycle_pattern * num_repeats)[:total_frames_to_record]
            print(f"  Generated full focus pattern with {len(full_focus_pattern)} steps for the entire video.")
        else:
            print("  Warning: Calculated total frames to record is zero. Focus pattern will be empty.")

        any_skipped = acquire_and_save_frames(
            camera, lens, full_focus_pattern,
            actual_frame_width, actual_frame_height, watermark_base_x, watermark_first_line_y,
            run_folder_path, log_file_path, current_date_str,
            PLATE_ID, WELL_ID, START_DPT, config,
            final_fps=final_fps, actual_exposure_us=actual_exposure_us,
            process_as_color=process_as_color, debayer_code=debayer_code
        )
        acquisition_results.update({'final_fps': final_fps, 'actual_exposure_us': actual_exposure_us,
                                    'any_frames_skipped': any_skipped, 'status': 'success'})
        print(f"\n=== Acquisition finished successfully for Plate {PLATE_ID}, Well {WELL_ID} ===")

    except (pylon.GenericException, ConnectionError, ValueError, AttributeError, optoICC.NoEccError) as e:
        print(f"\nERROR during setup or acquisition for Plate {PLATE_ID}, Well {WELL_ID}:")
        print(f"  Type: {type(e).__name__}\n  Message: {e}")
        traceback.print_exc(file=sys.stdout)
        print(f"\n=== Acquisition FAILED for Plate {PLATE_ID}, Well {WELL_ID} ===")
        acquisition_results.update({'final_fps': final_fps, 'actual_exposure_us': actual_exposure_us})
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during acquisition for Plate {PLATE_ID}, Well {WELL_ID}:")
        print(f"  Type: {type(e).__name__}\n  Message: {e}")
        traceback.print_exc(file=sys.stdout)
        print(f"\n=== Acquisition FAILED for Plate {PLATE_ID}, Well {WELL_ID} ===")
        acquisition_results.update({'final_fps': final_fps, 'actual_exposure_us': actual_exposure_us})

    finally:
        print("\n--- Cleaning up resources ---")
        if camera:
            if camera.IsGrabbing():
                print("Stopping camera grabbing (in finally block)...")
                try: camera.StopGrabbing()
                except Exception as e: print(f"  Error stopping camera grabbing: {e}")
            if camera.IsOpen():
                try:
                    print("Attempting to reset ROI to full sensor size before closing camera...")
                    reset_roi_to_full(camera)
                except Exception as e_reset_roi: print(f"  Error resetting ROI to full: {e_reset_roi}")
                print("Closing camera...")
                try: camera.Close()
                except Exception as e: print(f"  Error closing camera: {e}")
                else: print("  Camera closed.")
            else: print("Camera was not open, skipping close related actions.")
        else: print("Camera object not created or connection failed.")
        
        if ecc1c:
            print("Disconnecting from ECC-1C...")
            try:
                if hasattr(ecc1c, 'disconnect') and callable(ecc1c.disconnect): ecc1c.disconnect()
                elif hasattr(ecc1c, 'close') and callable(ecc1c.close): ecc1c.close()
                else: print("  Warning: ecc1c object lacks known 'disconnect()' or 'close()' method.")
                print("  Disconnected from ECC-1C.")
            except Exception as e: print(f"  Error during ECC-1C disconnection: {e}")
        else: print("ECC-1C controller object not created or connection failed.")
        print("--- Cleanup finished ---")
        return acquisition_results

if __name__ == "__main__":
    print("Executing acquisition module directly (example run)...")
    loaded_config = None
    try:
        loaded_config = load_config()
        loaded_config['loaded_from_path'] = os.path.abspath(DEFAULT_CONFIG_PATH)
        # For direct execution test, we assume monochrome. The main script run_acquire.py will handle the choice.
        results = run_acquisition(1, 1, -1.8, loaded_config, process_as_color=False)
        print(f"Example run acquisition results: {results}")
    except SystemExit:
        print("Exiting due to configuration loading error.")
    except Exception as main_e:
        print(f"\nFATAL ERROR during script execution: {main_e}")
        traceback.print_exc(file=sys.stdout)
    print("Acquisition module execution complete.")
