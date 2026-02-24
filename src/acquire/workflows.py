from __future__ import annotations

import copy
import csv
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from itertools import product

try:
    from zaber.serial.unexpectedreplyerror import UnexpectedReplyError
except Exception:  # pragma: no cover - hardware dependency unavailable in test env
    class UnexpectedReplyError(Exception):
        pass

from .camera_capabilities import check_camera_is_color
from .config import get_plate_settings
from .experiment_logging import append_to_experiment_log, initialize_experiment_log
from .stage_controller import StageController
from .well_selector_gui import well_selection_gui


def _get_acquisition_api():
    from .wellscanner_acquisition_module import load_config, run_acquisition
    return load_config, run_acquisition


def select_wells(acq_conf, default_plate_24, default_plate_96, csv_filename):
    print("Starting well selection GUI...")
    try:
        plate_type, plate_settings = get_plate_settings(acq_conf, default_plate_24, default_plate_96)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False
    well_selection_gui(plate_type, plate_settings, csv_filename)
    print(f"Selected wells (if any) saved to '{csv_filename}'.")
    return True


def decide_color_mode(acq_conf):
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
    else:
        if is_color_camera:
            while True:
                response = input("Color camera detected. Save images in color? (y/n): ").lower().strip()
                if response == 'y':
                    process_images_in_color = True
                    break
                if response == 'n':
                    process_images_in_color = False
                    break
                print("Invalid input. Please enter 'y' or 'n'.")
        else:
            print("Monochrome camera detected. Proceeding with monochrome acquisition.")
            process_images_in_color = False

    print(f"FINAL DECISION: Image acquisition will be processed in {'COLOR' if process_images_in_color else 'MONOCHROME'}.")
    return process_images_in_color


def load_acquisition_config(acquisition_config_path):
    print("\nLoading acquisition configuration...")
    acq_conf = load_config(acquisition_config_path)
    acq_conf['loaded_from_path'] = os.path.abspath(acquisition_config_path)
    return acq_conf


def validate_selection_csv(csv_filename):
    if not os.path.exists(csv_filename):
        print(f"Well selection CSV ('{csv_filename}') not found. Exiting.")
        return False
    try:
        with open(csv_filename, 'r') as f:
            if len(f.readlines()) <= 1:
                print(f"'{csv_filename}' is empty or contains no well data. Exiting.")
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
                return False
    except Exception as e:
        print(f"Error reading '{csv_filename}': {e}. Exiting.")
        return False
    return True


def prompt_plate_and_dpt():
    while True:
        try:
            plate_num = int(input("Enter the plate number (integer): "))
        except ValueError:
            print("Invalid input. Plate number must be an integer.")
        else:
            break

    while True:
        try:
            start_dpt_val = round(float(input("Enter dioptries at 0.0mm (-2.00 to +3.00): ")), 2)
            if not (-2.00 <= start_dpt_val <= 3.00):
                raise ValueError("Out of range.")
        except ValueError as e:
            print(f"Invalid input: {e}. Range -2.00 to 3.00.")
        else:
            break

    print(f"Accepted Plate No: {plate_num}, Start DPT: {start_dpt_val:.2f}\n---------------------------\n")
    return plate_num, start_dpt_val


def preflight_delete_or_exit(base_out_dir, candidate_dirs, check_step_title):
    print(check_step_title)
    if candidate_dirs:
        print("\n--- WARNING: Existing Data Found ---")
        for line in candidate_dirs['description_lines']:
            print(line)
        for d in candidate_dirs['dirs']:
            print(f"  - {d}")
        print("\nChoose an option:")
        print(candidate_dirs['option_1'])
        print("  2) EXIT: Stop the program without making any changes.")
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                print("\nProceeding. Deleting contents of existing directories...")
                try:
                    for d in candidate_dirs['dirs']:
                        shutil.rmtree(d)
                        print(f"  - Cleared: {d}")
                    print("All targeted directories have been cleared.")
                except Exception as e:
                    print(f"\nFATAL ERROR: Could not delete directories: {e}")
                    if candidate_dirs.get('fatal_suffix'):
                        print(candidate_dirs['fatal_suffix'])
                    sys.exit(1)
                break
            elif choice == '2':
                print("Exiting program as requested. No measurements will be performed.")
                sys.exit(0)
            else:
                print("Invalid input. Please enter 1 or 2.")
    else:
        print(candidate_dirs['none_message'])


def finalize_stage_cleanup(initialization_successful, zab_port, zab_device_x, zab_device_y, stage_controller):
    if initialization_successful:
        print("\n--- Attempting Final Homing Sequence ---")
        if zab_port and hasattr(zab_port, 'is_open') and zab_port.is_open:
            stage_controller.home_device(zab_device_x, "X", zab_port)
            stage_controller.home_device(zab_device_y, "Y", zab_port)
            print("Final homing attempts finished.")
        elif zab_port:
            print("Final Homing: Zaber port was initialized but is not open. Cannot home.")
        else:
            print("Final Homing: Zaber port was not properly initialized. Cannot home.")

    print("\n--- Entering Final Resource Cleanup ---")
    if zab_port:
        print("Zaber port object exists. Checking if open for closure...")
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


def rename_files_after_run(run_folder_path, string_to_remove, config):
    if not os.path.isdir(run_folder_path):
        print(f"  Warning: Cannot rename files, directory not found: {run_folder_path}")
        return

    print(f"  Post-processing: Renaming files in {os.path.basename(run_folder_path)}...")
    renamed_count = 0
    image_formats = {'.png', '.bmp', '.jpg', '.jpeg', '.tiff'}

    try:
        filenames = os.listdir(run_folder_path)
    except FileNotFoundError:
        print(f"  Warning: Directory disappeared before renaming: {run_folder_path}")
        return

    for filename in filenames:
        name, ext = os.path.splitext(filename)
        if ext.lower() in image_formats and string_to_remove in name:
            try:
                old_path = os.path.join(run_folder_path, filename)
                new_filename = name.replace(string_to_remove, "") + ext
                new_path = os.path.join(run_folder_path, new_filename)
                os.rename(old_path, new_path)
                renamed_count += 1
            except Exception as e:
                print(f"    Error renaming {filename}: {e}")

    print(f"  Renamed {renamed_count} image files.")
    metadata_log_path = os.path.join(run_folder_path, config['paths']['log_filename'])
    if os.path.exists(metadata_log_path):
        print(f"  Updating '{config['paths']['log_filename']}' with new filenames...")
        try:
            with open(metadata_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(metadata_log_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    if line.strip().startswith('#') or 'timestamp,frame_count' in line:
                        f.write(line)
                    else:
                        f.write(line.replace(string_to_remove, ""))
            print("  Log file updated successfully.")
        except Exception as e:
            print(f"  ERROR updating {metadata_log_path}: {e}")
    else:
        print(f"  Warning: Could not find log file to update at {metadata_log_path}")


def run_multiwell_workflow(*, acquisition_config_path, csv_filename, default_plate_24, default_plate_96, serial_port,
                           device_x_number, device_y_number, device_x_microstep_size, device_y_microstep_size,
                           position_tolerance):
    try:
        acq_conf = load_acquisition_config(acquisition_config_path)
    except SystemExit:
        return
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}. Exiting.")
        return

    print("Step 1: Well Selection")
    if not select_wells(acq_conf, default_plate_24, default_plate_96, csv_filename):
        return
    if not validate_selection_csv(csv_filename):
        return

    process_images_in_color = decide_color_mode(acq_conf)

    summary_log_fn = acq_conf.get('paths', {}).get('experiment_summary_log_filename', 'experiment_summary_log.csv')
    base_out_dir = acq_conf.get('paths', {}).get('output_folder_base', '.')
    os.makedirs(base_out_dir, exist_ok=True)
    exp_summary_log_pth = os.path.join(base_out_dir, summary_log_fn)
    initialize_experiment_log(
        exp_summary_log_pth,
        [
            "date", "time", "plate_id", "well_id", "well_name",
            "start_dpt", "exposure_time_us", "final_acquisition_fps",
            "any_frames_skipped", "acquisition_status", "run_folder",
        ],
    )

    plate_num, start_dpt_val = prompt_plate_and_dpt()

    try:
        with open(csv_filename, newline='') as csv_f:
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
        preflight_delete_or_exit(
            base_out_dir,
            {
                'dirs': existing_dirs,
                'description_lines': [
                    "The following output directories for today's date and plate number already exist:",
                ],
                'option_1': "  1) PROCEED: Delete the contents of these directories and start the new acquisition.",
                'none_message': "No existing directories found for this plate and date. Proceeding.",
                'fatal_suffix': "Please check file permissions and close any programs that might be using these folders.",
            },
            "Step 3: Checking for existing output directories...",
        )
    except FileNotFoundError:
        print(f"ERROR: Cannot perform directory check because '{csv_filename}' was not found.")
        sys.exit(1)

    input("Ensure:\n1) Plate orientation (A1 lower right)\n2) Lid\n3) Light source\n4) Devices connected\n(Enter to start)\n")

    stage_controller = StageController(
        serial_port,
        device_x_number,
        device_y_number,
        device_x_microstep_size,
        device_y_microstep_size,
        position_tolerance,
    )
    zab_port, zab_device_x, zab_device_y = None, None, None
    initialization_successful = False

    try:
        print("\nStep 4: Stage Initialization and Main Processing")
        zab_port, zab_device_x, zab_device_y = stage_controller.initialize()
        initialization_successful = True
        current_x_pos, current_y_pos = None, None

        print(f"\nLoading coordinates from '{csv_filename}'...")
        with open(csv_filename, newline='') as csv_f:
            all_wells_data = list(csv.DictReader(csv_f))
            for idx, well_row in enumerate(all_wells_data):
                wellname = well_row.get('wellname', 'N/A')
                print(f"\n--- Processing well {idx+1}/{len(all_wells_data)}: {wellname} (WellNr: {well_row.get('wellnr', 'N/A')}) ---")
                try:
                    x_mm, y_mm = float(well_row['X']), float(well_row['Y'])
                    well_num = int(well_row['wellnr'])
                except (ValueError, KeyError) as err:
                    print(f"Skipping row {well_row} due to error: {err}")
                    run_fldr = f"plate{plate_num}_well{well_row.get('wellnr', 'err')}_data_error"
                    append_to_experiment_log(
                        exp_summary_log_pth,
                        lambda now: [
                            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), plate_num, well_row.get('wellnr', -1), wellname,
                            f"{start_dpt_val:.2f}", 'N/A', f"{0.0:.2f}", True, 'data_error', run_fldr,
                        ],
                    )
                    continue

                if zab_port:
                    zab_port.flush()
                current_x_pos, current_y_pos = stage_controller.move_to(
                    x_mm, y_mm, current_x_pos, current_y_pos, zab_device_x, zab_device_y, zab_port,
                )

                print(f"Starting acquisition for Plate {plate_num}, Well {well_num}...")
                date_fldr_str = datetime.now().strftime("%Y%m%d")
                run_fldr_name = os.path.join(
                    acq_conf['paths']['output_folder_base'],
                    f"{date_fldr_str}_plate{plate_num}_well{well_num}",
                )

                _, run_acquisition = _get_acquisition_api()
                well_acq_results = run_acquisition(
                    plate_num,
                    well_num,
                    start_dpt_val,
                    acq_conf,
                    process_as_color=process_images_in_color,
                )

                append_to_experiment_log(
                    exp_summary_log_pth,
                    lambda now, _r=well_acq_results: [
                        now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), plate_num, well_num, wellname,
                        f"{start_dpt_val:.2f}", _r.get('actual_exposure_us', 'N/A'), f"{_r.get('final_fps', 0.0):.2f}",
                        _r.get('any_frames_skipped', True), _r.get('status', 'failed'), run_fldr_name,
                    ],
                )
                if well_acq_results.get('status') == 'success':
                    print(f"Acquisition for {wellname} successful. Results: {well_acq_results}")
                else:
                    print(f"Acquisition for {wellname} FAILED. Results: {well_acq_results}")

                if idx < len(all_wells_data) - 1:
                    print("Waiting before next well...")
                    time.sleep(5)

            print("\nAll wells processed.")

    except FileNotFoundError:
        print(f"ERROR: CSV file '{csv_filename}' not found.")
    except UnexpectedReplyError as zab_err_main:
        print(f"Zaber Stage Communication Error during main processing: {zab_err_main}.")
    except Exception as gen_err_main:
        print(f"An error occurred: {gen_err_main}")
        traceback.print_exc()
    finally:
        finalize_stage_cleanup(initialization_successful, zab_port, zab_device_x, zab_device_y, stage_controller)


def run_experiment_sweep_workflow(*, acquisition_config_path, csv_filename, default_plate_24, default_plate_96, serial_port,
                                  device_x_number, device_y_number, device_x_microstep_size, device_y_microstep_size,
                                  position_tolerance, scan_steps_um_options, focusing_pattern_options):
    try:
        acq_conf = load_acquisition_config(acquisition_config_path)
    except SystemExit:
        return
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}. Exiting.")
        return

    print("Step 1: Well Selection")
    if not select_wells(acq_conf, default_plate_24, default_plate_96, csv_filename):
        return
    if not validate_selection_csv(csv_filename):
        return

    process_images_in_color = decide_color_mode(acq_conf)

    summary_log_fn = acq_conf.get('paths', {}).get('experiment_summary_log_filename', 'experiment_summary_log.csv')
    base_out_dir = acq_conf.get('paths', {}).get('output_folder_base', '.')
    os.makedirs(base_out_dir, exist_ok=True)
    exp_summary_log_pth = os.path.join(base_out_dir, summary_log_fn)
    initialize_experiment_log(
        exp_summary_log_pth,
        [
            "date", "time", "plate_id", "well_id", "well_name",
            "scan_step_um", "focus_pattern",
            "start_dpt", "exposure_time_us", "final_acquisition_fps",
            "any_frames_skipped", "acquisition_status", "run_folder",
        ],
    )

    plate_num, start_dpt_val = prompt_plate_and_dpt()

    try:
        with open(csv_filename, newline='') as csv_f:
            all_wells_data_for_check = list(csv.DictReader(csv_f))

        date_str = datetime.now().strftime("%Y%m%d")
        existing_combination_dirs = []
        for well_row in all_wells_data_for_check:
            try:
                well_num = int(well_row['wellnr'])
                for step_um in scan_steps_um_options:
                    for pattern in focusing_pattern_options:
                        unique_well_id_str = f"{well_num}_{step_um}um_{pattern}"
                        dir_to_check = os.path.join(base_out_dir, f"{date_str}_plate{plate_num}_well{unique_well_id_str}")
                        if os.path.isdir(dir_to_check):
                            existing_combination_dirs.append(dir_to_check)
            except (ValueError, KeyError):
                continue

        preflight_delete_or_exit(
            base_out_dir,
            {
                'dirs': existing_combination_dirs,
                'description_lines': [
                    "The following output directories for today's date, plate, and parameters already exist:",
                ],
                'option_1': "  1) PROCEED: Delete the contents of these specific directories and start the new acquisition.",
                'none_message': "No existing directories found for this plate, date, and parameters. Proceeding.",
                'fatal_suffix': None,
            },
            "Step 3: Checking for existing output directories for all parameter combinations...",
        )

    except FileNotFoundError:
        print(f"ERROR: Cannot perform directory check because '{csv_filename}' was not found.")
        sys.exit(1)

    input("Ensure:\n1) Plate orientation (A1 lower right)\n2) Lid\n3) Light source\n4) Devices connected\n(Enter to start)\n")

    stage_controller = StageController(
        serial_port,
        device_x_number,
        device_y_number,
        device_x_microstep_size,
        device_y_microstep_size,
        position_tolerance,
    )
    zab_port, zab_device_x, zab_device_y = None, None, None
    initialization_successful = False

    try:
        print("\nStep 4: Stage Initialization and Main Processing")
        zab_port, zab_device_x, zab_device_y = stage_controller.initialize()
        initialization_successful = True

        current_x_pos, current_y_pos = None, None

        with open(csv_filename, newline='') as csv_f:
            all_wells_data = list(csv.DictReader(csv_f))

            for idx, well_row in enumerate(all_wells_data):
                wellname = well_row.get('wellname', 'N/A')
                print(f"\n--- Processing well {idx+1}/{len(all_wells_data)}: {wellname} (WellNr: {well_row.get('wellnr', 'N/A')}) ---")
                try:
                    x_mm, y_mm = float(well_row['X']), float(well_row['Y'])
                    well_num = int(well_row['wellnr'])
                except (ValueError, KeyError) as err:
                    print(f"Skipping row {well_row} due to error: {err}")
                    for step_um, pattern in product(scan_steps_um_options, focusing_pattern_options):
                        unique_well_id_str = f"{well_row.get('wellnr', 'err')}_{step_um}um_{pattern}"
                        run_fldr = f"plate{plate_num}_well{unique_well_id_str}_data_error"
                        append_to_experiment_log(
                            exp_summary_log_pth,
                            lambda now, _s=step_um, _p=pattern: [
                                now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), plate_num, well_row.get('wellnr', -1), wellname,
                                _s, _p, f"{start_dpt_val:.2f}", 'N/A', f"{0.0:.2f}", True, 'data_error', run_fldr,
                            ],
                        )
                    continue

                if zab_port:
                    zab_port.flush()
                current_x_pos, current_y_pos = stage_controller.move_to(
                    x_mm, y_mm, current_x_pos, current_y_pos, zab_device_x, zab_device_y, zab_port,
                )

                parameter_combinations = list(product(scan_steps_um_options, focusing_pattern_options))
                random.shuffle(parameter_combinations)
                print(f"  Randomized parameter order for this well: {parameter_combinations}")

                for current_step_um, current_pattern in parameter_combinations:
                    print(f"\n  -> Starting acquisition for {wellname} with Step: {current_step_um}um, Pattern: {current_pattern}")

                    parameter_identifier_str = f"_{current_step_um}um_{current_pattern}"
                    unique_well_id_str = f"{well_num}{parameter_identifier_str}"

                    modified_acq_conf = copy.deepcopy(acq_conf)
                    modified_acq_conf['focusing']['scan_step_um'] = current_step_um
                    modified_acq_conf['focusing']['pattern'] = current_pattern

                    _, run_acquisition = _get_acquisition_api()
                    well_acq_results = run_acquisition(
                        plate_num,
                        unique_well_id_str,
                        start_dpt_val,
                        modified_acq_conf,
                        process_as_color=process_images_in_color,
                    )

                    date_fldr_str = datetime.now().strftime("%Y%m%d")
                    run_fldr_name = f"{date_fldr_str}_plate{plate_num}_well{unique_well_id_str}"
                    unique_run_folder_path = os.path.join(base_out_dir, run_fldr_name)

                    if well_acq_results.get('status') == 'success':
                        rename_files_after_run(unique_run_folder_path, parameter_identifier_str, modified_acq_conf)

                    append_to_experiment_log(
                        exp_summary_log_pth,
                        lambda now, _r=well_acq_results, _s=current_step_um, _p=current_pattern: [
                            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), plate_num, well_num, wellname,
                            _s, _p, f"{start_dpt_val:.2f}", _r.get('actual_exposure_us', 'N/A'), f"{_r.get('final_fps', 0.0):.2f}",
                            _r.get('any_frames_skipped', True), _r.get('status', 'failed'), unique_run_folder_path,
                        ],
                    )

                    if well_acq_results.get('status') == 'success':
                        print(f"  Acquisition and renaming for {wellname} ({current_step_um}um, {current_pattern}) successful.")
                    else:
                        print(f"  Acquisition for {wellname} ({current_step_um}um, {current_pattern}) FAILED.")

                    time.sleep(2)

                if idx < len(all_wells_data) - 1:
                    print("\nWaiting before next well...")
                    time.sleep(5)

            print("\nAll wells and parameter combinations processed.")

    except FileNotFoundError:
        print(f"ERROR: CSV file '{csv_filename}' not found.")
    except UnexpectedReplyError as zab_err_main:
        print(f"Zaber Stage Communication Error during main processing: {zab_err_main}.")
    except Exception as gen_err_main:
        print(f"An error occurred: {gen_err_main}")
        traceback.print_exc()
    finally:
        finalize_stage_cleanup(initialization_successful, zab_port, zab_device_x, zab_device_y, stage_controller)


def run_single_well_workflow(*, acquisition_config_path, plate_no=0, well_no=0, start_dpt=-2.0):
    load_config, run_acquisition = _get_acquisition_api()
    acq_conf = load_config(acquisition_config_path)
    run_acquisition(plate_no, well_no, start_dpt, acq_conf)
