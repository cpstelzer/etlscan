#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-guided flat-field and dark-frame acquisition.

This script loads camera settings from config.yml, guides the user through
capturing a flat-field image (averaged over multiple frames), then captures a
dark frame with the shutter closed. Outputs are saved as flat.png and dark.png
inside <output_folder_base>/1_flat_field.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from pypylon import pylon

from wellscanner_acquisition_module import (
    BAYER_TO_CV2_CODE,
    load_config,
    reset_roi_to_full,
    set_centered_roi,
)


DEFAULT_CONFIG_PATH = "config/config.yml"


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

        available_formats = camera.PixelFormat.Symbolics
        is_color = any(fmt.startswith("Bayer") for fmt in available_formats)

        print(
            f"CAMERA CHECK: Found '{camera.GetDeviceInfo().GetModelName()}'. "
            f"Current format: '{camera.PixelFormat.GetValue()}'."
        )
        print(
            "CAMERA CHECK: Sensor is "
            f"{'COLOR (Bayer capable)' if is_color else 'MONOCHROME'}."
        )

        camera.Close()
        return is_color

    except Exception as e:
        print(f"CAMERA CHECK: Error while checking camera sensor type: {e}")
        if camera and camera.IsOpen():
            camera.Close()
        return False


def determine_color_mode(acq_conf):
    print("\nStep 1: Determining acquisition mode")
    process_images_in_color = False
    is_color_camera = check_camera_is_color()
    color_mode_config = acq_conf.get("camera", {}).get("color_mode", "Ask").capitalize()

    if color_mode_config == "Color":
        if not is_color_camera:
            print(
                "WARNING: Config requests 'Color' mode, but a monochrome camera "
                "was detected. Forcing monochrome."
            )
        else:
            print("Configured for 'Color' mode.")
            process_images_in_color = True
    elif color_mode_config == "Mono":
        print("Configured for 'Mono' mode.")
        process_images_in_color = False
    else:
        if is_color_camera:
            while True:
                response = input("Color camera detected. Save images in color? (y/n): ").lower().strip()
                if response == "y":
                    process_images_in_color = True
                    break
                if response == "n":
                    process_images_in_color = False
                    break
                print("Invalid input. Please enter 'y' or 'n'.")
        else:
            print("Monochrome camera detected. Proceeding with monochrome acquisition.")
            process_images_in_color = False

    print(
        "FINAL DECISION: Image acquisition will be processed in "
        f"{'COLOR' if process_images_in_color else 'MONOCHROME'}."
    )
    return process_images_in_color


def configure_camera(camera, config, process_as_color):
    print("\nConnecting to camera...")
    camera.Open()
    print(f"Connected to camera: {camera.GetDeviceInfo().GetModelName()}")

    print("Configuring camera settings (Pixel Format, Exposure, Buffers, ROI, FPS)...")
    current_format_str = camera.PixelFormat.GetValue()
    is_currently_bayer = current_format_str.startswith("Bayer")

    if process_as_color:
        if not is_currently_bayer:
            print(
                f"  Requesting COLOR, but camera is in MONO format ('{current_format_str}')."
            )
            print("  Searching for a compatible Bayer format from config preferences...")
            format_set = False
            for fmt_pref in config["camera"]["pixel_format_preference"]:
                if fmt_pref.startswith("Bayer") and fmt_pref in camera.PixelFormat.Symbolics:
                    try:
                        camera.PixelFormat.SetValue(fmt_pref)
                        print(f"  Successfully set Pixel Format to: {fmt_pref}")
                        format_set = True
                        break
                    except Exception as e:
                        print(f"  Warning: Could not set format to {fmt_pref}: {e}")
            if not format_set:
                print("ERROR: Failed to set a Bayer format on the camera. Cannot proceed.")
                raise ValueError("Could not configure camera for Bayer pixel format.")
        else:
            print(f"  Camera is already in a compatible Bayer format: {current_format_str}")
    else:
        if is_currently_bayer:
            print(
                f"  Requesting MONO, but camera is in COLOR format ('{current_format_str}')."
            )
            print("  Searching for a compatible Mono format from config preferences...")
            format_set = False
            for fmt_pref in config["camera"]["pixel_format_preference"]:
                if fmt_pref.startswith("Mono") and fmt_pref in camera.PixelFormat.Symbolics:
                    try:
                        camera.PixelFormat.SetValue(fmt_pref)
                        print(f"  Successfully set Pixel Format to: {fmt_pref}")
                        format_set = True
                        break
                    except Exception as e:
                        print(f"  Warning: Could not set format to {fmt_pref}: {e}")
            if not format_set:
                print(
                    "  Warning: Could not switch camera to a Mono format. "
                    "Will proceed with current Bayer format."
                )
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
                raise ValueError(
                    f"Unsupported Bayer format '{final_pixel_format}' for color conversion. "
                    "Check BAYER_TO_CV2_CODE map."
                )
            print(f"  Will use debayering for '{final_pixel_format}'.")
        else:
            print(
                "WARNING: Color processing was requested, but the final camera format "
                "is not Bayer. Images will be monochrome."
            )
            process_as_color = False

    try:
        exposure_to_set = config["camera"]["exposure_us"]
        if hasattr(camera.ExposureTime, "Min") and hasattr(camera.ExposureTime, "Max"):
            min_exp, max_exp = camera.ExposureTime.Min, camera.ExposureTime.Max
            if exposure_to_set < min_exp:
                print(
                    f"  Warning: Requested exposure {exposure_to_set}us < min {min_exp}us. "
                    "Clamping."
                )
                exposure_to_set = min_exp
            if exposure_to_set > max_exp:
                print(
                    f"  Warning: Requested exposure {exposure_to_set}us > max {max_exp}us. "
                    "Clamping."
                )
                exposure_to_set = max_exp
        camera.ExposureTime.SetValue(exposure_to_set)
        actual_exposure_us = camera.ExposureTime.GetValue()
        print(
            "  Exposure Time set to: "
            f"{actual_exposure_us} us (Requested: {config['camera']['exposure_us']} us)"
        )
    except Exception as e:
        actual_exposure_us = (
            camera.ExposureTime.GetValue()
            if camera.ExposureTime.IsReadable()
            else config["camera"]["exposure_us"]
        )
        print(f"Warning: Could not set exposure time: {e}. Using current/config: {actual_exposure_us} us")

    try:
        buffer_count = max(config["camera"]["max_num_buffer_min"], int(config["camera"]["fps"]))
        camera.MaxNumBuffer = buffer_count
        print(f"  MaxNumBuffer set to: {camera.MaxNumBuffer.GetValue()}")
    except Exception as e:
        fallback = camera.MaxNumBuffer.GetValue() if camera.MaxNumBuffer.IsReadable() else "N/A"
        print(f"Warning: Could not set MaxNumBuffer: {e}. Using default: {fallback}")

    try:
        actual_frame_width, actual_frame_height = set_centered_roi(
            camera, config["roi"]["width"], config["roi"]["height"]
        )
        print(f"ROI set. Actual acquisition dimensions: {actual_frame_width}x{actual_frame_height}")
    except Exception as e:
        print(f"ERROR: Failed to set camera ROI. Error: {e}")
        if camera and camera.IsOpen():
            actual_frame_width = camera.Width.GetValue()
            actual_frame_height = camera.Height.GetValue()
            print(
                "Using current camera dimensions after ROI failure: "
                f"{actual_frame_width}x{actual_frame_height}"
            )
        raise

    final_fps = 0.0
    try:
        desired_fps = config["camera"]["fps"]
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(desired_fps)
        final_fps = camera.AcquisitionFrameRate.GetValue()
        print(f"  Final Acquisition Frame Rate set to: {final_fps:.2f} FPS")
    except Exception as e:
        final_fps = camera.AcquisitionFrameRate.GetValue() if camera.AcquisitionFrameRate.IsReadable() else 0
        print(f"Warning: Could not configure frame rate: {e}. Using current value: {final_fps:.2f} FPS")

    return actual_frame_width, actual_frame_height, debayer_code, process_as_color, final_fps, actual_exposure_us


def grab_average_frame(camera, num_frames, process_as_color, debayer_code, grab_strategy):
    if num_frames <= 0:
        raise ValueError("Number of frames to average must be positive.")

    print(f"  Capturing {num_frames} frame(s)...")
    camera.StartGrabbingMax(num_frames, grab_strategy)

    accumulator = None
    frame_count = 0
    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            try:
                if grab_result.GrabSucceeded():
                    frame = grab_result.Array
                    if process_as_color and debayer_code is not None:
                        frame = cv2.cvtColor(frame, debayer_code)
                    if accumulator is None:
                        accumulator = np.zeros_like(frame, dtype=np.float32)
                        output_dtype = frame.dtype
                    accumulator += frame.astype(np.float32)
                    frame_count += 1
                else:
                    print("  Warning: Frame grab failed.")
            finally:
                grab_result.Release()
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()

    if frame_count == 0:
        raise RuntimeError("No frames were captured.")

    averaged = accumulator / frame_count
    dtype_info = np.iinfo(output_dtype) if np.issubdtype(output_dtype, np.integer) else None
    if dtype_info:
        averaged = np.clip(averaged, dtype_info.min, dtype_info.max)
    return averaged.astype(output_dtype)


def prompt_user(step_title, instructions):
    print("\n" + step_title)
    for line in instructions:
        print(f"  - {line}")
    input("\nPress Enter to continue...")


def main():
    parser = argparse.ArgumentParser(
        description="Acquire flat-field and dark-frame calibration images."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to unified configuration YAML (default: config/config.yml)",
    )
    parser.add_argument(
        "--flat-frames",
        type=int,
        default=3,
        help="Number of frames to average for the flat-field image (default: 3)",
    )
    args = parser.parse_args()

    print("Loading acquisition configuration...")
    try:
        acq_conf = load_config(args.config)
        acq_conf["loaded_from_path"] = os.path.abspath(args.config)
    except SystemExit:
        return
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}. Exiting.")
        return

    process_as_color = determine_color_mode(acq_conf)

    base_out_dir = acq_conf.get("paths", {}).get("output_folder_base", ".")
    flat_dir = os.path.join(base_out_dir, "1_flat_field")
    os.makedirs(flat_dir, exist_ok=True)
    flat_path = os.path.join(flat_dir, "flat.png")
    dark_path = os.path.join(flat_dir, "dark.png")

    camera = None
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise ConnectionError("No Pylon compatible camera found.")
        camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

        (
            _frame_width,
            _frame_height,
            debayer_code,
            process_as_color,
            final_fps,
            actual_exposure_us,
        ) = configure_camera(camera, acq_conf, process_as_color)

        print(
            "\nCamera configured. "
            f"Exposure: {actual_exposure_us} us | FPS: {final_fps:.2f} | Output folder: {flat_dir}"
        )

        prompt_user(
            "Step 2: Acquire flat-field image",
            [
                "Remove samples/particles from the field of view.",
                "Ensure uniform illumination across the sensor.",
                f"The flat-field will average {args.flat_frames} frame(s).",
            ],
        )
        flat_image = grab_average_frame(
            camera,
            args.flat_frames,
            process_as_color,
            debayer_code,
            acq_conf["camera"]["grab_strategy"],
        )
        if not cv2.imwrite(flat_path, flat_image):
            raise IOError(f"Failed to write flat-field image to {flat_path}")
        print(f"Flat-field saved to: {flat_path}")

        prompt_user(
            "Step 3: Acquire dark frame",
            [
                "Close the camera shutter or block the light path.",
                "Keep all other camera settings unchanged.",
            ],
        )
        dark_image = grab_average_frame(
            camera,
            1,
            process_as_color,
            debayer_code,
            acq_conf["camera"]["grab_strategy"],
        )
        if not cv2.imwrite(dark_path, dark_image):
            raise IOError(f"Failed to write dark-frame image to {dark_path}")
        print(f"Dark frame saved to: {dark_path}")

        print("\nFlat-field acquisition complete.")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        if camera:
            if camera.IsGrabbing():
                try:
                    camera.StopGrabbing()
                except Exception as e:
                    print(f"Warning: Error stopping camera grabbing: {e}")
            if camera.IsOpen():
                try:
                    reset_roi_to_full(camera)
                except Exception as e:
                    print(f"Warning: Error resetting ROI: {e}")
                camera.Close()


if __name__ == "__main__":
    main()
