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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:   # also runnable as `python src/acquire/ws_flat_field.py`
    sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pypylon import pylon  # noqa: E402

from src.acquire.camera_capabilities import check_camera_is_color  # noqa: E402
from src.acquire.hardware_setup import close_camera, configure_camera, open_first_camera, set_frame_rate  # noqa: E402
from src.acquire.wellscanner_acquisition_module import load_config  # noqa: E402

DEFAULT_CONFIG_PATH = "config/config.yml"


def determine_color_mode(acq_conf):
    print("\nStep 1: Determining acquisition mode")
    process_images_in_color = False
    is_color_camera = check_camera_is_color()
    color_mode_config = acq_conf["camera"]["color_mode"].capitalize()

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


def grab_average_frame(camera, num_frames, process_as_color, debayer_code, grab_strategy, frame_timeout_ms):
    """Average ``num_frames`` frames; each must arrive within ``frame_timeout_ms``."""
    if num_frames <= 0:
        raise ValueError("Number of frames to average must be positive.")

    print(f"  Capturing {num_frames} frame(s)...")
    camera.StartGrabbingMax(num_frames, grab_strategy)

    accumulator = None
    frame_count = 0
    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(frame_timeout_ms, pylon.TimeoutHandling_ThrowException)
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

    base_out_dir = acq_conf["paths"]["output_folder_base"]
    flat_dir = os.path.join(base_out_dir, "1_flat_field")
    os.makedirs(flat_dir, exist_ok=True)
    flat_path = os.path.join(flat_dir, "flat.png")
    dark_path = os.path.join(flat_dir, "dark.png")

    camera = None
    try:
        print("\nConnecting to camera...")
        camera = open_first_camera(pylon)
        camera_setup = configure_camera(camera, acq_conf, process_as_color, acq_conf["camera"]["fps"])
        process_as_color = camera_setup.process_as_color
        debayer_code = camera_setup.debayer_code
        actual_exposure_us = camera_setup.exposure_us
        final_fps = set_frame_rate(camera, acq_conf["camera"]["fps"])
        frame_timeout_ms = acq_conf["camera"]["single_frame_timeout_ms"]

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
            frame_timeout_ms,
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
            frame_timeout_ms,
        )
        if not cv2.imwrite(dark_path, dark_image):
            raise IOError(f"Failed to write dark-frame image to {dark_path}")
        print(f"Dark frame saved to: {dark_path}")

        print("\nFlat-field acquisition complete.")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        if camera is not None:
            close_camera(camera)


if __name__ == "__main__":
    main()
