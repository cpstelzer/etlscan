# src/acquire/hardware_setup.py
"""Connect and configure the camera and the ETL controller: the one implementation.

Used by the batch acquisition (``wellscanner_acquisition_module.run_acquisition``), the
realtime camera source (``src/realtime/acquisition.py``), the flat-field tool
(``ws_flat_field.py``) and the diagnosis script, so that a recording, a realtime session
and a calibration image are taken with the same camera settings.

The vendor modules (``pypylon.pylon``, ``optoICC``) are passed in by the caller and never
imported here: the realtime mode imports them only when a camera session starts, and the
tests substitute simulated ones. Progress messages go to ``emit`` (``print`` for the
interactive acquisition scripts, a logger method for the realtime session).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2

from .roi import reset_roi_to_full, set_centered_roi

Emit = Callable[[str], None]

# Debayering code per Bayer pixel format the camera may deliver.
BAYER_TO_CV2_CODE = {
    'BayerRG8': cv2.COLOR_BAYER_RG2BGR,
    'BayerBG8': cv2.COLOR_BAYER_BG2BGR,
    'BayerGB8': cv2.COLOR_BAYER_GB2BGR,
    'BayerGR8': cv2.COLOR_BAYER_GR2BGR,
    'BayerRG12': cv2.COLOR_BAYER_RG2BGR,
    'BayerBG12': cv2.COLOR_BAYER_BG2BGR,
    'BayerGB12': cv2.COLOR_BAYER_GB2BGR,
    'BayerGR12': cv2.COLOR_BAYER_GR2BGR,
}


@dataclass(frozen=True)
class CameraSetup:
    """What :func:`configure_camera` set on the camera."""

    frame_width: int
    frame_height: int
    exposure_us: float
    pixel_format: str
    process_as_color: bool          # False when color was requested but the camera delivers mono
    debayer_code: Optional[int]     # cv2 conversion code of the Bayer format, None for mono


# --------------------------------------------------------------------------- camera
def open_first_camera(pylon: Any, emit: Emit = print) -> Any:
    """Open the first pylon camera found and return it.

    Raises:
        ConnectionError: no camera is connected.
    """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise ConnectionError("No Pylon compatible camera found.")
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    emit(f"Connected to camera: {camera.GetDeviceInfo().GetModelName()}")
    return camera


def select_pixel_format(
    camera: Any, pixel_format_preference: Sequence[str], process_as_color: bool, emit: Emit = print
) -> str:
    """Set the first format of ``pixel_format_preference`` that fits the request and the camera.

    Color needs a Bayer format, monochrome a Mono format; the order of
    ``acquisition.camera.pixel_format_preference`` decides among the formats the camera
    offers. Returns the format that is active afterwards.

    Raises:
        ValueError: color was requested and the camera offers none of the preferred Bayer formats.
    """
    wanted_kind = "Bayer" if process_as_color else "Mono"
    current_format = camera.PixelFormat.GetValue()
    available_formats = list(camera.PixelFormat.Symbolics)
    for preferred_format in pixel_format_preference:
        if not preferred_format.startswith(wanted_kind) or preferred_format not in available_formats:
            continue
        if preferred_format == current_format:
            emit(f"  Camera is already in the preferred {wanted_kind} format: {current_format}")
            return current_format
        try:
            camera.PixelFormat.SetValue(preferred_format)
            emit(f"  Successfully set Pixel Format to: {preferred_format} (was {current_format})")
            return preferred_format
        except Exception as e:  # noqa: BLE001 - try the next preferred format
            emit(f"  Warning: Could not set format to {preferred_format}: {e}")
    if process_as_color:
        emit("ERROR: Failed to set a Bayer format on the camera. Cannot proceed with color acquisition.")
        raise ValueError("Could not configure camera for Bayer pixel format.")
    emit(f"  Warning: Could not switch camera to a preferred Mono format. Will proceed with {current_format}.")
    return camera.PixelFormat.GetValue()


def _node_limit(node: Any, name: str) -> Optional[float]:
    """Min / Max / Inc of a camera node, or None when the node does not have it."""
    try:
        value = getattr(node, name)
    except Exception:  # noqa: BLE001 - pypylon raises its own error for an absent limit
        return None
    return None if value is None else float(value)


def set_exposure(camera: Any, exposure_us: float, emit: Emit = print) -> float:
    """Set the exposure time, clamped to the camera's range and increment; return the active value."""
    exposure_to_set = exposure_us
    try:
        min_exposure = _node_limit(camera.ExposureTime, 'Min')
        max_exposure = _node_limit(camera.ExposureTime, 'Max')
        increment = _node_limit(camera.ExposureTime, 'Inc')
        if min_exposure is not None and exposure_to_set < min_exposure:
            emit(f"  Warning: Requested exposure {exposure_to_set}us < min {min_exposure}us. Clamping.")
            exposure_to_set = min_exposure
        if max_exposure is not None and exposure_to_set > max_exposure:
            emit(f"  Warning: Requested exposure {exposure_to_set}us > max {max_exposure}us. Clamping.")
            exposure_to_set = max_exposure
        if increment is not None and increment > 0:
            base = min_exposure if min_exposure is not None else 0.0
            on_increment = base + round((exposure_to_set - base) / increment) * increment
            if not math.isclose(on_increment, exposure_to_set):
                emit(f"  Warning: Requested exposure {exposure_to_set}us is not on the camera's "
                     f"{increment}us increment. Using {on_increment}us.")
                exposure_to_set = on_increment
        camera.ExposureTime.SetValue(exposure_to_set)
        actual_exposure_us = camera.ExposureTime.GetValue()
        emit(f"  Exposure Time set to: {actual_exposure_us} us (Requested: {exposure_us} us)")
    except Exception as e:  # noqa: BLE001 - the recording goes on with the exposure the camera has
        actual_exposure_us = camera.ExposureTime.GetValue() if camera.ExposureTime.IsReadable() else exposure_us
        emit(f"Warning: Could not set exposure time: {e}. Using current/config: {actual_exposure_us} us")
    return actual_exposure_us


def set_buffer_count(camera: Any, max_num_buffer_min: int, frame_rate_fps: float, emit: Emit = print) -> None:
    """Give the camera at least ``max_num_buffer_min`` buffers, and one second's worth of frames."""
    try:
        camera.MaxNumBuffer = max(max_num_buffer_min, int(frame_rate_fps))
        emit(f"  MaxNumBuffer set to: {camera.MaxNumBuffer.GetValue()}")
    except Exception as e:  # noqa: BLE001 - the camera keeps its buffer count
        current = camera.MaxNumBuffer.GetValue() if camera.MaxNumBuffer.IsReadable() else 'N/A'
        emit(f"Warning: Could not set MaxNumBuffer: {e}. Using default: {current}")


def configure_camera(
    camera: Any,
    acq_conf: Dict[str, Any],
    process_as_color: bool,
    frame_rate_fps: float,
    emit: Emit = print,
) -> CameraSetup:
    """Set pixel format, exposure, buffer count and the centred ROI from the acquisition config.

    ``frame_rate_fps`` is the rate the caller will run at (it sizes the buffer queue); the
    frame rate itself is set with :func:`set_frame_rate` or by the trigger loop.

    Raises:
        ValueError: no usable Bayer format for a color run, or an ROI the sensor cannot deliver.
    """
    camera_config = acq_conf['camera']
    emit("Configuring camera settings (Pixel Format, Exposure, Buffers)...")
    pixel_format = select_pixel_format(camera, camera_config['pixel_format_preference'], process_as_color, emit)
    emit(f"  Final active Pixel Format is: {pixel_format}")

    debayer_code = None
    if process_as_color:
        if pixel_format.startswith("Bayer"):
            debayer_code = BAYER_TO_CV2_CODE.get(pixel_format)
            if debayer_code is None:
                raise ValueError(
                    f"Unsupported Bayer format '{pixel_format}' for color conversion. Check BAYER_TO_CV2_CODE map."
                )
            emit(f"  Will use debayering for '{pixel_format}'.")
        else:
            emit("WARNING: Color processing was requested, but the final camera format is not Bayer. "
                 "Images will be monochrome.")
            process_as_color = False

    exposure_us = set_exposure(camera, camera_config['exposure_us'], emit)
    set_buffer_count(camera, camera_config['max_num_buffer_min'], frame_rate_fps, emit)

    try:
        frame_width, frame_height = set_centered_roi(
            camera, acq_conf['roi']['width'], acq_conf['roi']['height'], emit=emit
        )
        emit(f"ROI set. Actual acquisition dimensions: {frame_width}x{frame_height}")
    except Exception as e:
        emit(f"ERROR: Failed to set camera ROI. Error: {e}")
        raise

    return CameraSetup(
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        exposure_us=exposure_us,
        pixel_format=pixel_format,
        process_as_color=process_as_color,
        debayer_code=debayer_code,
    )


def set_frame_rate(camera: Any, desired_fps: float, emit: Emit = print) -> float:
    """Set the free-run frame rate, limited to what the camera can deliver; return the active rate.

    Raises:
        ValueError: the camera cannot deliver a positive frame rate.
    """
    emit("Configuring camera frame rate...")
    max_possible_fps = None
    try:
        camera.AcquisitionFrameRateEnable.SetValue(True)
        emit(f"  Attempting to set desired FPS: {desired_fps:.2f}")
        try:
            if hasattr(camera, "AcquisitionFrameRate") and camera.AcquisitionFrameRate.HasMax:
                max_possible_fps = camera.AcquisitionFrameRate.Max
                emit(f"  Camera's AcquisitionFrameRate.Max: {max_possible_fps:.2f} FPS")
            elif hasattr(camera, "ResultingFrameRate") and camera.ResultingFrameRate.IsReadable():
                max_possible_fps = camera.ResultingFrameRate.Value
                emit(f"  Camera's current ResultingFrameRate: {max_possible_fps:.2f} FPS")
            else:
                emit(f"  Warning: Cannot directly query max FPS. Will attempt to set desired FPS "
                     f"({desired_fps:.2f}) and read back.")
        except Exception as e_fps_query:  # noqa: BLE001 - set the desired rate and read it back
            emit(f"  Warning: Error querying max FPS capabilities: {e_fps_query}. Proceeding cautiously.")
            max_possible_fps = None

        fps_to_set = desired_fps
        if max_possible_fps is not None and desired_fps > max_possible_fps:
            fps_to_set = math.floor(max_possible_fps)
            emit(f"Warning: Desired FPS ({desired_fps:.2f}) exceeds reported/estimated max possible "
                 f"({max_possible_fps:.2f}).")
            emit(f"  Adjusting FPS to {fps_to_set:.2f} FPS.")
            if fps_to_set <= 0:
                raise ValueError(
                    f"The camera reports a maximum frame rate of {max_possible_fps:.2f} FPS with these "
                    "settings: no whole frame rate is possible."
                )
        camera.AcquisitionFrameRate.SetValue(fps_to_set)
        final_fps = camera.AcquisitionFrameRate.GetValue()
        emit(f"  Final Acquisition Frame Rate set to: {final_fps:.2f} FPS")
        if abs(final_fps - desired_fps) > 0.1 and (max_possible_fps is None or desired_fps <= max_possible_fps):
            emit(f"  Note: Final FPS ({final_fps:.2f}) differs slightly from desired ({desired_fps:.2f}) "
                 "due to camera constraints.")
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - keep the rate the camera has, if it has one
        final_fps = camera.AcquisitionFrameRate.GetValue() if camera.AcquisitionFrameRate.IsReadable() else 0
        emit(f"Warning: Could not configure frame rate: {e}. Using current value: {final_fps:.2f} FPS")
        if camera.AcquisitionFrameRateEnable.IsReadable() and not camera.AcquisitionFrameRateEnable.GetValue():
            emit("  Note: Frame rate control might be disabled on camera.")

    if final_fps <= 0:
        emit("ERROR: Final FPS is zero or negative. Cannot proceed.")
        raise ValueError("Final FPS is not positive.")
    return final_fps


def close_camera(camera: Any, emit: Emit = print, restore_full_roi: bool = True) -> None:
    """Stop grabbing, optionally reset the ROI to the full sensor, and close the camera."""
    if camera is None:
        emit("Camera object not created or connection failed.")
        return
    if camera.IsGrabbing():
        emit("Stopping camera grabbing...")
        try:
            camera.StopGrabbing()
        except Exception as e:  # noqa: BLE001 - go on closing
            emit(f"  Error stopping camera grabbing: {e}")
    if not camera.IsOpen():
        emit("Camera was not open, skipping close related actions.")
        return
    if restore_full_roi:
        try:
            emit("Attempting to reset ROI to full sensor size before closing camera...")
            reset_roi_to_full(camera)
        except Exception as e_reset_roi:  # noqa: BLE001 - go on closing
            emit(f"  Error resetting ROI to full: {e_reset_roi}")
    emit("Closing camera...")
    try:
        camera.Close()
    except Exception as e:  # noqa: BLE001 - nothing left to do with this camera
        emit(f"  Error closing camera: {e}")
    else:
        emit("  Camera closed.")


# --------------------------------------------------------------------------- lens
def connect_lens(optoICC: Any, port: str, emit: Emit = print) -> Tuple[Any, Any]:
    """Connect to the ECC-1C lens controller on ``port``; return (controller, lens).

    Raises:
        ConnectionError: the controller reports no lens.
        optoICC.NoEccError: no controller on that port.
    """
    emit("Connecting to liquid lens controller...")
    ecc1c = optoICC.connectEcc(port=port)
    lens = ecc1c.Lens
    if lens is None:
        raise ConnectionError("Failed to get valid Lens object from ECC-1C.")
    emit(f"Connected to ECC-1C on {port}.")
    return ecc1c, lens


def prefocus_lens(lens: Any, start_dpt: float, settle_sec: float, emit: Emit = print) -> None:
    """Drive the lens to the first focal plane and wait ``settle_sec`` before the recording starts."""
    emit(f"\n--- Pre-focusing lens to starting position: {start_dpt:.3f} dpt ---")
    lens.StaticInput.SetAsInput()
    lens.StaticInput.SetFocalPower(start_dpt)
    emit(f"Waiting {settle_sec}s for lens to settle...")
    time.sleep(settle_sec)
    emit("Lens pre-focused and settled.")


def disconnect_lens(ecc1c: Any, emit: Emit = print) -> None:
    """Disconnect from the lens controller (no-op for None)."""
    if ecc1c is None:
        emit("ECC-1C controller object not created or connection failed.")
        return
    emit("Disconnecting from ECC-1C...")
    try:
        if hasattr(ecc1c, 'disconnect') and callable(ecc1c.disconnect):
            ecc1c.disconnect()
        elif hasattr(ecc1c, 'close') and callable(ecc1c.close):
            ecc1c.close()
        else:
            emit("  Warning: ecc1c object lacks known 'disconnect()' or 'close()' method.")
        emit("  Disconnected from ECC-1C.")
    except Exception as e:  # noqa: BLE001 - nothing left to do with this controller
        emit(f"  Error during ECC-1C disconnection: {e}")
