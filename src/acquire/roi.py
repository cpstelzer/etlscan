from __future__ import annotations

from typing import Callable


def set_centered_roi(camera, roi_width: int, roi_height: int, *, emit: Callable[[str], None] = print):
    emit(f"\n--- Setting Centered ROI ({roi_width}x{roi_height}) ---")
    try:
        sensor_width = camera.WidthMax.Value
        sensor_height = camera.HeightMax.Value
        width_inc = camera.Width.Inc if hasattr(camera.Width, 'Inc') else 1
        height_inc = camera.Height.Inc if hasattr(camera.Height, 'Inc') else 1
        offset_x_inc = camera.OffsetX.Inc if hasattr(camera.OffsetX, 'Inc') else 1
        offset_y_inc = camera.OffsetY.Inc if hasattr(camera.OffsetY, 'Inc') else 1
        min_width = camera.Width.Min if hasattr(camera.Width, 'Min') else 1
        min_height = camera.Height.Min if hasattr(camera.Height, 'Min') else 1

        emit(f"Sensor Max Dimensions: {sensor_width}x{sensor_height}")
        emit(f"Parameter Increments: W({width_inc}), H({height_inc}), Ox({offset_x_inc}), Oy({offset_y_inc})")

        if not (min_width <= roi_width <= sensor_width):
            raise ValueError(f"Requested roi_width {roi_width} out of range [{min_width}-{sensor_width}]")
        if not (min_height <= roi_height <= sensor_height):
            raise ValueError(f"Requested roi_height {roi_height} out of range [{min_height}-{sensor_height}]")

        if roi_width % width_inc != 0:
            original_width = roi_width
            roi_width = (roi_width // width_inc) * width_inc
            emit(f"Warning: Requested roi_width {original_width} not multiple of {width_inc}. Adjusted to {roi_width}.")
            if roi_width < min_width:
                raise ValueError(f"Adjusted roi_width {roi_width} is below minimum {min_width}.")
        if roi_height % height_inc != 0:
            original_height = roi_height
            roi_height = (roi_height // height_inc) * height_inc
            emit(f"Warning: Requested roi_height {original_height} not multiple of {height_inc}. Adjusted to {roi_height}.")
            if roi_height < min_height:
                raise ValueError(f"Adjusted roi_height {roi_height} is below minimum {min_height}.")

        emit("Attempting to reset ROI to maximum before setting new ROI...")
        try:
            camera.OffsetX.Value = camera.OffsetX.Min
            camera.OffsetY.Value = camera.OffsetY.Min
            camera.Width.Value = sensor_width
            camera.Height.Value = sensor_height
            emit(
                f"  Reset successful: W={camera.Width.Value}, H={camera.Height.Value}, "
                f"Ox={camera.OffsetX.Value}, Oy={camera.OffsetY.Value}"
            )
        except Exception as e:  # noqa: BLE001
            emit(f"  Warning: Could not fully reset ROI. Error: {e}. Proceeding...")

        offset_x = (sensor_width - roi_width) // 2
        offset_y = (sensor_height - roi_height) // 2
        offset_x = (offset_x // offset_x_inc) * offset_x_inc
        offset_y = (offset_y // offset_y_inc) * offset_y_inc

        emit(f"Setting centered ROI: {roi_width}x{roi_height} with Offset ({offset_x},{offset_y})")
        camera.Width.Value = roi_width
        camera.Height.Value = roi_height
        emit(f"  Actual Width set: {camera.Width.Value}, Actual Height set: {camera.Height.Value}")
        camera.OffsetX.Value = offset_x
        camera.OffsetY.Value = offset_y
        emit(f"  Actual OffsetX set: {camera.OffsetX.Value}, Actual OffsetY set: {camera.OffsetY.Value}")
        emit("ROI successfully configured.")
        return camera.Width.Value, camera.Height.Value
    except (ValueError, AttributeError) as e:
        emit(f"ERROR setting ROI: {type(e).__name__} - {e}")
        raise


def reset_roi_to_full(camera, *, emit: Callable[[str], None] = print):
    emit("\n--- Resetting ROI to Full Sensor Size ---")
    if not camera or not camera.IsOpen():
        emit("Warning: Camera not open or available for ROI reset.")
        return
    try:
        min_offset_x = camera.OffsetX.Min
        min_offset_y = camera.OffsetY.Min
        max_width = camera.WidthMax.Value
        max_height = camera.HeightMax.Value
        camera.OffsetX.Value = min_offset_x
        camera.OffsetY.Value = min_offset_y
        emit(f"  Offsets reset to: Ox={camera.OffsetX.Value}, Oy={camera.OffsetY.Value}")
        camera.Width.Value = max_width
        camera.Height.Value = max_height
        emit(f"  Size reset to: W={camera.Width.Value}, H={camera.Height.Value}")
        emit("ROI successfully reset to full.")
    except AttributeError as e:
        emit(f"ERROR resetting ROI: {type(e).__name__} - {e}")
    except Exception as e:  # noqa: BLE001
        emit(f"Unexpected ERROR resetting ROI: {type(e).__name__} - {e}")

