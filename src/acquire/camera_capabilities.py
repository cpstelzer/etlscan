from __future__ import annotations



def check_camera_is_color():
    from pypylon import pylon
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

        print(f"CAMERA CHECK: Found '{camera.GetDeviceInfo().GetModelName()}'. Current format: '{camera.PixelFormat.GetValue()}'.")
        print(f"CAMERA CHECK: Sensor is {'COLOR (Bayer capable)' if is_color else 'MONOCHROME'}.")

        camera.Close()
        return is_color

    except Exception as e:
        print(f"CAMERA CHECK: Error while checking camera sensor type: {e}")
        if camera and camera.IsOpen():
            camera.Close()
        return False
