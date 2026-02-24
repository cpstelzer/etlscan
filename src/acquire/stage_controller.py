from __future__ import annotations

import time
from math import isclose



class StageController:
    def __init__(self, serial_port, device_x_number, device_y_number, device_x_microstep_size, device_y_microstep_size,
                 position_tolerance):
        self.serial_port = serial_port
        self.device_x_number = device_x_number
        self.device_y_number = device_y_number
        self.device_x_microstep_size = device_x_microstep_size
        self.device_y_microstep_size = device_y_microstep_size
        self.position_tolerance = position_tolerance

    def initialize(self):
        from zaber.serial import BinarySerial, BinaryDevice, BinaryCommand
        from zaber.serial.unexpectedreplyerror import UnexpectedReplyError

        print(f"Initializing Zaber stage on port {self.serial_port}...")
        port = None
        try:
            port = BinarySerial(self.serial_port, baud=115200, timeout=60)
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
            device_x = BinaryDevice(port, self.device_x_number)
            print(f"  X-axis device (ID {self.device_x_number}) instantiated.")
            device_y = BinaryDevice(port, self.device_y_number)
            print(f"  Y-axis device (ID {self.device_y_number}) instantiated.")
            print("  Performing initial homing sequence...")
            self.home_device(device_x, "X", port)
            self.home_device(device_y, "Y", port)
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

    def home_device(self, device, device_name_str, port_obj):
        from zaber.serial.unexpectedreplyerror import UnexpectedReplyError

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

    def _move_axis_if_needed(self, device, target_steps, last_value, axis_name, port_obj):
        from zaber.serial.unexpectedreplyerror import UnexpectedReplyError

        if last_value is None or not isclose(last_value, target_steps, abs_tol=self.position_tolerance):
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

        print(f"  Zaber {axis_name}-axis already at {target_steps} microsteps (within tolerance); skipping move.")
        port_obj.flush()
        return last_value

    def move_to(self, x_mm, y_mm, last_x_steps, last_y_steps, dev_x, dev_y, port_obj):
        x_target_s = int(x_mm * 1000 / self.device_x_microstep_size)
        y_target_s = int(y_mm * 1000 / self.device_y_microstep_size)
        print(f"\nMoving Zaber stage to X = {x_mm:.2f} mm ({x_target_s} steps), Y = {y_mm:.2f} mm ({y_target_s} steps)")

        new_lx = self._move_axis_if_needed(dev_x, x_target_s, last_x_steps, "X", port_obj)
        time.sleep(0.2)
        new_ly = self._move_axis_if_needed(dev_y, y_target_s, last_y_steps, "Y", port_obj)

        print("  Waiting for Zaber stage to settle at position...")
        time.sleep(2)
        port_obj.flush()
        return new_lx, new_ly
