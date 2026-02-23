# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:30:40 2025
Reads current position of ZaberXY-table
(for troubleshooting)
@author: Stelzer Lab
"""

import zaber.serial

def main():
    # Create a BinarySerial connection on COM7 with baud rate 115200
    port = zaber.serial.BinarySerial("COM4", baud=115200, timeout=5)
    
    # Create BinaryDevice instances for X and Y axes
    # Here, we assume device number 1 corresponds to the X axis and 2 to the Y axis.
    x_axis = zaber.serial.BinaryDevice(port, 1)
    y_axis = zaber.serial.BinaryDevice(port, 2)
    
    # Query the current positions (in microsteps)
    x_position_steps = x_axis.get_position()  # Returns integer microsteps for X axis
    y_position_steps = y_axis.get_position()  # Returns integer microsteps for Y axis
    
    # Convert microsteps to micrometers using the given conversion factor
    microstep_size = 0.15625  # micrometers per microstep
    x_position_mm = x_position_steps * microstep_size / 1000
    y_position_mm = y_position_steps * microstep_size / 1000
    
    # Display the positions in micrometers
    print(f"X position: {x_position_mm} mm")
    print(f"Y position: {y_position_mm} mm")

if __name__ == "__main__":
    main()
