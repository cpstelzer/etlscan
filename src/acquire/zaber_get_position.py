# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:30:40 2025
Reads current position of ZaberXY-table
(for troubleshooting; run as `python -m src.acquire.zaber_get_position`)
@author: Stelzer Lab
"""

import zaber.serial

from src.acquire.stage_controller import STAGE_CONTROLLER_CONFIG_KEYS
from src.utils.config_loader import load_config_once, require_config_sections


def main():
    acq_conf = load_config_once()['acquisition']
    require_config_sections(acq_conf, {'stage_controller': STAGE_CONTROLLER_CONFIG_KEYS}, 'acquisition')
    stage_config = acq_conf['stage_controller']
    port = zaber.serial.BinarySerial(
        stage_config['serial_port'], baud=stage_config['baud_rate'], timeout=stage_config['serial_timeout_sec']
    )

    x_axis = zaber.serial.BinaryDevice(port, stage_config['device_x_number'])
    y_axis = zaber.serial.BinaryDevice(port, stage_config['device_y_number'])

    # Current positions in microsteps, converted to mm with the profile's microstep size.
    x_position_steps = x_axis.get_position()
    y_position_steps = y_axis.get_position()
    microstep_size_um = stage_config['microstep_size_um']
    x_position_mm = x_position_steps * microstep_size_um / 1000
    y_position_mm = y_position_steps * microstep_size_um / 1000

    print(f"X position: {x_position_mm} mm")
    print(f"Y position: {y_position_mm} mm")


if __name__ == "__main__":
    main()
