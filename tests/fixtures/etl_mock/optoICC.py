"""Fake optoICC for dry runs."""
import time

from sim_hw import LENS, lens_temp


class NoEccError(Exception):
    pass


class _Static:
    def SetAsInput(self):
        pass

    def SetFocalPower(self, dpt):
        LENS.command(float(dpt))


class _Lens:
    def __init__(self):
        self.StaticInput = _Static()


class _TempManager:
    def GetDeviceTemperature(self):
        time.sleep(0.004)        # a serial round trip
        return lens_temp()


class _Ecc:
    def __init__(self):
        self.Lens = _Lens()
        import os
        if not os.environ.get("SIM_NO_TEMP"):
            self.TemperatureManager = _TempManager()

    def disconnect(self):
        pass


def connectEcc(port=None):
    return _Ecc()
