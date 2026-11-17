"""The shared camera / lens setup (src/acquire/hardware_setup.py) and the stage controller's
config, on the simulated devices of tests/fixtures/etl_mock (no rig needed)."""
import importlib
import os
import sys
from pathlib import Path

import pytest

from src.acquire import hardware_setup
from src.acquire.focus_pattern import build_focus_cycle_pattern
from src.acquire.stage_controller import STAGE_CONTROLLER_CONFIG_KEYS, StageController
from src.utils.common_exceptions import ConfigError
from src.utils.config_loader import load_config_once

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = PROJECT_ROOT / "tests" / "fixtures" / "etl_mock"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"
_MOCK_MODULES = ("sim_hw", "optoICC", "pypylon", "pypylon.pylon")


@pytest.fixture()
def sim():
    """The simulated pypylon.pylon and optoICC modules (the real ones are restored afterwards)."""
    saved_modules = {name: sys.modules.pop(name) for name in _MOCK_MODULES if name in sys.modules}
    sys.path.insert(0, str(MOCK_DIR))
    try:
        yield importlib.import_module("pypylon.pylon"), importlib.import_module("optoICC")
    finally:
        sys.path.remove(str(MOCK_DIR))
        for name in _MOCK_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


@pytest.fixture()
def acq_conf():
    load_config_once.cache_clear()
    yield load_config_once(str(CONFIG_PATH))["acquisition"]
    load_config_once.cache_clear()


def _messages():
    lines = []
    return lines, lines.append


# --- camera ---------------------------------------------------------------------

def test_open_first_camera_opens_it(sim):
    pylon, _ = sim
    lines, emit = _messages()

    camera = hardware_setup.open_first_camera(pylon, emit=emit)

    assert camera.IsOpen()
    assert any("Connected to camera" in line for line in lines)


def test_configure_camera_applies_format_exposure_buffers_and_roi(sim, acq_conf):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)
    camera.PixelFormat.SetValue("Mono12")          # left over from another program
    acq_conf["camera"]["exposure_us"] = 900
    acq_conf["roi"] = {"width": 2000, "height": 1000}

    setup = hardware_setup.configure_camera(camera, acq_conf, False, 25.0, emit=lambda line: None)

    # Mono8 precedes Mono12 in acquisition.camera.pixel_format_preference.
    assert setup.pixel_format == "Mono8" and camera.PixelFormat.GetValue() == "Mono8"
    assert setup.exposure_us == 900 and camera.ExposureTime.GetValue() == 900
    assert (setup.frame_width, setup.frame_height) == (2000, 1000)
    assert camera.MaxNumBuffer.GetValue() == max(acq_conf["camera"]["max_num_buffer_min"], 25)
    assert setup.process_as_color is False and setup.debayer_code is None


def test_color_request_on_a_mono_camera_fails(sim, acq_conf):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)

    with pytest.raises(ValueError, match="Bayer"):
        hardware_setup.configure_camera(camera, acq_conf, True, 25.0, emit=lambda line: None)


@pytest.mark.parametrize("requested_us, expected_us", [(5.0, 20.0), (5e6, 1e6), (750, 750)])
def test_exposure_is_clamped_to_the_camera_range(sim, requested_us, expected_us):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)

    assert hardware_setup.set_exposure(camera, requested_us, emit=lambda line: None) == expected_us


def test_exposure_is_set_on_the_camera_increment(sim):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)
    camera.ExposureTime.Inc = 25.0                 # range starts at 20 us: 20, 45, 70, ...

    assert hardware_setup.set_exposure(camera, 750, emit=lambda line: None) == pytest.approx(745.0)


def test_frame_rate_above_the_camera_maximum_is_reduced(sim):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)   # simulated maximum: 27 fps

    assert hardware_setup.set_frame_rate(camera, 25.0, emit=lambda line: None) == 25.0
    assert hardware_setup.set_frame_rate(camera, 40.0, emit=lambda line: None) == 27.0


def test_frame_rate_the_camera_cannot_deliver_fails(sim):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)
    camera.AcquisitionFrameRate.Max = 0.4          # e.g. a very long exposure

    with pytest.raises(ValueError, match="maximum frame rate"):
        hardware_setup.set_frame_rate(camera, 25.0, emit=lambda line: None)


def test_close_camera_closes_and_tolerates_none(sim):
    pylon, _ = sim
    camera = hardware_setup.open_first_camera(pylon, emit=lambda line: None)

    hardware_setup.close_camera(camera, emit=lambda line: None)
    hardware_setup.close_camera(None, emit=lambda line: None)

    assert not camera.IsOpen()


# --- lens -----------------------------------------------------------------------

def test_lens_is_connected_prefocused_and_disconnected(sim, monkeypatch):
    _, optoICC = sim
    sim_hw = sys.modules["sim_hw"]
    waited = []
    monkeypatch.setattr(hardware_setup.time, "sleep", waited.append)

    ecc1c, lens = hardware_setup.connect_lens(optoICC, "COM11", emit=lambda line: None)
    hardware_setup.prefocus_lens(lens, -1.25, 0.7, emit=lambda line: None)
    hardware_setup.disconnect_lens(ecc1c, emit=lambda line: None)
    hardware_setup.disconnect_lens(None, emit=lambda line: None)

    assert sim_hw.LENS.last_cmd == -1.25
    assert waited == [0.7]                         # acquisition.focusing.prefocus_settle_sec, not a literal


def test_unknown_focus_pattern_fails():
    with pytest.raises(ValueError, match="Unknown focus pattern"):
        build_focus_cycle_pattern(-2.0, "spiral", 3.2, 200, 3.36)


# --- XY stage -------------------------------------------------------------------

def test_stage_controller_takes_every_setting_from_the_profile(acq_conf):
    stage_config = acq_conf["stage_controller"]

    controller = StageController.from_config(stage_config)

    assert controller.serial_port == stage_config["serial_port"]
    assert controller.device_x_microstep_size == controller.device_y_microstep_size == stage_config["microstep_size_um"]
    assert controller.position_tolerance == stage_config["position_tolerance_microsteps"]
    assert (controller.baud_rate, controller.serial_timeout_sec) == (stage_config["baud_rate"], stage_config["serial_timeout_sec"])
    assert (controller.renumber_wait_sec, controller.homing_wait_sec, controller.axis_move_wait_sec,
            controller.between_axes_wait_sec, controller.settle_wait_sec) == tuple(
        stage_config[key] for key in ("renumber_wait_sec", "homing_wait_sec", "axis_move_wait_sec",
                                      "between_axes_wait_sec", "settle_wait_sec"))


@pytest.mark.parametrize("missing_key", STAGE_CONTROLLER_CONFIG_KEYS)
def test_stage_controller_needs_every_profile_key(acq_conf, missing_key):
    stage_config = {key: value for key, value in acq_conf["stage_controller"].items() if key != missing_key}

    with pytest.raises(ConfigError, match=f"stage_hardware_profile.{missing_key}"):
        StageController.from_config(stage_config)


def test_stage_moves_wait_the_configured_times(acq_conf, monkeypatch):
    """move_to sends both axis commands and waits the profile's times in between."""
    from src.acquire import stage_controller as stage_controller_module

    waited = []
    monkeypatch.setattr(stage_controller_module.time, "sleep", waited.append)
    monkeypatch.setitem(sys.modules, "zaber", type(sys)("zaber"))
    monkeypatch.setitem(sys.modules, "zaber.serial", type(sys)("zaber.serial"))
    unexpected_reply = type(sys)("zaber.serial.unexpectedreplyerror")
    unexpected_reply.UnexpectedReplyError = type("UnexpectedReplyError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "zaber.serial.unexpectedreplyerror", unexpected_reply)

    class _Axis:
        def __init__(self, number):
            self.number, self.targets = number, []

        def move_abs(self, steps):
            self.targets.append(steps)

    class _Port:
        def flush(self):
            pass

    stage_config = dict(acq_conf["stage_controller"], axis_move_wait_sec=0.11, between_axes_wait_sec=0.22,
                        settle_wait_sec=0.33)
    controller = StageController.from_config(stage_config)
    axis_x, axis_y = _Axis(1), _Axis(2)

    controller.move_to(1.0, 2.0, None, None, axis_x, axis_y, _Port())

    microstep_um = stage_config["microstep_size_um"]
    assert axis_x.targets == [int(1000 / microstep_um)] and axis_y.targets == [int(2000 / microstep_um)]
    assert waited == [0.11, 0.22, 0.11, 0.33]
