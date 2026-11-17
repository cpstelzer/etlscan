"""Batch acquisition on simulated hardware (tests/fixtures/etl_mock): fixed-grid trigger loop
against the legacy free-run loop.

The mock camera stamps every exposure with ``perf_counter * 1e9 + CLOCK_OFFSET_TICKS`` and
``sim_hw.LENS.commanded_at(t)`` is the lens setting a frame exposed at ``t`` really shows, so
each saved frame's label can be checked against what the camera saw.
"""
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.utils.acquisition_metadata import read_frames_lost_at_acquisition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = PROJECT_ROOT / "tests" / "fixtures" / "etl_mock"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"
_MOCK_MODULES = ("sim_hw", "optoICC", "pypylon", "pypylon.pylon")
_ACQUISITION_MODULE = "src.acquire.wellscanner_acquisition_module"
_FRAME_FILE_RE = re.compile(r"^\d{8}_plate3_well7_f(\d{5})_dpt(-?\d+\.\d{3})_fd(-?\d+\.\d{2})mm\.png$")
_LEGACY_COLUMNS = [
    "timestamp", "frame_count", "pylon_frame_id", "skipped_frame_warning",
    "diopter", "focusing_distance_mm", "lens_set_time_ms", "lens_settle_wait_ms",
    "frame_grab_time_ms", "image_processing_time_ms", "image_save_submit_time_ms",
    "total_loop_iteration_time_ms", "output_filename",
]
_TIMING_COLUMNS = [
    "grid_time_s", "lens_cmd_start_s", "lens_cmd_end_s", "trigger_time_s", "trigger_late_ms",
    "block_id", "camera_timestamp_ns", "exposure_minus_trigger_ms", "exposure_minus_lens_cmd_ms",
    "lens_temp_c",
]


@pytest.fixture()
def hw():
    """Simulated pypylon + optoICC at the front of sys.path, acquisition module imported against it."""
    import src.acquire as acquire_package
    from src.utils.config_loader import load_config_once

    # acquisition load_config converts values of the cached config in place (font, grab
    # strategy of the simulated pylon): give every test, and the tests after these, a fresh one
    load_config_once.cache_clear()
    names = (*_MOCK_MODULES, _ACQUISITION_MODULE)
    saved_modules = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    saved_attribute = acquire_package.__dict__.get("wellscanner_acquisition_module")
    sys.path.insert(0, str(MOCK_DIR))
    try:
        yield SimpleNamespace(sim_hw=importlib.import_module("sim_hw"),
                              pylon=importlib.import_module("pypylon.pylon"),
                              module=importlib.import_module(_ACQUISITION_MODULE))
    finally:
        load_config_once.cache_clear()
        sys.path.remove(str(MOCK_DIR))
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_attribute is None:
            acquire_package.__dict__.pop("wellscanner_acquisition_module", None)
        else:
            acquire_package.wellscanner_acquisition_module = saved_attribute


def _config(hw, tmp_path, trigger_mode, pattern="zigzag", duration_s=1.6):
    acq_config = hw.module.load_config(str(CONFIG_PATH))
    acq_config["paths"]["output_folder_base"] = str(tmp_path)
    acq_config["camera"]["video_duration_sec"] = duration_s
    acq_config["camera"]["trigger_mode"] = trigger_mode
    acq_config["focusing"]["pattern"] = pattern
    acq_config["focusing"]["scan_step_um"] = 550       # 26 planes over the 14 mm sweep
    return acq_config


def _expected_pattern(hw, acq_config, start_dpt):
    from src.acquire.focus_pattern import build_focus_cycle_pattern

    focusing = acq_config["focusing"]
    cycle = build_focus_cycle_pattern(start_dpt, focusing["pattern"], focusing["scan_height_mm"],
                                      focusing["scan_step_um"], focusing["scan_ratio"])
    n_frames = int(acq_config["camera"]["video_duration_sec"] * acq_config["camera"]["fps"])
    return (cycle * (n_frames // len(cycle) + 1))[:n_frames]


def _read_metadata(run_folder):
    with open(os.path.join(run_folder, "metadata_log.csv"), encoding="utf-8") as log_file:
        lines = log_file.read().splitlines()
    n_header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    header = {}
    for line in lines[:n_header]:
        key, sep, value = line[1:].strip().partition(":")
        if sep and not line.startswith("# ==="):
            header[key.strip()] = value.strip()
    frame_log = pd.read_csv(os.path.join(run_folder, "metadata_log.csv"), skiprows=n_header)
    return header, frame_log


def _saved_frames(run_folder):
    frames = {}
    for name in os.listdir(run_folder):
        match = _FRAME_FILE_RE.match(name)
        if match:
            frames[int(match.group(1))] = float(match.group(2))
    return frames


def _shown_setting(hw, camera_timestamp_ns):
    """Lens setting the frame really shows (mock tick frequency: 1 GHz)."""
    exposure_s = (int(camera_timestamp_ns) - hw.sim_hw.CLOCK_OFFSET_TICKS) / 1e9
    return hw.sim_hw.LENS.commanded_at(exposure_s)


def _record(hw, tmp_path, trigger_mode, start_dpt=-2.0, **kwargs):
    acq_config = _config(hw, tmp_path, trigger_mode, **kwargs)
    result = hw.module.run_acquisition(3, 7, start_dpt, acq_config)
    run_folder = next(str(p) for p in Path(tmp_path).iterdir() if p.name.endswith("_plate3_well7"))
    header, frame_log = _read_metadata(run_folder)
    return SimpleNamespace(result=result, config=acq_config, folder=run_folder, header=header,
                           log=frame_log, frames=_saved_frames(run_folder),
                           pattern=_expected_pattern(hw, acq_config, start_dpt))


def test_grid_mode_frame_shows_its_own_setting(hw, tmp_path, capsys):
    run = _record(hw, tmp_path, "software_grid")
    n_frames = len(run.pattern)

    assert run.result["status"] == "success" and run.result["any_frames_skipped"] is False
    # one saved frame per pattern entry, frame 0 included, file names unchanged
    assert sorted(run.frames) == list(range(n_frames))
    assert [run.frames[i] for i in range(n_frames)] == pytest.approx(run.pattern)
    # every frame shows the setting it is labelled with (no lag)
    for _, row in run.log.iterrows():
        assert _shown_setting(hw, row.camera_timestamp_ns) == pytest.approx(run.frames[row.frame_count])
    # contiguous block IDs, one per entry
    assert run.log.block_id.diff().dropna().eq(1).all()
    assert (run.log.pylon_frame_id == run.log.block_id).all()
    # each exposure starts at least min_settle_ms after its own lens command returned
    assert run.log.exposure_minus_lens_cmd_ms.min() >= run.config["focusing"]["min_settle_ms"] - 0.5
    assert run.log.trigger_late_ms.notna().all() and run.log.grid_time_s.notna().all()
    assert (run.log.grid_time_s.diff().dropna() - 0.04).abs().max() < 2e-6
    # lens temperature once per second (25 frames at 25 fps)
    assert list(run.log.index[run.log.lens_temp_c.notna()]) == list(range(0, n_frames, 25))

    assert list(run.log.columns) == _LEGACY_COLUMNS + _TIMING_COLUMNS
    assert run.header["Trigger Mode"] == "software_grid"
    assert run.header["Min Settle (ms)"] == "10"
    assert {"Lens Settle Time (s)", "Lens Temperature Start (C)", "Lens Temperature End (C)"} <= set(run.header)
    out = capsys.readouterr().out
    assert f"frames received {n_frames}/{n_frames}" in out and "timestamp mismatches 0" in out
    assert "Warning: Loop" not in out          # no per-frame loop warnings


def test_grid_mode_large_jump_delays_only_its_trigger(hw, tmp_path):
    run = _record(hw, tmp_path, "software_grid", pattern="sawtooth", duration_s=1.6)
    flyback = next(i for i in range(1, len(run.pattern)) if run.pattern[i] - run.pattern[i - 1] < -3.0)

    assert sorted(run.frames) == list(range(len(run.pattern)))   # no sacrificial frame
    assert run.log.block_id.diff().dropna().eq(1).all()
    for _, row in run.log.iterrows():
        assert _shown_setting(hw, row.camera_timestamp_ns) == pytest.approx(run.frames[row.frame_count])
    jump = run.log.iloc[flyback]
    large_jump_settle_ms = 1000 * run.config["focusing"]["large_jump_settle_sec"]
    assert 1000 * (jump.trigger_time_s - jump.lens_cmd_end_s) >= large_jump_settle_ms - 0.1
    # the grid keeps its schedule: later triggers are due at t0 + i / fps again
    assert (run.log.grid_time_s.diff().dropna() - 0.04).abs().max() < 2e-6
    assert run.log.trigger_late_ms.iloc[-1] < 2.0


def test_grid_mode_maps_frames_by_block_id_after_a_lost_frame(hw, tmp_path, capsys):
    hw.sim_hw.DROP_BLOCK_IDS.add(10)          # exposure of entry 9 never reaches the PC
    run = _record(hw, tmp_path, "software_grid")

    assert run.result["any_frames_skipped"] is True
    assert sorted(run.frames) == [i for i in range(len(run.pattern)) if i != 9]
    for _, row in run.log.dropna(subset=["camera_timestamp_ns"]).iterrows():
        assert _shown_setting(hw, row.camera_timestamp_ns) == pytest.approx(run.frames[row.frame_count])
    assert run.log.loc[9, "output_filename"] == "Error/NA"
    assert bool(run.log.loc[10, "skipped_frame_warning"]) is True
    # Stage 1 (external BG) reads exactly this frame back as lost at acquisition.
    assert read_frames_lost_at_acquisition(os.path.join(run.folder, "metadata_log.csv")) == {9}
    out = capsys.readouterr().out
    assert f"frames received {len(run.pattern) - 1}/{len(run.pattern)}" in out and "block-ID gaps 1" in out


def test_grid_mode_skipped_trigger_does_not_shift_later_frames(hw, tmp_path, monkeypatch):
    camera_class = hw.pylon.InstantCamera
    original_wait = camera_class.WaitForFrameTriggerReady
    calls = []

    def wait_failing_once(self, timeout_ms, handling=0):
        calls.append(1)
        if len(calls) == 6:
            raise hw.pylon.TimeoutException("simulated: camera not ready")
        return original_wait(self, timeout_ms, handling)

    monkeypatch.setattr(camera_class, "WaitForFrameTriggerReady", wait_failing_once)
    run = _record(hw, tmp_path, "software_grid")

    assert sorted(run.frames) == [i for i in range(len(run.pattern)) if i != 5]
    assert run.log.loc[5, "output_filename"].startswith("ERROR_IN_LOOP")
    assert read_frames_lost_at_acquisition(os.path.join(run.folder, "metadata_log.csv")) == {5}
    assert run.log.block_id.dropna().diff().dropna().eq(1).all()   # no exposure for entry 5
    for _, row in run.log.dropna(subset=["camera_timestamp_ns"]).iterrows():
        assert _shown_setting(hw, row.camera_timestamp_ns) == pytest.approx(run.frames[row.frame_count])


def test_free_run_mode_shows_the_previous_setting(hw, tmp_path):
    """Documents the legacy loop: at 25 fps a frame is exposed before its own lens command
    (the mock's transfer takes 38 ms of the 40 ms frame interval), so it shows the setting
    commanded for the frame before."""
    run = _record(hw, tmp_path, "free_run")
    pattern = run.pattern

    assert sorted(run.frames) == list(range(len(pattern)))
    shown = [_shown_setting(hw, ns) for ns in run.log.camera_timestamp_ns]
    changed = [i for i in range(1, len(pattern)) if pattern[i] != pattern[i - 1]]
    assert all(shown[i] != pytest.approx(pattern[i]) for i in changed)          # never its own setting
    lag_one = sum(shown[i] == pytest.approx(pattern[i - 1]) for i in changed)
    assert lag_one >= 0.9 * len(changed)
    assert run.header["Trigger Mode"] == "free_run"
    assert "Min Settle (ms)" not in run.header
    assert run.log.exposure_minus_lens_cmd_ms.iloc[1:].median() < 4.0       # mock lens settles in 4 ms
    assert run.log.trigger_time_s.isna().all() and run.log.lens_temp_c.isna().all()


def test_diagnosis_production_test_sees_every_lens_command_and_grab(hw, tmp_path):
    """run_acquire_diagnosis.Run.production wraps acquire_and_save_frames with logging proxies:
    with the grid loop they still log one SetFocalPower and one successful RetrieveResult per
    frame, and the frames carry no lag."""
    spec = importlib.util.spec_from_file_location("run_acquire_diagnosis",
                                                  PROJECT_ROOT / "scripts" / "run_acquire_diagnosis.py")
    diagnosis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnosis)
    acq_config = hw.module.load_config(str(CONFIG_PATH))
    ecc = sys.modules["optoICC"].connectEcc(port=None)
    camera = hw.pylon.InstantCamera()
    camera.Open()
    exposure_us, width, height = diagnosis.configure_camera(camera, acq_config)
    run = diagnosis.Run(str(tmp_path), "20261021", 3, 7, 0.0, acq_config, camera, ecc.Lens,
                        width, height, exposure_us)
    sequence = diagnosis.toggle_sequence(0.0, 40)

    run.production("prod_toggle_25fps", "toggle", sequence, 25.0, 0.0)

    folder = tmp_path / "prod_toggle_25fps"
    lens_log = pd.read_csv(folder / "proxy_lens.csv")
    grab_log = pd.read_csv(folder / "proxy_grabs.csv")
    grabs = grab_log[grab_log.grab_ok.astype(str) == "True"].reset_index(drop=True)
    assert lens_log.dpt.tolist() == pytest.approx(sequence)
    assert len(grabs) == len(sequence)
    assert [_shown_setting(hw, ticks) for ticks in grabs.cam_ticks] == pytest.approx(sequence)
    header, _ = _read_metadata(str(folder))
    assert header["Trigger Mode"] == "software_grid"
    assert camera.TriggerMode.GetValue() == "Off"          # free-run restored for the next test


# --- recording check and clock mapping (no hardware) ------------------------------

def test_summary_uses_block_id_span_and_expected_steps():
    from src.acquire.triggered_acquisition import summarize_recording

    ticks = [0, 40_000_000, 120_000_000, 160_000_000]            # ns; the third frame follows a discarded one
    summary = summarize_recording(5, [1, 2, 4, 5], ticks, 1e9, expected_block_id_steps=[1, 2, 1])
    assert summary.block_id_gaps == 0 and summary.frames_received == 4
    assert summary.mean_frame_interval_ms == pytest.approx(40.0)
    assert summarize_recording(5, [1, 2, 4, 5], ticks, 1e9).block_id_gaps == 1
    assert summary.triggers_late is None and "n/a (free-run)" in summary.format(2.0)


def test_summary_counts_late_triggers_and_timestamp_mismatches():
    from src.acquire.triggered_acquisition import summarize_recording

    summary = summarize_recording(4, [7, 8, 9, 10], [0, 40, 80, 120], 1e3,
                                  trigger_late_ms=[0.1, 7.3, 2.5, 0.2], late_is_large_jump=[False, True, False, False],
                                  exposure_minus_trigger_ms=[1.8, 1.7, 41.9, 1.9],
                                  late_threshold_ms=2.0, mismatch_tolerance_ms=10.0)
    assert (summary.triggers_late, summary.triggers_late_large_jump) == (2, 1)
    assert summary.timestamp_mismatches == 1
    assert summary.exposure_minus_trigger_ms_median == pytest.approx(1.85)


def test_camera_clock_map_interpolates_the_offset_between_latches():
    from src.acquire.triggered_acquisition import ClockSample, build_camera_clock_map

    # camera clock = PC clock + 100 s, drifting by +1 ms over the 10 s between the two latch sets
    before = [ClockSample(10.0, 10.0, int(110.0 * 1e9))]
    after = [ClockSample(20.0, 20.0, int(120.001 * 1e9))]
    clock_map = build_camera_clock_map(before, after, 1e9)
    assert clock_map.host_time_s(int(115.0005 * 1e9)) == pytest.approx(15.0, abs=1e-9)
    assert build_camera_clock_map([], [], 1e9) is None


# --- realtime camera source (src/realtime/acquisition.py) ------------------------

def _realtime_source(trigger_mode, target_fps=10.0, max_frames=12, control=None, adaptive=False,
                     queue_size=2, drop=True):
    import copy

    from src.realtime.acquisition import CameraFrameSource
    from src.utils.config_loader import load_config_once

    acquisition_config = copy.deepcopy(load_config_once(str(CONFIG_PATH))["acquisition"])
    acquisition_config["camera"]["trigger_mode"] = trigger_mode
    return CameraFrameSource(
        target_fps, max_frames, 150, adaptive, 0.2, 3, 1, 750.0, None, None,
        etl_control_enabled=True, start_dpt=-2.0, focus_pattern="zigzag", scan_height_mm=14.0,
        scan_step_um=550.0, scan_ratio=3.36, lens_settle_time_sec=0.025, large_jump_threshold_dpt=-3.0,
        large_jump_settle_sec=0.040, lens_controller_port="COM11", acquisition_config=acquisition_config,
        queue_size=queue_size, drop_frame_if_backlogged=drop, max_consecutive_missing_frames=15,
        acquisition_control=control)


def _realtime_cycle():
    from src.acquire.focus_pattern import build_focus_cycle_pattern

    return build_focus_cycle_pattern(-2.0, "zigzag", 14.0, 550.0, 3.36)


@pytest.fixture()
def cameras(hw, monkeypatch):
    """Every simulated camera the code under test opens."""
    created = []
    original_init = hw.pylon.InstantCamera.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(hw.pylon.InstantCamera, "__init__", init)
    return created


def _exposure_intervals_ms(packets):
    ns = [p.camera_timestamp_ns for p in packets]
    return [(b - a) / 1e6 for a, b in zip(ns[:-1], ns[1:])]


def test_realtime_grid_frames_show_their_labels(hw, cameras):
    packets = list(_realtime_source("software_grid"))

    assert [p.frame_index for p in packets] == list(range(12))
    assert [p.diopter for p in packets] == pytest.approx(_realtime_cycle()[:12])
    assert [_shown_setting(hw, p.camera_timestamp_ns) for p in packets] == pytest.approx([p.diopter for p in packets])
    assert min(_exposure_intervals_ms(packets)) == pytest.approx(100.0, abs=3.0)      # 10 fps grid
    assert cameras[0].TriggerMode.GetValue() == "Off"                                 # free-run restored


def test_realtime_grid_adaptive_request_applies_two_frames_later(hw, cameras):
    """The lens command for frame k + 1 is sent right after frame k's exposure, before frame k
    reaches the consumer, so a request made on frame k applies from frame k + 2."""
    from src.realtime.acquisition import AcquisitionControl

    control = AcquisitionControl()
    packets = []
    for packet in _realtime_source("software_grid", control=control, adaptive=True):
        packets.append(packet)
        if packet.frame_index == 4:
            control.submit_adaptive_center_diopter(0.5)
    rounds = [p.focus_round_id for p in packets]

    assert rounds.index(1) == 6
    assert [_shown_setting(hw, p.camera_timestamp_ns) for p in packets] == pytest.approx([p.diopter for p in packets])


def test_realtime_grid_pause_keeps_every_frame_and_restarts_the_grid(hw, cameras, caplog):
    import logging
    import time

    packets = []
    with caplog.at_level(logging.INFO, logger="src.realtime.acquisition"):
        for packet in _realtime_source("software_grid", queue_size=1, drop=False):
            packets.append(packet)
            if packet.frame_index == 3:
                time.sleep(0.5)                  # processing falls behind

    assert [p.frame_index for p in packets] == list(range(12))          # nothing dropped
    assert [_shown_setting(hw, p.camera_timestamp_ns) for p in packets] == pytest.approx([p.diopter for p in packets])
    assert min(_exposure_intervals_ms(packets)) > 90.0                  # no catch-up burst after the pause
    assert "grid restarted 1 times" in caplog.text


def test_realtime_grid_drops_frames_for_slow_processing_on_request(hw, cameras):
    import time

    packets = []
    for packet in _realtime_source("software_grid", queue_size=1, drop=True):
        packets.append(packet)
        if packet.frame_index == 3:
            time.sleep(0.5)

    indices = [p.frame_index for p in packets]
    assert indices[:4] == [0, 1, 2, 3] and len(indices) < 12 and indices[-1] == 11   # frames dropped, grid kept
    assert [_shown_setting(hw, p.camera_timestamp_ns) for p in packets] == pytest.approx([p.diopter for p in packets])


def test_realtime_stop_request_ends_the_grid_and_restores_free_run(hw, cameras):
    from src.realtime.acquisition import AcquisitionControl

    control = AcquisitionControl()
    packets = []
    for packet in _realtime_source("software_grid", max_frames=None, control=control):
        packets.append(packet)
        if packet.frame_index == 2:
            control.request_stop()

    assert 3 <= len(packets) <= 5
    assert cameras[0].TriggerMode.GetValue() == "Off"


def test_realtime_free_run_keeps_the_legacy_loop(hw, cameras):
    """Legacy realtime loop at 25 fps on the mock: same plane sequence, one-frame lag."""
    packets = list(_realtime_source("free_run", target_fps=25.0, max_frames=20))
    cycle = _realtime_cycle()

    assert [p.diopter for p in packets] == pytest.approx(cycle[:20])
    shown = [_shown_setting(hw, p.camera_timestamp_ns) for p in packets]
    assert sum(shown[i] == pytest.approx(cycle[i - 1]) for i in range(1, 20)) >= 17
