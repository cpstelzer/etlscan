"""Diagnosis of the ETL/camera timing (axial localization: up-down offset).

Runs a fixed suite of short recordings on one well that holds a static sample
(e.g. fixed rotifers) which is in focus at --start-dpt, and logs for every
frame when the lens was commanded and when the camera exposed the frame.

Usage (from the repository root, like run_acquire_single_sweep.py):
    python scripts/run_acquire_diagnosis.py --plate 3 --well 7 --start-dpt 0

Two suites are available (--tests). suite2 is the default.

suite2: fixed-grid loop and direction offset over the production sweep
  zzcal_pre                   One zigzag cycle over the production sweep range
                              (--range-start-dpt, scan_height_mm of the config)
                              at the first step size, after a 1 s hold at the
                              range start, as at the start of a recording. Tests
                              whether the first sweep after idling differs.
  zzcal150 ... zzcal550       Zigzag over the production sweep range at every
                              step size of experiment_sweep.scan_steps_um_options,
                              --cal-cycles cycles each, fixed-grid loop at 25 fps.
                              Up-down offset of the specimens at --start-dpt.
  grid_toggle_25fps           Fixed-grid loop on the pseudo-random toggle sequence
                              (lag and frame rate check).
  saw150, saw450              Sawtooth over the production sweep range (the
                              repository's sawtooth pattern), --saw-cycles cycles.
The fixed-grid loop sets the lens, triggers each exposure on a fixed 40 ms grid
(no accumulating overhead), sends SetAsInput() once before the loop and moves
the lens only after the exposure has ended.

stability (--tests stability): is the direction offset stable over a session?
  Measurement blocks at --stab-times-min (default 0, 5, 10, 20, 30, 45, 60 min
  after the start). Each block records the production-range zigzag at 150 and
  550 µm (stabNN_zz150, stabNN_zz550; --cal-cycles cycles). Between blocks the
  lens keeps sweeping the 150 µm zigzag as during a recording (--stab-between
  sweep, default) or is parked at the sweep start (idle). The lens temperature
  is read once per second throughout (lens_temperature_log.csv). To save disk,
  only frames within --save-window-dpt of --start-dpt are written; all frames
  are acquired and logged. A grid_toggle_25fps test at the end serves the tile
  selection. Start it on a cold rig, before any other recording.
    python scripts/run_acquire_diagnosis.py --plate 3 --well 8 --start-dpt 1.5 --tests stability

loopcmp (--tests loopcmp): does the per-frame SetAsInput() call change the direction offset?
  After --warmup-min minutes of lens sweeping (default 45), three loops record the
  production-range zigzag at 150 and 550 µm (--cal-cycles cycles), in the order
  grid1 - gridE - prod, then prod - gridE - grid1, so a slow drift cancels in the mean:
    grid1  fixed-grid loop, SetAsInput() once before the loop (the candidate fix)
    gridE  fixed-grid loop, SetAsInput() before every SetFocalPower()
    prod   production loop (wellscanner_acquisition_module.acquire_and_save_frames with
           the trigger mode of the config or --prod-trigger-mode; with free_run the
           one-frame lag is removed in the analysis using the camera timestamps)
  Test names lc<rep>_<loop>_zz<step>. A grid_toggle_25fps test at the end serves the
  tile selection.
    python scripts/run_acquire_diagnosis.py --plate 3 --well 8 --start-dpt 1.536 --tests loopcmp

Lens temperature: python scripts/run_acquire_diagnosis.py --probe-temperature
connects to the lens controller only and reports which call returns the lens
temperature (Optotune documents controller.TemperatureManager.GetDeviceTemperature()
for C#; the Python names are probed). All grid tests log the temperature when a
call is found.

suite1: the first diagnosis (tests below, all settings relative to --start-dpt):
  1  prod_zz150, prod_zz450   Production loop (wellscanner_acquisition_module.
                              acquire_and_save_frames, called as run_acquisition
                              does) on a zigzag of +-2 mm around the focus, 6
                              passes, 25 fps.
  2  prod_toggle_25/15/10fps  Production loop on a pseudo-random sequence of two
                              settings: in focus (start) and defocused (start + 0.5
                              dpt), at 25, 15 and 10 fps.
     The production loop uses camera.trigger_mode of the config (software_grid, the
     fixed-grid trigger loop, by default); --prod-trigger-mode free_run records the
     legacy free-run loop for comparison. Logging proxies record every SetFocalPower
     and RetrieveResult call of the loop; the lens temperature is passed to it when
     a temperature call is found, as run_acquisition does.
  3  trig_toggle_settle25/10/2  Simple triggered loop: set lens, wait (25/10/2 ms),
                              software-trigger one exposure, wait for that frame.
  4  pipe_toggle_25fps        Pipelined triggered loop (candidate fix): set lens,
                              wait 10 ms, trigger, change lens only after the
                              exposure has ended; frames retrieved in a thread.
  5  hysteresis               Slow triggered sweep start-0.4 -> start+0.4 dpt in
                              0.02 dpt steps, up and down, twice; 300 ms settle.
  6  pipe_zz150, pipe_zz450   Pipelined triggered loop on the zigzag of test 1.

Every frame is saved as PNG with the production file-name scheme. Each test
folder holds test_info.json, clock_sync.json and either diag_frames.csv (own
loops) or proxy_lens.csv + proxy_grabs.csv + metadata_log.csv (production loop).
diagnosis_manifest.json in the run folder lists all tests. Analyse the run with
scripts/analyze_diagnosis.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pypylon import pylon  # noqa: E402
import optoICC  # noqa: E402

from src.acquire.focus_pattern import build_focus_cycle_pattern  # noqa: E402
from src.acquire.roi import reset_roi_to_full, set_centered_roi  # noqa: E402
from src.acquire.wellscanner_acquisition_module import (  # noqa: E402
    acquire_and_save_frames, load_config)

pc = time.perf_counter

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
ZIGZAG_HALF_RANGE_MM = 2.0      # zigzag from focus - 2 mm to focus + 2 mm
ZIGZAG_PASSES = 6
ZIGZAG_STEPS_UM = (150, 450)
TOGGLE_DEFOCUS_DPT = 0.5        # about 1.7 mm in water with 3.36 mm/dpt
TOGGLE_FRAMES = 150
TOGGLE_MAX_RUN = 3              # longest run of identical settings in the sequence
TOGGLE_SEED = 20261020
PROD_TOGGLE_FPS = (25.0, 15.0, 10.0)
TRIG_SETTLE_MS = (25.0, 10.0, 2.0)
TRIG_TOGGLE_FRAMES = 100
PIPE_FPS = 25.0
PIPE_SETTLE_MS = 10.0
PIPE_POST_EXPOSURE_MS = 2.0     # wait after the exposure before the lens may move
HYST_HALF_RANGE_DPT = 0.4
HYST_STEP_DPT = 0.02
HYST_SETTLE_S = 0.3
HYST_FRAMES_PER_SETTING = 2
HYST_REPEATS = 2
CLOCK_SYNC_SAMPLES = 20

# suite2
GRID_FPS = 25.0
GRID_SETTLE_MS = 10.0           # minimum time from the end of the lens command to the trigger
GRID_POST_EXPOSURE_MS = 2.0     # wait after the exposure before the lens may move
GRID_READY_LEAD_MS = 8.0        # ask the camera whether it is ready this long before the grid time
                                # (the check took 5.4-5.8 ms on the rig, 20.10.2026)
GRID_SPIN_MS = 2.0              # busy-wait instead of sleeping for the last 2 ms before a trigger
GRID_SET_AS_INPUT_EACH_FRAME = False   # production sends SetAsInput() with every lens command
CAL_RANGE_START_DPT = -2.0      # production start of the focus sweep (z = 0)
CAL_STEPS_UM_DEFAULT = (150, 250, 350, 450, 550)
CAL_CYCLES = 2                  # zigzag cycles per step size: 2 upward and 2 downward passes
CAL_PRE_CYCLES = 1
SAW_STEPS_UM = (150, 450)
SAW_CYCLES = 3
FLYBACK_DPT = 1.0               # a downward jump larger than this is a sawtooth fly-back
RANGE_MARGIN_DPT = 0.3          # specimens should be at least this far inside the sweep range

# loop comparison
LOOPCMP_STEPS_UM = (150, 550)
LOOPCMP_WARMUP_MIN = 45.0       # lens sweeps this long before the first recording
LOOPCMP_ORDER = (("grid1", "gridE", "prod"), ("prod", "gridE", "grid1"))   # A-B-C, then C-B-A

# stability test
STAB_TIMES_MIN = "0,5,10,20,30,45,60"   # start of each measurement block, minutes after the run start
STAB_STEPS_UM = (150, 550)
STAB_SAVE_WINDOW_DPT = 0.9      # save only frames within this distance of --start-dpt (all are acquired)
TEMP_EVERY_FRAMES = 25          # lens temperature read once per second during recordings
TEMP_PATHS = (                  # candidate optoICC calls, tried in this order
    "ecc.TemperatureManager.GetDeviceTemperature",
    "lens.TemperatureManager.GetDeviceTemperature",
    "ecc.Lens.TemperatureManager.GetDeviceTemperature",
    "ecc.TemperatureManager.GetTemperature",
    "lens.TemperatureManager.GetTemperature",
    "lens.GetTemperature",
    "ecc.GetTemperature",
)


# ---------------------------------------------------------------------------
# Helpers: camera timestamps, clock sync, logging proxies
# ---------------------------------------------------------------------------
def grab_timestamp(res):
    for attr in ("TimeStamp", "GetTimeStamp"):
        try:
            v = getattr(res, attr)
            return int(v() if callable(v) else v)
        except Exception:  # noqa: BLE001
            continue
    return None


def grab_block_id(res):
    try:
        return int(res.GetBlockID())
    except Exception:  # noqa: BLE001
        return None


def ready_buffers(camera):
    try:
        return int(camera.NumReadyBuffers.GetValue())
    except Exception:  # noqa: BLE001
        return None


def tick_frequency_hz(camera):
    try:
        return float(camera.GevTimestampTickFrequency.GetValue())
    except Exception:  # noqa: BLE001
        return 1e9   # SFNC cameras (e.g. Basler ace 2) count nanoseconds


def latch_camera_clock(camera):
    """Return camera timestamp ticks latched now, or None if not supported."""
    for cmd, val in (("TimestampLatch", "TimestampLatchValue"),
                     ("GevTimestampControlLatch", "GevTimestampValue")):
        try:
            getattr(camera, cmd).Execute()
            return int(getattr(camera, val).GetValue())
        except Exception:  # noqa: BLE001
            continue
    return None


def clock_sync(camera, n=CLOCK_SYNC_SAMPLES):
    samples = []
    for _ in range(n):
        t0 = pc()
        ticks = latch_camera_clock(camera)
        t1 = pc()
        if ticks is None:
            break
        samples.append({"host_before_s": t0, "host_after_s": t1, "cam_ticks": ticks})
    return samples


class _StaticInputProxy:
    def __init__(self, static_input, log):
        self._si = static_input
        self._log = log

    def SetFocalPower(self, dpt):  # noqa: N802 (optoICC naming)
        t0 = pc()
        result = self._si.SetFocalPower(dpt)
        t1 = pc()
        self._log.append({"cmd_index": len(self._log), "dpt": float(dpt),
                          "t_cmd_start_s": t0, "t_cmd_end_s": t1})
        return result

    def __getattr__(self, name):
        return getattr(self._si, name)


class LensProxy:
    """Passes every call to the real lens; logs SetFocalPower times."""

    def __init__(self, lens, log):
        self._lens = lens
        self.StaticInput = _StaticInputProxy(lens.StaticInput, log)

    def __getattr__(self, name):
        return getattr(self._lens, name)


class CameraProxy:
    """Passes every call to the real camera; logs each RetrieveResult."""

    def __init__(self, camera, log):
        self._cam = camera
        self._log = log

    def RetrieveResult(self, *args, **kwargs):  # noqa: N802 (pylon naming)
        ready = ready_buffers(self._cam)
        t0 = pc()
        res = self._cam.RetrieveResult(*args, **kwargs)
        t1 = pc()
        ok = False
        try:
            ok = bool(res.GrabSucceeded())
        except Exception:  # noqa: BLE001
            pass
        self._log.append({"grab_index": len(self._log), "t_retrieve_start_s": t0,
                          "t_retrieve_end_s": t1, "ready_buffers_before": ready,
                          "block_id": grab_block_id(res) if ok else None,
                          "cam_ticks": grab_timestamp(res) if ok else None,
                          "grab_ok": ok})
        return res

    def __getattr__(self, name):
        return getattr(self._cam, name)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Focus sequences
# ---------------------------------------------------------------------------
def zigzag_sequence(start_dpt, scan_ratio, step_um, half_range_mm, passes):
    step_dpt = step_um / 1000.0 / scan_ratio
    n_half = int(round(half_range_mm * 1000 / step_um))
    up = [round(start_dpt + k * step_dpt, 3) for k in range(-n_half, n_half + 1)]
    cycle = up + up[-2:0:-1]
    n_frames = passes * (len(up) - 1) + 1
    seq = (cycle * (passes // 2 + 2))[:n_frames]
    return seq, up[0]


def toggle_sequence(start_dpt, n, seed=TOGGLE_SEED, max_run=TOGGLE_MAX_RUN):
    rng = random.Random(seed)
    states, run = [], 0
    for i in range(n):
        s = rng.random() < 0.5
        if i > 0 and s == states[-1]:
            run += 1
            if run >= max_run:
                s = not s
                run = 0
        else:
            run = 0
        states.append(s)
    return [round(start_dpt if s else start_dpt + TOGGLE_DEFOCUS_DPT, 3) for s in states]


def sweep_sequence(range_start_dpt, height_mm, scan_ratio, step_um, pattern, cycles):
    """Production focus cycle (focus_pattern.build_focus_cycle_pattern), repeated.

    A zigzag gets one extra frame at the start plane, so the lens ends where the
    next test begins. Returns (sequence, number of planes, range end in dpt).
    """
    cycle = build_focus_cycle_pattern(range_start_dpt, pattern, height_mm, step_um, scan_ratio)
    seq = list(cycle) * cycles
    if pattern == "zigzag":
        seq.append(cycle[0])
    return seq, len(set(cycle)), max(cycle)


def label_passes(seq):
    """Direction of every frame (up / down / flyback) and a pass index that
    increases whenever the direction changes."""
    directions = []
    for i in range(len(seq)):
        d = (seq[1] - seq[0] if len(seq) > 1 else 0.0) if i == 0 else seq[i] - seq[i - 1]
        if d < -FLYBACK_DPT:
            lab = "flyback"
        elif d > 0:
            lab = "up"
        elif d < 0:
            lab = "down"
        else:
            lab = directions[-1] if directions else "up"
        directions.append(lab)
    passes = [0]
    for i in range(1, len(seq)):
        passes.append(passes[-1] + int(directions[i] != directions[i - 1]))
    return directions, passes


def fine_timer(on):
    """Windows: 1 ms timer resolution while a loop runs (time.sleep granularity)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        winmm = ctypes.windll.winmm
        (winmm.timeBeginPeriod if on else winmm.timeEndPeriod)(1)
    except Exception:  # noqa: BLE001
        pass


def sleep_until(t, spin_s=GRID_SPIN_MS / 1000.0):
    """Sleep until perf_counter() >= t; busy-wait the last spin_s seconds."""
    while True:
        rem = t - pc()
        if rem <= 0:
            return
        if rem > spin_s:
            time.sleep(rem - spin_s)
        else:
            while pc() < t:
                pass
            return


# ---------------------------------------------------------------------------
# Lens temperature
# ---------------------------------------------------------------------------
def _safe_dir(obj):
    try:
        return [a for a in dir(obj) if not a.startswith("_")]
    except Exception:  # noqa: BLE001
        return []


def _call_float(fn):
    v = fn()
    if isinstance(v, (list, tuple)):
        v = v[0]
    return float(v)


def _resolve(path, ecc, lens):
    root, *attrs = path.split(".")
    obj = {"ecc": ecc, "lens": lens}[root]
    for a in attrs:
        obj = getattr(obj, a)
    return obj


def find_temperature_reader(ecc, lens):
    """Find an optoICC call that returns the lens temperature in °C.

    Tries TEMP_PATHS (Optotune's C# example reads it with
    controller.TemperatureManager.GetDeviceTemperature()), then any callable whose name
    contains 'temp', up to two attribute levels below the controller and the lens.
    Returns (function, path, first value, notes); function is None if nothing worked.
    """
    notes = []
    for path in TEMP_PATHS:
        try:
            fn = _resolve(path, ecc, lens)
            v = _call_float(fn)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{path}: {type(e).__name__}")
            continue
        if -30.0 < v < 120.0:
            return fn, path, v, notes
        notes.append(f"{path}: implausible value {v}")
    for root_name, root in (("ecc", ecc), ("lens", lens)):
        for a in _safe_dir(root):
            if "temp" not in a.lower():
                continue
            obj = getattr(root, a, None)
            cands = [(f"{root_name}.{a}", obj)] + [(f"{root_name}.{a}.{b}", getattr(obj, b, None))
                                                   for b in _safe_dir(obj) if "temp" in b.lower()]
            for path, fn in cands:
                if not callable(fn) or path in TEMP_PATHS:
                    continue
                try:
                    v = _call_float(fn)
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{path}: {type(e).__name__}")
                    continue
                if -30.0 < v < 120.0:
                    return fn, path, v, notes
                notes.append(f"{path}: implausible value {v}")
    return None, None, None, notes


def temperature_attribute_listing(ecc, lens):
    """Names containing 'temp' below the controller and the lens (for the probe printout)."""
    out = []
    for root_name, root in (("ecc", ecc), ("lens", lens), ("ecc.Lens", getattr(ecc, "Lens", None))):
        if root is None:
            continue
        for a in _safe_dir(root):
            if "temp" in a.lower():
                obj = getattr(root, a, None)
                out.append(f"{root_name}.{a}: " + ", ".join(_safe_dir(obj)[:40]))
    return out


class TempLogger:
    """Reads the lens temperature on demand and keeps every reading."""

    def __init__(self, fn, path):
        self.fn, self.path = fn, path
        self.rows = []

    def read(self, phase):
        t0 = pc()
        try:
            v = _call_float(self.fn)
        except Exception:  # noqa: BLE001
            v = float("nan")
        t1 = pc()
        self.rows.append({"t_s": t0, "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                          "phase": phase, "lens_temp_c": v, "read_ms": 1000 * (t1 - t0)})
        return v, 1000 * (t1 - t0)

    def last(self):
        return self.rows[-1]["lens_temp_c"] if self.rows else float("nan")


def hysteresis_sequence(start_dpt):
    n = int(round(HYST_HALF_RANGE_DPT / HYST_STEP_DPT))
    up = [round(start_dpt + k * HYST_STEP_DPT, 3) for k in range(-n, n + 1)]
    segments = []
    for _ in range(HYST_REPEATS):
        segments.append(("up", up))
        segments.append(("down", up[::-1]))
    return segments


# ---------------------------------------------------------------------------
# Camera configuration (mirrors run_acquisition; mono only)
# ---------------------------------------------------------------------------
def configure_camera(camera, acq):
    fmt = camera.PixelFormat.GetValue()
    if fmt.startswith("Bayer"):
        for pref in acq["camera"]["pixel_format_preference"]:
            if pref.startswith("Mono") and pref in camera.PixelFormat.Symbolics:
                camera.PixelFormat.SetValue(pref)
                break
    camera.ExposureTime.SetValue(acq["camera"]["exposure_us"])
    exposure_us = float(camera.ExposureTime.GetValue())
    camera.MaxNumBuffer = max(acq["camera"]["max_num_buffer_min"], int(acq["camera"]["fps"]))
    width, height = set_centered_roi(camera, acq["roi"]["width"], acq["roi"]["height"])
    return exposure_us, int(width), int(height)


def set_free_run(camera, fps):
    try:
        camera.TriggerSelector.SetValue("FrameStart")
    except Exception:  # noqa: BLE001
        pass
    camera.TriggerMode.SetValue("Off")
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(fps)
    return float(camera.AcquisitionFrameRate.GetValue())


def set_software_trigger(camera):
    camera.AcquisitionFrameRateEnable.SetValue(False)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Software")


def resulting_fps(camera):
    try:
        return float(camera.ResultingFrameRate.GetValue())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, base_dir, date_str, plate, well, start_dpt, acq, camera, lens,
                 width, height, exposure_us):
        self.base_dir, self.date_str = base_dir, date_str
        self.plate, self.well, self.start_dpt = plate, well, start_dpt
        self.acq, self.camera, self.lens = acq, camera, lens
        self.width, self.height, self.exposure_us = width, height, exposure_us
        self.scan_ratio = float(acq["focusing"]["scan_ratio"])
        self.tick_hz = tick_frequency_hz(camera)
        self.tests = []

    def folder(self, name):
        path = os.path.join(self.base_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def filename(self, name, i, dpt):
        fd = (dpt - self.start_dpt) * self.scan_ratio
        return (f"{self.date_str}_plate{self.plate}_well{self.well}_{name}_"
                f"f{i:05d}_dpt{dpt:.3f}_fd{fd:.2f}mm.png")

    def _finish(self, name, folder, info, sync_before):
        info["clock_sync"] = "clock_sync.json"
        write_json(os.path.join(folder, "clock_sync.json"),
                   {"tick_frequency_hz": self.tick_hz, "before": sync_before,
                    "after": clock_sync(self.camera)})
        write_json(os.path.join(folder, "test_info.json"), info)
        self.tests.append({"name": name, "folder": name, "kind": info["kind"], "loop": info["loop"]})
        print(f"  -> {name}: done ({info.get('n_frames', '?')} frames)")

    # --- 1/2: production loop with logging proxies ---------------------------
    def production(self, name, kind, sequence, fps, start_for_fd, step_um=None, extra_info=None,
                   temp_logger=None):
        trigger_mode = self.acq["camera"]["trigger_mode"]
        print(f"\n=== {name}: production loop, {len(sequence)} frames, {fps:g} fps, {trigger_mode} ===")
        folder = self.folder(name)
        final_fps = set_free_run(self.camera, fps)
        lens_log, grab_log = [], []
        cam_proxy, lens_proxy = CameraProxy(self.camera, grab_log), LensProxy(self.lens, lens_log)
        conf = copy.deepcopy({k: v for k, v in self.acq.items() if k not in ("watermark", "camera")})
        conf["watermark"] = self.acq["watermark"]          # holds cv2 constants: no deepcopy
        conf["camera"] = dict(self.acq["camera"])          # holds a pylon constant
        conf["camera"]["fps"] = fps
        self.lens.StaticInput.SetAsInput()
        self.lens.StaticInput.SetFocalPower(sequence[0])
        time.sleep(1.0)
        temp_start = temp_logger.read(f"{name}:start")[0] if temp_logger else None
        sync = clock_sync(self.camera)
        wm_x = max(10, self.width - self.acq["watermark"]["offset_horizontal"])
        wm_y = max(10, min(self.height - 30, self.height - self.acq["watermark"]["offset_vertical"]))
        acquire_and_save_frames(
            cam_proxy, lens_proxy, list(sequence), self.width, self.height, wm_x, wm_y,
            folder, os.path.join(folder, "metadata_log.csv"), self.date_str,
            self.plate, f"{self.well}_{name}", start_for_fd, conf,
            final_fps=final_fps, actual_exposure_us=self.exposure_us,
            process_as_color=False, debayer_code=None,
            lens_temperature_reader=(lambda: _call_float(temp_logger.fn)) if temp_logger else None)
        write_csv(os.path.join(folder, "proxy_lens.csv"), lens_log)
        write_csv(os.path.join(folder, "proxy_grabs.csv"), grab_log)
        temp_end = temp_logger.read(f"{name}:end")[0] if temp_logger else None
        grid = trigger_mode == "software_grid"
        info = {"name": name, "kind": kind, "loop": "production", "trigger_mode": trigger_mode, "fps_set": fps,
                "lens_temp_start_c": temp_start, "lens_temp_end_c": temp_end, **(extra_info or {}),
                "fps_camera": final_fps, "fps_resulting": resulting_fps(self.camera),
                "settle_ms": (self.acq["focusing"]["min_settle_ms"] if grid
                              else 1000 * self.acq["focusing"]["lens_settle_time_sec"]),
                "step_um": step_um, "start_dpt": self.start_dpt,
                "defocus_dpt": TOGGLE_DEFOCUS_DPT if kind == "toggle" else None,
                "scan_ratio": self.scan_ratio, "exposure_us": self.exposure_us,
                "sequence_dpt": list(sequence), "n_frames": len(sequence),
                "frame_log": "proxy", "fd_reference_dpt": start_for_fd}
        self._finish(name, folder, info, sync)

    # --- 3/5: simple triggered loop ------------------------------------------
    def triggered(self, name, kind, sequence, settle_ms, frames_per_setting=1,
                  segment_labels=None, step_um=None):
        print(f"\n=== {name}: triggered loop, {len(sequence)} settings, settle {settle_ms:g} ms ===")
        folder = self.folder(name)
        set_software_trigger(self.camera)
        cam = self.camera
        timeout = 1000
        rows, futures = [], []
        self.lens.StaticInput.SetAsInput()
        self.lens.StaticInput.SetFocalPower(sequence[0])
        time.sleep(1.0)
        sync = clock_sync(cam)
        cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        frame_i = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            for cmd_i, dpt in enumerate(sequence):
                t0 = pc()
                self.lens.StaticInput.SetAsInput()
                self.lens.StaticInput.SetFocalPower(dpt)
                t1 = pc()
                time.sleep(settle_ms / 1000.0)
                for rep in range(frames_per_setting):
                    cam.WaitForFrameTriggerReady(timeout, pylon.TimeoutHandling_ThrowException)
                    t2 = pc()
                    cam.ExecuteSoftwareTrigger()
                    t3 = pc()
                    res = cam.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                    t4 = pc()
                    ok = res.GrabSucceeded()
                    fname = self.filename(name, frame_i, dpt)
                    rows.append({"frame_index": frame_i, "cmd_index": cmd_i, "rep": rep, "dpt": dpt,
                                 "segment": segment_labels[cmd_i] if segment_labels else "",
                                 "t_cmd_start_s": t0, "t_cmd_end_s": t1, "t_trigger_start_s": t2,
                                 "t_trigger_end_s": t3, "t_retrieve_end_s": t4,
                                 "block_id": grab_block_id(res) if ok else None,
                                 "cam_ticks": grab_timestamp(res) if ok else None,
                                 "grab_ok": ok, "filename": fname})
                    if ok:
                        futures.append(ex.submit(cv2.imwrite, os.path.join(folder, fname), res.Array))
                    res.Release()
                    frame_i += 1
            cam.StopGrabbing()
            for f in futures:
                f.result()
        write_csv(os.path.join(folder, "diag_frames.csv"), rows)
        info = {"name": name, "kind": kind, "loop": "triggered", "settle_ms": settle_ms,
                "frames_per_setting": frames_per_setting, "step_um": step_um,
                "start_dpt": self.start_dpt, "scan_ratio": self.scan_ratio,
                "defocus_dpt": TOGGLE_DEFOCUS_DPT if kind == "toggle" else None,
                "exposure_us": self.exposure_us, "sequence_dpt": list(sequence),
                "n_frames": frame_i, "frame_log": "diag_frames.csv"}
        self._finish(name, folder, info, sync)

    # --- 4/6: pipelined triggered loop (candidate fix) -----------------------
    def pipelined(self, name, kind, sequence, fps=PIPE_FPS, settle_ms=PIPE_SETTLE_MS, step_um=None):
        print(f"\n=== {name}: pipelined triggered loop, {len(sequence)} frames, {fps:g} fps ===")
        folder = self.folder(name)
        set_software_trigger(self.camera)
        cam = self.camera
        period = 1.0 / fps
        post_exposure_s = self.exposure_us / 1e6 + PIPE_POST_EXPOSURE_MS / 1000.0
        grabs = []                     # metadata of received frames, in arrival order
        stop = threading.Event()
        ex = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        futures = []

        def grabber():
            while True:
                if stop.is_set() and not ready_buffers(cam):
                    break
                try:
                    res = cam.RetrieveResult(200, pylon.TimeoutHandling_Return)
                except Exception:  # noqa: BLE001
                    continue
                if res is None or not res.IsValid():
                    continue
                t = pc()
                ok = res.GrabSucceeded()
                k = len(grabs)
                dpt = sequence[k] if k < len(sequence) else float("nan")
                fname = self.filename(name, k, dpt)
                if ok:
                    futures.append(ex.submit(cv2.imwrite, os.path.join(folder, fname), res.Array))
                grabs.append({"t_retrieve_end_s": t, "grab_ok": ok,
                              "block_id": grab_block_id(res) if ok else None,
                              "cam_ticks": grab_timestamp(res) if ok else None, "filename": fname})
                res.Release()

        self.lens.StaticInput.SetAsInput()
        self.lens.StaticInput.SetFocalPower(sequence[0])
        time.sleep(1.0)
        sync = clock_sync(cam)
        cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        th = threading.Thread(target=grabber, daemon=True)
        th.start()
        cmd_rows = []
        next_trigger = pc()
        for i, dpt in enumerate(sequence):
            t0 = pc()
            self.lens.StaticInput.SetAsInput()
            self.lens.StaticInput.SetFocalPower(dpt)
            t1 = pc()
            ready_at = max(t1 + settle_ms / 1000.0, next_trigger)
            while pc() < ready_at:
                time.sleep(0.0005)
            cam.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException)
            t2 = pc()
            cam.ExecuteSoftwareTrigger()
            t3 = pc()
            next_trigger = t2 + period
            time.sleep(post_exposure_s)          # lens must not move during the exposure
            cmd_rows.append({"frame_index": i, "cmd_index": i, "rep": 0, "dpt": dpt, "segment": "",
                             "t_cmd_start_s": t0, "t_cmd_end_s": t1,
                             "t_trigger_start_s": t2, "t_trigger_end_s": t3})
        deadline = pc() + 5.0
        while len(grabs) < len(sequence) and pc() < deadline:
            time.sleep(0.05)
        stop.set()
        th.join(timeout=5.0)
        cam.StopGrabbing()
        for f in futures:
            f.result()
        ex.shutdown()
        rows = []
        for i, c in enumerate(cmd_rows):
            g = grabs[i] if i < len(grabs) else {"t_retrieve_end_s": None, "grab_ok": False,
                                                 "block_id": None, "cam_ticks": None, "filename": ""}
            rows.append({**c, **g})
        if len(grabs) != len(sequence):
            print(f"  WARNING: {len(sequence)} triggers but {len(grabs)} frames received")
        write_csv(os.path.join(folder, "diag_frames.csv"), rows)
        info = {"name": name, "kind": kind, "loop": "pipelined", "fps_set": fps,
                "settle_ms": settle_ms, "post_exposure_ms": PIPE_POST_EXPOSURE_MS, "step_um": step_um,
                "start_dpt": self.start_dpt, "scan_ratio": self.scan_ratio,
                "defocus_dpt": TOGGLE_DEFOCUS_DPT if kind == "toggle" else None,
                "exposure_us": self.exposure_us, "sequence_dpt": list(sequence),
                "n_frames": len(sequence), "n_received": len(grabs), "frame_log": "diag_frames.csv"}
        self._finish(name, folder, info, sync)

    # --- suite2: fixed-grid triggered loop (fix candidate) -----------------------
    def grid(self, name, kind, sequence, fps=GRID_FPS, settle_ms=GRID_SETTLE_MS, step_um=None,
             extra_info=None, temp_logger=None, temp_every=TEMP_EVERY_FRAMES, save_window_dpt=None,
             set_as_input_each=None):
        """Lens command, then one software trigger per frame on a fixed time grid.

        Trigger i is due at t0 + i / fps, so the overhead of WaitForFrameTriggerReady and
        ExecuteSoftwareTrigger does not add up from frame to frame (it did in pipelined()).
        The readiness check starts GRID_READY_LEAD_MS before the grid time; the lens is
        moved only after the exposure has ended; frames are retrieved in a thread.
        With temp_logger, the lens temperature is read every temp_every frames, right after
        the lens command (in the settling time). With save_window_dpt, every frame is
        acquired but only frames within that distance of --start-dpt are saved.
        set_as_input_each (default GRID_SET_AS_INPUT_EACH_FRAME) sends SetAsInput() before every
        SetFocalPower(), as the production loop does.
        """
        each = GRID_SET_AS_INPUT_EACH_FRAME if set_as_input_each is None else bool(set_as_input_each)
        print(f"\n=== {name}: fixed-grid triggered loop, {len(sequence)} frames, {fps:g} fps ===")
        folder = self.folder(name)
        set_software_trigger(self.camera)
        cam, si = self.camera, self.lens.StaticInput
        period = 1.0 / fps
        settle_s = settle_ms / 1000.0
        lead_s = GRID_READY_LEAD_MS / 1000.0
        post_exposure_s = self.exposure_us / 1e6 + GRID_POST_EXPOSURE_MS / 1000.0
        directions, passes = label_passes(sequence)
        grabs, futures = [], []
        pending = {"now": 0, "max": 0}
        plock = threading.Lock()
        stop = threading.Event()
        ex = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

        def saved(_f):
            with plock:
                pending["now"] -= 1

        def grabber():
            while True:
                if stop.is_set() and not ready_buffers(cam):
                    break
                try:
                    res = cam.RetrieveResult(200, pylon.TimeoutHandling_Return)
                except Exception:  # noqa: BLE001
                    continue
                if res is None or not res.IsValid():
                    continue
                t = pc()
                ok = res.GrabSucceeded()
                k = len(grabs)
                dpt = sequence[k] if k < len(sequence) else float("nan")
                save = ok and (save_window_dpt is None or abs(dpt - self.start_dpt) <= save_window_dpt)
                fname = self.filename(name, k, dpt) if save else ""
                if save:
                    with plock:
                        pending["now"] += 1
                        pending["max"] = max(pending["max"], pending["now"])
                    fut = ex.submit(cv2.imwrite, os.path.join(folder, fname), res.Array)
                    fut.add_done_callback(saved)
                    futures.append(fut)
                grabs.append({"t_retrieve_end_s": t, "grab_ok": ok,
                              "block_id": grab_block_id(res) if ok else None,
                              "cam_ticks": grab_timestamp(res) if ok else None, "filename": fname})
                res.Release()

        si.SetAsInput()
        si.SetFocalPower(sequence[0])
        time.sleep(1.0)
        temp_start = temp_logger.read(f"{name}:start")[0] if temp_logger else None
        sync = clock_sync(cam)
        cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        th = threading.Thread(target=grabber, daemon=True)
        th.start()
        cmd_rows = []
        old_switch = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)      # the grabber thread must not hold up a trigger for 5 ms
        fine_timer(True)
        try:
            t_grid0 = pc() + 0.05
            for i, dpt in enumerate(sequence):
                t_grid = t_grid0 + i * period
                t0 = pc()
                if each:
                    si.SetAsInput()
                si.SetFocalPower(dpt)
                t1 = pc()
                temp_c = temp_ms = None
                if temp_logger is not None and temp_every and i % temp_every == 0:
                    temp_c, temp_ms = temp_logger.read(name)
                t_fire = max(t_grid, t1 + settle_s)
                sleep_until(t_fire - lead_s)
                tw0 = pc()
                cam.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException)
                tw1 = pc()
                sleep_until(t_fire)
                t2 = pc()
                cam.ExecuteSoftwareTrigger()
                t3 = pc()
                cmd_rows.append({"frame_index": i, "cmd_index": i, "rep": 0, "dpt": dpt, "segment": "",
                                 "direction": directions[i], "pass_index": passes[i], "t_grid_s": t_grid,
                                 "t_cmd_start_s": t0, "t_cmd_end_s": t1, "t_wait_start_s": tw0,
                                 "t_wait_end_s": tw1, "t_trigger_start_s": t2, "t_trigger_end_s": t3,
                                 "lens_temp_c": temp_c, "temp_read_ms": temp_ms})
                sleep_until(t3 + post_exposure_s)          # lens must not move during the exposure
        finally:
            fine_timer(False)
            sys.setswitchinterval(old_switch)
        deadline = pc() + 5.0
        while len(grabs) < len(sequence) and pc() < deadline:
            time.sleep(0.05)
        stop.set()
        th.join(timeout=5.0)
        cam.StopGrabbing()
        for f in futures:
            f.result()
        ex.shutdown()
        rows = []
        for i, c in enumerate(cmd_rows):
            g = grabs[i] if i < len(grabs) else {"t_retrieve_end_s": None, "grab_ok": False,
                                                 "block_id": None, "cam_ticks": None, "filename": ""}
            rows.append({**c, **g})
        write_csv(os.path.join(folder, "diag_frames.csv"), rows)
        temp_end = temp_logger.read(f"{name}:end")[0] if temp_logger else None
        trig = np.array([c["t_trigger_start_s"] for c in cmd_rows])
        late_ms = 1000 * np.array([c["t_trigger_start_s"] - c["t_grid_s"] for c in cmd_rows])
        n_saved = sum(1 for g in grabs if g["filename"])
        print(f"  trigger interval mean {1000 * np.mean(np.diff(trig)):.2f} ms (target {1000 * period:.1f}); "
              f"triggers late by > 2 ms: {int(np.sum(late_ms > 2))} of {len(cmd_rows)}; "
              f"frames received {len(grabs)} of {len(sequence)}, saved {n_saved}; "
              f"max frames waiting to be saved {pending['max']}"
              + (f"; lens {temp_start:.2f} -> {temp_end:.2f} °C" if temp_logger else ""))
        if len(grabs) != len(sequence):
            print(f"  WARNING: {len(sequence)} triggers but {len(grabs)} frames received")
        info = {"name": name, "kind": kind, "loop": "grid", "fps_set": fps, "settle_ms": settle_ms,
                "save_window_dpt": save_window_dpt, "n_saved": n_saved,
                "lens_temp_start_c": temp_start, "lens_temp_end_c": temp_end,
                "post_exposure_ms": GRID_POST_EXPOSURE_MS, "ready_lead_ms": GRID_READY_LEAD_MS,
                "set_as_input_each_frame": each, "step_um": step_um,
                "start_dpt": self.start_dpt, "scan_ratio": self.scan_ratio,
                "defocus_dpt": TOGGLE_DEFOCUS_DPT if kind == "toggle" else None,
                "exposure_us": self.exposure_us, "sequence_dpt": list(sequence),
                "n_frames": len(sequence), "n_received": len(grabs), "pending_saves_max": pending["max"],
                "frame_log": "diag_frames.csv", **(extra_info or {})}
        self._finish(name, folder, info, sync)

    # --- stability test: between two measurement blocks ------------------------
    def keep_cycling(self, cycle, until_t, temp_logger, label, mode):
        """Until perf_counter() reaches until_t: 'sweep' drives the lens through the zigzag
        cycle at 25 settings per second (as during a recording, camera idle); 'idle' parks it
        at the sweep start. Reads the lens temperature once per second."""
        si = self.lens.StaticInput
        period = 1.0 / GRID_FPS
        remaining = (until_t - pc()) / 60
        print(f"\n--- {label}: lens {'sweeping' if mode == 'sweep' else 'parked at the sweep start'} "
              f"for {remaining:.1f} min ---")
        fine_timer(True)
        try:
            t0 = pc()
            next_temp, next_print, i = t0, t0 + 60.0, 0
            if mode == "idle":
                si.SetFocalPower(cycle[0])
            while pc() < until_t:
                if mode == "sweep":
                    si.SetFocalPower(cycle[i % len(cycle)])
                now = pc()
                if temp_logger is not None and now >= next_temp:
                    temp_logger.read(label)
                    next_temp += 1.0
                if now >= next_print:
                    temp = f", lens {temp_logger.last():.2f} °C" if temp_logger is not None else ""
                    print(f"  {(until_t - now) / 60:.1f} min to the next block{temp}")
                    next_print += 60.0
                i += 1
                sleep_until(min(t0 + i * period, until_t))
        finally:
            fine_timer(False)


# ---------------------------------------------------------------------------
def probe_temperature(acq):
    """Connect to the lens controller only, report which call returns the lens
    temperature, and read it while the lens moves between the ends of its range."""
    ecc = optoICC.connectEcc(port=acq["focusing"]["lens_controller_port"])
    try:
        lens = ecc.Lens
        fn, path, v, notes = find_temperature_reader(ecc, lens)
        if fn is None:
            print("No temperature call found. Tried:")
            for n in notes:
                print("  ", n)
            print("Attributes containing 'temp':")
            for line in temperature_attribute_listing(ecc, lens) or ["  (none)"]:
                print("  ", line)
            return
        print(f"Lens temperature: {v:.2f} °C via {path}")
        lens.StaticInput.SetAsInput()
        for dpt in (-2.0, 2.0, -2.0, 2.0, 0.0):
            lens.StaticInput.SetFocalPower(dpt)
            time.sleep(0.5)
            t0 = pc()
            val = _call_float(fn)
            print(f"  at {dpt:+.1f} dpt: {val:.2f} °C (read took {1000 * (pc() - t0):.1f} ms)")
    finally:
        for m in ("disconnect", "close"):
            if callable(getattr(ecc, m, None)):
                try:
                    getattr(ecc, m)()
                except Exception:  # noqa: BLE001
                    pass
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yml")
    ap.add_argument("--plate", type=int)
    ap.add_argument("--well", type=int)
    ap.add_argument("--start-dpt", type=float,
                    help="ETL setting at which the sample is in focus")
    ap.add_argument("--tests", default="suite2",
                    help="suite2 (default), suite1 (first diagnosis), stability, loopcmp, all (suite1 + suite2), or a "
                         "comma-separated list of test names, e.g. zzcal150,zzcal450")
    ap.add_argument("--probe-temperature", action="store_true",
                    help="only connect to the lens controller and report how the lens temperature is read")
    ap.add_argument("--stab-times-min", default=STAB_TIMES_MIN,
                    help=f"stability: start of each measurement block in minutes (default {STAB_TIMES_MIN})")
    ap.add_argument("--stab-between", choices=("sweep", "idle"), default="sweep",
                    help="stability: between blocks the lens keeps sweeping (default) or is parked at the sweep start")
    ap.add_argument("--save-window-dpt", type=float, default=STAB_SAVE_WINDOW_DPT,
                    help=f"stability, loopcmp: save only frames of the fixed-grid loop within this distance "
                         f"of --start-dpt (default {STAB_SAVE_WINDOW_DPT}; 0 saves all)")
    ap.add_argument("--warmup-min", type=float, default=LOOPCMP_WARMUP_MIN,
                    help=f"loopcmp: minutes of lens sweeping before the first recording "
                         f"(default {LOOPCMP_WARMUP_MIN:g}; 0 if the lens is already warm)")
    ap.add_argument("--range-start-dpt", type=float, default=CAL_RANGE_START_DPT,
                    help="suite2: start of the production focus sweep in dpt (default -2.0)")
    ap.add_argument("--range-height-mm", type=float, default=None,
                    help="suite2: height of the production focus sweep (default: focusing.scan_height_mm)")
    ap.add_argument("--cal-cycles", type=int, default=CAL_CYCLES,
                    help=f"suite2: zigzag cycles per step size (default {CAL_CYCLES})")
    ap.add_argument("--saw-cycles", type=int, default=SAW_CYCLES,
                    help=f"suite2: sawtooth cycles per step size (default {SAW_CYCLES})")
    ap.add_argument("--tag", default=None,
                    help="suffix of the output folder name (default s2, or stab for --tests stability; "
                         "use --tag \"\" for none)")
    ap.add_argument("--prod-trigger-mode", choices=("software_grid", "free_run"), default=None,
                    help="trigger mode of the production-loop tests (prod_*, loopcmp prod); default: "
                         "camera.trigger_mode of the config")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    acq = load_config(args.config)
    acq["loaded_from_path"] = os.path.abspath(args.config)
    if args.prod_trigger_mode is not None:
        acq["camera"]["trigger_mode"] = args.prod_trigger_mode
    if args.probe_temperature:
        probe_temperature(acq)
        return
    missing = [n for n, v in (("--plate", args.plate), ("--well", args.well), ("--start-dpt", args.start_dpt))
               if v is None]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")
    date_str = date.today().strftime("%Y%m%d")
    tag = args.tag if args.tag is not None else {"stability": "stab", "loopcmp": "loopcmp"}.get(args.tests, "s2")
    suffix = f"_{tag}" if tag else ""
    base_dir = os.path.join(acq["paths"]["output_folder_base"],
                            f"{date_str}_plate{args.plate}_well{args.well}_diagnosis{suffix}")
    if os.path.isdir(base_dir) and os.listdir(base_dir):
        sys.exit(f"ERROR: output folder exists and is not empty: {base_dir}\n"
                 f"Move or delete it, or use another --tag.")
    scan_ratio = float(acq["focusing"]["scan_ratio"])
    start = args.start_dpt
    range_start = args.range_start_dpt
    height_mm = float(args.range_height_mm if args.range_height_mm is not None
                      else acq["focusing"]["scan_height_mm"])
    cal_steps = [int(s) for s in (acq.get("experiment_sweep", {}).get("scan_steps_um_options")
                                  or CAL_STEPS_UM_DEFAULT)]

    suite1 = (["prod_zz150", "prod_zz450"] + [f"prod_toggle_{int(f)}fps" for f in PROD_TOGGLE_FPS]
              + [f"trig_toggle_settle{int(s)}" for s in TRIG_SETTLE_MS] + ["pipe_toggle_25fps", "hysteresis",
                                                                           "pipe_zz150", "pipe_zz450"])
    suite2 = (["zzcal_pre"] + [f"zzcal{s}" for s in cal_steps] + ["grid_toggle_25fps"]
              + [f"saw{s}" for s in SAW_STEPS_UM])
    stab_times, loopcmp = [], args.tests == "loopcmp"
    if loopcmp:
        plan = ["loopcmp"]
    elif args.tests == "stability":
        plan = ["stability"]
        stab_times = sorted(float(t) for t in args.stab_times_min.split(",") if t.strip())
    elif args.tests in ("suite1", "suite2", "all"):
        plan = {"suite1": suite1, "suite2": suite2, "all": suite1 + suite2}[args.tests]
    else:
        wanted = [t.strip() for t in args.tests.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in suite1 + suite2]
        if unknown:
            sys.exit(f"ERROR: unknown test(s) {unknown}. Available: {', '.join(suite1 + suite2)}")
        plan = [t for t in suite1 + suite2 if t in wanted]

    print("Diagnosis run\n  output:", base_dir, "\n  tests:", ", ".join(plan))
    if any(t.startswith("prod") for t in plan) or loopcmp:
        print(f"  production loop trigger mode: {acq['camera']['trigger_mode']}")
    print(f"  sample must be in focus at {start:+.3f} dpt")
    if any(t in suite1 for t in plan):
        print(f"  suite1 lens range: {start - ZIGZAG_HALF_RANGE_MM / scan_ratio:+.3f} to "
              f"{start + ZIGZAG_HALF_RANGE_MM / scan_ratio:+.3f} dpt")
    if any(t in suite2 for t in plan):
        _, _, range_end = sweep_sequence(range_start, height_mm, scan_ratio, cal_steps[0], "zigzag", 1)
        n_frames = 0
        for t in plan:
            if t == "zzcal_pre":
                n_frames += len(sweep_sequence(range_start, height_mm, scan_ratio, cal_steps[0], "zigzag",
                                               CAL_PRE_CYCLES)[0])
            elif t.startswith("zzcal"):
                n_frames += len(sweep_sequence(range_start, height_mm, scan_ratio, int(t[5:]), "zigzag",
                                               args.cal_cycles)[0])
            elif t.startswith("saw"):
                n_frames += len(sweep_sequence(range_start, height_mm, scan_ratio, int(t[3:]), "sawtooth",
                                               args.saw_cycles)[0])
            elif t == "grid_toggle_25fps":
                n_frames += TOGGLE_FRAMES
        print(f"  production sweep: {range_start:+.3f} to {range_end:+.3f} dpt ({height_mm:g} mm), "
              f"steps {', '.join(str(s) for s in cal_steps)} µm; about {n_frames} frames "
              f"({n_frames / GRID_FPS:.0f} s of recording)")
        if not (range_start + RANGE_MARGIN_DPT <= start <= range_end - RANGE_MARGIN_DPT):
            print(f"  WARNING: --start-dpt {start:+.3f} is less than {RANGE_MARGIN_DPT} dpt inside the sweep "
                  f"range; the zigzag and sawtooth tests need the specimens well inside it.")
    if loopcmp:
        _, _, range_end = sweep_sequence(range_start, height_mm, scan_ratio, LOOPCMP_STEPS_UM[0], "zigzag", 1)
        n_rec = sum(len(sweep_sequence(range_start, height_mm, scan_ratio, st, "zigzag", args.cal_cycles)[0])
                    for st in LOOPCMP_STEPS_UM) * 2 * 3
        print(f"  loop comparison: {args.warmup_min:g} min warm-up (lens sweeping), then "
              f"{' / '.join(' - '.join(o) for o in LOOPCMP_ORDER)}, each loop as a zigzag at "
              f"{' + '.join(str(st) for st in LOOPCMP_STEPS_UM)} µm ({args.cal_cycles} cycles) over "
              f"{range_start:+.3f} to {range_end:+.3f} dpt; {n_rec} frames ({n_rec / GRID_FPS:.0f} s of recording); "
              f"total about {args.warmup_min + 6:.0f} min")
        print("    grid1 = fixed-grid loop, SetAsInput() once; gridE = fixed-grid loop, SetAsInput() before every "
              f"command; prod = production loop ({acq['camera']['trigger_mode']}; a free-run lag is removed in "
              "the analysis)")
        if not (range_start + RANGE_MARGIN_DPT <= start <= range_end - RANGE_MARGIN_DPT):
            print(f"  WARNING: --start-dpt {start:+.3f} is less than {RANGE_MARGIN_DPT} dpt inside the sweep range.")
    if stab_times:
        _, _, range_end = sweep_sequence(range_start, height_mm, scan_ratio, STAB_STEPS_UM[0], "zigzag", 1)
        print(f"  stability: blocks at {', '.join(f'{t:g}' for t in stab_times)} min, each zigzag "
              f"{' + '.join(str(s) for s in STAB_STEPS_UM)} µm ({args.cal_cycles} cycles) over "
              f"{range_start:+.3f} to {range_end:+.3f} dpt; between blocks the lens is "
              f"{'sweeping' if args.stab_between == 'sweep' else 'parked'}; total about "
              f"{stab_times[-1] + 1.5:.0f} min")
        if not (range_start + RANGE_MARGIN_DPT <= start <= range_end - RANGE_MARGIN_DPT):
            print(f"  WARNING: --start-dpt {start:+.3f} is less than {RANGE_MARGIN_DPT} dpt inside the sweep range.")
    os.makedirs(base_dir, exist_ok=True)
    if not args.yes:
        input("Ensure:\n1) Lid\n2) Light source\n3) Devices connected\n4) Sample in focus at --start-dpt\n(Enter to start)\n")

    ecc, camera, run, templog, manifest = None, None, None, None, None
    try:
        ecc = optoICC.connectEcc(port=acq["focusing"]["lens_controller_port"])
        lens = ecc.Lens
        t_fn, t_path, t_val, t_notes = find_temperature_reader(ecc, lens)
        if t_fn is not None:
            templog = TempLogger(t_fn, t_path)
            print(f"Lens temperature {t_val:.2f} °C (read via {t_path})")
        else:
            print("WARNING: no lens-temperature call found; continuing without temperature. Tried: "
                  + "; ".join(t_notes))
            for line in temperature_attribute_listing(ecc, lens):
                print("   ", line)
        tl = pylon.TlFactory.GetInstance()
        devices = tl.EnumerateDevices()
        if not devices:
            raise ConnectionError("No Pylon camera found.")
        camera = pylon.InstantCamera(tl.CreateDevice(devices[0]))
        camera.Open()
        exposure_us, width, height = configure_camera(camera, acq)
        run = Run(base_dir, date_str, args.plate, args.well, start, acq, camera, lens,
                  width, height, exposure_us)

        manifest = {"created": datetime.now().isoformat(), "plate": args.plate, "well": args.well,
                    "start_dpt": start, "scan_ratio": scan_ratio, "exposure_us": exposure_us,
                    "tests_requested": args.tests, "range_start_dpt": range_start,
                    "range_height_mm": height_mm, "cal_steps_um": cal_steps, "cal_cycles": args.cal_cycles,
                    "saw_cycles": args.saw_cycles, "temperature_call": t_path,
                    "stab_times_min": stab_times, "stab_between": args.stab_between if stab_times else None,
                    "loopcmp_warmup_min": args.warmup_min if loopcmp else None,
                    "prod_trigger_mode": acq["camera"]["trigger_mode"],
                    "roi": [width, height], "camera_model": camera.GetDeviceInfo().GetModelName(),
                    "camera_serial": camera.GetDeviceInfo().GetSerialNumber(),
                    "tick_frequency_hz": run.tick_hz,
                    "timestamp_latch_supported": latch_camera_clock(camera) is not None,
                    "config": acq["loaded_from_path"], "tests": run.tests}

        for step in ZIGZAG_STEPS_UM:
            if f"prod_zz{step}" in plan:
                seq, bottom = zigzag_sequence(start, scan_ratio, step, ZIGZAG_HALF_RANGE_MM, ZIGZAG_PASSES)
                run.production(f"prod_zz{step}", "zigzag", seq, 25.0, bottom, step_um=step, temp_logger=templog)
        for fps in PROD_TOGGLE_FPS:
            if f"prod_toggle_{int(fps)}fps" in plan:
                run.production(f"prod_toggle_{int(fps)}fps", "toggle", toggle_sequence(start, TOGGLE_FRAMES),
                               fps, start, temp_logger=templog)
        for settle in TRIG_SETTLE_MS:
            if f"trig_toggle_settle{int(settle)}" in plan:
                run.triggered(f"trig_toggle_settle{int(settle)}", "toggle",
                              toggle_sequence(start, TRIG_TOGGLE_FRAMES), settle)
        if "pipe_toggle_25fps" in plan:
            run.pipelined("pipe_toggle_25fps", "toggle", toggle_sequence(start, TOGGLE_FRAMES))
        if "hysteresis" in plan:
            segs = hysteresis_sequence(start)
            seq = [d for _, s in segs for d in s]
            labels = [f"{lab}{k // 2 + 1}" for k, (lab, s) in enumerate(segs) for _ in s]
            run.triggered("hysteresis", "hysteresis", seq, HYST_SETTLE_S * 1000,
                          frames_per_setting=HYST_FRAMES_PER_SETTING, segment_labels=labels)
        for step in ZIGZAG_STEPS_UM:
            if f"pipe_zz{step}" in plan:
                seq, _ = zigzag_sequence(start, scan_ratio, step, ZIGZAG_HALF_RANGE_MM, ZIGZAG_PASSES)
                run.pipelined(f"pipe_zz{step}", "zigzag", seq, step_um=step)

        # suite2
        def sweep_info(pattern, step, cycles, n_planes, end):
            return {"pattern": pattern, "range_start_dpt": range_start, "range_end_dpt": end,
                    "range_height_mm": height_mm, "n_planes": n_planes, "cycles": cycles}
        if "zzcal_pre" in plan:
            seq, n_pl, end = sweep_sequence(range_start, height_mm, scan_ratio, cal_steps[0], "zigzag",
                                            CAL_PRE_CYCLES)
            run.grid("zzcal_pre", "zigzag", seq, step_um=cal_steps[0],
                     extra_info={**sweep_info("zigzag", cal_steps[0], CAL_PRE_CYCLES, n_pl, end),
                                 "precycle_of": f"zzcal{cal_steps[0]}"})
        for step in cal_steps:
            if f"zzcal{step}" in plan:
                seq, n_pl, end = sweep_sequence(range_start, height_mm, scan_ratio, step, "zigzag",
                                                args.cal_cycles)
                run.grid(f"zzcal{step}", "zigzag", seq, step_um=step,
                         extra_info=sweep_info("zigzag", step, args.cal_cycles, n_pl, end))
        if "grid_toggle_25fps" in plan:
            run.grid("grid_toggle_25fps", "toggle", toggle_sequence(start, TOGGLE_FRAMES), temp_logger=templog)
        for step in SAW_STEPS_UM:
            if f"saw{step}" in plan:
                seq, n_pl, end = sweep_sequence(range_start, height_mm, scan_ratio, step, "sawtooth",
                                                args.saw_cycles)
                run.grid(f"saw{step}", "sawtooth", seq, step_um=step, temp_logger=templog,
                         extra_info=sweep_info("sawtooth", step, args.saw_cycles, n_pl, end))

        # stability
        if stab_times:
            window = args.save_window_dpt if args.save_window_dpt > 0 else None
            cycle = build_focus_cycle_pattern(range_start, "zigzag", height_mm, STAB_STEPS_UM[0], scan_ratio)
            run_t0 = pc()
            manifest["stab_t0_perf_s"] = run_t0
            for b, t_min in enumerate(stab_times, 1):
                due = run_t0 + 60.0 * t_min
                if pc() < due:
                    run.keep_cycling(cycle, due, templog, f"between_{b - 1:02d}_{b:02d}", args.stab_between)
                for step in STAB_STEPS_UM:
                    seq, n_pl, end = sweep_sequence(range_start, height_mm, scan_ratio, step, "zigzag",
                                                    args.cal_cycles)
                    run.grid(f"stab{b:02d}_zz{step}", "zigzag", seq, step_um=step, temp_logger=templog,
                             save_window_dpt=window,
                             extra_info={**sweep_info("zigzag", step, args.cal_cycles, n_pl, end),
                                         "stab_block": b, "stab_planned_min": t_min,
                                         "t_since_run_start_s": pc() - run_t0})
            run.grid("grid_toggle_25fps", "toggle", toggle_sequence(start, TOGGLE_FRAMES), temp_logger=templog)

        # loop comparison
        if loopcmp:
            window = args.save_window_dpt if args.save_window_dpt > 0 else None
            run_t0 = pc()
            manifest["stab_t0_perf_s"] = run_t0
            if args.warmup_min > 0:
                cycle = build_focus_cycle_pattern(range_start, "zigzag", height_mm, LOOPCMP_STEPS_UM[0], scan_ratio)
                run.keep_cycling(cycle, run_t0 + 60.0 * args.warmup_min, templog, "warmup", "sweep")
            for rep_i, variants in enumerate(LOOPCMP_ORDER, 1):
                for variant in variants:
                    for step in LOOPCMP_STEPS_UM:
                        seq, n_pl, end = sweep_sequence(range_start, height_mm, scan_ratio, step, "zigzag",
                                                        args.cal_cycles)
                        name = f"lc{rep_i}_{variant}_zz{step}"
                        extra = {**sweep_info("zigzag", step, args.cal_cycles, n_pl, end),
                                 "loopcmp_rep": rep_i, "loopcmp_variant": variant,
                                 "t_since_run_start_s": pc() - run_t0}
                        if variant == "prod":
                            run.production(name, "zigzag", seq, GRID_FPS, range_start, step_um=step,
                                           extra_info=extra, temp_logger=templog)
                        else:
                            run.grid(name, "zigzag", seq, step_um=step, temp_logger=templog,
                                     save_window_dpt=window, set_as_input_each=(variant == "gridE"),
                                     extra_info=extra)
            run.grid("grid_toggle_25fps", "toggle", toggle_sequence(start, TOGGLE_FRAMES), temp_logger=templog)
        manifest["finished"] = datetime.now().isoformat()
        write_json(os.path.join(base_dir, "diagnosis_manifest.json"), manifest)
        print(f"\nDiagnosis finished. Analyse with:\n  python scripts/analyze_diagnosis.py --diag_dir \"{base_dir}\"")
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        if run is not None:
            write_json(os.path.join(base_dir, "diagnosis_manifest.json"),
                       {"aborted": str(e), "start_dpt": start, "scan_ratio": scan_ratio,
                        "tests": run.tests})
    finally:
        if templog is not None and templog.rows:
            # time axis: seconds since the start of the stability run, else since the first reading
            t_ref = manifest.get("stab_t0_perf_s") if isinstance(manifest, dict) else None
            t_ref = templog.rows[0]["t_s"] if t_ref is None else t_ref
            write_csv(os.path.join(base_dir, "lens_temperature_log.csv"),
                      [{**r, "t_rel_s": r["t_s"] - t_ref} for r in templog.rows])
        if camera is not None:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                set_free_run(camera, float(acq["camera"]["fps"]))
                reset_roi_to_full(camera)
                camera.Close()
            except Exception as e:  # noqa: BLE001
                print(f"Camera cleanup: {e}")
        if ecc is not None:
            for m in ("disconnect", "close"):
                if callable(getattr(ecc, m, None)):
                    try:
                        getattr(ecc, m)()
                    except Exception:  # noqa: BLE001
                        pass
                    break


if __name__ == "__main__":
    main()
