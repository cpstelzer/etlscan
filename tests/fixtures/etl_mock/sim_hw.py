"""Simulated ETL + Basler camera for dry-running the diagnosis script.

Physics of the simulation (what the analysis should recover):
  (env: SIM_FOCUS_DPT focus of the sample, SIM_BLUR_PER_DPT focus-curve width,
   SIM_TEMP_RISE_C / SIM_TEMP_TAU_S lens warm-up; camera timestamps are
   perf_counter * 1e9 + CLOCK_OFFSET_TICKS, so exposure time = (ticks - offset) / 1e9
   and LENS.commanded_at(exposure time) is the setting a frame really shows)
  - free-run camera: exposures every 1/fps; a frame becomes retrievable
    TRANSFER_S after its exposure (OneByOne queue)
  - software trigger: exposure at trigger time, retrievable TRANSFER_S later,
    next trigger accepted MIN_PERIOD_S after the previous one
  - lens: reaches a new setting SETTLE_S after the command; static hysteresis:
    actual = commanded - sign(last move) * HYST_DPT / 2
"""
import os
import threading
import time

import cv2
import numpy as np

TRANSFER_S = 0.038
MIN_PERIOD_S = 0.037
SETTLE_S = 0.004
DROP_BLOCK_IDS = set()   # tests: exposures with these block IDs never reach the PC (a lost frame)
HYST_DPT = 0.03          # at 0 dpt; grows with lens power (see hyst_at)
CLOCK_OFFSET_TICKS = 987_654_321_000
IMG = 384
BLUR_PER_DPT = float(os.environ.get("SIM_BLUR_PER_DPT", "9"))
FOCUS_DPT = float(os.environ.get("SIM_FOCUS_DPT", "0"))   # sample in focus here (tilted)
pc = time.perf_counter


T0 = pc()
TEMP_RISE_C = float(os.environ.get("SIM_TEMP_RISE_C", "4"))      # warm-up of the simulated lens
TEMP_TAU_S = float(os.environ.get("SIM_TEMP_TAU_S", "120"))


def lens_temp():
    import math
    return 24.0 + TEMP_RISE_C * (1 - math.exp(-(pc() - T0) / TEMP_TAU_S))


def hyst_at(dpt):
    # offset grows with lens power and (simulated) with temperature: +10 % per degree
    return max(0.005, HYST_DPT * (1 + 0.5 * dpt)) * (1 + 0.10 * (lens_temp() - 24.0))


class LensState:
    def __init__(self):
        self.lock = threading.Lock()
        self.hist = [(-1e9, 0.0, 0.0)]   # (t_effective, commanded, actual)
        self.last_cmd = 0.0
        self.direction = 1.0

    def command(self, dpt):
        with self.lock:
            if dpt != self.last_cmd:
                self.direction = 1.0 if dpt > self.last_cmd else -1.0
            self.last_cmd = dpt
            actual = dpt - self.direction * hyst_at(dpt) / 2
            self.hist.append((pc() + SETTLE_S, dpt, actual))

    def commanded_at(self, t):
        """Setting commanded (and settled, SETTLE_S after the command) at perf_counter time t."""
        with self.lock:
            val = self.hist[0][1]
            for te, cmd, _ in self.hist:
                if te <= t:
                    val = cmd
                else:
                    break
            return val

    def actual_at(self, t):
        with self.lock:
            val = self.hist[0][2]
            for te, _, a in self.hist:
                if te <= t:
                    val = a
                else:
                    break
            return val


LENS = LensState()
rng = np.random.default_rng(1)
SPEC = [(int(x), int(y)) for x, y in rng.uniform(40, IMG - 40, size=(14, 2))]


def render(actual_dpt):
    img = np.full((IMG, IMG), 170, np.float32)
    for (x, y) in SPEC:
        focus = FOCUS_DPT + 0.02 * (x - IMG / 2) / (IMG / 2)
        sigma = 0.6 + BLUR_PER_DPT * abs(actual_dpt - focus)
        patch = np.zeros((81, 81), np.float32)
        cv2.ellipse(patch, (40, 40), (14, 9), 30, 0, 360, 90, -1)
        cv2.circle(patch, (36, 38), 3, 40, -1)
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)
        y0, x0 = y - 40, x - 40
        img[y0:y0 + 81, x0:x0 + 81] -= patch
    img += np.random.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
