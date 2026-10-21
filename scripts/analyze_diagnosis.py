"""Analyse diagnosis runs written by run_acquire_diagnosis.py.

Usage (from the repository root):
    python scripts/analyze_diagnosis.py --diag_dir "C:/wellscanner/test/20261021_plate3_well7_diagnosis_s2"

Several runs at once (each is analysed; a combined table and figure of the
direction offset against lens power is written next to the run folders):
    python scripts/analyze_diagnosis.py --diag_dir RUN_A RUN_B RUN_C

What it does
  1. Sharpness: every frame is split into square tiles (--tile px) and the
     variance of the Laplacian is computed per tile (cached per test in
     _tile_sharpness_<tile>.npz). Tiles that contain specimens are selected once
     per run, from a toggle test, as those whose sharpness in focus clearly
     exceeds their sharpness when defocused. The watermark corner is excluded.
  2. Toggle tests: which lens setting did each frame actually see? The
     sharpness index is regressed (non-negative least squares) on the command
     sequence shifted by 0..3 frames. The weights give the lag in frames.
  3. Zigzag tests: per tile and pass, the focus peak is interpolated (three-point
     Gaussian) within --zz_window_dpt of --start-dpt; the difference between
     upward and downward passes is the direction offset. A second estimate
     (curve fit) shifts each pass against the tile's sharpness curve from the
     finest-step zigzag of the same loop; it does not depend on where the peak
     falls between two planes, so it separates interpolation bias from lens
     behaviour at coarse steps. For the production-range zigzags (suite2) also:
     specimen position in dpt, single-peak scatter within one direction, and the
     first sweep after idling (zzcal_pre) against the passes recorded after it.
  4. Sawtooth tests (suite2): upward peaks compared with the upward and
     downward zigzag peaks of the same tiles and step size.
  5. Hysteresis test (suite1): per tile, the focus peak of each slow upward and
     downward sweep; the up-down difference is static hysteresis.
  6. Timing: camera exposure times (camera timestamps mapped to the PC clock
     with the latched clock samples) against the times the lens was commanded:
     which command was active at each exposure, frame intervals, dropped frames
     (block-ID gaps), retrieval delay, buffer backlog; for the fixed-grid loop
     also trigger punctuality and the duration of each step of the loop.
Writes diagnosis_report.md, diagnosis_summary.csv and figures into
<diag_dir>/analysis/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, nnls

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FRAME_RE = re.compile(r"_f(\d{5})_dpt(-?\d+\.\d+)_fd")
WATERMARK_BOX = (700, 340)      # width, height (px) excluded at the bottom-right corner
ASSUMED_SETTLE_S = 0.005        # lens settling assumed when deciding which command was active
MAX_LAG = 3
ORDINAL_BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light -> dark
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_tests(diag_dir):
    man_path = os.path.join(diag_dir, "diagnosis_manifest.json")
    manifest = {}
    if os.path.isfile(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
    names = [t["name"] for t in manifest.get("tests", [])]
    if not names:   # aborted run: use every folder with a test_info.json
        names = sorted(d for d in os.listdir(diag_dir)
                       if os.path.isfile(os.path.join(diag_dir, d, "test_info.json")))
    tests = []
    for n in names:
        folder = os.path.join(diag_dir, n)
        with open(os.path.join(folder, "test_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        info["folder_path"] = folder
        tests.append(info)
    if not tests:
        sys.exit(f"ERROR: no tests found in {diag_dir}")
    return manifest, tests


def frame_table(info):
    """One row per saved frame: frame_index, dpt (commanded), path, plus timing columns."""
    folder = info["folder_path"]
    if info["loop"] == "production":
        rows = []
        for fn in os.listdir(folder):
            m = FRAME_RE.search(fn)
            if m and fn.lower().endswith((".png", ".bmp", ".tif", ".tiff")):
                rows.append({"frame_index": int(m.group(1)), "dpt": float(m.group(2)),
                             "path": os.path.join(folder, fn)})
        df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
        df["cmd_index"] = df.frame_index
        df["segment"] = ""
        g = _read_csv(os.path.join(folder, "proxy_grabs.csv"))
        lens = _read_csv(os.path.join(folder, "proxy_lens.csv"))
        if g is not None:
            g = g[g.grab_ok.astype(str) == "True"].reset_index(drop=True)
            g["frame_index"] = np.arange(len(g))
            df = df.merge(g[["frame_index", "t_retrieve_start_s", "t_retrieve_end_s",
                             "ready_buffers_before", "block_id", "cam_ticks"]], on="frame_index", how="left")
        if lens is not None:
            # the first logged command is the frame-0 setting; pre-focus happens before the proxy
            df = df.merge(lens[["cmd_index", "t_cmd_start_s", "t_cmd_end_s"]], on="cmd_index", how="left")
        commands = lens
    else:
        log = _read_csv(os.path.join(folder, info.get("frame_log", "diag_frames.csv")))
        if log is None:
            raise FileNotFoundError(f"no frame log in {folder}")
        log["path"] = [os.path.join(folder, f) if isinstance(f, str) and f else "" for f in log.filename]
        df = log[log.grab_ok.astype(str) == "True"].reset_index(drop=True)
        commands = log.groupby("cmd_index")[["dpt", "t_cmd_start_s", "t_cmd_end_s"]].first().reset_index()
    timing = df.copy()                     # every grabbed frame (also frames not saved)
    df = df[[os.path.isfile(p) for p in df.path]].reset_index(drop=True)
    return df, timing, commands


def _read_csv(path):
    return pd.read_csv(path) if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Sharpness per tile
# ---------------------------------------------------------------------------
def tile_sharpness(path, tile):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lap = cv2.Laplacian(img, cv2.CV_32F)
    h, w = lap.shape
    ny, nx = h // tile, w // tile
    blk = lap[:ny * tile, :nx * tile].reshape(ny, tile, nx, tile)
    m = blk.mean(axis=(1, 3), dtype=np.float64)
    m2 = (blk.astype(np.float64) ** 2).mean(axis=(1, 3))
    return m2 - m ** 2, (h, w)


def sharpness_stack(df, folder, tile, workers, recompute=False):
    cache = os.path.join(folder, f"_tile_sharpness_{tile}.npz")
    if os.path.isfile(cache) and not recompute:
        z = np.load(cache)
        if len(z["frame_index"]) == len(df) and np.array_equal(z["frame_index"], df.frame_index.values):
            return z["S"], tuple(z["shape"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda p: tile_sharpness(p, tile), df.path))
    shape = next(o[1] for o in out if o is not None)
    ny, nx = shape[0] // tile, shape[1] // tile
    S = np.stack([o[0] if o is not None else np.full((ny, nx), np.nan) for o in out])
    np.savez_compressed(cache, S=S, frame_index=df.frame_index.values, shape=np.array(shape))
    return S, shape


def watermark_mask(shape, tile):
    h, w = shape
    ny, nx = h // tile, w // tile
    keep = np.ones((ny, nx), bool)
    if w < 2 * WATERMARK_BOX[0] or h < 2 * WATERMARK_BOX[1]:
        return keep            # small (test) frames: no watermark exclusion
    for iy in range(ny):
        for ix in range(nx):
            if (ix + 1) * tile > w - WATERMARK_BOX[0] and (iy + 1) * tile > h - WATERMARK_BOX[1]:
                keep[iy, ix] = False
    return keep


def select_tiles(S, df, info, keep, min_ratio):
    start = info["start_dpt"]
    if info["kind"] == "toggle":
        cmd = np.isclose(df.dpt.values, start)
        infocus = cmd & np.r_[False, cmd[:-1]]          # in focus now and one frame before
        outfocus = (~cmd) & np.r_[False, ~cmd[:-1]]
    else:                                               # zigzag, sawtooth or hysteresis frames
        dist = np.abs(df.dpt.values - start)
        infocus = dist <= 0.03
        outfocus = (dist >= 0.3) & (dist <= 0.8)
    a = np.nanmedian(S[infocus], axis=0)
    b = np.nanmedian(S[outfocus], axis=0)
    ratio = a / np.maximum(b, 1e-9)
    ok = keep & np.isfinite(ratio) & (ratio > 0)
    lr = np.log(ratio[ok])
    med = np.median(lr)
    mad = 1.4826 * np.median(np.abs(lr - med))
    thr = max(np.log(min_ratio), med + 4 * mad)      # clear outliers above the background tiles
    sel = ok & (np.log(np.where(ok, ratio, 1)) >= thr)
    if sel.sum() < 5:                                  # fallback: best 10 % of tiles with ratio > min_ratio
        cut = np.nanpercentile(ratio[ok], 90)
        sel = ok & (ratio >= max(cut, min_ratio))
    return sel, ratio


def sharpness_index(S, sel):
    """Per-frame index: per tile scaled to its 10th..90th percentile within the test, median over tiles."""
    X = S[:, sel]
    lo = np.nanpercentile(X, 10, axis=0)
    hi = np.nanpercentile(X, 90, axis=0)
    Xn = (X - lo) / np.maximum(hi - lo, 1e-9)
    return np.nanmedian(Xn, axis=1)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def clock_offset(folder):
    path = os.path.join(folder, "clock_sync.json")
    if not os.path.isfile(path):
        return None, None, None
    with open(path, encoding="utf-8") as f:
        cs = json.load(f)
    f_hz = cs.get("tick_frequency_hz", 1e9)
    offs, unc = [], []
    for s in cs.get("before", []) + cs.get("after", []):
        mid = 0.5 * (s["host_before_s"] + s["host_after_s"])
        offs.append(s["cam_ticks"] / f_hz - mid)
        unc.append(0.5 * (s["host_after_s"] - s["host_before_s"]))
    if not offs:
        return None, f_hz, None
    before = [s["cam_ticks"] / f_hz - 0.5 * (s["host_before_s"] + s["host_after_s"]) for s in cs.get("before", [])]
    after = [s["cam_ticks"] / f_hz - 0.5 * (s["host_before_s"] + s["host_after_s"]) for s in cs.get("after", [])]
    drift = (np.median(after) - np.median(before)) if before and after else None
    return float(np.median(offs)), f_hz, {"latch_half_width_ms": 1000 * float(np.median(unc)),
                                          "drift_ms": None if drift is None else 1000 * drift}


def timing_analysis(df, commands, folder):
    off, f_hz, sync = clock_offset(folder)
    out = {"clock_sync": sync}
    if "t_grid_s" in df:                               # fixed-grid loop: punctuality of each step
        late = 1000 * (df.t_trigger_start_s - df.t_grid_s)
        out["trigger_late_ms_median"] = float(late.median())
        out["trigger_late_ms_max"] = float(late.max())
        out["triggers_late_over_2ms"] = int((late > 2).sum())
        out["trigger_interval_ms_median"] = 1000 * float(df.t_trigger_start_s.diff().median())
        out["ready_wait_ms_median"] = 1000 * float((df.t_wait_end_s - df.t_wait_start_s).median())
        out["lens_call_ms_median"] = 1000 * float((df.t_cmd_end_s - df.t_cmd_start_s).median())
    if "cam_ticks" not in df or df.cam_ticks.isna().all():
        return out, None
    ticks = df.cam_ticks.astype(float).values
    intervals = np.diff(ticks) / f_hz
    out["frame_interval_ms_median"] = 1000 * float(np.nanmedian(intervals)) if len(intervals) else None
    out["frame_interval_ms_mean"] = 1000 * float(np.nanmean(intervals)) if len(intervals) else None
    if len(intervals):
        out["frame_interval_ms_max"] = 1000 * float(np.nanmax(intervals))
    if "block_id" in df and df.block_id.notna().any():
        bid = df.block_id.astype(float).values
        gaps = np.diff(bid)
        out["block_id_gaps"] = int(np.sum(gaps[np.isfinite(gaps)] != 1))
    if "ready_buffers_before" in df and df.ready_buffers_before.notna().any():
        rb = df.ready_buffers_before.astype(float)
        out["backlog_max"] = int(np.nanmax(rb))
        out["backlog_frames_with_queue"] = int(np.nansum(rb > 0))
    if off is None or commands is None or "t_cmd_end_s" not in commands:
        return out, None
    t_exp = ticks / f_hz - off
    df = df.assign(t_exposure_s=t_exp)
    if "t_retrieve_end_s" in df:
        out["exposure_to_retrieve_ms_median"] = 1000 * float(np.nanmedian(df.t_retrieve_end_s - t_exp))
    if "t_trigger_start_s" in df:
        d = 1000 * (t_exp - df.t_trigger_start_s.values)
        med = float(np.nanmedian(d))
        out["exposure_minus_trigger_ms_median"] = med
        out["exposure_trigger_mismatches"] = int(np.sum(np.abs(d - med) > 10))   # a frame matched to the wrong trigger
    cmd_t = commands.sort_values("cmd_index")
    eff = np.searchsorted(cmd_t.t_cmd_end_s.values + ASSUMED_SETTLE_S, t_exp, side="right") - 1
    eff_cmd = cmd_t.cmd_index.values[np.clip(eff, 0, len(cmd_t) - 1)]
    timing_lag = df.cmd_index.values - eff_cmd
    df = df.assign(timing_lag=timing_lag)
    vals, counts = np.unique(timing_lag, return_counts=True)
    out["timing_lag_counts"] = {int(v): int(c) for v, c in zip(vals, counts)}
    out["timing_lag_mean"] = float(np.mean(timing_lag))
    # where in the frame interval does the lens command fall, relative to the exposure of that frame?
    own = cmd_t.set_index("cmd_index").t_cmd_end_s
    dt = t_exp - own.reindex(df.cmd_index).values
    out["exposure_minus_own_command_ms_median"] = 1000 * float(np.nanmedian(dt))
    out["exposure_minus_own_command_ms_min"] = 1000 * float(np.nanmin(dt))
    return out, df


# ---------------------------------------------------------------------------
# Test analyses
# ---------------------------------------------------------------------------
def toggle_lag(df, info, s):
    start = info["start_dpt"]
    c = np.isclose(df.dpt.values, start).astype(float)
    n = len(c)
    X = np.column_stack([np.r_[np.zeros(k), c[:n - k]] for k in range(MAX_LAG + 1)])[MAX_LAG:]
    y = s[MAX_LAG:]
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    A = np.column_stack([X, np.ones(len(y)), -np.ones(len(y))])   # free intercept via +/- columns
    w, _ = nnls(A, y)
    wk = w[:MAX_LAG + 1]
    total = wk.sum()
    corr = [np.corrcoef(X[:, k], y)[0, 1] for k in range(MAX_LAG + 1)]
    thr = 0.5 * (np.nanpercentile(y, 10) + np.nanpercentile(y, 90))
    agree = [float(np.mean((y > thr) == (X[:, k] > 0.5))) for k in range(MAX_LAG + 1)]
    return {"lag_frames": float((np.arange(MAX_LAG + 1) * wk).sum() / total) if total > 0 else np.nan,
            "weights": [round(float(v / total), 3) if total > 0 else np.nan for v in wk],
            "best_integer_lag": int(np.argmax(corr)), "agreement_at_lag": [round(a, 3) for a in agree],
            "contrast": float(total)}


def passes_from_sequence(dpts):
    d = np.sign(np.diff(dpts))
    pass_id = np.zeros(len(dpts), int)
    for i in range(1, len(dpts)):
        pass_id[i] = pass_id[i - 1] + (1 if i >= 2 and d[i - 1] != d[i - 2] and d[i - 1] != 0 else 0)
    return pass_id


def relabel_by_timestamps(df, dft, commands):
    """Production-loop frames: replace each frame's commanded dpt by the setting that was active
    at its exposure (command index minus the timestamp-derived lag of that frame), so the lag is
    removed frame by frame, also where it changes within a recording."""
    lag = dft.set_index("frame_index").timing_lag.reindex(df.frame_index).fillna(0).astype(int).values
    cmd_dpt = commands.set_index("cmd_index").dpt
    src = np.maximum(df.cmd_index.values - lag, cmd_dpt.index.min())
    out = df.copy()
    out["dpt_as_labelled"] = out.dpt
    out["dpt"] = cmd_dpt.reindex(src).values
    out["timing_lag_used"] = lag
    return out


def pass_labels(df):
    """Pass index and direction per frame: from the frame log (suite2) or from the sequence."""
    if {"pass_index", "direction"} <= set(df.columns) and df.pass_index.notna().all():
        return df.pass_index.astype(int).values, df.direction.astype(str).values
    dpts = df.dpt.values
    pid = passes_from_sequence(dpts)
    dirs = np.empty(len(dpts), object)
    for p in np.unique(pid):
        idx = np.nonzero(pid == p)[0]
        dirs[idx] = "up" if dpts[idx[-1]] > dpts[idx[0]] else "down"
    return pid, dirs


def gauss3(z, y, i):
    yl, y0, yr = np.log(np.maximum([y[i - 1], y[i], y[i + 1]], 1e-12))
    den = yl - 2 * y0 + yr
    if not np.isfinite(den) or den >= 0:
        return np.nan
    return z[i] + 0.5 * (z[i + 1] - z[i]) * (yl - yr) / den


def pass_peaks(df, info, S, sel, window_dpt):
    """Focus peak (three-point Gaussian) per specimen tile and pass, searched within
    window_dpt of --start-dpt. Fly-back frames are skipped."""
    dpts = df.dpt.values
    pid, dirs = pass_labels(df)
    centre, ratio = info["start_dpt"], info["scan_ratio"]
    passes = [(p, np.nonzero(pid == p)[0]) for p in np.unique(pid)]
    rows = []
    for iy, ix in zip(*np.nonzero(sel)):
        y = S[:, iy, ix]
        for p, idx in passes:
            direction = dirs[idx[0]]
            if direction not in ("up", "down"):
                continue
            zz, yy = dpts[idx], y[idx]
            w = np.abs(zz - centre) <= window_dpt
            zz, yy = zz[w], yy[w]
            if len(zz) < 5 or not np.isfinite(yy).all():
                continue
            order = np.argsort(zz)
            zz, yy = zz[order], yy[order]
            if len(np.unique(zz)) < len(zz):          # a plane seen twice (relabelled frames): average
                g = pd.Series(yy).groupby(zz).mean()
                zz, yy = g.index.values, g.values
                if len(zz) < 5:
                    continue
            k = int(np.argmax(yy))
            if k == 0 or k == len(zz) - 1:
                continue
            peak = gauss3(zz, yy, k)
            if np.isfinite(peak):
                rows.append({"tile": f"{iy}_{ix}", "pass": int(p), "direction": direction,
                             "peak_dpt": peak, "peak_mm": peak * ratio})
    return pd.DataFrame(rows, columns=["tile", "pass", "direction", "peak_dpt", "peak_mm"])


def flag_outliers(P, thr_dpt):
    """Mark single-pass peaks more than thr_dpt from the median of all peaks of their tile
    (a noise maximum picked instead of the specimen). Returns P with a 'rejected' column."""
    if P.empty:
        return P.assign(rejected=pd.Series(dtype=bool))
    med = P.groupby("tile").peak_dpt.transform("median")
    return P.assign(rejected=(P.peak_dpt - med).abs() > thr_dpt)


def within_direction(P):
    """Deviation of each single peak from its tile x direction mean; pooled SD."""
    g = P.groupby(["tile", "direction"]).peak_mm
    P = P.assign(dev_mm=P.peak_mm - g.transform("mean"), n=g.transform("size"))
    multi = P[P.n > 1]
    # robust SD (1.4826 x median absolute deviation, corrected for estimating the mean from n
    # passes); the pooled SD is dominated by a few noisy tiles
    n_corr = float(np.sqrt((multi.n / (multi.n - 1)).mean())) if len(multi) else np.nan
    sd = float(1.4826 * np.median(np.abs(multi.dev_mm)) * n_corr) if len(multi) else np.nan
    per_pass = (P.groupby(["pass", "direction"])
                .agg(peak_mm_median=("peak_mm", "median"), dev_mm_median=("dev_mm", "median"),
                     n_tiles=("tile", "size")).reset_index())
    return sd, per_pass


def zigzag_updown(P, info):
    step_mm = (info.get("step_um") or np.nan) / 1000
    if P.empty:
        return {"n_tiles": 0}, pd.DataFrame()
    sd, per_pass = within_direction(P)
    t = P.groupby(["tile", "direction"]).peak_mm.mean().unstack()
    if not {"up", "down"} <= set(t.columns):
        return {"n_tiles": 0, "single_peak_sd_mm": sd}, per_pass
    t = t.dropna(subset=["up", "down"])
    if t.empty:
        return {"n_tiles": 0, "single_peak_sd_mm": sd}, per_pass
    ud = t["up"] - t["down"]
    mid_dpt = 0.5 * (t["up"] + t["down"]) / info["scan_ratio"]
    return {"n_tiles": int(len(ud)), "specimen_dpt": float(mid_dpt.median()),
            "up_minus_down_mm_median": float(ud.median()),
            "up_minus_down_mm_iqr": [float(ud.quantile(0.25)), float(ud.quantile(0.75))],
            "lag_equiv_frames": float(ud.median() / (2 * step_mm)) if np.isfinite(step_mm) else None,
            "single_peak_sd_mm": sd,
            "n_passes": {d: int(per_pass[per_pass.direction == d]["pass"].nunique()) for d in ("up", "down")}}, per_pass


def reference_curves(df, S, sel, P_ref, half_domain=0.8):
    """Per tile: cubic spline of the plane-averaged log sharpness of the upward passes of a
    fine-step zigzag (the reference curve), over its peak +- half_domain dpt."""
    pid, dirs = pass_labels(df)
    z = df.dpt.values
    up = dirs == "up"
    centres = P_ref[P_ref.direction == "up"].groupby("tile").peak_dpt.median()
    refs = {}
    for tile, c in centres.items():
        iy, ix = map(int, tile.split("_"))
        if not sel[iy, ix]:
            continue
        m = up & (np.abs(z - c) <= half_domain)
        g = (pd.DataFrame({"z": z[m], "y": np.log(np.maximum(S[m, iy, ix], 1e-12))})
             .groupby("z").y.mean())
        if len(g) < 7 or not np.isfinite(g.values).all():
            continue
        refs[tile] = (CubicSpline(g.index.values, g.values, extrapolate=False), float(c))
    return refs


def template_shifts(df, S, refs, half_width=0.45, max_shift=0.45):
    """Shift of every pass against its tile's reference curve: the pass's log sharpness
    samples y_k are fitted as R(z_k - delta) + const. Uses the whole curve shape, so the
    result does not depend on where the peak falls between two planes."""
    pid, dirs = pass_labels(df)
    z = df.dpt.values
    grid = np.arange(-max_shift, max_shift + 1e-9, 0.002)
    rows = []
    for tile, (spl, c) in refs.items():
        iy, ix = map(int, tile.split("_"))
        y_all = np.log(np.maximum(S[:, iy, ix], 1e-12))
        for p in np.unique(pid):
            idx = np.nonzero(pid == p)[0]
            d = dirs[idx[0]]
            if d not in ("up", "down"):
                continue
            idx = idx[np.abs(z[idx] - c) <= half_width]
            if len(idx) < 4 or not np.isfinite(y_all[idx]).all():
                continue
            zk, yk = z[idx], y_all[idx]

            def cost(delta):
                r = spl(zk - delta)
                ok = np.isfinite(r)
                if ok.sum() < 4:
                    return np.inf
                res = yk[ok] - r[ok]
                return float(np.mean((res - res.mean()) ** 2))

            costs = np.array([cost(dl) for dl in grid])
            if not np.isfinite(costs).any():
                continue
            k = int(np.nanargmin(costs))
            lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
            best = minimize_scalar(cost, bounds=(lo, hi), method="bounded")
            rows.append({"tile": tile, "pass": int(p), "direction": d,
                         "shift_dpt": float(best.x if np.isfinite(best.fun) else grid[k]),
                         "at_search_limit": bool(k in (0, len(grid) - 1))})
    return pd.DataFrame(rows, columns=["tile", "pass", "direction", "shift_dpt", "at_search_limit"])


def template_updown(T, ratio):
    T = T[~T.at_search_limit]
    t = T.groupby(["tile", "direction"]).shift_dpt.mean().unstack()
    if not {"up", "down"} <= set(t.columns):
        return {}
    ud = (t["up"] - t["down"]).dropna() * ratio
    if ud.empty:
        return {}
    return {"up_minus_down_template_mm": float(ud.median()),
            "up_minus_down_template_mm_iqr": [float(ud.quantile(0.25)), float(ud.quantile(0.75))],
            "n_tiles_template": int(len(ud))}


def reference_test(name, data):
    """Fine-step zigzag of the same family that serves as reference curve for this test."""
    family = re.sub(r"\d+$", "", name)
    if family in ("saw", "zzcal_pre"):
        family = "zzcal"
    cands = [(info.get("step_um") or 1e9, n) for n, (info, *_rest) in data.items()
             if info["kind"] == "zigzag" and re.sub(r"\d+$", "", n) == family]
    return min(cands)[1] if cands else None


def precycle_check(P_pre, P_ref):
    """Each pass of zzcal_pre against the tile x direction means of the test recorded right
    after it with the same step size."""
    ref = P_ref.groupby(["tile", "direction"]).peak_mm.mean().rename("ref_mm")
    X = P_pre.join(ref, on=["tile", "direction"]).dropna(subset=["ref_mm"])
    X = X.assign(dev_mm=X.peak_mm - X.ref_mm)
    return (X.groupby(["pass", "direction"]).agg(dev_mm_median=("dev_mm", "median"),
                                                 n_tiles=("dev_mm", "size")).reset_index())


def saw_vs_zigzag(P_saw, P_zz):
    """Upward sawtooth peaks against the zigzag peaks (same tiles, same step size)."""
    saw = P_saw[P_saw.direction == "up"].groupby("tile").peak_mm.mean().rename("saw")
    z = P_zz.groupby(["tile", "direction"]).peak_mm.mean().unstack()
    if not {"up", "down"} <= set(z.columns):
        return {}
    X = pd.concat([saw, z[["up", "down"]]], axis=1, join="inner").dropna()
    if X.empty:
        return {}
    return {"n_tiles_matched": int(len(X)),
            "saw_minus_zz_up_mm": float((X.saw - X.up).median()),
            "saw_minus_zz_down_mm": float((X.saw - X.down).median()),
            "saw_minus_zz_mid_mm": float((X.saw - 0.5 * (X.up + X.down)).median())}


def hysteresis_analysis(df, info, S, sel):
    ratio = info["scan_ratio"]
    rows = []
    segs = [s for s in df.segment.unique() if isinstance(s, str) and s]
    for iy, ix in zip(*np.nonzero(sel)):
        for seg in segs:
            m = (df.segment == seg).values
            g = pd.DataFrame({"dpt": df.dpt.values[m], "s": S[m, iy, ix]}).groupby("dpt").s.mean()
            z, y = g.index.values, g.values
            k = int(np.nanargmax(y))
            lo, hi = max(0, k - 3), min(len(z), k + 4)
            if hi - lo < 5:
                continue
            c = np.polyfit(z[lo:hi] - z[k], np.log(np.maximum(y[lo:hi], 1e-12)), 2)
            if c[0] >= 0:
                continue
            peak = z[k] - c[1] / (2 * c[0])
            rows.append({"tile": f"{iy}_{ix}", "segment": seg, "direction": seg.rstrip("0123456789"),
                         "peak_dpt": peak})
    P = pd.DataFrame(rows)
    if P.empty:
        return {"n_tiles": 0}, P
    t = P.groupby(["tile", "direction"]).peak_dpt.mean().unstack().dropna()
    d = t["up"] - t["down"]
    rep = P.groupby(["tile", "segment"]).peak_dpt.mean().unstack()
    rep_sd = None
    if {"up1", "up2"} <= set(rep.columns):
        rep_sd = float((rep["up1"] - rep["up2"]).std() / np.sqrt(2) * ratio)
    return {"n_tiles": int(len(d)), "up_minus_down_dpt_median": float(d.median()),
            "up_minus_down_mm_median": float(d.median() * ratio),
            "up_minus_down_mm_iqr": [float(d.quantile(0.25) * ratio), float(d.quantile(0.75) * ratio)],
            "repeat_sd_mm": rep_sd}, P


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def fmt_signed(v, nd=3):
    return "–" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def fmt_iqr(r, key, iqr_key, nd=3):
    v, iqr = r.get(key), r.get(iqr_key)
    if v is None:
        return "–"
    return f"{fmt(v, nd)} [{fmt(iqr[0], nd)}, {fmt(iqr[1], nd)}]" if iqr else fmt(v, nd)


def _mean_or_none(*vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else None


def is_zzcal(r):
    return (r["kind"] == "zigzag" and r["loop"] == "grid" and not r.get("stab_block")
            and not r.get("loopcmp_variant"))


def is_stab(r):
    return r["kind"] == "zigzag" and bool(r.get("stab_block"))


def interpret(results, extras):
    lines = []
    lag = {r["name"]: r.get("lag_frames") for r in results if r["kind"] == "toggle"}
    p25, p15, p10 = (lag.get("prod_toggle_25fps"), lag.get("prod_toggle_15fps"), lag.get("prod_toggle_10fps"))
    if p25 is not None and np.isfinite(p25):
        if p25 > 0.7 and all(v is not None and v < 0.3 for v in (p15, p10)):
            lines.append("- Production loop: lag of about one frame at 25 fps, none at 15 and 10 fps. "
                         "The lag is a timing effect: at 25 fps the next exposure happens before the loop has "
                         "set the lens, because a frame reaches the PC only about one frame interval after its "
                         "exposure. Not an indexing error.")
        elif p25 > 0.7 and all(v is not None and v > 0.7 for v in (p15, p10)):
            lines.append("- Production loop: lag of about one frame at every frame rate. This points to a "
                         "constant off-by-one between frame and setting (buffering or indexing), independent of timing.")
        else:
            lines.append(f"- Production loop lag (frames): 25 fps {fmt(p25)}, 15 fps {fmt(p15)}, 10 fps {fmt(p10)}.")
    trig = {r["name"]: r.get("lag_frames") for r in results if r["name"].startswith("trig_toggle")}
    if trig:
        lines.append("- Triggered loop lag by settle time: " +
                     ", ".join(f"{k.replace('trig_toggle_settle', '')} ms: {fmt(v)}" for k, v in trig.items()) +
                     ". A lag above zero only at short settle times measures the lens settling time.")
    pipe = lag.get("pipe_toggle_25fps")
    if pipe is not None:
        lines.append(f"- Pipelined triggered loop at 25 fps (candidate fix): lag {fmt(pipe)} frames"
                     + (" -> fix works." if np.isfinite(pipe) and abs(pipe) < 0.15 else "."))
    grid = [r for r in results if r["loop"] == "grid"]
    if grid:
        g = next((r for r in grid if r["kind"] == "toggle"), None)
        fi = [r.get("timing_frame_interval_ms_mean") for r in grid if r.get("timing_frame_interval_ms_mean")]
        late = sum(r.get("timing_triggers_late_over_2ms") or 0 for r in grid)
        n = sum(r.get("n_frames") or 0 for r in grid)
        lagged = sum(sum(c for k, c in (r.get("timing_timing_lag_counts") or {}).items() if int(k) != 0)
                     for r in grid)
        mism = sum(r.get("timing_exposure_trigger_mismatches") or 0 for r in grid)
        txt = "- Fixed-grid loop: "
        if g is not None:
            txt += f"toggle lag {fmt(g.get('lag_frames'))} frames; "
        txt += (f"mean camera frame interval {fmt(float(np.mean(fi)), 2) if fi else '–'} ms (target 40.00); "
                f"triggers late by more than 2 ms: {late} of {n}; frames exposed under another command than "
                f"their own (timestamps): {lagged}; frames matched to the wrong trigger: {mism}.")
        lines.append(txt)
    zz = [r for r in results if r["kind"] == "zigzag" and r["loop"] != "grid" and not r.get("loopcmp_variant")]
    hy = next((r.get("up_minus_down_mm_median") for r in results if r["kind"] == "hysteresis"), None)
    if zz:
        lines.append("- Zigzag (±2 mm) up minus down (mm): " +
                     ", ".join(f"{r['name']}: {fmt(r.get('up_minus_down_mm_median'), 3)}" for r in zz)
                     + f"; static hysteresis: {fmt(hy, 3)} mm.")
    cal = [r for r in results if is_zzcal(r) and r["name"] != "zzcal_pre"]
    if cal:
        dpt = np.nanmedian([r.get("specimen_dpt", np.nan) for r in cal])
        lines.append(f"- Production-range zigzag, specimens at {fmt(float(dpt), 2)} dpt: up − down " +
                     ", ".join(f"{fmt(r.get('step_um'), 0)} µm {fmt(r.get('up_minus_down_mm_median'), 3)} mm"
                               for r in cal) +
                     "; curve fit " + ", ".join(fmt(r.get("up_minus_down_template_mm"), 3) for r in cal) +
                     " mm; single-peak spread (robust SD) within one direction " +
                     ", ".join(f"{fmt(r.get('single_peak_sd_mm'), 3)}" for r in cal) + " mm. "
                     "With no timing lag, this offset is lens hysteresis.")
    stab = [r for r in results if is_stab(r)]
    if stab:
        parts = []
        for s in sorted({int(r["step_um"]) for r in stab}):
            v = [r.get("up_minus_down_template_mm", r.get("up_minus_down_mm_median")) for r in stab
                 if int(r["step_um"]) == s]
            v = [x for x in v if x is not None and np.isfinite(x)]
            if v:
                parts.append(f"{s} µm {min(v):.3f} to {max(v):.3f} mm")
        temps = [r.get("lens_temp_mean_c") for r in stab if r.get("lens_temp_mean_c") is not None]
        lines.append(f"- Stability over {len({r['stab_block'] for r in stab})} blocks: up − down (curve fit) "
                     + "; ".join(parts) + (f"; lens {min(temps):.2f} to {max(temps):.2f} °C" if temps else "")
                     + ". Details in the stability section.")
    pre = extras.get("precycle")
    if pre is not None and not pre.empty:
        lines.append("- First sweep after the 1 s hold (zzcal_pre), deviation from the passes recorded after it: " +
                     ", ".join(f"pass {int(r['pass'])} {r['direction']} {fmt_signed(r['dev_mm_median'])} mm"
                               for _, r in pre.iterrows()) + ".")
    saw = [r for r in results if r["kind"] == "sawtooth" and "saw_minus_zz_up_mm" in r]
    if saw:
        lines.append("- Sawtooth upward peaks minus zigzag peaks: " +
                     "; ".join(f"{fmt(r.get('step_um'), 0)} µm: vs up {fmt_signed(r['saw_minus_zz_up_mm'])}, "
                               f"vs down {fmt_signed(r['saw_minus_zz_down_mm'])}, "
                               f"vs mid {fmt_signed(r['saw_minus_zz_mid_mm'])} mm" for r in saw) + ".")
    return lines


def plot_toggle(ax, df, s, info, title):
    n = min(60, len(s))
    c = np.isclose(df.dpt.values[:n], info["start_dpt"]).astype(float)
    ax.step(np.arange(n), c, where="mid", color="0.6", label="commanded in focus")
    ax.plot(np.arange(n), s[:n], "o-", ms=3, color="#2a78d6", label="sharpness index")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("frame", fontsize=7)
    ax.tick_params(labelsize=7)


def report_lines(diag_dir, manifest, results, extras, sel, tile, ref_name):
    L = ["# ETL timing diagnosis", "",
         f"Run: `{diag_dir}`  ",
         f"Camera: {manifest.get('camera_model', '?')}, exposure {manifest.get('exposure_us', '?')} µs, "
         f"timestamp latch supported: {manifest.get('timestamp_latch_supported', '?')}  ",
         f"Specimens in focus at (--start-dpt): {manifest.get('start_dpt', '?')} dpt  ",
         f"Specimen tiles: {int(sel.sum())} (tile {tile} px, reference {ref_name})", "",
         "## Interpretation", ""] + interpret(results, extras)
    tog = [r for r in results if r["kind"] == "toggle"]
    if tog:
        L += ["", "## Toggle tests (which setting did a frame see?)", "",
              "| test | loop | fps | settle ms | lag (frames) | weights lag 0..3 | best integer lag | "
              "timing lag counts | frame interval ms | exposure→retrieve ms | backlog max | ID gaps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in tog:
            L.append(f"| {r['name']} | {r['loop']} | {fmt(r.get('fps_set'), 0)} | {fmt(r.get('settle_ms'), 0)} | "
                     f"{fmt(r.get('lag_frames'))} | {r.get('weights')} | {r.get('best_integer_lag')} | "
                     f"{r.get('timing_timing_lag_counts', '–')} | {fmt(r.get('timing_frame_interval_ms_median'), 1)} | "
                     f"{fmt(r.get('timing_exposure_to_retrieve_ms_median'), 1)} | {r.get('timing_backlog_max', '–')} | "
                     f"{r.get('timing_block_id_gaps', '–')} |")
    grid = [r for r in results if r["loop"] == "grid"]
    if grid:
        L += ["", "## Fixed-grid loop timing", "",
              "| test | frames | camera frame interval ms (mean / median / max) | trigger interval ms | "
              "trigger late ms (median / max) | late > 2 ms | ready check ms | lens command ms | "
              "lens command → exposure ms (median / min) | timing lag counts | wrong trigger | ID gaps | "
              "max frames waiting to be saved |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in grid:
            L.append(f"| {r['name']} | {r.get('n_frames')} | {fmt(r.get('timing_frame_interval_ms_mean'), 2)} / "
                     f"{fmt(r.get('timing_frame_interval_ms_median'), 1)} / "
                     f"{fmt(r.get('timing_frame_interval_ms_max'), 1)} | {fmt(r.get('timing_trigger_interval_ms_median'), 1)} | "
                     f"{fmt(r.get('timing_trigger_late_ms_median'), 1)} / {fmt(r.get('timing_trigger_late_ms_max'), 1)} | "
                     f"{r.get('timing_triggers_late_over_2ms', '–')} | {fmt(r.get('timing_ready_wait_ms_median'), 1)} | "
                     f"{fmt(r.get('timing_lens_call_ms_median'), 1)} | "
                     f"{fmt(r.get('timing_exposure_minus_own_command_ms_median'), 1)} / "
                     f"{fmt(r.get('timing_exposure_minus_own_command_ms_min'), 1)} | "
                     f"{r.get('timing_timing_lag_counts', '–')} | {r.get('timing_exposure_trigger_mismatches', '–')} | "
                     f"{r.get('timing_block_id_gaps', '–')} | {r.get('pending_saves_max', '–')} |")
    zz = [r for r in results if r["kind"] == "zigzag" and r["loop"] != "grid" and not r.get("loopcmp_variant")]
    if zz:
        L += ["", "## Zigzag tests (±2 mm around the specimens)", "",
              "| test | loop | step µm | tiles | up − down (mm), median [IQR] | up − down, curve fit (mm) [IQR] | "
              "equivalent lag (frames) | timing lag counts | backlog max | ID gaps |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in zz:
            L.append(f"| {r['name']} | {r['loop']} | {fmt(r.get('step_um'), 0)} | {r.get('n_tiles')} | "
                     f"{fmt_iqr(r, 'up_minus_down_mm_median', 'up_minus_down_mm_iqr')} | "
                     f"{fmt_iqr(r, 'up_minus_down_template_mm', 'up_minus_down_template_mm_iqr')} | "
                     f"{fmt(r.get('lag_equiv_frames'))} | {r.get('timing_timing_lag_counts', '–')} | "
                     f"{r.get('timing_backlog_max', '–')} | {r.get('timing_block_id_gaps', '–')} |")
    cal = [r for r in results if is_zzcal(r)]
    if cal:
        L += ["", "## Direction offset over the production sweep range (zigzag, fixed-grid loop)", "",
              "| test | step µm | sweep range dpt | tiles | specimens at dpt | up − down (mm), median [IQR] | "
              "up − down, curve fit (mm) [IQR] | single-peak spread within one direction, robust SD (mm) | passes up / down | "
              "peaks rejected |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in cal:
            npass = r.get("n_passes") or {}
            L.append(f"| {r['name']} | {fmt(r.get('step_um'), 0)} | {fmt(r.get('range_start_dpt'), 2)} to "
                     f"{fmt(r.get('range_end_dpt'), 2)} | {r.get('n_tiles')} | {fmt(r.get('specimen_dpt'), 3)} | "
                     f"{fmt_iqr(r, 'up_minus_down_mm_median', 'up_minus_down_mm_iqr')} | "
                     f"{fmt_iqr(r, 'up_minus_down_template_mm', 'up_minus_down_template_mm_iqr')} | "
                     f"{fmt(r.get('single_peak_sd_mm'), 3)} | {npass.get('up', '–')} / {npass.get('down', '–')} | "
                     f"{r.get('peaks_rejected', '–')} |")
        L += ["", "Up − down: three-point Gaussian peak per pass (the production estimator). Curve fit: shift "
                  "of each pass against the tile's sharpness curve from the finest-step zigzag; it does not "
                  "depend on where the peak falls between two planes. If the two differ at coarse steps, the "
                  "difference is interpolation bias, not lens behaviour."]
    pre = extras.get("precycle")
    if pre is not None and not pre.empty:
        ref = extras.get("precycle_ref", "")
        L += ["", "## First sweep after idling (zzcal_pre)", "",
              f"Peak of each zzcal_pre pass minus the mean of the same tile and direction in {ref} "
              f"(recorded right after it, same step size). The last column gives the passes of {ref} "
              "against their own mean, for scale.", "",
              "| pass | direction | zzcal_pre deviation (mm) | tiles |", "|---|---|---|---|"]
        for _, r in pre.iterrows():
            L.append(f"| {int(r['pass'])} | {r['direction']} | {fmt_signed(r['dev_mm_median'])} | {int(r['n_tiles'])} |")
        own = extras.get("precycle_ref_passes")
        if own is not None and not own.empty:
            L += ["", f"{ref}, passes against their own mean: " +
                  ", ".join(f"pass {int(r['pass'])} {r['direction']} {fmt_signed(r['dev_mm_median'])}"
                            for _, r in own.iterrows()) + " mm."]
    saw = [r for r in results if r["kind"] == "sawtooth"]
    if saw:
        L += ["", "## Sawtooth (upward passes only)", "",
              "| test | step µm | tiles | specimens at dpt | single-peak spread, robust SD (mm) | first pass deviation (mm) | "
              "saw − zigzag up (mm) | saw − zigzag down (mm) | saw − zigzag mid (mm) | "
              "curve fit: saw − zigzag up / down (mm) | peaks rejected |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in saw:
            L.append(f"| {r['name']} | {fmt(r.get('step_um'), 0)} | {r.get('n_tiles')} | {fmt(r.get('specimen_dpt'), 3)} | "
                     f"{fmt(r.get('single_peak_sd_mm'), 3)} | {fmt_signed(r.get('first_pass_dev_mm', np.nan))} | "
                     f"{fmt_signed(r.get('saw_minus_zz_up_mm', np.nan))} | "
                     f"{fmt_signed(r.get('saw_minus_zz_down_mm', np.nan))} | "
                     f"{fmt_signed(r.get('saw_minus_zz_mid_mm', np.nan))} | "
                     f"{fmt_signed(r.get('saw_minus_zz_up_template_mm', np.nan))} / "
                     f"{fmt_signed(r.get('saw_minus_zz_down_template_mm', np.nan))} | {r.get('peaks_rejected', '–')} |")
    for r in results:
        if r["kind"] == "hysteresis":
            iqr = r.get("up_minus_down_mm_iqr")
            L += ["", "## Static hysteresis", "",
                  f"Up − down focus position: {fmt(r.get('up_minus_down_dpt_median'), 4)} dpt = "
                  f"{fmt(r.get('up_minus_down_mm_median'), 3)} mm (median of {r.get('n_tiles')} tiles, "
                  f"IQR {fmt(iqr[0], 3) if iqr else '–'} to {fmt(iqr[1], 3) if iqr else '–'} mm); "
                  f"repeatability between the two repeats (SD): {fmt(r.get('repeat_sd_mm'), 3)} mm."]
    L += ["", "## Notes", "",
          "- Timing lag = labelled command index minus the last command completed "
          f"{1000 * ASSUMED_SETTLE_S:.0f} ms before the exposure (camera timestamp mapped to the PC clock).",
          "- Weights are the share of each frame's sharpness explained by the command 0, 1, 2 or 3 frames earlier.",
          "- Up − down: mean upward minus mean downward peak per tile, median over tiles. Single-peak spread: "
          "scatter of single passes around their tile and direction mean, as a robust SD (1.4826 × median "
          "absolute deviation) over all tiles. Single-pass peaks "
          "more than --outlier_dpt from their tile's median are rejected before both (count in the tables; "
          "all peaks with a 'rejected' flag in *_peaks.csv).",
          "- Figures: tile_selection.png, toggle_tests.png. Per-test tables: *_frames.csv, *_peaks.csv, "
          "*_per_pass.csv, *_timing.csv."]
    return L


# ---------------------------------------------------------------------------
# Stability test
# ---------------------------------------------------------------------------
def stability_analysis(results, diag_dir, out_dir, ratio):
    """Per measurement block: direction offset (both estimators) at each step size, lens
    temperature, and the shift of the up/down midpoint relative to the first block.
    Writes stability_summary.csv and stability.png; returns report lines."""
    stab = [r for r in results if is_stab(r)]
    if not stab:
        return []
    rows = []
    for r in stab:
        rows.append({"block": int(r["stab_block"]), "planned_min": r.get("stab_planned_min"),
                     "t_min": (r.get("t_since_run_start_s") or np.nan) / 60, "step_um": int(r["step_um"]),
                     "lens_temp_c": r.get("lens_temp_mean_c"), "n_tiles": r.get("n_tiles"),
                     "up_minus_down_mm": r.get("up_minus_down_mm_median"),
                     "up_minus_down_curve_fit_mm": r.get("up_minus_down_template_mm", np.nan),
                     "midpoint_mm": (r.get("specimen_dpt") or np.nan) * ratio,
                     "single_peak_sd_mm": r.get("single_peak_sd_mm")})
    B = pd.DataFrame(rows).sort_values(["block", "step_um"])
    first = B[B.block == B.block.min()].set_index("step_um").midpoint_mm
    B["midpoint_shift_mm"] = B.midpoint_mm - B.step_um.map(first)
    B.to_csv(os.path.join(out_dir, "stability_summary.csv"), index=False)

    temp_path = os.path.join(diag_dir, "lens_temperature_log.csv")
    T = pd.read_csv(temp_path) if os.path.isfile(temp_path) else None
    steps = sorted(B.step_um.unique())
    L = ["", "## Stability over the session", "",
         "Per block and step size: direction offset (up − down, three-point Gaussian / curve fit), lens "
         "temperature (mean of the readings during the block), and the shift of the up/down midpoint of "
         "the specimens against the first block (focus drift).", "",
         "| block | planned min | start min | lens °C | step µm | up − down (mm) | curve fit (mm) | "
         "midpoint shift (mm) | single-peak spread (mm) | tiles |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, b in B.iterrows():
        L.append(f"| {int(b.block)} | {fmt(b.planned_min, 0)} | {fmt(b.t_min, 1)} | {fmt(b.lens_temp_c, 2)} | "
                 f"{int(b.step_um)} | {fmt(b.up_minus_down_mm, 3)} | {fmt(b.up_minus_down_curve_fit_mm, 3)} | "
                 f"{fmt_signed(b.midpoint_shift_mm)} | {fmt(b.single_peak_sd_mm, 3)} | "
                 f"{int(b.n_tiles) if pd.notna(b.n_tiles) else '–'} |")
    for s in steps:
        d = B[B.step_um == s]
        col = "up_minus_down_curve_fit_mm" if d.up_minus_down_curve_fit_mm.notna().any() else "up_minus_down_mm"
        L.append(f"- {s} µm: up − down from {fmt(d[col].min(), 3)} to {fmt(d[col].max(), 3)} mm "
                 f"(first block {fmt(d[col].iloc[0], 3)}, last {fmt(d[col].iloc[-1], 3)}); midpoint shift up to "
                 f"{fmt(d.midpoint_shift_mm.abs().max(), 3)} mm.")
    if T is not None and T.lens_temp_c.notna().any():
        L.append(f"- Lens temperature from {fmt(T.lens_temp_c.min(), 2)} to {fmt(T.lens_temp_c.max(), 2)} °C "
                 f"(lens_temperature_log.csv, {len(T)} readings; median read time {fmt(T.read_ms.median(), 1)} ms).")

    cols = [ORDINAL_BLUES[1], ORDINAL_BLUES[4]] if len(steps) == 2 else ORDINAL_BLUES
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    ax_t, ax_h, ax_ht, ax_m = axes.ravel()

    def style(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=8, color=INK)
        ax.set_ylabel(ylabel, fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK_MUTED)
        ax.tick_params(labelsize=7, colors=INK_MUTED)

    if T is not None and T.lens_temp_c.notna().any():
        tcol = "t_rel_s" if "t_rel_s" in T else T.columns[0]
        ax_t.plot(T[tcol] / 60, T.lens_temp_c, color=ORDINAL_BLUES[3], lw=1.5)
    else:
        ax_t.text(0.5, 0.5, "no temperature log", ha="center", va="center", transform=ax_t.transAxes,
                  color=INK_MUTED, fontsize=8)
    for t0 in B.drop_duplicates("block").t_min:
        ax_t.axvline(t0, color=GRID, lw=1)
    style(ax_t, "minutes since start", "lens temperature (°C)", "A  Lens temperature (blocks: grey lines)")
    ycol = "up_minus_down_curve_fit_mm" if B.up_minus_down_curve_fit_mm.notna().any() else "up_minus_down_mm"
    for i, s in enumerate(steps):
        d = B[B.step_um == s]
        kw = dict(color=cols[i % len(cols)], lw=2, marker=MARKERS[i % len(MARKERS)], ms=7, mec="white", mew=1.5,
                  label=f"{s} µm")
        ax_h.plot(d.t_min, d[ycol], **kw)
        ax_ht.plot(d.lens_temp_c, d[ycol], **kw)
        ax_m.plot(d.t_min, d.midpoint_shift_mm, **kw)
        for ax, x, y in ((ax_h, d.t_min, d[ycol]), (ax_m, d.t_min, d.midpoint_shift_mm)):
            if len(d):
                ax.annotate(f"{s} µm", (x.iloc[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7, color=INK_MUTED)
    style(ax_h, "minutes since start", "up − down (mm)", "B  Direction offset over time")
    style(ax_ht, "lens temperature (°C)", "up − down (mm)", "C  Direction offset against temperature")
    style(ax_m, "minutes since start", "midpoint shift (mm)", "D  Focus drift of the up/down midpoint")
    ax_m.axhline(0, color=INK_MUTED, lw=0.8)
    for ax in (ax_h, ax_ht, ax_m):
        ax.legend(title="step size", fontsize=7, title_fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stability.png"), dpi=170)
    plt.close(fig)
    L.append("- Figure: stability.png (curve-fit offsets; three-point Gaussian values in stability_summary.csv).")
    return L


# ---------------------------------------------------------------------------
# Loop comparison
# ---------------------------------------------------------------------------
LOOP_LABELS = {"grid1": "fixed grid, SetAsInput once", "gridE": "fixed grid, SetAsInput every frame",
               "prod": "production loop, lag removed"}


def loopcmp_table(results):
    """One row per loop-comparison recording. The lag (from camera timestamps) is removed:
    offset = up − down − 2 × step × mean lag."""
    rows = []
    for r in results:
        if not r.get("loopcmp_variant") or r["kind"] != "zigzag":
            continue
        step_mm = r["step_um"] / 1000
        lag = r.get("timing_timing_lag_mean")
        lag = 0.0 if lag is None or not np.isfinite(lag) else float(lag)
        residual_lag = 0.0 if r.get("relabelled_by_timestamps") else lag
        ud_g = r.get("up_minus_down_mm_median", np.nan)
        ud_t = r.get("up_minus_down_template_mm", np.nan)
        rows.append({"name": r["name"], "rep": int(r["loopcmp_rep"]), "loop": r["loopcmp_variant"],
                     "step_um": int(r["step_um"]), "t_min": (r.get("t_since_run_start_s") or np.nan) / 60,
                     "lens_temp_c": r.get("lens_temp_mean_c"), "lag_frames": lag,
                     "lag_counts": r.get("timing_timing_lag_counts"), "n_tiles": r.get("n_tiles"),
                     "up_minus_down_mm": ud_g, "up_minus_down_curve_fit_mm": ud_t,
                     "offset_mm": ud_g - 2 * step_mm * residual_lag,
                     "offset_curve_fit_mm": ud_t - 2 * step_mm * residual_lag,
                     "frames_lag_over_1": r.get("frames_relabelled_by_more_than_1"),
                     "single_peak_sd_mm": r.get("single_peak_sd_mm")})
    return pd.DataFrame(rows)


def loopcmp_analysis(results, out_dir):
    C = loopcmp_table(results)
    if C.empty:
        return [], None
    C.to_csv(os.path.join(out_dir, "loop_comparison.csv"), index=False)
    ycol = "offset_curve_fit_mm" if C.offset_curve_fit_mm.notna().any() else "offset_mm"
    loops = [v for v in ("grid1", "gridE", "prod") if v in set(C.loop)]
    steps = sorted(C.step_um.unique())
    M = (C.groupby(["loop", "step_um"])
         .agg(offset=(ycol, "mean"), offset_g=("offset_mm", "mean"),
              rep_diff=(ycol, lambda x: x.max() - x.min()), lag=("lag_frames", "mean"),
              temp=("lens_temp_c", "mean"), n=(ycol, "size")).reset_index())
    ref = M[M.loop == "grid1"].set_index("step_um").offset
    M["minus_grid1"] = M.offset - M.step_um.map(ref)

    L = ["", "## Loop comparison (direction offset with the lag removed)", "",
         "Offset = up − down after removing the lag. The fixed-grid loop has none. For the production loop, "
         "each frame is assigned the setting that was active at its exposure (camera timestamps), so the "
         "lag column shows what was removed; up − down of those tests is already after removal. "
         "Recorded in the order "
         "grid1 - gridE - prod, then prod - gridE - grid1, so a linear drift cancels in the mean of the two "
         "repeats.", "",
         "| test | min | lens °C | lag (frames) | frames with lag > 1 | up − down (mm) | curve fit (mm) | "
         "offset (mm) | offset, curve fit (mm) | tiles |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in C.sort_values("t_min").iterrows():
        lag_over = "–" if pd.isna(r.frames_lag_over_1) else int(r.frames_lag_over_1)
        L.append(f"| {r['name']} | {fmt(r.t_min, 1)} | {fmt(r.lens_temp_c, 2)} | {fmt(r.lag_frames, 2)} | {lag_over} | "
                 f"{fmt(r.up_minus_down_mm, 3)} | {fmt(r.up_minus_down_curve_fit_mm, 3)} | {fmt(r.offset_mm, 3)} | "
                 f"{fmt(r.offset_curve_fit_mm, 3)} | {r.n_tiles} |")
    L += ["", "Mean of the two repeats (" + ("curve fit" if ycol == "offset_curve_fit_mm" else
                                             "three-point Gaussian") + "):", "",
          "| loop | step µm | offset (mm) | difference to grid1 (mm) | repeat difference (mm) | "
          "offset, three-point Gaussian (mm) | lag (frames) |", "|---|---|---|---|---|---|---|"]
    for _, m in M.sort_values(["step_um", "loop"], key=lambda c: c.map(
            {v: i for i, v in enumerate(loops)}) if c.name == "loop" else c).iterrows():
        L.append(f"| {LOOP_LABELS.get(m.loop, m.loop)} | {int(m.step_um)} | {fmt(m.offset, 3)} | "
                 f"{fmt_signed(m.minus_grid1)} | {fmt(m.rep_diff, 3)} | {fmt(m.offset_g, 3)} | {fmt(m.lag, 2)} |")
    L.append("- The difference between loops is meaningful when it exceeds the repeat difference of both loops.")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cols = [ORDINAL_BLUES[1], ORDINAL_BLUES[4]] if len(steps) == 2 else ORDINAL_BLUES
    width = 0.44
    for i, s in enumerate(steps):
        col = cols[i % len(cols)]
        dx = (i - (len(steps) - 1) / 2) * width
        for j, v in enumerate(loops):
            d = C[(C.step_um == s) & (C.loop == v)]
            ax.scatter(np.full(len(d), j + dx), d[ycol], s=40, color=col, marker=MARKERS[i % len(MARKERS)],
                       edgecolors="white", linewidths=1.2, zorder=3,
                       label=f"{s} µm (single recordings)" if j == 0 else None)
            m = d[ycol].mean()
            ax.hlines(m, j + dx - 0.08, j + dx + 0.08, color=col, lw=2.5, zorder=2)
            ax.annotate(f"{m:.2f}", (j + dx + 0.09, m), va="center", fontsize=7, color=INK_MUTED)
    ax.set_xticks(range(len(loops)))
    ax.set_xticklabels([LOOP_LABELS.get(v, v).replace(", ", ",\n") for v in loops], fontsize=8, color=INK)
    ax.set_ylabel("direction offset, up − down (mm)", fontsize=9, color=INK)
    ax.set_title("Direction offset by acquisition loop (bars: mean of two repeats)", fontsize=9, color=INK,
                 loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK_MUTED)
    ax.tick_params(labelsize=7, colors=INK_MUTED)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loop_comparison.png"), dpi=170)
    plt.close(fig)
    L.append("- Figure: loop_comparison.png; all values in loop_comparison.csv.")
    return L, M


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def analyze_run(diag_dir, args):
    out_dir = os.path.join(diag_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    manifest, tests = load_tests(diag_dir)
    print(f"\n### {diag_dir}\nComputing tile sharpness ...")
    data, timing_frames = {}, {}
    for info in tests:
        df, timing_df, commands = frame_table(info)
        if df.empty:
            print(f"  {info['name']}: no frames, skipped")
            continue
        S, shape = sharpness_stack(df, info["folder_path"], args.tile, args.workers, args.recompute)
        data[info["name"]] = (info, df, commands, S, shape)
        timing_frames[info["name"]] = timing_df
        print(f"  {info['name']}: {len(df)} frames" +
              (f" (of {len(timing_df)} acquired)" if len(timing_df) != len(df) else ""))

    ref_name = next((n for n in ("trig_toggle_settle25", "grid_toggle_25fps", "pipe_toggle_25fps",
                                 "prod_toggle_10fps", "hysteresis") if n in data), next(iter(data)))
    info, df, _, S, shape = data[ref_name]
    keep = watermark_mask(shape, args.tile)
    sel, ratio = select_tiles(S, df, info, keep, args.min_ratio)
    print(f"Specimen tiles: {int(sel.sum())} of {sel.size} (reference test {ref_name})")
    if sel.sum() < 3:
        sys.exit("ERROR: fewer than 3 specimen tiles; lower --min_ratio or check focus at --start-dpt")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.log10(np.maximum(ratio, 1e-3)), cmap="Greys")
    ys, xs = np.nonzero(sel)
    ax.scatter(xs, ys, s=12, facecolors="none", edgecolors="#eb6834")
    ax.set_title("log10 sharpness ratio in/out of focus; circles = tiles used", fontsize=8)
    fig.savefig(os.path.join(out_dir, "tile_selection.png"), dpi=150)
    plt.close(fig)

    results, toggles, peaks = [], [], {}
    for name, (info, df, commands, S, shape) in data.items():
        tf = timing_frames[name]
        temps = pd.to_numeric(tf["lens_temp_c"], errors="coerce") if "lens_temp_c" in tf else pd.Series(dtype=float)
        r = {"name": name, "kind": info["kind"], "loop": info["loop"], "fps_set": info.get("fps_set"),
             "settle_ms": info.get("settle_ms"), "step_um": info.get("step_um"), "n_frames": len(tf),
             "n_frames_saved": len(df),
             "range_start_dpt": info.get("range_start_dpt"), "range_end_dpt": info.get("range_end_dpt"),
             "pending_saves_max": info.get("pending_saves_max"), "stab_block": info.get("stab_block"),
             "stab_planned_min": info.get("stab_planned_min"),
             "t_since_run_start_s": info.get("t_since_run_start_s"),
             "loopcmp_rep": info.get("loopcmp_rep"), "loopcmp_variant": info.get("loopcmp_variant"),
             "lens_temp_mean_c": float(temps.mean()) if temps.notna().any() else _mean_or_none(
                 info.get("lens_temp_start_c"), info.get("lens_temp_end_c"))}
        timing, dft = timing_analysis(timing_frames[name], commands, info["folder_path"])
        r.update({f"timing_{k}": v for k, v in timing.items()})
        if info["kind"] == "toggle":
            s = sharpness_index(S, sel)
            r.update(toggle_lag(df, info, s))
            toggles.append((name, df, s, info))
            pd.DataFrame({"frame_index": df.frame_index, "dpt": df.dpt, "sharpness_index": s}).to_csv(
                os.path.join(out_dir, f"{name}_frames.csv"), index=False)
        elif info["kind"] in ("zigzag", "sawtooth"):
            if info.get("loopcmp_variant") == "prod" and dft is not None and "timing_lag" in dft:
                df = relabel_by_timestamps(df, dft, commands)
                data[name] = (info, df, commands, S, shape)
                r["relabelled_by_timestamps"] = True
                r["frames_relabelled_by_more_than_1"] = int((df.timing_lag_used > 1).sum())
            P = flag_outliers(pass_peaks(df, info, S, sel, args.zz_window_dpt), args.outlier_dpt)
            P.to_csv(os.path.join(out_dir, f"{name}_peaks.csv"), index=False)
            r["peaks_rejected"] = f"{int(P.rejected.sum())} of {len(P)}"
            P = P[~P.rejected].drop(columns="rejected")
            peaks[name] = P
            if info["kind"] == "zigzag":
                z, per_pass = zigzag_updown(P, info)
                r.update(z)
            else:
                sd, per_pass = within_direction(P) if not P.empty else (np.nan, pd.DataFrame())
                up = P[P.direction == "up"]
                r.update({"n_tiles": int(up.tile.nunique()), "single_peak_sd_mm": sd,
                          "specimen_dpt": float(up.groupby("tile").peak_dpt.mean().median()) if len(up) else np.nan})
                if not per_pass.empty:
                    first = per_pass[per_pass.direction == "up"].sort_values("pass").head(1)
                    r["first_pass_dev_mm"] = float(first.dev_mm_median.iloc[0]) if len(first) else np.nan
            if not per_pass.empty:
                per_pass.to_csv(os.path.join(out_dir, f"{name}_per_pass.csv"), index=False)
        elif info["kind"] == "hysteresis":
            h, P = hysteresis_analysis(df, info, S, sel)
            r.update(h)
            P.to_csv(os.path.join(out_dir, f"{name}_peaks.csv"), index=False)
        if dft is not None:
            dft.drop(columns=["path"], errors="ignore").to_csv(os.path.join(out_dir, f"{name}_timing.csv"), index=False)
        results.append(r)
        print(f"  {name}: done")

    # second estimator of the direction offset: shift against a fine-step reference curve
    ref_cache, templates = {}, {}
    for r in results:
        if r["kind"] not in ("zigzag", "sawtooth"):
            continue
        ref = reference_test(r["name"], data)
        if ref is None or ref not in peaks or peaks[ref].empty:
            continue
        if ref not in ref_cache:
            _, df_r, _, S_r, _ = data[ref]
            ref_cache[ref] = reference_curves(df_r, S_r, sel, peaks[ref])
        if not ref_cache[ref]:
            continue
        info, df, _, S, _ = data[r["name"]]
        T = template_shifts(df, S, ref_cache[ref])
        T.to_csv(os.path.join(out_dir, f"{r['name']}_template_shifts.csv"), index=False)
        templates[r["name"]] = T
        r["template_reference"] = ref
        if r["kind"] == "zigzag":
            r.update(template_updown(T, info["scan_ratio"]))
    for r in results:
        zz_name = f"zzcal{int(r['step_um'])}" if r["kind"] == "sawtooth" and r.get("step_um") else None
        if zz_name in templates and r["name"] in templates:
            ratio = data[r["name"]][0]["scan_ratio"]
            sw = templates[r["name"]]
            sw = sw[(sw.direction == "up") & ~sw.at_search_limit].groupby("tile").shift_dpt.mean()
            zt = templates[zz_name]
            zt = zt[~zt.at_search_limit].groupby(["tile", "direction"]).shift_dpt.mean().unstack()
            if {"up", "down"} <= set(zt.columns):
                X = pd.concat([sw.rename("saw"), zt[["up", "down"]]], axis=1, join="inner").dropna()
                if len(X):
                    r["saw_minus_zz_up_template_mm"] = float((X.saw - X.up).median() * ratio)
                    r["saw_minus_zz_down_template_mm"] = float((X.saw - X.down).median() * ratio)

    extras = {}
    pre_info = data.get("zzcal_pre", (None,))[0]
    if pre_info is not None:
        ref = pre_info.get("precycle_of")
        if ref in peaks and not peaks[ref].empty and not peaks["zzcal_pre"].empty:
            extras["precycle"] = precycle_check(peaks["zzcal_pre"], peaks[ref])
            extras["precycle_ref"] = ref
            extras["precycle_ref_passes"] = within_direction(peaks[ref])[1]
            extras["precycle"].to_csv(os.path.join(out_dir, "zzcal_pre_vs_later.csv"), index=False)
    for r in results:
        if r["kind"] == "sawtooth":
            zz_name = f"zzcal{int(r['step_um'])}" if r.get("step_um") else None
            if zz_name in peaks and not peaks[zz_name].empty and not peaks[r["name"]].empty:
                r.update(saw_vs_zigzag(peaks[r["name"]], peaks[zz_name]))

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(out_dir, "diagnosis_summary.csv"), index=False)
    cal = summary[(summary.kind == "zigzag") & (summary.loop == "grid") & (summary.name != "zzcal_pre")]
    if not cal.empty:
        cols = [c for c in ("name", "step_um", "n_tiles", "specimen_dpt", "up_minus_down_mm_median",
                            "up_minus_down_mm_iqr", "up_minus_down_template_mm", "up_minus_down_template_mm_iqr",
                            "single_peak_sd_mm", "range_start_dpt", "range_end_dpt")
                if c in cal.columns]
        c = cal[cols].copy()
        c.insert(0, "run", os.path.basename(os.path.normpath(diag_dir)))
        c.to_csv(os.path.join(out_dir, "zzcal_summary.csv"), index=False)

    if toggles:
        fig, axes = plt.subplots(len(toggles), 1, figsize=(7, 1.6 * len(toggles)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (name, df, s, info) in zip(axes, toggles):
            plot_toggle(ax, df, s, info, name)
        axes[0].legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "toggle_tests.png"), dpi=150)
        plt.close(fig)

    lines = report_lines(diag_dir, manifest, results, extras, sel, args.tile, ref_name)
    stab_lines = stability_analysis(results, diag_dir, out_dir, float(manifest.get("scan_ratio") or
                                                                      next(iter(data.values()))[0]["scan_ratio"]))
    lc_lines, lc_means = loopcmp_analysis(results, out_dir)
    for extra in (stab_lines, lc_lines):
        if extra:
            k = lines.index("## Notes") if "## Notes" in lines else len(lines)
            lines = lines[:k] + extra[1:] + [""] + lines[k:]
    if lc_means is not None and not lc_means.empty:
        k = lines.index("## Interpretation") + 2 if "## Interpretation" in lines else len(lines)
        parts = []
        for s in sorted(lc_means.step_um.unique()):
            d = lc_means[lc_means.step_um == s].set_index("loop")
            parts.append(f"{int(s)} µm " + ", ".join(f"{v} {d.offset[v]:.3f}" for v in ("grid1", "gridE", "prod")
                                                     if v in d.index) + " mm")
        lines.insert(k, "- Loop comparison, direction offset with the lag removed (mean of two repeats): "
                        + "; ".join(parts) + ". Compare the differences with the repeat differences in the "
                        "loop-comparison section.")
    with open(os.path.join(out_dir, "diagnosis_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport written to {os.path.join(out_dir, 'diagnosis_report.md')}")
    return results


# ---------------------------------------------------------------------------
# Several runs: direction offset against lens power
# ---------------------------------------------------------------------------
def combine(diag_dirs, all_results, out_dir):
    rows = []
    for d, results in zip(diag_dirs, all_results):
        for r in results:
            if is_zzcal(r) and r["name"] != "zzcal_pre" and r.get("n_tiles"):
                iqr = r.get("up_minus_down_mm_iqr") or [np.nan, np.nan]
                tiq = r.get("up_minus_down_template_mm_iqr") or [np.nan, np.nan]
                rows.append({"run": os.path.basename(os.path.normpath(d)), "step_um": int(r["step_um"]),
                             "specimen_dpt": r.get("specimen_dpt"), "n_tiles": r.get("n_tiles"),
                             "up_minus_down_mm": r.get("up_minus_down_mm_median"),
                             "iqr_low_mm": iqr[0], "iqr_high_mm": iqr[1],
                             "up_minus_down_curve_fit_mm": r.get("up_minus_down_template_mm", np.nan),
                             "curve_fit_iqr_low_mm": tiq[0], "curve_fit_iqr_high_mm": tiq[1],
                             "single_peak_sd_mm": r.get("single_peak_sd_mm")})
    if not rows:
        print("\nNo production-range zigzag tests (zzcal...) found; nothing to combine.")
        return
    C = pd.DataFrame(rows)
    C["run_dpt"] = C.groupby("run").specimen_dpt.transform("median")
    C = C.sort_values(["run_dpt", "step_um"])
    os.makedirs(out_dir, exist_ok=True)
    C.drop(columns="run_dpt").to_csv(os.path.join(out_dir, "diagnosis_combined.csv"), index=False)

    steps = sorted(C.step_um.unique())

    def table(col, title):
        piv = C.pivot_table(index=["run_dpt", "run"], columns="step_um", values=col)
        out = ["", title, "", "| specimens at dpt | run | " + " | ".join(f"{s} µm" for s in steps) + " |",
               "|---|---|" + "---|" * len(steps)]
        for (dpt, run), row in piv.iterrows():
            out.append(f"| {dpt:+.2f} | {run} | " + " | ".join(fmt(row.get(s, np.nan), 3) for s in steps) + " |")
        return out

    L = ["# Direction offset against lens power (combined runs)", "",
         "Fixed-grid loop, zigzag over the production sweep range; medians over specimen tiles."]
    L += table("up_minus_down_mm", "Up − down (mm), three-point Gaussian peaks (production estimator):")
    if C.up_minus_down_curve_fit_mm.notna().any():
        L += table("up_minus_down_curve_fit_mm", "Up − down (mm), curve fit against the finest-step zigzag "
                                                  "(independent of where the peak falls between planes):")
    L += table("single_peak_sd_mm", "Single-peak spread within one direction, robust SD (mm):")
    with open(os.path.join(out_dir, "diagnosis_combined.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))

    use_fit = C.up_minus_down_curve_fit_mm.notna().any()
    ycol, lo_col, hi_col = (("up_minus_down_curve_fit_mm", "curve_fit_iqr_low_mm", "curve_fit_iqr_high_mm")
                            if use_fit else ("up_minus_down_mm", "iqr_low_mm", "iqr_high_mm"))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, s in enumerate(steps):
        d = C[C.step_um == s].dropna(subset=[ycol]).sort_values("specimen_dpt")
        col = ORDINAL_BLUES[min(i, len(ORDINAL_BLUES) - 1)]
        ax.vlines(d.specimen_dpt, d[lo_col], d[hi_col], color=col, lw=1)
        ax.plot(d.specimen_dpt, d[ycol], "-", color=col, lw=2, marker=MARKERS[i % len(MARKERS)],
                ms=7, mec="white", mew=1.5, label=f"{s} µm")
        if len(d):
            ax.annotate(f"{s} µm", (d.specimen_dpt.iloc[-1], d[ycol].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_MUTED)
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("lens power at the specimens (dpt)", fontsize=9, color=INK)
    ax.set_ylabel("up − down (mm)", fontsize=9, color=INK)
    ax.set_title("Direction offset (" + ("curve fit" if use_fit else "three-point Gaussian peaks") +
                 "; median, IQR over specimen tiles)", fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK_MUTED)
    ax.tick_params(labelsize=8, colors=INK_MUTED)
    ax.legend(title="step size", fontsize=8, title_fontsize=8, frameon=False, loc="upper left")
    xmin, xmax = C.specimen_dpt.min(), C.specimen_dpt.max()
    ax.set_xlim(xmin - 0.2, xmax + 0.45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diagnosis_combined.png"), dpi=200)
    plt.close(fig)
    print(f"\nCombined table and figure written to {out_dir} (diagnosis_combined.md/.csv/.png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag_dir", required=True, nargs="+", help="one or more run folders")
    ap.add_argument("--out_dir", default=None,
                    help="folder for the combined table and figure (default: parent of the first run folder)")
    ap.add_argument("--tile", type=int, default=192, help="tile size in px")
    ap.add_argument("--min_ratio", type=float, default=1.1,
                    help="minimum in-focus / defocused sharpness ratio of a specimen tile "
                         "(tiles must also stand out from the background tiles)")
    ap.add_argument("--zz_window_dpt", type=float, default=0.6,
                    help="zigzag and sawtooth peaks are searched within this distance of --start-dpt")
    ap.add_argument("--outlier_dpt", type=float, default=0.25,
                    help="single-pass peaks further than this from their tile's median are rejected")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--recompute", action="store_true", help="ignore cached tile sharpness")
    args = ap.parse_args()
    for d in args.diag_dir:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: folder not found: {d}")
    all_results = [analyze_run(d, args) for d in args.diag_dir]
    if len(args.diag_dir) > 1:
        combine(args.diag_dir, all_results,
                args.out_dir or os.path.dirname(os.path.normpath(args.diag_dir[0])))


if __name__ == "__main__":
    main()
