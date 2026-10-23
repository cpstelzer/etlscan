#!/usr/bin/env python
"""
Axial localization experiment: focus peaks with sub-step interpolation.

For every recording listed in exp_design.csv and every reference specimen
(specimen_XY_coordinates.csv), the script
  1) finds the specimen's track in *all_particle_features.csv by its XY
     position (nearest track within --tol_px; fragments of the same
     specimen without overlapping frames are merged),
  2) splits the zigzag scan into passes (monotonic runs of z_mm),
  3) takes the frame of maximal sharpness in each pass (the focus peak,
     identical to the peaks in filtered_in_focus.csv),
  4) interpolates the peak position from the peak frame and its two
     neighbouring planes:
        z_parabola  parabola through the three sharpness values
        z_gauss     parabola through log(sharpness), i.e. a Gaussian peak
     Both give z_peak + delta * step, with |delta| <= 0.5.
One output row per specimen and pass. Passes whose peak lies on the first
or last plane of the pass (turning point of the zigzag), or whose
neighbouring planes are missing, keep z_peak but get no interpolated value
(see column flag).

Usage (from the repository folder, in the pipeline's conda environment):
  python scripts/extract_focus_peaks_interp.py ^
      --output_dir C:/sandbox/data/output/test ^
      --design C:/path/to/exp_design.csv ^
      --reference C:/path/to/specimen_XY_coordinates.csv ^
      --exclude_depths 14
Output: focus_peaks_interp.csv in output_dir (or --out).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHARP = "variance_laplacian"
NEEDED = ["frame", "X", "Y", "z_mm", "dioptries", "track_id", SHARP]


def die(msg):
    sys.exit("ERROR: " + msg)


def read_checked(path, what):
    if not path.is_file():
        die(f"{what} not found: {path}")
    return pd.read_csv(path)


def norm_cols(df, mapping, where):
    """Rename columns case-insensitively, e.g. {'specimen': ['specimen', 'specimen_id']}."""
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    for target, cands in mapping.items():
        hit = next((lower[c.lower()] for c in cands if c.lower() in lower), None)
        if hit is None:
            die(f"{where}: no column for '{target}' (tried {cands}); found {list(df.columns)}")
        ren[hit] = target
    return df.rename(columns=ren)


def find_features_file(output_dir, plate, well):
    recs = sorted(p for p in output_dir.glob(f"*_plate{plate}_well{well}") if p.is_dir())
    if len(recs) != 1:
        return None, ("recording_not_found" if not recs else "multiple_recordings")
    hits = sorted(recs[0].rglob("*all_particle_features*.csv"))
    if len(hits) != 1:
        return None, ("features_file_not_found" if not hits else "multiple_features_files")
    return hits[0], "ok"


def scan_schedule(feat):
    """frame -> z, dpt, pass index, direction, plus the step size (mm)."""
    sched = (feat.groupby("frame")[["z_mm", "dioptries"]].first()
             .sort_index().reset_index())
    dz = np.diff(sched.z_mm.values)
    step = float(np.round(np.median(np.abs(dz[dz != 0])), 4))
    sign = np.sign(dz)
    pass_id = np.zeros(len(sched), int)
    for i in range(1, len(sched)):
        # a new pass starts after the direction of z changes
        pass_id[i] = pass_id[i - 1] + (1 if i >= 2 and sign[i - 1] != sign[i - 2] else 0)
    sched["pass_index"] = pass_id + 1
    first_last = sched.groupby("pass_index").z_mm.agg(["first", "last"])
    sched["direction"] = sched.pass_index.map(pd.Series(
        np.where(first_last["last"] > first_last["first"], "up", "down"),
        index=first_last.index))
    return sched, step


def match_specimen(feat, x0, y0, tol):
    """Track ids of the specimen: nearest track within tol, merged with
    other tracks within tol that share no frames with it (fragments)."""
    d = np.hypot(feat.X - x0, feat.Y - y0)
    per_track = d.groupby(feat.track_id).min().sort_values()
    within = per_track[per_track <= tol]
    if within.empty:
        return [], (per_track.iloc[0] if len(per_track) else np.nan), "no_match_within_tol"
    chosen = [within.index[0]]
    frames = set(feat.loc[feat.track_id == chosen[0], "frame"])
    status = "ok"
    for tid in within.index[1:]:
        f = set(feat.loc[feat.track_id == tid, "frame"])
        if frames & f:
            status = "ok_other_object_within_tol"   # second object nearby, not merged
        else:
            chosen.append(tid)
            frames |= f
    if len(chosen) > 1 and status == "ok":
        status = "ok_fragments_merged"
    return chosen, within.iloc[0], status


def vertex(zm, z0, zp, sm, s0, sp, log=False):
    """Vertex of the parabola through (zm,sm), (z0,s0), (zp,sp)."""
    y = np.log([sm, s0, sp]) if log else np.array([sm, s0, sp], float)
    den = y[0] - 2 * y[1] + y[2]
    if not np.all(np.isfinite(y)) or den >= 0:
        return np.nan, np.nan
    delta = 0.5 * (y[0] - y[2]) / den          # in units of the step
    return z0 + delta * (zp - z0), delta


def peaks_for_specimen(series, sched, step):
    """series: rows of one specimen (frame, z, sharpness, ...) -> one row per pass."""
    s = series.merge(sched[["frame", "pass_index", "direction"]], on="frame")
    out = []
    for p, g in s.groupby("pass_index"):
        g = g.sort_values("z_mm").reset_index(drop=True)
        i = int(g[SHARP].values.argmax())
        pk = g.iloc[i]
        row = {"pass_index": p, "direction": g.direction.iloc[0],
               "n_planes_in_pass": len(g), "peak_frame": pk.frame,
               "z_peak": pk.z_mm, "dpt_peak": pk.dioptries, "S_peak": pk[SHARP],
               "S_lower": np.nan, "S_upper": np.nan,
               "z_parabola": np.nan, "z_gauss": np.nan,
               "delta_parabola": np.nan, "delta_gauss": np.nan,
               "dpt_gauss": np.nan, "flag": "interpolated"}
        plan_edges = sched.loc[sched.pass_index == p, "z_mm"]
        if pk.z_mm in (plan_edges.min(), plan_edges.max()):
            row["flag"] = "peak_at_pass_end"   # turning point of the zigzag
        elif i == 0 or i == len(g) - 1:
            row["flag"] = "neighbour_missing"
        else:
            lo, hi = g.iloc[i - 1], g.iloc[i + 1]
            if (abs(pk.z_mm - lo.z_mm - step) > 0.01 * step
                    or abs(hi.z_mm - pk.z_mm - step) > 0.01 * step):
                row["flag"] = "neighbour_missing"
            else:
                row["S_lower"], row["S_upper"] = lo[SHARP], hi[SHARP]
                row["z_parabola"], row["delta_parabola"] = vertex(
                    lo.z_mm, pk.z_mm, hi.z_mm, lo[SHARP], pk[SHARP], hi[SHARP])
                row["z_gauss"], row["delta_gauss"] = vertex(
                    lo.z_mm, pk.z_mm, hi.z_mm, lo[SHARP], pk[SHARP], hi[SHARP], log=True)
                if not np.isfinite(row["delta_gauss"]):
                    row["flag"] = "not_concave"
                else:
                    row["dpt_gauss"] = pk.dioptries + row["delta_gauss"] * (hi.dioptries - pk.dioptries)
        row["X_peak"], row["Y_peak"] = pk.X, pk.Y
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tol_px", type=float, default=10.0)
    ap.add_argument("--exclude_depths", type=float, nargs="*", default=[])
    args = ap.parse_args()

    if not args.output_dir.is_dir():
        die(f"output_dir not found: {args.output_dir}")
    design = norm_cols(read_checked(args.design, "design file"),
                       {"plate": ["plate"], "well": ["well"],
                        "water_depth_mm": ["water_depth_mm", "water_depth"],
                        "scan_step_um": ["scan_step_um", "scan_step", "step_um"]},
                       "design file")
    ref = norm_cols(read_checked(args.reference, "reference file"),
                    {"water_depth_mm": ["water_depth_mm"],
                     "specimen": ["specimen", "specimen_id"],
                     "X_ref": ["X", "x"], "Y_ref": ["Y", "y"]}, "reference file")
    if design.duplicated(["water_depth_mm", "scan_step_um"]).any():
        die("design file has more than one recording per water depth and step size")
    design = design[~design.water_depth_mm.isin(args.exclude_depths)]
    out_path = args.out or args.output_dir / "focus_peaks_interp.csv"

    rows = []
    for _, d in design.sort_values(["water_depth_mm", "scan_step_um"]).iterrows():
        plate, well = int(d.plate), int(d.well)
        base = {"water_depth_mm": d.water_depth_mm, "scan_step_um": d.scan_step_um,
                "plate": plate, "well": well}
        path, status = find_features_file(args.output_dir, plate, well)
        refs = ref[ref.water_depth_mm == d.water_depth_mm]
        if path is None:
            rows += [dict(base, specimen=r.specimen, status=status) for _, r in refs.iterrows()]
            print(f"{d.water_depth_mm:g} mm, {d.scan_step_um:g} um: {status}")
            continue
        feat = pd.read_csv(path)
        missing = [c for c in NEEDED if c not in feat.columns]
        if missing:
            die(f"{path.name} lacks columns {missing}")
        sched, step = scan_schedule(feat)
        if abs(step * 1000 - d.scan_step_um) > 1:
            print(f"WARNING: {path.name}: step in data {step * 1000:g} um, design says {d.scan_step_um:g} um")
        for _, r in refs.iterrows():
            tids, dist, st = match_specimen(feat, r.X_ref, r.Y_ref, args.tol_px)
            info = dict(base, specimen=r.specimen, X_ref=r.X_ref, Y_ref=r.Y_ref,
                        step_mm_data=step, status=st, match_dist_px=dist,
                        track_ids=";".join(str(int(t)) for t in tids))
            if not tids:
                rows.append(info)
                continue
            ser = (feat[feat.track_id.isin(tids)][["frame", "X", "Y", "z_mm", "dioptries", SHARP]]
                   .sort_values(SHARP, ascending=False).drop_duplicates("frame"))
            for pr in peaks_for_specimen(ser, sched, step):
                rows.append(dict(info, **pr))
        print(f"{d.water_depth_mm:g} mm, {d.scan_step_um:g} um: {path.name}, step {step * 1000:g} um, "
              f"{sched.pass_index.max()} passes")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, na_rep="NA")
    print(f"\nWrote {out_path} ({len(out)} rows)")

    ok = out[out.get("z_gauss").notna()] if "z_gauss" in out else out.iloc[0:0]
    if len(ok):
        # quick look: up - down difference per depth and step (mean over specimens)
        m = ok.groupby(["water_depth_mm", "scan_step_um", "specimen", "direction"]).z_gauss.mean().unstack()
        if {"up", "down"} <= set(m.columns):
            diff = (m.up - m.down).groupby(level=[0, 1]).agg(["mean", "std", "count"])
            diff["lag_frames"] = diff["mean"] / (2 * diff.index.get_level_values(1) / 1000)
            print("\nUp minus down pass (z_gauss, mm), mean over specimens:")
            print(diff.round(3).to_string())
    print("\nFlags:")
    lab = out["flag"].fillna(out["status"]) if "flag" in out else out["status"]
    print(lab.value_counts().to_string())


if __name__ == "__main__":
    main()
