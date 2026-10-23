"""Correct the one-frame lag of recordings made with the free-running acquisition loop.

Background (ETL timing diagnosis, 20.10.2026): with the free-running production loop
at 25 fps, every frame was exposed about 9 ms before the lens command of its own label.
Frame k therefore shows the focal plane commanded for frame k-1. This script writes a
corrected copy of a batch in which frame k carries the _dpt and _fd label of frame k-1.
Frame 0 keeps its label: it was exposed after the 1 s pre-focus at the start setting,
which is its own label.

The copy needs no extra disk space: every image is a hard link to the original file
(input and output must be on the same volume; --copy makes real copies instead).
metadata_log.csv is rewritten with the per-frame diopter, focusing_distance_mm and
output_filename columns shifted the same way, a column diopter_as_recorded holding the
original label, and one header line "# Frame Label Shift (frames): 1". Every other file
(1_flat_field, design tables, ...) is linked unchanged. The input batch is not modified.

Checks per recording (a folder with metadata_log.csv and frame images). A recording that
fails a check is linked unchanged and listed in the summary:
  - Final Acquisition FPS >= 24: at 15 and 10 fps the diagnosis found no lag
  - Focusing Pattern zigzag: in sawtooth runs the lag is not one frame after a fly-back
  - frame labels in the file names agree with the per-frame settings in metadata_log.csv
--force relabels recordings that fail the first two checks.
Loop iterations longer than 1.5 frame intervals are counted per recording: after such an
iteration a few frames can lag by two frames instead of one. They are reported, not
corrected.

Usage (from the repository root):
    python scripts/relabel_frame_lag.py --input_batch_dir <batch> --output_batch_dir <batch>_lag1
    python scripts/relabel_frame_lag.py --input_batch_dir <batch> --output_batch_dir <batch>_lag1 \\
        --samples 20250611_plate3_well5,20250611_plate3_well6 --dry_run
Then run the pipeline on <batch>_lag1 exactly as on the original batch.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sys

FRAME_RE = re.compile(r"^(?P<prefix>.+)_f(?P<frame>\d{5})_dpt(?P<dpt>[-+]?\d*\.?\d+)_fd(?P<fd>[-+]?\d*\.?\d+)mm"
                      r"\.(?P<ext>png|bmp|tif|tiff|jpg|jpeg)$", re.IGNORECASE)
LOG_NAME = "metadata_log.csv"
FRAME_DATA_BANNER = "# === Frame Data ==="
SHIFT_HEADER = "# Frame Label Shift (frames): 1"
MIN_FPS = 24.0
LONG_ITERATION_FACTOR = 1.5


def read_log(path):
    """Header lines (list), header fields (dict) and frame rows (list of dicts)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    header, k = [], 0
    while k < len(lines) and lines[k].startswith("#"):
        header.append(lines[k])
        k += 1
    fields = {}
    for line in header:
        content = line[1:].strip()
        if ":" in content and not content.startswith(("===", "---")):
            key, value = content.split(":", 1)
            fields[key.strip()] = value.strip()
    rows = list(csv.DictReader(io.StringIO("".join(lines[k:]))))
    return header, fields, rows


def plan_recording(folder, files):
    """Return (status, reason, mapping, log) for one recording folder."""
    header, fields, rows = read_log(os.path.join(folder, LOG_NAME))
    frames = {}
    for fn in files:
        m = FRAME_RE.match(fn)
        if m:
            frames[int(m.group("frame"))] = (fn, m)
    info = {"fps": fields.get("Final Acquisition FPS", "?"), "pattern": fields.get("Focusing Pattern", "?"),
            "settle_s": fields.get("Lens Settle Time (s)", "?"), "n_frames": len(frames)}
    log_by_frame = {}
    for r in rows:
        try:
            log_by_frame[int(r["frame_count"])] = r
        except (KeyError, ValueError):
            continue
    try:
        fps = float(info["fps"])
    except ValueError:
        fps = float("nan")
    long_iters = []
    for i, r in log_by_frame.items():
        try:
            if i > 0 and float(r["total_loop_iteration_time_ms"]) > LONG_ITERATION_FACTOR * 1000 / fps:
                long_iters.append(i)
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    info["long_iterations"] = sorted(long_iters)

    reasons = []
    if not fps >= MIN_FPS:
        reasons.append(f"fps {info['fps']} < {MIN_FPS:g}")
    if info["pattern"].strip().lower() != "zigzag":
        reasons.append(f"pattern {info['pattern']}")
    # labels of the files must agree with the logged settings
    mismatches = []
    for i, (fn, m) in frames.items():
        r = log_by_frame.get(i)
        if r is None:
            continue
        try:
            if abs(float(m.group("dpt")) - float(r["diopter"])) > 1e-6 or \
               abs(float(m.group("fd")) - float(r["focusing_distance_mm"])) > 1e-6:
                mismatches.append(i)
        except (KeyError, ValueError):
            pass
    if mismatches:
        return "error", f"file labels differ from metadata_log.csv at frames {mismatches[:5]}", None, info
    if frames and sorted(frames) != list(range(min(frames), max(frames) + 1)):
        gaps = sorted(set(range(min(frames), max(frames) + 1)) - set(frames))
        info["missing_frames"] = gaps[:10]

    def label(i):
        """Label tokens (dpt, fd) commanded for frame i: from its file name, else from the log."""
        if i in frames:
            return frames[i][1].group("dpt"), frames[i][1].group("fd")
        r = log_by_frame.get(i)
        if r is None:
            return None
        try:
            return f"{float(r['diopter']):.3f}", f"{float(r['focusing_distance_mm']):.2f}"
        except (KeyError, ValueError):
            return None

    mapping = {}
    for i, (fn, m) in frames.items():
        if i == 0:
            mapping[fn] = fn
            continue
        prev = label(i - 1)
        if prev is None:
            return "error", f"no label for frame {i - 1}", None, info
        mapping[fn] = f"{m.group('prefix')}_f{i:05d}_dpt{prev[0]}_fd{prev[1]}mm.{m.group('ext')}"
    info["header"], info["rows"], info["label"] = header, rows, label
    return ("skip" if reasons else "relabel"), "; ".join(reasons), mapping, info


def write_log(path_out, info, mapping, out_folder):
    header, rows, label = info["header"], info["rows"], info["label"]
    new_name_by_frame = {}
    for old, new in mapping.items():
        m = FRAME_RE.match(old)
        new_name_by_frame[int(m.group("frame"))] = new
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        for line in header:
            if line.strip() == FRAME_DATA_BANNER:
                f.write(SHIFT_HEADER + "\n")
            f.write(line)
        if not rows:
            return
        fieldnames = list(rows[0].keys()) + ["diopter_as_recorded"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["diopter_as_recorded"] = r.get("diopter", "")
            try:
                i = int(r["frame_count"])
            except (KeyError, ValueError):
                w.writerow(r)
                continue
            if i > 0:
                prev = label(i - 1)
                if prev is not None:
                    r["diopter"], r["focusing_distance_mm"] = prev
            if i in new_name_by_frame and r.get("output_filename"):
                r["output_filename"] = os.path.join(out_folder, new_name_by_frame[i])
            w.writerow(r)


def link_or_copy(src, dst, copy):
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as e:
        sys.exit(f"ERROR: cannot hard-link {src} -> {dst} ({e}).\n"
                 f"Put --output_batch_dir on the same volume as the input, or use --copy.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_batch_dir", required=True)
    ap.add_argument("--output_batch_dir", required=True)
    ap.add_argument("--samples", default=None,
                    help="comma-separated recording folder names to include (default: all); other files "
                         "outside recording folders (e.g. 1_flat_field) are always included")
    ap.add_argument("--copy", action="store_true", help="copy images instead of hard-linking them")
    ap.add_argument("--force", action="store_true", help="relabel recordings that fail the fps or pattern check")
    ap.add_argument("--dry_run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    src_root = os.path.abspath(args.input_batch_dir)
    dst_root = os.path.abspath(args.output_batch_dir)
    if not os.path.isdir(src_root):
        sys.exit(f"ERROR: input folder not found: {src_root}")
    if dst_root == src_root or dst_root.startswith(src_root + os.sep):
        sys.exit("ERROR: the output folder must not be the input folder or lie inside it.")
    if os.path.isdir(dst_root) and os.listdir(dst_root) and not args.dry_run:
        sys.exit(f"ERROR: output folder exists and is not empty: {dst_root}")
    wanted = {s.strip() for s in args.samples.split(",")} if args.samples else None

    summary, n_links = [], 0
    for folder, dirs, files in os.walk(src_root):
        dirs.sort()
        rel = os.path.relpath(folder, src_root)
        out_folder = os.path.normpath(os.path.join(dst_root, rel))
        is_recording = LOG_NAME in files and any(FRAME_RE.match(f) for f in files)
        if is_recording and wanted is not None and os.path.basename(folder) not in wanted:
            dirs[:] = []
            continue
        if not args.dry_run:
            os.makedirs(out_folder, exist_ok=True)
        if not is_recording:
            for fn in files:
                if not args.dry_run:
                    link_or_copy(os.path.join(folder, fn), os.path.join(out_folder, fn), args.copy)
                n_links += 1
            continue

        status, reason, mapping, info = plan_recording(folder, files)
        if status == "skip" and args.force:
            status, reason = "relabel", f"forced ({reason})"
        name = os.path.basename(folder)
        summary.append((rel, status, reason, info))
        if status == "error":
            print(f"ERROR in {rel}: {reason}. Recording not included in the output.")
            continue
        relabel = status == "relabel"
        for fn in files:
            src = os.path.join(folder, fn)
            if fn == LOG_NAME and relabel:
                if not args.dry_run:
                    write_log(os.path.join(out_folder, fn), info, mapping, out_folder)
                continue
            new = mapping.get(fn, fn) if relabel else fn
            if not args.dry_run:
                link_or_copy(src, os.path.join(out_folder, new), args.copy)
            n_links += 1
        if relabel and not args.dry_run:
            with open(os.path.join(out_folder, "frame_lag_relabel.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame", "filename_as_recorded", "filename_relabelled"])
                for old in sorted(mapping):
                    w.writerow([int(FRAME_RE.match(old).group("frame")), old, mapping[old]])
        if wanted is not None:
            wanted.discard(name)

    print(f"\n{'Dry run: nothing written. ' if args.dry_run else ''}Output batch: {dst_root}")
    print(f"{'Files that would be linked' if args.dry_run else 'Files linked' if not args.copy else 'Files copied'}: "
          f"{n_links}\n")
    print("| recording | result | frames | fps | pattern | settle s | long loop iterations |")
    print("|---|---|---|---|---|---|---|")
    for rel, status, reason, info in summary:
        result = {"relabel": "relabelled", "skip": f"unchanged ({reason})", "error": f"NOT INCLUDED ({reason})"}[status]
        if status == "relabel" and reason:
            result = f"relabelled, {reason}"
        li = info.get("long_iterations", [])
        li_txt = f"{len(li)}" + (f" (frames {', '.join(map(str, li[:8]))}{', ...' if len(li) > 8 else ''})" if li else "")
        miss = info.get("missing_frames")
        if miss:
            result += f"; missing frames {miss}"
        print(f"| {rel} | {result} | {info.get('n_frames')} | {info.get('fps')} | {info.get('pattern')} | "
              f"{info.get('settle_s')} | {li_txt} |")
    if wanted:
        print(f"\nWARNING: requested samples not found: {', '.join(sorted(wanted))}")


if __name__ == "__main__":
    main()
