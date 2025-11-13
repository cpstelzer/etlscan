import os
import shutil
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Optional

# ------------------------
# User-configurable parameters
# ------------------------
# Focusing pattern: sawtooth 250um step size
#input_csv = "D:/sandbox/data/output/size_monitoring_test/20250909_plate1_well2_sawtooth/stage2_characterization/20250909_plate1_well2_sawtooth_stage2_all_particle_features.csv"

# Focusing pattern: zigzag 250um step size
#input_csv = "D:/sandbox/data/output/size_monitoring_test/20250828_plate1_well1/stage2_characterization/20250828_plate1_well1_stage2_all_particle_features.csv"
input_csv = "D:/sandbox/data/output/size_monitoring_test/20250923_plate1_well1_zigzag_250um/stage2_characterization/20250923_plate1_well1_zigzag_250um_stage2_all_particle_features.csv"

# Focusing pattern: sawtooth 500um step size
#input_csv = "D:/sandbox/data/output/size_monitoring_test/20250909_plate1_well2/stage2_characterization/20250909_plate1_well2_stage2_all_particle_features.csv"

plot_dir = "diagnostics_plots"

# Selection / detection parameters
frame_col = "frame"
track_col = "track_id"
sharp_col = "variance_laplacian"
z_col = "z_mm"
min_sharpness_frac = 0.3
min_prominence_frac = 0.05
smooth_window = 3
sharpness_min_threshold = 80

# Diagnostics / plotting
max_tracks_to_plot: Optional[int] = 15   # set to None for all tracks, or integer for N longest
save_plots = True
show_plots = False

# Misc
verbose = True

# Derived output paths
output_csv = os.path.join(plot_dir, "filtered_in_focus.csv")
diagnostics_csv = os.path.join(plot_dir, "diagnostics_selected_peaks.csv")


def smooth_series(s: pd.Series, w: int):
    if w <= 1:
        return s.values
    return s.rolling(window=w, center=True, min_periods=1).mean().values


def detect_peaks_in_track(g: pd.DataFrame):
    y = g[sharp_col].values
    if len(y) == 0:
        return np.array([], dtype=int)
    max_val = np.nanmax(y)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.array([], dtype=int)
    threshold = min_sharpness_frac * max_val
    smoothed = smooth_series(pd.Series(y), smooth_window)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] \
           and smoothed[i] >= threshold and smoothed[i] >= sharpness_min_threshold:
            local_window = smoothed[max(0, i - 5):min(len(smoothed), i + 6)]
            prominence = smoothed[i] - np.median(local_window)
            if prominence >= min_prominence_frac * max_val:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def filter_focused_frames(df: pd.DataFrame):
    filtered_rows = []
    diagnostics = []
    for tid, g in df.groupby(track_col):
        g = g.sort_values(frame_col)
        peak_idx = detect_peaks_in_track(g)
        if len(peak_idx) > 0:
            sel = g.iloc[peak_idx]
            filtered_rows.append(sel)
            diagnostics.append({
                track_col: tid,
                "n_frames": g.shape[0],
                "n_selected": sel.shape[0]
            })
        else:
            diagnostics.append({
                track_col: tid,
                "n_frames": g.shape[0],
                "n_selected": 0
            })
    if len(filtered_rows) > 0:
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        filtered_df = pd.DataFrame(columns=df.columns)
    diagnostics_df = pd.DataFrame(diagnostics)
    return filtered_df, diagnostics_df


def plot_track_with_collage(df: pd.DataFrame, tid: int, crops_dir: str, prefix: str, outpath: str):
    g = df[df[track_col] == tid].sort_values(frame_col)
    if g.empty:
        return None

    peak_idx = detect_peaks_in_track(g)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(g[frame_col], g[sharp_col], color="tab:blue", linewidth=1.2)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Sharpness", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(g[frame_col], g[z_col], color="tab:orange", linewidth=1.0)
    ax2.set_ylabel("z_mm", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    maxima_frames = []
    for j, p in enumerate(peak_idx):
        ax2.scatter(g.iloc[p][frame_col], g.iloc[p][z_col], color="red", marker="o", s=40, zorder=5)
        offset_x = 2
        offset_y = 0.02 * (g[z_col].max() - g[z_col].min())
        ax2.text(g.iloc[p][frame_col] + offset_x, g.iloc[p][z_col] + offset_y,
                 str(j+1), color="black", fontsize=8)
        maxima_frames.append((j+1, int(g.iloc[p][frame_col]), int(g.iloc[p]['X']), int(g.iloc[p]['Y'])))

    ax1.set_title(f"Track {tid} (length={len(g)})")

    ax_collage = fig.add_subplot(gs[1])
    ax_collage.axis("off")

    imgs = []
    for num, frame_int, xpix, ypix in maxima_frames:
        pattern = os.path.join(crops_dir, f"{prefix}_track{tid}_frame{frame_int}_X{xpix}_Y{ypix}.png")
        print(f"Searching for maxima image: {pattern}")
        files = glob.glob(pattern)
        if files:
            print(f"  Found: {files[0]}")
            img = mpimg.imread(files[0])
            imgs.append((num, img))
        else:
            print("  No match found.")

    if imgs:
        n = len(imgs)
        cols = min(5, n)
        rows = math.ceil(n / cols)
        gs_collage = GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[1])
        for i, (num, img) in enumerate(imgs):
            r = i // cols
            c = i % cols
            ax_img = fig.add_subplot(gs_collage[r, c])
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax_img.imshow(img, cmap="gray", aspect="equal")
            else:
                ax_img.imshow(img, aspect="equal")
            ax_img.axis("off")
            ax_img.text(2, 10, str(num), color="yellow", fontsize=10,
                        ha="left", va="top",
                        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)
    return outpath


def plot_example_tracks_with_collages(df: pd.DataFrame, crops_dir: str, prefix: str, outdir: str, n_tracks: Optional[int] = None):
    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    if n_tracks is None:
        track_ids = track_lengths.index
    else:
        track_ids = track_lengths.index[:n_tracks]

    saved = []
    for tid in track_ids:
        outpath = os.path.join(outdir, f"track_{tid}_diagnostic_with_collage.png")
        res = plot_track_with_collage(df, tid, crops_dir, prefix, outpath)
        if res:
            saved.append(res)
    return saved


def plot_3d_reconstructed(df: pd.DataFrame, filtered_df: pd.DataFrame, outpath: str, n_tracks: Optional[int] = 8):
    track_lengths = df.groupby(track_col)[frame_col].count().sort_values(ascending=False)
    if n_tracks is None:
        track_ids = track_lengths.index
    else:
        track_ids = track_lengths.index[:n_tracks]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('tab20')
    ncol = max(1, min(20, len(track_ids)))

    plotted = 0
    for i, tid in enumerate(track_ids):
        g = df[df[track_col] == tid].sort_values(frame_col)
        sel = filtered_df[filtered_df[track_col] == tid]
        if g.shape[0] == 0 or sel.shape[0] == 0:
            continue

        x = g['x_mm'].values
        y = g['y_mm'].values
        z_real = sel[z_col].mean()
        z_line = np.full_like(x, z_real, dtype=float)

        color = cmap(i % ncol)
        ax.plot(x, y, z_line, linewidth=0.8, alpha=0.7, color=color)
        ax.scatter(sel['x_mm'].values, sel['y_mm'].values, sel[z_col].values,
                   marker='o', s=18, color=color, edgecolor='k', linewidth=0.3)
        plotted += 1

    ax.set_xlabel('x_mm')
    ax.set_ylabel('y_mm')
    ax.set_zlabel('z_mm (real)')
    ax.set_title(f'Reconstructed trajectories of {plotted} tracks')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def plot_diagnostics_summary(diagnostics_df: pd.DataFrame, outpath: str):
    df_sorted = diagnostics_df.sort_values("n_frames", ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df_sorted.index, df_sorted["n_frames"], color="tab:blue", marker="o", label="Track length (n_frames)")
    ax1.set_xlabel("Track (sorted by length)")
    ax1.set_ylabel("Track length (frames)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.bar(df_sorted.index, df_sorted["n_selected"], alpha=0.4, color="tab:orange", label="Sharpness maxima (n_selected)")
    ax2.set_ylabel("Sharpness maxima count", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Track lengths and sharpness maxima across all tracks")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if verbose:
        print(f"Loaded {len(df)} rows from {input_csv}")

    crops_dir = os.path.join(os.path.dirname(input_csv), "stage2_all_particle_crops")
    if not os.path.exists(crops_dir):
        if verbose:
            print(f"Warning: crops directory not found: {crops_dir}")

    basename = os.path.basename(input_csv)
    prefix = basename.split("_stage2")[0]
    if verbose:
        print(f"Using image filename prefix: {prefix}")

    filtered_df, diagnostics_df = filter_focused_frames(df)

    filtered_df.to_csv(output_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    if verbose:
        print(f"Wrote filtered CSV with {len(filtered_df)} rows to {output_csv}")
        print(f"Wrote diagnostics CSV with {len(diagnostics_df)} track entries to {diagnostics_csv}")

    if save_plots:
        saved = plot_example_tracks_with_collages(df, crops_dir, prefix, plot_dir, n_tracks=max_tracks_to_plot)
        if verbose:
            print(f"Saved {len(saved)} diagnostic+collage plots to {plot_dir}")

        three_d_plot = os.path.join(plot_dir, "reconstructed_trajectories_3d.png")
        out3 = plot_3d_reconstructed(df, filtered_df, three_d_plot, n_tracks=max_tracks_to_plot)
        if out3 is not None and verbose:
            print(f"Saved 3D reconstructed trajectories plot to {out3}")

        summary_plot = os.path.join(plot_dir, "diagnostics_summary.png")
        plot_diagnostics_summary(diagnostics_df, summary_plot)
        if verbose:
            print(f"Saved diagnostics summary plot to {summary_plot}")


if __name__ == '__main__':
    main()
