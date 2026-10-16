import csv
import os

from src.acquire.workflows import rename_files_after_run

PARAMETER_SUFFIX = "_100um_zigzag"
LOG_FIELDNAMES = ["timestamp", "frame_count", "diopter", "output_filename"]
CONFIG = {"paths": {"log_filename": "metadata_log.csv"}}


def _write_run(tmp_path, num_frames=3):
    """Mimic acquire_and_save_frames: images named after the folder, then the metadata log."""
    run_folder_name = f"20261015_plate3_well7{PARAMETER_SUFFIX}"
    run_folder_path = os.path.join(str(tmp_path), run_folder_name)
    os.makedirs(run_folder_path)
    comment_lines = [
        "# === Acquisition Run Metadata ===\n",
        f"# Well ID: 7{PARAMETER_SUFFIX}\n",
        f"# Output Folder: {run_folder_path}\n",
        "# === Frame Data ===\n",
    ]
    rows = []
    for frame in range(num_frames):
        filename = f"{run_folder_name}_f{frame:05d}_dpt-1.500_fd0.00mm.png"
        open(os.path.join(run_folder_path, filename), "wb").close()
        rows.append({
            "timestamp": f"2026-10-15T16:21:0{frame}",
            "frame_count": frame,
            "diopter": "-1.500",
            "output_filename": os.path.join(run_folder_path, filename),
        })
    rows.append({"timestamp": "2026-10-15T16:21:09", "frame_count": num_frames, "diopter": "-1.500",
                 "output_filename": "ERROR_IN_LOOP: grab timeout"})
    log_path = os.path.join(run_folder_path, CONFIG["paths"]["log_filename"])
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        f.writelines(comment_lines)
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return run_folder_path, log_path, comment_lines, rows


def _read_log(log_path):
    with open(log_path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    comment_lines = [line for line in lines if line.startswith("#")]
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    return comment_lines, rows


def test_logged_paths_point_to_renamed_files(tmp_path):
    run_folder_path, log_path, original_comments, original_rows = _write_run(tmp_path)

    rename_files_after_run(run_folder_path, PARAMETER_SUFFIX, CONFIG)

    comment_lines, rows = _read_log(log_path)
    image_rows = rows[:-1]
    assert len(rows) == len(original_rows)
    for row, original in zip(image_rows, original_rows):
        logged_path = row["output_filename"]
        assert os.path.dirname(logged_path) == run_folder_path
        assert os.path.isfile(logged_path)
        assert PARAMETER_SUFFIX not in os.path.basename(logged_path)
        assert {k: v for k, v in row.items() if k != "output_filename"} == \
            {k: str(v) for k, v in original.items() if k != "output_filename"}
    assert rows[-1]["output_filename"] == "ERROR_IN_LOOP: grab timeout"
    assert comment_lines == original_comments


def test_images_renamed_and_nothing_else_touched(tmp_path):
    run_folder_path, _, _, _ = _write_run(tmp_path)

    rename_files_after_run(run_folder_path, PARAMETER_SUFFIX, CONFIG)

    images = sorted(name for name in os.listdir(run_folder_path) if name.endswith(".png"))
    assert images == [f"20261015_plate3_well7_f{frame:05d}_dpt-1.500_fd0.00mm.png" for frame in range(3)]
    assert os.path.isfile(os.path.join(run_folder_path, CONFIG["paths"]["log_filename"]))
