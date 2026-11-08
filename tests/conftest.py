import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def write_metadata_log():
    """Factory writing a ``metadata_log.csv`` in the acquisition's header format.

    ``run_overrides`` / ``config_overrides`` replace fields of the run-metadata and
    configuration sections; the defaults mirror a real single-well sweep run.
    ``output_filenames`` maps frame index -> logged ``output_filename`` and replaces the
    default one-row frame table.
    """
    def _write(directory, *, run_overrides=None, config_overrides=None, filename="metadata_log.csv",
               output_filenames=None):
        run_fields = {
            "Timestamp (Start)": "2026-10-16T10:32:18.488994",
            "Plate ID": "3",
            "Well ID": "7_250um_zigzag",
            "Start Diopter (dpt)": "-2.000",
            "Output Folder": "C:/wellscanner/test/20261016_plate3_well7_250um_zigzag",
            "Date String": "20261016",
        }
        config_fields = {
            "Config File Path": r"C:\CodeLab\well_plate_analyzer\config\config.yml",
            "Processed as Color": "False",
            "Target Duration (s)": "20.2",
            "Configured FPS": "25.00",
            "Final Acquisition FPS": "25.00",
            "Actual Exposure (us)": "750.0",
            "Actual ROI (WxH)": "4512x4512",
            "Pixel Format": "Mono8",
            "Image Format": ".png",
            "Lens Controller Port": "COM11",
            "Focusing Pattern": "zigzag",
            "Lens Settle Time (s)": "0.025",
            "Scan Height (mm)": "14",
            "Scan Step (um)": "250",
            "Scan Ratio (mm/dpt)": "3.36",
        }
        run_fields.update(run_overrides or {})
        config_fields.update(config_overrides or {})
        lines = ["# === Acquisition Run Metadata ==="]
        lines += [f"# {key}: {value}" for key, value in run_fields.items()]
        lines.append("# --- Configuration ---")
        lines += [f"# {key}: {value}" for key, value in config_fields.items()]
        if output_filenames is None:
            output_filenames = {0: "C:/run/20261016_plate3_well7_f00000_dpt-2.000_fd0.00mm.png"}
        lines += ["# === Frame Data ===", "timestamp,frame_count,diopter,output_filename"]
        lines += [
            f"2026-10-16T10:32:18.52,{frame},-2.000,{output_filename}"
            for frame, output_filename in output_filenames.items()
        ]
        os.makedirs(directory, exist_ok=True)
        log_path = os.path.join(str(directory), filename)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(lines) + "\n")
        return log_path
    return _write
