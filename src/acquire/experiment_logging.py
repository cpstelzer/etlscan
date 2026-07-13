"""Helpers for writing a per-experiment summary CSV log.

The log is a plain CSV file that accumulates one row per completed well or
acquisition event.  :func:`initialize_experiment_log` creates the file with a
header row if it does not yet exist; :func:`append_to_experiment_log` appends a
single row built by the caller-supplied factory function.

Typical usage in an acquisition workflow::

    log_path = os.path.join(output_dir, "experiment_summary.csv")
    initialize_experiment_log(log_path, ["timestamp", "well", "frames", "status"])
    ...
    append_to_experiment_log(
        log_path,
        lambda ts: [ts.isoformat(), well_name, frame_count, "ok"],
    )
"""
from __future__ import annotations

import csv
import os
from datetime import datetime


def initialize_experiment_log(log_path: str, headers: list) -> None:
    """Create the summary CSV with a header row if it does not already exist.

    If *log_path* already exists the file is left untouched so that a resumed
    acquisition run can continue appending to the same log.

    Args:
        log_path: Absolute or relative path to the CSV file to create.
        headers: Column names written as the first row.
    """
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Initialized experiment summary log: {log_path}")


def append_to_experiment_log(log_path: str, row_builder) -> None:
    """Append one row to the summary CSV log.

    *row_builder* is called with the current :class:`~datetime.datetime` and
    must return a sequence whose length matches the header written by
    :func:`initialize_experiment_log`.  Errors are caught and printed rather
    than raised so that a logging failure never aborts an acquisition run.

    Args:
        log_path: Path to an existing CSV log file.
        row_builder: Callable ``(datetime) -> list`` that produces the row values.
    """
    try:
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_builder(datetime.now()))
    except Exception as e:
        print(f"Error writing to experiment summary log {log_path}: {e}")
