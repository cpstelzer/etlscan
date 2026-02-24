from __future__ import annotations

import csv
import os
from datetime import datetime


def initialize_experiment_log(log_path, headers):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Initialized experiment summary log: {log_path}")


def append_to_experiment_log(log_path, row_builder):
    try:
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_builder(datetime.now()))
    except Exception as e:
        print(f"Error writing to experiment summary log {log_path}: {e}")
