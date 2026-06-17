#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for experiment-sweep acquisition workflow."""

import os
import sys


def _ensure_repo_root_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()

from src.acquire.workflows import run_experiment_sweep_workflow

CSV_FILENAME = "wellscanner_coords.csv"
# Zaber stage settings now live in hardware_profiles and are read from the
# merged config by the workflow.
ACQUISITION_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yml"
)
SCAN_STEPS_UM_OPTIONS = [100, 200, 300, 400, 500, 600, 700]
FOCUSING_PATTERN_OPTIONS = ['zigzag', 'sawtooth']


def main():
    run_experiment_sweep_workflow(
        acquisition_config_path=ACQUISITION_CONFIG_PATH,
        csv_filename=CSV_FILENAME,
        scan_steps_um_options=SCAN_STEPS_UM_OPTIONS,
        focusing_pattern_options=FOCUSING_PATTERN_OPTIONS,
    )


if __name__ == "__main__":
    main()
