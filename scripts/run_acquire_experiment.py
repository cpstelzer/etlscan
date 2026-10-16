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
# The parameter grid (scan steps x focusing patterns) and the optional variable
# recording duration live in acquisition.experiment_sweep in the config.


def main():
    run_experiment_sweep_workflow(
        acquisition_config_path=ACQUISITION_CONFIG_PATH,
        csv_filename=CSV_FILENAME,
    )


if __name__ == "__main__":
    main()
