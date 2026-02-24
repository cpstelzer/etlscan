"""Compatibility shim for single-sample acquisition workflow."""

import os
import sys


def _ensure_repo_root_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()

from src.acquire.workflows import run_single_well_workflow

PLATE_NO = 0
WELL_NO = 0
START_DPT = -2.0
ACQUISITION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ws_config.yaml")


def main():
    run_single_well_workflow(
        acquisition_config_path=ACQUISITION_CONFIG_PATH,
        plate_no=PLATE_NO,
        well_no=WELL_NO,
        start_dpt=START_DPT,
    )


if __name__ == "__main__":
    main()
