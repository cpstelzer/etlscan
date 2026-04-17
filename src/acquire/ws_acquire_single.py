"""Compatibility shim for single-sample acquisition workflow."""

import argparse
import os
import sys


def _ensure_repo_root_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()

from src.acquire.workflows import run_single_well_workflow


def main():
    parser = argparse.ArgumentParser(description="Acquire a single well using the unified config.")
    parser.add_argument("--config", default="config/config.yml", help="Path to the unified config.yml")
    parser.add_argument("--plate", type=int, required=True, help="Plate index")
    parser.add_argument("--well", type=int, required=True, help="Well index")
    parser.add_argument("--start-dpt", type=float, required=True, help="Starting diopter")
    args = parser.parse_args()

    run_single_well_workflow(
        acquisition_config_path=args.config,
        plate_no=args.plate,
        well_no=args.well,
        start_dpt=args.start_dpt,
    )


if __name__ == "__main__":
    main()
