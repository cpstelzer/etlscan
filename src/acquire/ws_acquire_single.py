"""Compatibility shim for single-sample acquisition workflow."""

from src.acquire.workflows import run_single_well_workflow

PLATE_NO = 0
WELL_NO = 0
START_DPT = -2.0
ACQUISITION_CONFIG_PATH = "ws_config.yaml"


def main():
    run_single_well_workflow(
        acquisition_config_path=ACQUISITION_CONFIG_PATH,
        plate_no=PLATE_NO,
        well_no=WELL_NO,
        start_dpt=START_DPT,
    )


if __name__ == "__main__":
    main()
