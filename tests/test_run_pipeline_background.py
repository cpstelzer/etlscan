"""run_pipeline: external-background sample layout, up-front input validation, and the
experiment-design columns added to per-sample outputs."""
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

from src.utils.common_exceptions import DataValidationError

RUN_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"


@pytest.fixture(scope="module")
def run_pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline_under_test", RUN_PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def config(tmp_path):
    return {
        "paths": {"pipeline_base": str(tmp_path / "pipeline")},
        "acquisition": {"paths": {"log_filename": "metadata_log.csv"}},
        "stage_01_initial_detection": {"external_background": {"fg_subdir": "fg", "bg_subdir": "bg"}},
    }


def test_external_layout_paths_and_batch_name(run_pipeline, config, tmp_path):
    root = str(tmp_path / "experiment")

    external = run_pipeline.prepare_sample_paths(config, root, "20261016_plate14_well2", "external")
    time_shift = run_pipeline.prepare_sample_paths(config, root, "20261016_plate14_well2", "time_shift")

    assert external.sample_input_dir == os.path.join(root, "fg", "20261016_plate14_well2")
    assert external.sample_bg_dir == os.path.join(root, "bg", "20261016_plate14_well2")
    assert external.batch_dir_name == "experiment"  # named after the root, not "fg"
    assert time_shift.sample_input_dir == os.path.join(root, "20261016_plate14_well2")
    assert time_shift.sample_bg_dir is None


def _design(tmp_path, rows):
    design_path = tmp_path / "exp_design.csv"
    pd.DataFrame(rows).to_csv(design_path, index=False)
    return str(design_path)


def test_validate_sample_inputs_returns_design_values(run_pipeline, config, tmp_path, write_metadata_log):
    root = tmp_path / "experiment"
    for side in ("fg", "bg"):
        write_metadata_log(root / side / "20261016_plate14_well2")
    design = _design(tmp_path, [{"date": 20261016, "plate": 14, "well": 2, "water_depth_mm": 14, "scan_step_um": 250}])

    values = run_pipeline.validate_sample_inputs(config, str(root), ["20261016_plate14_well2"], "external", design)

    assert values == {"20261016_plate14_well2": {"water_depth_mm": 14, "scan_step_um": 250}}


def test_validate_sample_inputs_collects_every_problem(run_pipeline, config, tmp_path, write_metadata_log):
    root = tmp_path / "experiment"
    # Sample A: BG recorded with another exposure. Sample B: no BG folder at all.
    write_metadata_log(root / "fg" / "20261016_plate14_well1")
    write_metadata_log(root / "bg" / "20261016_plate14_well1", config_overrides={"Actual Exposure (us)": "900.0"})
    write_metadata_log(root / "fg" / "20261016_plate14_well2")

    with pytest.raises(DataValidationError) as excinfo:
        run_pipeline.validate_sample_inputs(
            config, str(root), ["20261016_plate14_well1", "20261016_plate14_well2"], "external", None
        )

    message = str(excinfo.value)
    assert "2 sample(s)" in message
    assert "20261016_plate14_well1" in message and "Actual Exposure (us)" in message
    assert "20261016_plate14_well2" in message and "background folder missing" in message


def test_validate_sample_inputs_time_shift_without_design_reads_nothing(run_pipeline, config, tmp_path):
    # Regular runs need no metadata logs: nothing is read, nothing can fail.
    assert run_pipeline.validate_sample_inputs(config, str(tmp_path), ["20261016_plate1_well1"], "time_shift", None) == {}


def test_annotate_sample_outputs_puts_design_columns_first_and_is_idempotent(run_pipeline, tmp_path):
    csv_path = tmp_path / "filtered_good.csv"
    pd.DataFrame({"track_id": [1, 2], "area": [1200.5, 980.25]}).to_csv(csv_path, index=False)
    design_values = {"water_depth_mm": 14, "scan_step_um": 250}

    run_pipeline.annotate_sample_outputs([str(csv_path), str(tmp_path / "absent.csv")], design_values)
    run_pipeline.annotate_sample_outputs([str(csv_path)], design_values)

    annotated = pd.read_csv(csv_path)
    assert list(annotated.columns) == ["water_depth_mm", "scan_step_um", "track_id", "area"]
    assert annotated["water_depth_mm"].tolist() == [14, 14]
    assert annotated["area"].tolist() == [1200.5, 980.25]
