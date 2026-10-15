import copy
import csv

import pytest

from src.acquire import workflows
from src.acquire.focus_pattern import build_focus_cycle_pattern, count_frames_for_plane_visits
from src.acquire.workflows import (
    resolve_video_duration_sec,
    run_single_well_sweep_workflow,
    validate_experiment_sweep_settings,
)
from src.utils.common_exceptions import ConfigError
from src.utils.config_loader import load_config_once

SCAN_STEPS = [100, 200, 300, 400, 500, 600, 700]
PATTERNS = ["zigzag", "sawtooth"]


def _acq_conf(tmp_path, *, variable, min_duration_sec=15, plane_revisit_times=5):
    return {
        "paths": {
            "output_folder_base": str(tmp_path),
            "log_filename": "metadata_log.csv",
            "experiment_summary_log_filename": "experiment_summary_log.csv",
        },
        "focusing": {"pattern": "zigzag", "scan_step_um": 200, "scan_height_mm": 3.2, "scan_ratio": 3.36},
        "camera": {"fps": 25.0, "video_duration_sec": min_duration_sec},
        "experiment_sweep": {
            "scan_steps_um_options": list(SCAN_STEPS),
            "focusing_pattern_options": list(PATTERNS),
            "variable_recording_duration": variable,
            "plane_revisit_times": plane_revisit_times,
        },
    }


def _run_conf(acq_conf, step_um, pattern):
    run_conf = copy.deepcopy(acq_conf)
    run_conf["focusing"]["scan_step_um"] = step_um
    run_conf["focusing"]["pattern"] = pattern
    return run_conf


def test_repo_config_sweep_defaults():
    acq = load_config_once("config/config.yml")["acquisition"]
    sweep_conf = validate_experiment_sweep_settings(acq)
    assert sweep_conf["scan_steps_um_options"] == SCAN_STEPS
    assert sweep_conf["focusing_pattern_options"] == PATTERNS
    assert sweep_conf["variable_recording_duration"] is False
    assert sweep_conf["plane_revisit_times"] == 5
    assert acq["camera"]["video_duration_sec"] == 15


@pytest.mark.parametrize(
    "key, bad_value",
    [
        ("scan_steps_um_options", []),
        ("scan_steps_um_options", [100, -200]),
        ("focusing_pattern_options", ["zigzag", "spiral"]),
        ("variable_recording_duration", "yes"),
        ("plane_revisit_times", 0),
        ("plane_revisit_times", True),
    ],
)
def test_validate_rejects_bad_values(tmp_path, key, bad_value):
    acq_conf = _acq_conf(tmp_path, variable=True)
    acq_conf["experiment_sweep"][key] = bad_value
    with pytest.raises(ConfigError):
        validate_experiment_sweep_settings(acq_conf)


def test_validate_rejects_missing_keys(tmp_path):
    acq_conf = _acq_conf(tmp_path, variable=True)
    del acq_conf["experiment_sweep"]["plane_revisit_times"]
    with pytest.raises(ConfigError, match="plane_revisit_times"):
        validate_experiment_sweep_settings(acq_conf)
    del acq_conf["experiment_sweep"]
    with pytest.raises(ConfigError):
        validate_experiment_sweep_settings(acq_conf)


def test_fixed_duration_when_switch_off(tmp_path):
    acq_conf = _acq_conf(tmp_path, variable=False, min_duration_sec=15)
    for step_um in SCAN_STEPS:
        for pattern in PATTERNS:
            assert resolve_video_duration_sec(_run_conf(acq_conf, step_um, pattern), -1.0) == 15


def test_variable_duration_never_below_config_minimum(tmp_path):
    # With k = 5 the longest requirement on this rig is 289 frames (11.56 s), so all runs keep 15 s.
    acq_conf = _acq_conf(tmp_path, variable=True, min_duration_sec=15)
    for step_um in SCAN_STEPS:
        for pattern in PATTERNS:
            assert resolve_video_duration_sec(_run_conf(acq_conf, step_um, pattern), -1.0) == 15


@pytest.mark.parametrize("plane_revisit_times", [1, 5, 50])
def test_variable_duration_records_enough_frames(tmp_path, plane_revisit_times):
    acq_conf = _acq_conf(tmp_path, variable=True, min_duration_sec=0, plane_revisit_times=plane_revisit_times)
    fps = acq_conf["camera"]["fps"]
    for step_um in SCAN_STEPS:
        for pattern in PATTERNS:
            run_conf = _run_conf(acq_conf, step_um, pattern)
            cycle = build_focus_cycle_pattern(-1.0, pattern, 3.2, step_um, 3.36)
            required_frames = count_frames_for_plane_visits(cycle, plane_revisit_times)
            duration_sec = resolve_video_duration_sec(run_conf, -1.0)
            # Same frame-count expression as run_acquisition.
            assert int(duration_sec * fps) == required_frames


def test_variable_duration_example_values(tmp_path):
    acq_conf = _acq_conf(tmp_path, variable=True, min_duration_sec=2)
    assert resolve_video_duration_sec(_run_conf(acq_conf, 100, "zigzag"), -1.0) == pytest.approx(289 / 25)
    assert resolve_video_duration_sec(_run_conf(acq_conf, 100, "sawtooth"), -1.0) == pytest.approx(165 / 25)
    # 700 um sawtooth needs only 30 frames (1.2 s), so the 2 s minimum wins.
    assert resolve_video_duration_sec(_run_conf(acq_conf, 700, "sawtooth"), -1.0) == 2


@pytest.mark.parametrize("variable", [False, True])
def test_single_well_sweep_smoke(tmp_path, monkeypatch, variable):
    acq_conf = _acq_conf(tmp_path, variable=variable, min_duration_sec=2)
    calls = []

    def fake_run_acquisition(plate, well_id, start_dpt, conf, process_as_color=False):
        calls.append({
            "plate": plate,
            "well_id": well_id,
            "start_dpt": start_dpt,
            "step": conf["focusing"]["scan_step_um"],
            "pattern": conf["focusing"]["pattern"],
            "duration": conf["camera"]["video_duration_sec"],
        })
        return {"status": "success", "final_fps": 25.0, "actual_exposure_us": 750, "any_frames_skipped": False}

    monkeypatch.setattr(workflows, "_get_acquisition_api", lambda: (lambda path: copy.deepcopy(acq_conf), fake_run_acquisition))
    monkeypatch.setattr(workflows, "decide_color_mode", lambda conf: False)
    monkeypatch.setattr(workflows.time, "sleep", lambda seconds: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    run_single_well_sweep_workflow(acquisition_config_path="config.yml", plate_no=3, well_no=7, start_dpt=-1.0)

    assert len(calls) == len(SCAN_STEPS) * len(PATTERNS)
    assert {c["well_id"] for c in calls} == {f"7_{s}um_{p}" for s in SCAN_STEPS for p in PATTERNS}
    assert all(c["plate"] == 3 and c["start_dpt"] == -1.0 for c in calls)
    for c in calls:
        if variable:
            cycle = build_focus_cycle_pattern(-1.0, c["pattern"], 3.2, c["step"], 3.36)
            required_sec = count_frames_for_plane_visits(cycle, 5) / 25.0
            assert c["duration"] == pytest.approx(max(2, required_sec))
        else:
            assert c["duration"] == 2

    with open(tmp_path / "experiment_summary_log.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == workflows._SWEEP_LOG_HEADERS
    assert len(rows) == 1 + len(calls)
    assert all(row[2:5] == ["3", "7", "N/A"] and row[11] == "success" for row in rows[1:])
