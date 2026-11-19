"""Tests for src/utils/acquisition_metadata.py: header parsing, frames lost at acquisition,
FG/BG run compatibility, and the experiment-design cross-check used before a batch is
processed."""
import pandas as pd
import pytest

from src.utils.acquisition_metadata import (
    METADATA_LOOP_ERROR_PREFIX,
    METADATA_MISSING_VALUE,
    OPTICS_SOURCE_ACTIVE_PROFILE,
    OPTICS_SOURCE_RECORDING,
    load_experiment_design,
    read_acquisition_metadata_header,
    read_frames_lost_at_acquisition,
    read_recording_parameters,
    resolve_sample_design_values,
    validate_background_compatibility,
)
from src.utils.common_exceptions import ConfigError, DataValidationError


# --- header parsing -------------------------------------------------------------

def test_read_header_parses_sections_and_stops_at_frame_data(tmp_path, write_metadata_log):
    header = read_acquisition_metadata_header(write_metadata_log(tmp_path))

    assert set(header) == {"Acquisition Run Metadata", "Configuration"}
    # Values containing colons (ISO time, Windows drive) keep everything after the first one.
    assert header["Acquisition Run Metadata"]["Timestamp (Start)"] == "2026-10-16T10:32:18.488994"
    assert header["Configuration"]["Config File Path"] == r"C:\CodeLab\well_plate_analyzer\config\config.yml"
    assert header["Configuration"]["Scan Step (um)"] == "250"


def test_read_header_rejects_malformed_line(tmp_path):
    log_path = tmp_path / "metadata_log.csv"
    log_path.write_text("# === Acquisition Run Metadata ===\n# no separator here\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Malformed"):
        read_acquisition_metadata_header(str(log_path))


def test_read_header_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_acquisition_metadata_header(str(tmp_path / "absent.csv"))


# --- frames lost at acquisition -------------------------------------------------

def test_lost_frames_are_rows_without_a_saved_image(tmp_path, write_metadata_log):
    log_path = write_metadata_log(tmp_path, output_filenames={
        0: "C:/run/20261016_plate3_well7_f00000_dpt-2.000_fd0.00mm.png",
        1: METADATA_MISSING_VALUE,                          # exposure never reached the PC
        2: f"{METADATA_LOOP_ERROR_PREFIX} grab timeout",    # frame loop raised
        3: "C:/run/20261016_plate3_well7_f00003_dpt-1.256_fd2.50mm.png",
    })
    assert read_frames_lost_at_acquisition(log_path) == {1, 2}


def test_lost_frames_of_complete_run_is_empty(tmp_path, write_metadata_log):
    assert read_frames_lost_at_acquisition(write_metadata_log(tmp_path)) == set()


def test_lost_frames_need_the_frame_table_columns(tmp_path):
    log_path = tmp_path / "metadata_log.csv"
    log_path.write_text("# === Frame Data ===\ntimestamp,diopter\n2026-10-16T10:32:18.52,-2.000\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="frame_count"):
        read_frames_lost_at_acquisition(str(log_path))


def test_lost_frames_missing_log_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_frames_lost_at_acquisition(str(tmp_path / "absent.csv"))


# --- FG/BG compatibility --------------------------------------------------------

def test_compatible_runs_may_differ_in_run_identity(tmp_path, write_metadata_log):
    fg_log = write_metadata_log(tmp_path / "fg")
    bg_log = write_metadata_log(
        tmp_path / "bg",
        run_overrides={"Timestamp (Start)": "2026-10-16T10:08:49", "Plate ID": "9", "Well ID": "1"},
        config_overrides={"Config File Path": "D:/other.yml", "Final Acquisition FPS": "24.97"},
    )
    validate_background_compatibility(fg_log, bg_log)


@pytest.mark.parametrize(
    "run_overrides, config_overrides, field",
    [
        ({}, {"Actual Exposure (us)": "900.0"}, "Actual Exposure (us)"),
        ({}, {"Scan Step (um)": "350"}, "Scan Step (um)"),
        ({"Start Diopter (dpt)": "-1.500"}, {}, "Start Diopter (dpt)"),
    ],
)
def test_incompatible_runs_name_the_differing_field(tmp_path, write_metadata_log, run_overrides, config_overrides, field):
    fg_log = write_metadata_log(tmp_path / "fg")
    bg_log = write_metadata_log(tmp_path / "bg", run_overrides=run_overrides, config_overrides=config_overrides)
    with pytest.raises(DataValidationError, match=field.replace("(", r"\(").replace(")", r"\)")):
        validate_background_compatibility(fg_log, bg_log)


@pytest.mark.parametrize(
    "fg_trigger_mode, bg_trigger_mode, compatible",
    [
        (None, None, True),                       # two runs recorded before the grid loop
        ("free_run", None, True),                 # a missing field counts as free_run
        ("software_grid", "software_grid", True),
        ("software_grid", None, False),           # grid FG against an older (free-run) BG
        ("software_grid", "free_run", False),
    ],
)
def test_trigger_mode_must_match_and_defaults_to_free_run(tmp_path, write_metadata_log,
                                                           fg_trigger_mode, bg_trigger_mode, compatible):
    def _log(directory, trigger_mode):
        return write_metadata_log(directory, config_overrides={} if trigger_mode is None
                                  else {"Trigger Mode": trigger_mode})

    fg_log, bg_log = _log(tmp_path / "fg", fg_trigger_mode), _log(tmp_path / "bg", bg_trigger_mode)
    if compatible:
        validate_background_compatibility(fg_log, bg_log)
        validate_background_compatibility(bg_log, fg_log)
    else:
        with pytest.raises(DataValidationError, match=r"'Trigger Mode' FG='software_grid' BG='free_run'"):
            validate_background_compatibility(fg_log, bg_log)


# --- experiment design ----------------------------------------------------------

def _write_design(tmp_path, rows):
    design_path = tmp_path / "exp_design.csv"
    pd.DataFrame(rows).to_csv(design_path, index=False)
    return str(design_path)


_DESIGN_ROWS = [
    {"date": 20261016, "plate": 14, "well": 2, "water_depth_mm": 14, "scan_step_um": 250},
    {"date": 20261016, "plate": 4, "well": 2, "water_depth_mm": 4, "scan_step_um": 250},
]


def test_design_values_resolved_and_checked_against_every_log(tmp_path, write_metadata_log):
    design_df = load_experiment_design(_write_design(tmp_path, _DESIGN_ROWS))
    logs = [write_metadata_log(tmp_path / "fg"), write_metadata_log(tmp_path / "bg")]

    values = resolve_sample_design_values(design_df, "20261016_plate14_well2", logs)

    assert values == {"water_depth_mm": 14, "scan_step_um": 250}


def test_design_without_row_for_sample_raises(tmp_path, write_metadata_log):
    design_df = load_experiment_design(_write_design(tmp_path, _DESIGN_ROWS))
    with pytest.raises(DataValidationError, match="No experiment-design row"):
        resolve_sample_design_values(design_df, "20261016_plate6_well1", [write_metadata_log(tmp_path)])


def test_design_scan_step_disagreeing_with_bg_log_raises(tmp_path, write_metadata_log):
    design_df = load_experiment_design(_write_design(tmp_path, _DESIGN_ROWS))
    logs = [
        write_metadata_log(tmp_path / "fg"),
        write_metadata_log(tmp_path / "bg", config_overrides={"Scan Step (um)": "350"}),
    ]
    with pytest.raises(DataValidationError, match="scan step of 350"):
        resolve_sample_design_values(design_df, "20261016_plate14_well2", logs)


def test_design_requires_key_and_scan_step_columns(tmp_path):
    rows = [{"date": 20261016, "plate": 14, "well": 2, "water_depth_mm": 14}]
    with pytest.raises(DataValidationError, match="scan_step_um"):
        load_experiment_design(_write_design(tmp_path, rows))


def test_design_rejects_duplicate_sample_rows(tmp_path):
    with pytest.raises(DataValidationError, match="duplicate"):
        load_experiment_design(_write_design(tmp_path, _DESIGN_ROWS + _DESIGN_ROWS[:1]))


# --- recording parameters (frame rate, pixel scale) -----------------------------

# The rig as configured NOW: another frame rate and another lens than the recordings below.
CURRENT_RIG_CONFIG = {
    "acquisition": {
        "paths": {"log_filename": "metadata_log.csv"},
        "camera": {"fps": 10.0},
        "optics": {"pixel_size_um": 2.74, "magnification": 1.5},
    }
}


def test_recording_parameters_come_from_the_recording_not_the_current_config(tmp_path, write_metadata_log):
    write_metadata_log(tmp_path, config_overrides={
        "Final Acquisition FPS": "24.97", "Pixel Size (um)": "2.74", "Magnification": "0.638",
    })

    recording = read_recording_parameters(CURRENT_RIG_CONFIG, str(tmp_path))

    assert recording.frame_rate_fps == pytest.approx(24.97)
    assert recording.pixel_to_mm_factor == pytest.approx(2.74 / 0.638 / 1000)
    assert recording.optics_source == OPTICS_SOURCE_RECORDING


def test_recording_without_logged_optics_takes_the_active_profile(tmp_path, write_metadata_log):
    write_metadata_log(tmp_path)   # header as written before the optics were logged

    recording = read_recording_parameters(CURRENT_RIG_CONFIG, str(tmp_path))

    assert recording.frame_rate_fps == pytest.approx(25.0)   # still the recording's own frame rate
    assert recording.pixel_to_mm_factor == pytest.approx(2.74 / 1.5 / 1000)
    assert recording.optics_source == OPTICS_SOURCE_ACTIVE_PROFILE


def test_recording_without_logged_optics_needs_optics_in_the_profile(tmp_path, write_metadata_log):
    write_metadata_log(tmp_path)
    config_without_optics = {"acquisition": {"paths": {"log_filename": "metadata_log.csv"}}}

    with pytest.raises(ConfigError, match="acquisition.optics"):
        read_recording_parameters(config_without_optics, str(tmp_path))


def test_recording_with_only_one_optics_field_fails(tmp_path, write_metadata_log):
    write_metadata_log(tmp_path, config_overrides={"Magnification": "0.638"})

    with pytest.raises(DataValidationError, match="Magnification"):
        read_recording_parameters(CURRENT_RIG_CONFIG, str(tmp_path))


def test_recording_without_metadata_log_fails(tmp_path):
    with pytest.raises(DataValidationError, match="has no metadata_log.csv"):
        read_recording_parameters(CURRENT_RIG_CONFIG, str(tmp_path))


@pytest.mark.parametrize("logged_frame_rate", ["0.00", "-25.00", "fast"])
def test_recording_with_unusable_frame_rate_fails(tmp_path, write_metadata_log, logged_frame_rate):
    write_metadata_log(tmp_path, config_overrides={"Final Acquisition FPS": logged_frame_rate})

    with pytest.raises(DataValidationError, match="Final Acquisition FPS"):
        read_recording_parameters(CURRENT_RIG_CONFIG, str(tmp_path))
