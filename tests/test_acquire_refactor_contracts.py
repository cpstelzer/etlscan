import csv
from pathlib import Path

from src.acquire.config import get_plate_settings, normalize_plate_type
from src.acquire.experiment_logging import append_to_experiment_log, initialize_experiment_log
from src.acquire.plate_layout import calculate_coords_24wellplate, calculate_coords_96wellplate
from src.acquire.workflows import run_single_well_workflow
from src.utils.path_helpers import parse_date_plate_well


def test_normalize_plate_type_aliases():
    assert normalize_plate_type("24") == "24"
    assert normalize_plate_type("24-well") == "24"
    assert normalize_plate_type("24_well") == "24"
    assert normalize_plate_type("96well") == "96"
    assert normalize_plate_type(96) == "96"


def test_get_plate_settings_defaults():
    plate_type, settings = get_plate_settings({}, {"start_x": 1, "start_y": 2, "spacing": 3, "origin_x": 4, "origin_y": 5},
                                             {"start_x": 6, "start_y": 7, "spacing": 8, "origin_x": 9, "origin_y": 10})
    assert plate_type == "24"
    assert settings["start_x"] == 1


def test_plate_coordinate_snapshots():
    settings24 = {"start_x": 15.74, "start_y": 18.88, "spacing": 18.0, "origin_x": 7.89, "origin_y": 15.21}
    df24 = calculate_coords_24wellplate(settings24)
    assert len(df24) == 24
    first24 = df24.iloc[0].to_dict()
    last24 = df24.iloc[-1].to_dict()
    assert first24['wellname'] == 'A1' and first24['wellnr'] == 1 and first24['negX'] is False and first24['negY'] is False
    assert abs(first24['X'] - 7.85) < 1e-9 and abs(first24['Y'] - 3.67) < 1e-9
    assert last24['wellname'] == 'D6' and last24['wellnr'] == 24 and last24['negX'] is False and last24['negY'] is False
    assert abs(last24['X'] - 61.85) < 1e-9 and abs(last24['Y'] - 93.67) < 1e-9

    settings96 = {"start_x": 11.3, "start_y": 14.5, "spacing": 9.0, "origin_x": 17.096, "origin_y": 14.981}
    df96 = calculate_coords_96wellplate(settings96)
    assert len(df96) == 96
    first96 = df96.iloc[0].to_dict()
    last96 = df96.iloc[-1].to_dict()
    assert first96 == {'wellname': 'A1', 'wellnr': 1, 'X': 0.0, 'Y': 0.0, 'negX': True, 'negY': True}
    assert last96['wellname'] == 'H12' and last96['wellnr'] == 96 and last96['negX'] is False and last96['negY'] is False
    assert abs(last96['X'] - 57.204) < 1e-9 and abs(last96['Y'] - 98.519) < 1e-9


def test_summary_log_header_and_row_format(tmp_path):
    p = tmp_path / "summary.csv"
    headers = ["date", "time", "plate_id", "well_id", "well_name", "start_dpt", "exposure_time_us", "final_acquisition_fps", "any_frames_skipped", "acquisition_status", "run_folder"]
    initialize_experiment_log(str(p), headers)
    append_to_experiment_log(str(p), lambda now: [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), 1, 2, "A2", "-2.00", 1000, "12.34", False, "success", "folder"]) 
    rows = list(csv.reader(p.open()))
    assert rows[0] == headers
    assert rows[1][2:] == ['1', '2', 'A2', '-2.00', '1000', '12.34', 'False', 'success', 'folder']


def test_naming_contract_with_downstream_parser():
    date, plate, well = parse_date_plate_well("20251231_plate7_well42")
    assert (date, plate, well) == ("20251231", "7", "42")


def test_single_workflow_smoke(monkeypatch):
    called = {}

    def fake_load_config(path):
        called['config_path'] = path
        return {'ok': True}

    def fake_run_acquisition(plate, well, dpt, conf):
        called['args'] = (plate, well, dpt, conf)

    monkeypatch.setattr("src.acquire.workflows._get_acquisition_api", lambda: (fake_load_config, fake_run_acquisition))

    run_single_well_workflow(acquisition_config_path="ws_config.yaml", plate_no=0, well_no=0, start_dpt=-2.0)
    assert called['config_path'] == "ws_config.yaml"
    assert called['args'][0:3] == (0, 0, -2.0)


def test_load_acquisition_config_uses_lazy_api(monkeypatch):
    from src.acquire.workflows import load_acquisition_config

    called = {}

    def fake_load_config(path):
        called['path'] = path
        return {}

    monkeypatch.setattr("src.acquire.workflows._get_acquisition_api", lambda: (fake_load_config, lambda *args, **kwargs: None))

    conf = load_acquisition_config("ws_config.yaml")
    assert called['path'] == "ws_config.yaml"
    assert conf['loaded_from_path'].endswith("ws_config.yaml")
