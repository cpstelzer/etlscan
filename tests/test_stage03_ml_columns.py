import os
from pathlib import Path

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.stages.stage_03_filtering_and_classification import (
    run_stage_03_filtering_and_classification,
)
from src.utils import config_loader


FIXTURE_STAGE2 = Path('tests/fixtures/sample_stage2_final.csv')


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure each test starts with a clean configuration cache."""
    config_loader._config = {}
    yield
    config_loader._config = {}


def _write_config(tmp_path: Path, filter_on_boundary: bool = False) -> Path:
    """Write a minimal pipeline config for Stage 3 tests and return its path."""
    config_path = tmp_path / "config.yml"
    config_data = {
        'project_base_dir': str(tmp_path),
        'sample_subdir_name': 'sample_A',
        'stage_03_filtering_and_classification': {
            'model_type': 'random_forest_model',
            'size_invariant_features': False,
            'random_forest_model': {
                'model_filename': 'dummy_model.joblib',
                'scaler_filename': 'dummy_scaler.joblib',
            },
            'ml_filter': {'min_probability_threshold': 0.5},
            'quality_flags_filter': {
                'filter_on_boundary': filter_on_boundary,
                'filter_black_mask': False,
            },
        },
    }
    with open(config_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config_data, handle)
    return config_path


def _write_dummy_model_artifacts(tmp_path: Path) -> None:
    """Persist a small scaler and classifier to emulate trained artifacts."""
    models_dir = tmp_path / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    X = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.9],
            [0.9, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)

    joblib.dump(model, models_dir / 'dummy_model.joblib')
    joblib.dump(scaler, models_dir / 'dummy_scaler.joblib')

    feature_file = models_dir / 'dummy_model_feature_names.txt'
    with open(feature_file, 'w', encoding='utf-8') as handle:
        handle.write('feature_a\n')
        handle.write('feature_b\n')


def test_stage3_writes_ml_columns(tmp_path: Path):
    """Stage 3 should add ML columns and produce both filtered and legacy outputs."""
    config_path = _write_config(tmp_path)
    _write_dummy_model_artifacts(tmp_path)
    config = config_loader.load_config_once(str(config_path))

    stage2_df = pd.read_csv(FIXTURE_STAGE2).copy()
    stage2_df = pd.concat([stage2_df] * 4, ignore_index=True)
    stage2_df['final_crop_filename'] = [f'crop{i}.png' for i in range(len(stage2_df))]
    stage2_df['feature_a'] = [0.05, 0.15, 0.85, 0.95]
    stage2_df['feature_b'] = [0.05, 0.15, 0.75, 0.85]

    input_csv = tmp_path / 'stage2_final.csv'
    stage2_df.to_csv(input_csv, index=False)

    crops_dir = tmp_path / 'stage2_crops'
    crops_dir.mkdir()
    for crop_name in stage2_df['final_crop_filename']:
        (crops_dir / crop_name).write_bytes(b'00')

    stage3_output_dir = tmp_path / 'stage3'
    stage3_output_dir.mkdir()

    saved_csv = run_stage_03_filtering_and_classification(
        str(input_csv),
        str(stage3_output_dir),
        str(crops_dir),
        sample_id='sample_A',
        config=config,
    )

    assert saved_csv is not None
    result_df = pd.read_csv(saved_csv)
    assert 'ml_probability' in result_df.columns
    assert 'ml_label' in result_df.columns
    assert (result_df['ml_probability'] >= 0.5).all()
    assert set(result_df['ml_label'].unique()) == {'accept'}
    assert 0 < len(result_df) < len(stage2_df)

    filtered_good_csv = stage3_output_dir / 'filtered_good.csv'
    assert filtered_good_csv.exists()
    assert filtered_good_csv.as_posix() == saved_csv

    legacy_csv = stage3_output_dir / 'focussed_and_filtered.csv'
    assert legacy_csv.exists()
    legacy_df = pd.read_csv(legacy_csv)
    assert 'ml_probability' not in legacy_df.columns
    assert 'ml_label' not in legacy_df.columns
    assert len(legacy_df) == len(stage2_df)

    filtered_good_dir = stage3_output_dir / 'filtered_good'
    legacy_dir = stage3_output_dir / 'focussed_and_filtered'
    assert filtered_good_dir.is_dir()
    assert legacy_dir.is_dir()
    legacy_files = sorted(f.name for f in legacy_dir.iterdir())
    filtered_files = sorted(f.name for f in filtered_good_dir.iterdir())
    assert legacy_files
    assert filtered_files
    assert set(filtered_files).issubset(set(legacy_files))


@pytest.mark.parametrize("filter_on_boundary, expected_kept", [(False, 4), (True, 3)])
def test_stage3_obeys_quality_flags_filter(tmp_path: Path, filter_on_boundary: bool, expected_kept: int):
    """A particle on the crop boundary is dropped only when filter_on_boundary is switched on."""
    config_path = _write_config(tmp_path, filter_on_boundary=filter_on_boundary)
    _write_dummy_model_artifacts(tmp_path)
    config = config_loader.load_config_once(str(config_path))

    stage2_df = pd.concat([pd.read_csv(FIXTURE_STAGE2)] * 4, ignore_index=True)
    stage2_df['final_crop_filename'] = [f'crop{i}.png' for i in range(len(stage2_df))]
    stage2_df['feature_a'] = [0.05, 0.15, 0.85, 0.95]
    stage2_df['feature_b'] = [0.05, 0.15, 0.75, 0.85]
    stage2_df['is_on_crop_boundary'] = [False, False, False, True]

    input_csv = tmp_path / 'stage2_final.csv'
    stage2_df.to_csv(input_csv, index=False)
    crops_dir = tmp_path / 'stage2_crops'
    crops_dir.mkdir()
    for crop_name in stage2_df['final_crop_filename']:
        (crops_dir / crop_name).write_bytes(b'00')
    stage3_output_dir = tmp_path / 'stage3'
    stage3_output_dir.mkdir()

    run_stage_03_filtering_and_classification(
        str(input_csv), str(stage3_output_dir), str(crops_dir), sample_id='sample_A', config=config,
    )

    quality_filtered_df = pd.read_csv(stage3_output_dir / 'focussed_and_filtered.csv')
    assert len(quality_filtered_df) == expected_kept
