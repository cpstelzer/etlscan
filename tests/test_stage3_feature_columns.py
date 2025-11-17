import pandas as pd
import pytest

from src.stages.stage_03_filtering_and_classification import (
    BASE_NUMERIC_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    METADATA_PREFIX,
    _get_expected_numeric_feature_columns,
    _rename_metadata_columns,
    _validate_required_feature_columns,
)
from src.utils.common_exceptions import DataValidationError


FIXTURE_PATH = 'tests/fixtures/sample_stage2_final.csv'


def test_validate_required_feature_columns_passes_with_fixture():
    df = pd.read_csv(FIXTURE_PATH)
    # Should not raise
    _validate_required_feature_columns(df)


def test_validate_required_feature_columns_raises_on_missing():
    df = pd.DataFrame({'frame': [1]})
    with pytest.raises(DataValidationError):
        _validate_required_feature_columns(df)


def test_metadata_columns_are_prefixed():
    df = pd.read_csv(FIXTURE_PATH)
    renamed = _rename_metadata_columns(df.copy())
    for col in METADATA_COLUMNS:
        expected_name = f"{METADATA_PREFIX}{col}"
        assert expected_name in renamed.columns
        assert col not in renamed.columns


def test_lbp_columns_expand_with_config():
    custom_config = {
        'stage_02_characterization': {
            'shape_descriptors': {
                'lbp_radii': [1, 2, 3],
                'lbp_points': 16,
            }
        }
    }
    expected_columns = _get_expected_numeric_feature_columns(custom_config)
    lbp_columns = [col for col in expected_columns if col.startswith('lbp_')]
    assert len(lbp_columns) == 3 * (16 + 2)
    assert lbp_columns[-1] == 'lbp_53'

    df = pd.DataFrame({col: [0] for col in expected_columns})
    # Should not raise with the matching config
    _validate_required_feature_columns(df, config=custom_config)


def _build_numeric_df(lbp_count: int) -> pd.DataFrame:
    data = {col: [0] for col in BASE_NUMERIC_FEATURE_COLUMNS}
    for i in range(lbp_count):
        data[f'lbp_{i}'] = [i]
    return pd.DataFrame(data)


def test_validate_allows_superset_lbp_columns():
    df = _build_numeric_df(lbp_count=36)
    narrow_config = {
        'stage_02_characterization': {
            'shape_descriptors': {
                'lbp_radii': [1],
                'lbp_points': 8,
            }
        }
    }
    # Observed LBP columns exceed what the config expects; this should be allowed.
    _validate_required_feature_columns(df, config=narrow_config)


def test_validate_detects_missing_lbp_columns():
    df = _build_numeric_df(lbp_count=3)
    config = {
        'stage_02_characterization': {
            'shape_descriptors': {
                'lbp_radii': [1],
                'lbp_points': 4,  # Expect 6 columns
            }
        }
    }
    with pytest.raises(DataValidationError):
        _validate_required_feature_columns(df, config=config)
