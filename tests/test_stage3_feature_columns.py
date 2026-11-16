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
# Stage-2 config that produces no LBP columns, so only the base features are required.
NO_LBP_CONFIG = {'stage_02_characterization': {'shape_descriptors': {'lbp_radii': [], 'lbp_points': 8}}}


def test_validate_required_feature_columns_passes_with_fixture():
    """Fixture CSV should satisfy required feature column validation."""
    df = pd.read_csv(FIXTURE_PATH)
    # Should not raise
    _validate_required_feature_columns(df, config=NO_LBP_CONFIG)


def test_validate_required_feature_columns_raises_on_missing():
    """Validation should raise when key feature columns are absent."""
    df = pd.DataFrame({'frame': [1]})
    with pytest.raises(DataValidationError):
        _validate_required_feature_columns(df, config=NO_LBP_CONFIG)


def test_metadata_columns_are_prefixed():
    """Metadata columns should be renamed with the configured prefix."""
    df = pd.read_csv(FIXTURE_PATH)
    renamed = _rename_metadata_columns(df.copy())
    for col in METADATA_COLUMNS:
        expected_name = f"{METADATA_PREFIX}{col}"
        assert expected_name in renamed.columns
        assert col not in renamed.columns


def test_lbp_columns_expand_with_config():
    """LBP feature expectations should reflect radii and point settings in config."""
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
    """Build a dataframe containing a base feature set plus a configurable number of LBP columns."""
    data = {col: [0] for col in BASE_NUMERIC_FEATURE_COLUMNS}
    for i in range(lbp_count):
        data[f'lbp_{i}'] = [i]
    return pd.DataFrame(data)


def test_validate_allows_superset_lbp_columns():
    """Validation should allow more LBP columns than requested by config."""
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
    """Validation should raise when expected LBP features are missing."""
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
