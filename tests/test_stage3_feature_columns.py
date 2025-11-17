import pandas as pd
import pytest

from src.stages.stage_03_filtering_and_classification import (
    METADATA_COLUMNS,
    METADATA_PREFIX,
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
