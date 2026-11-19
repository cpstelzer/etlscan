"""The Stage-3 quality-flag filters obey ``quality_flags_filter`` in batch and realtime.

``apply_quality_flag_filters`` is the single implementation; both Stage-3 paths must
keep flagged particles when a flag's filter is switched off in the config.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.core.feature_calculators.quality_calculator import apply_quality_flag_filters
from src.realtime.stage3_rt import run_stage3_realtime
from src.utils import config_loader
from src.utils.common_exceptions import DataValidationError

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")


@pytest.fixture()
def particles_df() -> pd.DataFrame:
    """Five particles: one clean, two with a black-mask artifact, one on the boundary, one both."""
    return pd.DataFrame({
        "particle": ["clean", "artifact_a", "artifact_b", "boundary", "both"],
        "has_black_mask_artifact": [False, True, True, False, True],
        "is_on_crop_boundary": [False, False, False, True, True],
    })


def test_both_filters_enabled_drop_all_flagged_particles(particles_df):
    kept_df, removed_black_mask, removed_on_boundary = apply_quality_flag_filters(
        particles_df, filter_black_mask=True, filter_on_boundary=True
    )
    assert list(kept_df["particle"]) == ["clean"]
    assert removed_black_mask == 3
    # 'both' already went with the black-mask filter, so only 'boundary' is counted here.
    assert removed_on_boundary == 1
    assert list(kept_df.index) == [0]


def test_disabled_black_mask_filter_keeps_artifact_particles(particles_df):
    kept_df, removed_black_mask, removed_on_boundary = apply_quality_flag_filters(
        particles_df, filter_black_mask=False, filter_on_boundary=True
    )
    assert list(kept_df["particle"]) == ["clean", "artifact_a", "artifact_b"]
    assert removed_black_mask == 0
    assert removed_on_boundary == 2


def test_disabled_boundary_filter_keeps_boundary_particles(particles_df):
    kept_df, removed_black_mask, removed_on_boundary = apply_quality_flag_filters(
        particles_df, filter_black_mask=True, filter_on_boundary=False
    )
    assert list(kept_df["particle"]) == ["clean", "boundary"]
    assert removed_black_mask == 3
    assert removed_on_boundary == 0


def test_both_filters_disabled_keep_every_particle(particles_df):
    kept_df, removed_black_mask, removed_on_boundary = apply_quality_flag_filters(
        particles_df, filter_black_mask=False, filter_on_boundary=False
    )
    assert len(kept_df) == len(particles_df)
    assert (removed_black_mask, removed_on_boundary) == (0, 0)


def test_missing_flag_values_count_as_not_flagged():
    particles = pd.DataFrame({
        "has_black_mask_artifact": [None, True],
        "is_on_crop_boundary": [False, None],
    })
    kept_df, removed_black_mask, removed_on_boundary = apply_quality_flag_filters(
        particles, filter_black_mask=True, filter_on_boundary=True
    )
    assert len(kept_df) == 1
    assert (removed_black_mask, removed_on_boundary) == (1, 0)


def test_enabled_filter_without_its_column_fails(particles_df):
    with pytest.raises(DataValidationError, match="is_on_crop_boundary"):
        apply_quality_flag_filters(
            particles_df.drop(columns="is_on_crop_boundary"), filter_black_mask=True, filter_on_boundary=True
        )


def test_disabled_filter_does_not_need_its_column(particles_df):
    kept_df, _, _ = apply_quality_flag_filters(
        particles_df.drop(columns="is_on_crop_boundary"), filter_black_mask=True, filter_on_boundary=False
    )
    assert list(kept_df["particle"]) == ["clean", "boundary"]


def test_empty_input_passes_through():
    kept_df, removed_black_mask, removed_on_boundary = apply_quality_flag_filters(
        pd.DataFrame(), filter_black_mask=True, filter_on_boundary=True
    )
    assert kept_df.empty
    assert (removed_black_mask, removed_on_boundary) == (0, 0)


def test_realtime_stage3_uses_the_same_filters(particles_df):
    result = run_stage3_realtime(
        particles_df, {"enable_artifact_filter": True, "enable_boundary_filter": False}
    )
    assert list(result.accepted_df["particle"]) == ["clean", "boundary"]
    assert result.rejected_artifact_count == 3
    assert result.rejected_boundary_count == 0


def test_realtime_stage3_requires_both_switches(particles_df):
    with pytest.raises(KeyError):
        run_stage3_realtime(particles_df, {"enable_artifact_filter": True})


# ---------------------------------------------------------------------------
# Shipped config: the keys the training scripts read are where they read them.
# ---------------------------------------------------------------------------
def test_training_keys_live_under_training_params():
    config_loader.load_config_once.cache_clear()
    stage3_config = config_loader.load_config_once(CONFIG_PATH)["stage_03_filtering_and_classification"]
    training_params = stage3_config["training_params"]

    rf_training_keys = {
        "good_data_subdir", "bad_data_subdir", "test_split_size", "random_state",
        "exclude_feature_prefixes", "rf_param_grid",
    }
    assert rf_training_keys <= set(training_params)
    assert {"dataset_subdir", "default_epochs", "default_batch_size", "learning_rate",
            "weight_decay", "mixup_alpha"} <= set(training_params["efficient_net_training"])
    # Not duplicated one level up, where the scripts never looked.
    assert not (rf_training_keys | {"efficient_net_training"}) & set(stage3_config)
    # Every grid entry is a list of candidate values for GridSearchCV.
    assert all(isinstance(values, list) and values for values in training_params["rf_param_grid"].values())


def test_removed_stage3_keys_are_gone():
    config_loader.load_config_once.cache_clear()
    config = config_loader.load_config_once(CONFIG_PATH)
    stage3_config = config["stage_03_filtering_and_classification"]
    assert "sharpness_filter" not in stage3_config
    assert "debug_output" not in stage3_config
    egg_config = config["stage_04_biological_analysis"]["egg_detection"]
    assert "egg_min_eccentricity" not in egg_config
    assert "orientation_diff_tolerance_deg" not in egg_config
