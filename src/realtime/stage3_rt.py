from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.core.feature_calculators.quality_calculator import apply_quality_flag_filters


@dataclass
class Stage3Result:
    accepted_df: pd.DataFrame
    rejected_artifact_count: int = 0
    rejected_boundary_count: int = 0


def run_stage3_realtime(df: pd.DataFrame, config: Dict[str, bool]) -> Stage3Result:
    """Apply the quality-flag filters to one frame's Stage-2 particles.

    ``config`` holds ``enable_artifact_filter`` and ``enable_boundary_filter`` (from
    ``stage_03_filtering_and_classification.quality_flags_filter``); the filtering itself is
    the batch pipeline's :func:`apply_quality_flag_filters`.
    """
    if df is None or df.empty:
        return Stage3Result(accepted_df=pd.DataFrame())

    accepted_df, rejected_artifact, rejected_boundary = apply_quality_flag_filters(
        df.copy(),
        filter_black_mask=config["enable_artifact_filter"],
        filter_on_boundary=config["enable_boundary_filter"],
    )
    return Stage3Result(
        accepted_df=accepted_df,
        rejected_artifact_count=rejected_artifact,
        rejected_boundary_count=rejected_boundary,
    )
