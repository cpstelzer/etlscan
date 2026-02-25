from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class Stage3Result:
    accepted_df: pd.DataFrame
    rejected_artifact_count: int = 0
    rejected_boundary_count: int = 0


def run_stage3_realtime(df: pd.DataFrame, config: Dict[str, bool]) -> Stage3Result:
    if df is None or df.empty:
        return Stage3Result(accepted_df=pd.DataFrame())

    filtered = df.copy()
    rejected_artifact = 0
    rejected_boundary = 0

    if config.get("enable_artifact_filter", True) and "has_black_mask_artifact" in filtered.columns:
        artifact_mask = filtered["has_black_mask_artifact"].fillna(False).astype(bool)
        rejected_artifact = int(artifact_mask.sum())
        filtered = filtered[~artifact_mask]

    if config.get("enable_boundary_filter", True) and "is_on_crop_boundary" in filtered.columns:
        boundary_mask = filtered["is_on_crop_boundary"].fillna(False).astype(bool)
        rejected_boundary = int(boundary_mask.sum())
        filtered = filtered[~boundary_mask]

    return Stage3Result(
        accepted_df=filtered.reset_index(drop=True),
        rejected_artifact_count=rejected_artifact,
        rejected_boundary_count=rejected_boundary,
    )
