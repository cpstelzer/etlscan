from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class Stage3Result:
    accepted_df: pd.DataFrame


def run_stage3_realtime(df: pd.DataFrame, config: Dict[str, bool]) -> Stage3Result:
    if df is None or df.empty:
        return Stage3Result(accepted_df=pd.DataFrame())

    filtered = df.copy()
    if config.get("enable_artifact_filter", True) and "has_black_mask_artifact" in filtered.columns:
        filtered = filtered[~filtered["has_black_mask_artifact"].fillna(False).astype(bool)]
    if config.get("enable_boundary_filter", True) and "is_on_crop_boundary" in filtered.columns:
        filtered = filtered[~filtered["is_on_crop_boundary"].fillna(False).astype(bool)]
    return Stage3Result(accepted_df=filtered.reset_index(drop=True))
