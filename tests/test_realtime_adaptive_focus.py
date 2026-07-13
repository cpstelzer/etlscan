from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RT_ACQ_PATH = REPO_ROOT / "src" / "realtime" / "acquisition.py"


def _load_rt_acq_module():
    spec = importlib.util.spec_from_file_location("rtacq_test", RT_ACQ_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rtacq_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_adaptive_window_steps_uses_fraction_minimum_and_dispersion() -> None:
    mod = _load_rt_acq_module()
    steps = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    selected_d1 = mod.select_adaptive_window_steps(
        diopter_steps=steps,
        center_dpt=0.4,
        window_fraction=0.1,
        min_window_points=3,
        dispersion_coeff=1,
    )
    assert selected_d1 == [0.2, 0.4, 0.6]

    selected_d2 = mod.select_adaptive_window_steps(
        diopter_steps=steps,
        center_dpt=0.4,
        window_fraction=0.1,
        min_window_points=3,
        dispersion_coeff=2,
    )
    assert selected_d2 == [0.0, 0.4, 0.8]

    selected_d3 = mod.select_adaptive_window_steps(
        diopter_steps=steps,
        center_dpt=0.4,
        window_fraction=0.1,
        min_window_points=3,
        dispersion_coeff=3,
    )
    assert selected_d3 == [-0.2, 0.4, 1.0]


def test_select_adaptive_window_steps_keeps_odd_window() -> None:
    mod = _load_rt_acq_module()
    steps = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    selected = mod.select_adaptive_window_steps(
        diopter_steps=steps,
        center_dpt=0.2,
        window_fraction=0.5,
        min_window_points=3,
        dispersion_coeff=1,
    )
    assert len(selected) == 5
    assert 0.2 in selected
