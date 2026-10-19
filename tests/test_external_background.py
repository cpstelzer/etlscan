"""Stage 1 with an external (specimen-free, separately recorded) background.

The external BG mode exists to find NON-moving specimens, which a time-shifted BG from
the same recording subtracts away. Covers the strict frame pairing (frame index + focal
plane, never the plate/well tokens of renamed folders), the frame trimming, the BG
adaptation (illumination match + rim exclusion), the sample-level failed-subtraction
check, and an end-to-end run on synthetic frames.
"""
import copy
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)

try:
    import trackpy  # noqa: F401
    HAVE_TRACKPY = True
except ImportError:
    sys.modules.setdefault("trackpy", types.ModuleType("trackpy"))
    HAVE_TRACKPY = False

from src.stages.stage_01_initial_detection import (  # noqa: E402
    _build_external_background_pairs,
    _build_time_shift_pairs,
    _filter_image_df,
    _parse_image_filenames,
    _drop_lint_detections,
    _prepare_external_background,
    _sweep_period_frames,
    run_stage_01_initial_detection,
    select_background_frame,
)
from src.core import image_io_operations  # noqa: E402
from src.utils.common_exceptions import ConfigError, DataValidationError, ProcessingError  # noqa: E402
from src.utils.config_loader import load_config_once  # noqa: E402

FD_STEP_MM = 0.25


def _frame_name(prefix, frame, fd_mm):
    dpt = -2.0 + fd_mm / 3.36
    return f"{prefix}_f{frame:05d}_dpt{dpt:.3f}_fd{fd_mm:.2f}mm.png"


def _sweep_fd(frame):
    """Zigzag-like sweep: fd climbs 0..1.0 mm and returns to 0 at frame 8."""
    position = frame % 8
    return FD_STEP_MM * (position if position <= 4 else 8 - position)


def _touch_frames(directory, prefix, frames, fd_of=_sweep_fd):
    os.makedirs(directory, exist_ok=True)
    for frame in frames:
        open(os.path.join(directory, _frame_name(prefix, frame, fd_of(frame))), "wb").close()
    return _parse_image_filenames(str(directory))


# --- pairing --------------------------------------------------------------------

def test_external_pairs_match_by_frame_despite_renamed_filename_prefix(tmp_path):
    # Folders were renamed by hand: FG files still carry the acquisition's plate3_well7
    # prefix, the BG files another one. Pairing must not care.
    fg_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(1, 10))
    bg_df = _touch_frames(tmp_path / "bg", "20261016_plate9_well1", range(0, 10))

    pairs = _build_external_background_pairs(fg_df, bg_df)

    assert [int(fg["frame"]) for fg, _ in pairs] == list(range(1, 10))
    assert all(int(fg["frame"]) == int(bg["frame"]) and fg["fd_mm"] == bg["fd_mm"] for fg, bg in pairs)


def test_external_pairs_fail_on_missing_bg_frame(tmp_path):
    fg_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(1, 10))
    bg_df = _touch_frames(tmp_path / "bg", "20261016_plate3_well7", [f for f in range(1, 10) if f != 6])
    with pytest.raises(DataValidationError, match="no BG frame"):
        _build_external_background_pairs(fg_df, bg_df)


def test_external_pairs_fail_on_focal_plane_mismatch(tmp_path):
    fg_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(1, 10))
    # BG recorded with a different scan step: same frame indices, other focal planes.
    bg_df = _touch_frames(tmp_path / "bg", "20261016_plate3_well7", range(1, 10), fd_of=lambda f: 2 * _sweep_fd(f))
    with pytest.raises(DataValidationError, match="focal-plane mismatch"):
        _build_external_background_pairs(fg_df, bg_df)


def test_filter_keeps_tail_only_without_cycle_trim(tmp_path):
    images_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(0, 12))  # fd == 0 at frames 0 and 8

    trimmed = _filter_image_df(images_df.copy(), trim_to_last_fd0=True)
    untrimmed = _filter_image_df(images_df.copy(), trim_to_last_fd0=False)

    assert trimmed["frame"].tolist() == list(range(1, 9))     # frame 0 dropped, cut after last fd0
    assert untrimmed["frame"].tolist() == list(range(1, 12))  # frame 0 dropped, tail kept


def test_time_shift_pairs_unchanged(tmp_path):
    # Regression: the pair builder reproduces the former per-frame select_background_frame loop.
    images_df = _filter_image_df(
        _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(0, 17)), trim_to_last_fd0=True
    )
    config_s1 = {"BG_frame_offset_mode": "fixed", "bg_offset_val": 3}

    pairs = _build_time_shift_pairs(images_df, config_s1, recording_fps=25.0)

    expected = [select_background_frame(images_df, i, 3) for i in range(len(images_df))]
    assert [(fg["filename"], bg["filename"]) for fg, bg in pairs] == [
        (e["fg_row"]["filename"], e["bg_row"]["filename"]) for e in expected
    ]


# --- sweep period (tracking memory) and lint-shaped detections -------------------

def test_sweep_period_is_spacing_of_fd0_frames(tmp_path):
    images_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(0, 20))  # fd == 0 at 0, 8, 16
    assert _sweep_period_frames(images_df) == 8


def test_sweep_period_of_run_shorter_than_one_cycle_is_run_length(tmp_path):
    images_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(0, 6))  # only frame 0 at fd == 0
    assert _sweep_period_frames(images_df) == 6


def test_sweep_period_without_fd0_frame_raises(tmp_path):
    images_df = _touch_frames(tmp_path / "fg", "20261016_plate3_well7", range(1, 6))
    with pytest.raises(DataValidationError, match="sweep period"):
        _sweep_period_frames(images_df)


def test_drop_lint_detections_by_solidity_only():
    # Curled lint in focus (low solidity, any aspect ratio) goes; compact detections stay,
    # including an elongated but convex one and one whose solidity could not be measured.
    fused_df = pd.DataFrame([
        {"X": 100.0, "s1_aspect_ratio": 1.8, "s1_solidity": 0.85},    # animal
        {"X": 200.0, "s1_aspect_ratio": 4.5, "s1_solidity": 0.90},    # elongated, convex
        {"X": 300.0, "s1_aspect_ratio": 1.3, "s1_solidity": 0.35},    # curled lint (low AR)
        {"X": 400.0, "s1_aspect_ratio": 6.0, "s1_solidity": 0.50},    # thin lint, at the limit
        {"X": 500.0, "s1_aspect_ratio": 2.0, "s1_solidity": np.nan},  # unmeasured
    ])

    kept_df, dropped_df = _drop_lint_detections(fused_df, 0.5)

    assert kept_df["X"].tolist() == [100.0, 200.0, 500.0]
    assert dropped_df["X"].tolist() == [300.0, 400.0]


# --- BG adaptation --------------------------------------------------------------

@pytest.fixture()
def repo_config():
    return copy.deepcopy(load_config_once(os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")))


def test_prepare_external_background_insets_well_mask(repo_config):
    bg = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(bg, (200, 200), 150, 180, -1)  # lit well, dark plate around it
    fg = bg.copy()
    external_cfg = dict(repo_config["stage_01_initial_detection"]["external_background"], rim_exclusion_px=20)
    well_identification = dict(
        repo_config["stage_01_initial_detection"]["black_plate_well_analysis"]["well_identification"],
        min_expected_diameter_px=200, max_expected_diameter_px=400,
    )

    bg_matched, rim_mask = _prepare_external_background(fg, bg, external_cfg, well_identification)

    assert bg_matched.shape == bg.shape
    assert rim_mask[200, 200] == 255           # well interior kept
    assert rim_mask[200, 200 + 140] == 0       # 10 px inside the rim: excluded
    assert rim_mask[200, 200 + 125] == 255     # 25 px inside the rim: kept


# --- end to end -----------------------------------------------------------------

@pytest.fixture()
def no_flat_field_confirmed(monkeypatch):
    """Synthetic samples have no flat-field calibration: answer the one-time prompt with 'y'."""
    monkeypatch.setattr(image_io_operations, "_ffc_missing_confirmed", True)


def _write_sample(root, sample_id, frames, make_fg, make_bg, fd_of=_sweep_fd):
    for side, make in (("fg", make_fg), ("bg", make_bg)):
        sample_dir = os.path.join(root, side, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        for frame in frames:
            name = _frame_name("20261016_plate3_well7", frame, fd_of(frame))
            cv2.imwrite(os.path.join(sample_dir, name), make(frame))
    return os.path.join(root, "fg", sample_id), os.path.join(root, "bg", sample_id)


def _flat_frame(rng):
    return np.clip(rng.normal(180, 1.5, size=(600, 600)), 0, 255).astype(np.uint8)


def _with_static_specimen(frame_uint8):
    """A fixated (non-moving) dark specimen plus a darker lower half (illumination drift)."""
    frame_float = frame_uint8.astype(np.float32)
    frame_float[300:, :] -= 20
    frame_uint8 = np.clip(frame_float, 0, 255).astype(np.uint8)
    cv2.ellipse(frame_uint8, (300, 260), (30, 15), 0, 0, 360, 60, -1)
    return frame_uint8


@pytest.mark.skipif(not HAVE_TRACKPY, reason="trackpy required for the Stage-1 tracking step.")
def test_external_background_finds_static_specimen(tmp_path, repo_config, no_flat_field_confirmed):
    rng = np.random.default_rng(0)
    frames = range(0, 10)  # fd == 0 at frame 8: a time-shift run would trim frame 9
    fg_dir, bg_dir = _write_sample(
        str(tmp_path), "20261016_plate14_well2", frames,
        make_fg=lambda f: _with_static_specimen(_flat_frame(rng)),
        make_bg=lambda f: _flat_frame(rng),
    )
    output_dir = str(tmp_path / "out")

    csv_path = run_stage_01_initial_detection(fg_dir, output_dir, "20261016_plate14_well2", repo_config, bg_image_dir_path=bg_dir)

    tracked = pd.read_csv(csv_path)
    assert tracked["track_id"].nunique() == 1
    assert sorted(tracked["frame"].unique()) == list(range(1, 10))
    assert tracked["X"].between(270, 330).all() and tracked["Y"].between(245, 275).all()
    audit = pd.read_csv(os.path.join(output_dir, "20261016_plate14_well2_stage1_bg_pairing.csv"))
    assert len(audit) == 9 and not audit["subtraction_failed"].any()


@pytest.mark.skipif(not HAVE_TRACKPY, reason="trackpy required for the Stage-1 tracking step.")
def test_external_background_of_other_scene_fails_sample(tmp_path, repo_config, no_flat_field_confirmed):
    rng = np.random.default_rng(1)

    def _speckled(frame):
        # Fine dark speckle over 30% of the FG that the BG lacks: every frame floods the gate.
        frame_uint8 = _flat_frame(rng)
        frame_uint8[rng.random(frame_uint8.shape) < 0.3] = 120
        return frame_uint8

    fg_dir, bg_dir = _write_sample(
        str(tmp_path), "20261016_plate14_well2", range(0, 10), make_fg=_speckled, make_bg=lambda f: _flat_frame(rng)
    )
    output_dir = str(tmp_path / "out")

    with pytest.raises(ProcessingError, match="does not match the FG scene"):
        run_stage_01_initial_detection(fg_dir, output_dir, "20261016_plate14_well2", repo_config, bg_image_dir_path=bg_dir)
    audit = pd.read_csv(os.path.join(output_dir, "20261016_plate14_well2_stage1_bg_pairing.csv"))
    assert audit["subtraction_failed"].all()


def _long_sweep_fd(frame):
    """Zigzag with a 16-frame cycle: fd == 0 at frames 0, 16, 32."""
    position = frame % 16
    return FD_STEP_MM * (position if position <= 8 else 16 - position)


def _with_focus_dependent_specimen_and_fibre(frame, frame_uint8):
    """A static specimen detectable only while the sweep passes its plane (3 frames per
    cycle, 13-frame gaps > tracking.memory_frames) plus a curved lint fibre seen throughout."""
    if frame % 16 in (3, 4, 5):
        cv2.ellipse(frame_uint8, (200, 300), (30, 15), 0, 0, 360, 60, -1)
    curve_x = np.arange(330, 550)
    curve_y = (300 + 20 * np.sin((curve_x - 330) / 220 * 2 * np.pi)).astype(np.int32)
    cv2.polylines(frame_uint8, [np.column_stack([curve_x, curve_y]).astype(np.int32)], False, 60, thickness=5)
    return frame_uint8


@pytest.mark.skipif(not HAVE_TRACKPY, reason="trackpy required for the Stage-1 tracking step.")
def test_external_background_bridges_sweep_passes_and_rejects_lint(tmp_path, repo_config, no_flat_field_confirmed):
    rng = np.random.default_rng(2)
    fg_dir, bg_dir = _write_sample(
        str(tmp_path), "20261016_plate14_well2", range(0, 34),
        make_fg=lambda f: _with_focus_dependent_specimen_and_fibre(f, _flat_frame(rng)),
        make_bg=lambda f: _flat_frame(rng),
        fd_of=_long_sweep_fd,
    )
    output_dir = str(tmp_path / "out")

    csv_path = run_stage_01_initial_detection(fg_dir, output_dir, "20261016_plate14_well2", repo_config, bg_image_dir_path=bg_dir)

    tracked = pd.read_csv(csv_path)
    # With tracking.memory_frames (10) the two 3-frame passes would be separate stubs below
    # min_track_length and vanish; bridging one sweep period keeps one 6-detection track.
    assert tracked["track_id"].nunique() == 1
    assert sorted(tracked["frame"].unique()) == [3, 4, 5, 19, 20, 21]
    assert tracked["X"].between(170, 230).all()
    # The curled fibre is dropped detection by detection before tracking (all 33 frames).
    dropped = pd.read_csv(os.path.join(output_dir, "20261016_plate14_well2_stage1_rejected_lint_detections.csv"))
    assert len(dropped) == 33
    assert (dropped["X"] > 330).all() and (dropped["s1_solidity"] <= 0.5).all()


def test_external_background_requires_2d_tracking(tmp_path, repo_config):
    rng = np.random.default_rng(3)
    fg_dir, bg_dir = _write_sample(
        str(tmp_path), "20261016_plate14_well2", range(0, 3), make_fg=lambda f: _flat_frame(rng), make_bg=lambda f: _flat_frame(rng)
    )
    repo_config["stage_01_initial_detection"]["tracking"]["track_in_2d"] = False
    with pytest.raises(ConfigError, match="track_in_2d"):
        run_stage_01_initial_detection(fg_dir, str(tmp_path / "out"), "20261016_plate14_well2", repo_config, bg_image_dir_path=bg_dir)
