"""Stage-1 well-rim exclusion, shared by both background sources and the realtime Stage 1.

On focus-sweep recordings the well rim's image position moves with the focal plane
(focus breathing), so an FG frame and its time-shifted BG frame show the rim at different
radii. The subtraction then leaves a band along the rim that is detected and linked into
animal tracks (false Stage-2 focus peaks near the sweep start). The subtracted analysis
therefore searches only inside both frames' wells, rim_exclusion_px away from either rim.
"""
import copy
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required.", exc_type=ImportError)
pytest.importorskip("trackpy", reason="Stage 1 imports trackpy.")

from src.core.image_enhancement import subtract_images  # noqa: E402
from src.realtime.config import _resolve_stage1_config  # noqa: E402
from src.realtime.stage1_rt import run_stage1_realtime  # noqa: E402
from src.stages.stage_01_initial_detection import analyze_subtracted_image, prepare_masked_foreground  # noqa: E402
from src.utils.common_exceptions import ImageProcessingError  # noqa: E402
from src.utils.config_loader import load_config_once  # noqa: E402

FRAME_PX = 1200
WELL_CENTER = (600, 600)
SPECIMEN_FG = (600, 520)


@pytest.fixture()
def repo_config():
    config = copy.deepcopy(load_config_once(os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")))
    # The synthetic well (1000 px) is smaller than the rig's (~3200 px).
    config["stage_01_initial_detection"]["black_plate_well_analysis"]["well_identification"].update(
        min_expected_diameter_px=800, max_expected_diameter_px=1200
    )
    return config


def _frame(well_radius_px, specimen_center):
    """Lit well with a blurred rim on a black plate, plus one dark specimen."""
    frame = np.zeros((FRAME_PX, FRAME_PX), dtype=np.uint8)
    cv2.circle(frame, WELL_CENTER, well_radius_px, 180, -1)
    frame = cv2.GaussianBlur(frame, (0, 0), 4)
    cv2.ellipse(frame, specimen_center, (30, 15), 0, 0, 360, 60, -1)
    return frame


def _time_shift_pair():
    # The BG frame shows another focal plane: rim 8 px further out; the specimen moved.
    return _frame(500, SPECIMEN_FG), _frame(508, (600, 680))


def _subtracted_detections(config, fg, bg):
    config_s1 = config["stage_01_initial_detection"]
    prep = prepare_masked_foreground(
        fg,
        bg,
        enable_brightness_adjustment=config_s1["black_plate_well_analysis"]["enable_brightness_adjustment"],
        well_analysis_config=config_s1["black_plate_well_analysis"],
        well_masking_cfg=config_s1["well_masking"],
        system_cfg=config["system"],
        source_illumination=config["system"]["source_image_illumination_type"],
    )
    result = analyze_subtracted_image(
        subtract_images(prep["fg_for_sub"], prep["bg_for_sub"]),
        prep["subtraction_roi_mask"],
        config_s1["particle_analysis"],
        config["system"]["thresholding"],
        "s",
        "f.png",
        subtracted_foreground_config=config["system"]["subtracted_foreground"],
    )
    assert not result["subtraction_failed"]
    return result["particles_df"]


def _at_specimen(detections_df):
    return np.hypot(detections_df["X"] - SPECIMEN_FG[0], detections_df["Y"] - SPECIMEN_FG[1]) < 30


def test_time_shift_pair_without_inset_detects_the_rim(repo_config):
    # Guards the synthetic setup: the moved rim is a detection unless it is excluded. Here
    # it is one ring around the whole well, whose outer contour even swallows the specimen.
    repo_config["stage_01_initial_detection"]["black_plate_well_analysis"]["rim_exclusion_px"] = 0
    detections_df = _subtracted_detections(repo_config, *_time_shift_pair())
    assert (~_at_specimen(detections_df)).sum() >= 1
    assert detections_df["area"].max() > np.pi * 450 ** 2


def test_time_shift_pair_excludes_the_rim_and_keeps_the_specimen(repo_config):
    detections_df = _subtracted_detections(repo_config, *_time_shift_pair())
    assert len(detections_df) == 1
    assert _at_specimen(detections_df).all()


def test_realtime_stage1_excludes_the_rim(repo_config):
    stage1_config = _resolve_stage1_config(repo_config, repo_config["realtime_mode"])
    fg, bg = _time_shift_pair()
    detections_df = run_stage1_realtime(fg, bg, 0, 0.0, stage1_config).detections_df
    assert len(detections_df) == 1
    assert _at_specimen(detections_df).all()


def test_missing_well_mask_fails_fast(repo_config):
    well_analysis_config = repo_config["stage_01_initial_detection"]["black_plate_well_analysis"]
    well_analysis_config["well_identification"]["fallback_full_frame_on_failure"] = False
    blank = np.zeros((FRAME_PX, FRAME_PX), dtype=np.uint8)  # no well to find
    with pytest.raises(ImageProcessingError, match="rim"):
        prepare_masked_foreground(
            blank,
            blank,
            enable_brightness_adjustment=True,
            well_analysis_config=well_analysis_config,
            well_masking_cfg=repo_config["stage_01_initial_detection"]["well_masking"],
            system_cfg=repo_config["system"],
            source_illumination="collimated",
        )
