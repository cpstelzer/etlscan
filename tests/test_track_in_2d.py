# tests/test_track_in_2d.py
"""The Stage-1 tracker can link in the XY plane only (``track_in_2d``).

On focus-sweep acquisitions ``z`` is the ETL focal depth: identical for every
particle within a frame and stepping ~0.2 mm/frame. Feeding it to the linker
spends the search budget and throttles memory, fragmenting fast movers. Linking
in 2D fixes that while still carrying ``z`` through for the downstream depth merge.
"""
import pandas as pd

from src.stages.stage_01_initial_detection import _track_particles_3d


def _sweep_df() -> pd.DataFrame:
    """One fast xy mover (A) and one stationary particle (B) under a shared z sweep.

    A steps 0.30 mm/frame in x while the focal depth steps 0.20 mm/frame. With a
    0.33 mm search range the 2D step (0.30) links but the 3D step
    (sqrt(0.30^2 + 0.20^2) = 0.36) breaks -- so 2D keeps A whole while 3D fragments
    it into single-frame stubs. B is stationary, links either way, and anchors the
    frames so neither run collapses to an empty frame.
    """
    rows = []
    for frame in range(6):
        z = 5.0 + 0.2 * frame
        rows.append({"frame": frame, "x_mm": 0.30 * frame, "y_mm": 0.0, "z_mm": z})   # A: fast
        rows.append({"frame": frame, "x_mm": 10.0, "y_mm": 10.0, "z_mm": z})          # B: stationary
    return pd.DataFrame(rows)


_LINK_KW = dict(search_range_mm=0.33, memory_frames=0, min_track_length=5)


def test_2d_linking_keeps_fast_mover_that_3d_fragments():
    df = _sweep_df()

    tracks_3d = _track_particles_3d(df, **_LINK_KW, track_in_2d=False)
    tracks_2d = _track_particles_3d(df, **_LINK_KW, track_in_2d=True)

    # 3D: the fast mover fragments into sub-threshold stubs and is filtered out;
    # only the stationary particle survives. 2D: both survive as full tracks.
    assert tracks_3d["track_id"].nunique() == 1
    assert tracks_2d["track_id"].nunique() == 2
    # The fast track is continuous across all six frames only under 2D linking.
    assert tracks_2d.groupby("track_id").size().max() == 6


def test_2d_linking_retains_z_for_downstream_merge():
    df = _sweep_df()

    tracks_2d = _track_particles_3d(df, **_LINK_KW, track_in_2d=True)

    # run_particle_tracking merges tracks back onto the processed data using z, so z
    # must survive linking even though it is not part of the 2D link metric.
    assert "z" in tracks_2d.columns
    assert tracks_2d["z"].notna().all()


def test_3d_default_is_unchanged_when_flag_absent():
    df = _sweep_df()

    # Omitting the flag must reproduce legacy 3D behavior exactly (fast mover lost).
    tracks_default = _track_particles_3d(df, **_LINK_KW)
    tracks_3d = _track_particles_3d(df, **_LINK_KW, track_in_2d=False)

    assert tracks_default["track_id"].nunique() == tracks_3d["track_id"].nunique() == 1
