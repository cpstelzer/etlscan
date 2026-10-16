import pytest

from src.acquire.focus_pattern import build_focus_cycle_pattern, count_frames_for_plane_visits


def test_sawtooth_and_zigzag_cycles():
    assert build_focus_cycle_pattern(0.0, "sawtooth", 0.3, 100, 1.0) == [0.0, 0.1, 0.2, 0.3]
    assert build_focus_cycle_pattern(0.0, "zigzag", 0.3, 100, 1.0) == [0.0, 0.1, 0.2, 0.3, 0.2, 0.1]


def test_negative_scan_ratio_ramps_downward():
    assert build_focus_cycle_pattern(0.0, "sawtooth", 0.3, 100, -1.0) == [0.0, -0.1, -0.2, -0.3]


@pytest.mark.parametrize("scan_ratio, scan_step_um", [(0.0, 100), (1.0, 0.0)])
def test_degenerate_sweep_raises(scan_ratio, scan_step_um):
    with pytest.raises(ValueError):
        build_focus_cycle_pattern(0.0, "zigzag", 0.3, scan_step_um, scan_ratio)


@pytest.mark.parametrize(
    "scan_step_um, expected_planes",
    # TS1 rig (3.2 mm, 3.36 mm/dpt): the last plane is the step nearest the
    # range end, so 300/700 um overshoot and 600/700 um share N = 6.
    [(100, 33), (200, 17), (300, 12), (400, 9), (500, 7), (600, 6), (700, 6)],
)
def test_plane_count_rounds_to_nearest_step(scan_step_um, expected_planes):
    sawtooth = build_focus_cycle_pattern(-1.0, "sawtooth", 3.2, scan_step_um, 3.36)
    zigzag = build_focus_cycle_pattern(-1.0, "zigzag", 3.2, scan_step_um, 3.36)
    assert len(sawtooth) == expected_planes
    assert len(set(zigzag)) == expected_planes
    assert len(zigzag) == max(1, 2 * (expected_planes - 1))


@pytest.mark.parametrize("n_planes", [1, 2, 5, 33])
@pytest.mark.parametrize("plane_visits", [1, 5])
def test_count_frames_matches_closed_forms(n_planes, plane_visits):
    up_values = [round(0.1 * i, 3) for i in range(n_planes)]
    sawtooth = up_values
    zigzag = up_values + up_values[-2:0:-1]
    assert count_frames_for_plane_visits(sawtooth, plane_visits) == plane_visits * n_planes
    expected_zigzag = plane_visits if n_planes == 1 else (2 * plane_visits - 1) * (n_planes - 1) + 1
    assert count_frames_for_plane_visits(zigzag, plane_visits) == expected_zigzag


def test_count_frames_every_plane_reaches_target():
    cycle = build_focus_cycle_pattern(-1.0, "zigzag", 3.2, 300, 3.36)
    frames = count_frames_for_plane_visits(cycle, 5)
    recorded = [cycle[i % len(cycle)] for i in range(frames)]
    assert min(recorded.count(dpt) for dpt in set(cycle)) == 5
    shorter = recorded[:-1]
    assert min(shorter.count(dpt) for dpt in set(cycle)) == 4


@pytest.mark.parametrize("focus_cycle, plane_visits", [([], 5), ([0.0, 0.1], 0)])
def test_count_frames_rejects_bad_input(focus_cycle, plane_visits):
    with pytest.raises(ValueError):
        count_frames_for_plane_visits(focus_cycle, plane_visits)
