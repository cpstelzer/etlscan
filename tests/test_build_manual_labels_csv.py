"""Tests for scripts.build_manual_labels_csv."""

from pathlib import Path

import pandas as pd

from scripts.build_manual_labels_csv import (
    build_manual_labels_dataframe,
    parse_sample_id_from_filename,
)


def test_parse_sample_id_from_filename() -> None:
    """Sample IDs should be parsed from filenames that include a track suffix."""
    assert (
        parse_sample_id_from_filename('20251112_plate2_well19_track1_frame5')
        == '20251112_plate2_well19'
    )
    assert parse_sample_id_from_filename('sample_without_track') is None


def test_build_manual_labels_dataframe(tmp_path) -> None:
    """Curated crops should be matched to Stage-03 rows and labelled correctly."""
    stage3_root = tmp_path / 'output'
    sample_id = '20251112_plate2_well19'
    csv_dir = stage3_root / 'size_AS1_L1' / sample_id / 'stage3_filtering_classification'
    csv_dir.mkdir(parents=True)
    good_png = '20251112_plate2_well19_track1_frame5_X0_Y0.png'
    bad_png = '20251112_plate2_well19_track2_frame7_X0_Y0.png'

    df = pd.DataFrame(
        [
            {'SAMPLE_ID': sample_id, 'final_crop_filename': good_png, 'variance_laplacian': 123.0},
            {'SAMPLE_ID': sample_id, 'final_crop_filename': bad_png, 'variance_laplacian': 45.0},
        ]
    )
    csv_path = csv_dir / 'focussed_and_filtered.csv'
    df.to_csv(csv_path, index=False)

    good_dir = tmp_path / 'data' / 'training_data' / 'good'
    bad_dir = tmp_path / 'data' / 'training_data' / 'bad'
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    (good_dir / good_png).touch()
    (bad_dir / bad_png).touch()

    output_csv = tmp_path / 'manual_labels.csv'
    combined_df = build_manual_labels_dataframe(
        good_dir=good_dir,
        bad_dir=bad_dir,
        stage3_root=stage3_root,
        output_csv=output_csv,
    )

    assert output_csv.exists()
    assert len(combined_df) == 2
    assert sorted(combined_df['label']) == [0, 1]
    assert set(combined_df['variance_laplacian']) == {123.0, 45.0}
