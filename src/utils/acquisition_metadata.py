# src/utils/acquisition_metadata.py
"""Read and cross-check the acquisition metadata of pipeline input samples.

The acquisition writes a ``# Key: value`` header at the top of each run's
``metadata_log.csv`` (see ``src/acquire/wellscanner_acquisition_module.py``). The
pipeline uses it to verify, before any image is processed, that an external
background run was recorded with the same image-defining settings as its foreground
run, and that each sample matches its row in an optional experiment-design table.
Below the header, the frame table holds one row per focus-pattern entry; the pipeline
reads it to tell frames the camera lost during acquisition from files missing later.
The volume a recording covers is read from its frames themselves (frame size and the
focal planes in their filenames); its frame rate and pixel scale come from the header.
"""
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.common_exceptions import DataValidationError
from src.utils.config_loader import get_pixel_to_mm_factor, require_config_sections
from src.utils.path_helpers import parse_date_plate_well, parse_frame_filenames

logger = logging.getLogger(__name__)

_FRAME_DATA_SECTION = "Frame Data"
_SECTION_BANNER = re.compile(r"(?:===|---)\s*(.+?)\s*(?:===|---)")

# Frame-table values the acquisition writes for a frame it did not save: a field it could
# not fill (e.g. output_filename of an exposure that never reached the PC) holds
# METADATA_MISSING_VALUE, and a frame whose loop iteration raised has an output_filename
# starting with METADATA_LOOP_ERROR_PREFIX followed by the error.
METADATA_MISSING_VALUE = "Error/NA"
METADATA_LOOP_ERROR_PREFIX = "ERROR_IN_LOOP:"
_FRAME_INDEX_COLUMN = "frame_count"
_OUTPUT_FILENAME_COLUMN = "output_filename"

# Header fields that determine what each frame shows. A FG run and its external BG run
# must agree on all of them: camera settings set the image formation, and the ETL scan
# fields plus the start diopter set the focal plane of every frame index; the trigger mode
# decides whether frame k shows the plane of entry k (software_grid) or of entry k - 1
# (free_run at 25 fps). Run identity (timestamp, plate/well id, output folder), measured
# frame rates, lens temperatures and the config path legitimately differ between the two
# runs and are not compared.
_IMAGE_DEFINING_HEADER_FIELDS = (
    ("Acquisition Run Metadata", "Start Diopter (dpt)"),
    ("Configuration", "Processed as Color"),
    ("Configuration", "Actual Exposure (us)"),
    ("Configuration", "Actual ROI (WxH)"),
    ("Configuration", "Pixel Format"),
    ("Configuration", "Image Format"),
    ("Configuration", "Focusing Pattern"),
    ("Configuration", "Lens Settle Time (s)"),
    ("Configuration", "Scan Height (mm)"),
    ("Configuration", "Scan Step (um)"),
    ("Configuration", "Scan Ratio (mm/dpt)"),
    ("Configuration", "Trigger Mode"),
)
_SCAN_STEP_FIELD = ("Configuration", "Scan Step (um)")
# Header keys of the values the offline pipeline takes from the recording itself (shared
# with the acquisition writer). The frame rate has always been logged; pixel size and
# magnification are logged since the optics moved into the hardware profiles.
_CONFIGURATION_SECTION = "Configuration"
HEADER_FRAME_RATE_KEY = "Final Acquisition FPS"
HEADER_PIXEL_SIZE_KEY = "Pixel Size (um)"
HEADER_MAGNIFICATION_KEY = "Magnification"
OPTICS_SOURCE_RECORDING = "recording metadata"
OPTICS_SOURCE_ACTIVE_PROFILE = "active hardware profile (the recording predates logged optics)"
# Value of a header field that older acquisitions did not write: every run recorded before
# the fixed-grid trigger loop used the free-run loop.
_HEADER_FIELD_DEFAULTS = {
    ("Configuration", "Trigger Mode"): "free_run",
}

# Experiment-design table: one row per sample, keyed like the sample folder name
# (YYYYMMDD_plateX_wellY). scan_step_um is cross-checked against the acquisition
# metadata; every other non-key column is an experimental factor carried into outputs.
DESIGN_KEY_COLUMNS = ("date", "plate", "well")
DESIGN_SCAN_STEP_COLUMN = "scan_step_um"


def read_acquisition_metadata_header(metadata_log_path: str) -> Dict[str, Dict[str, str]]:
    """Parse the comment header of an acquisition ``metadata_log.csv``.

    Returns ``{section: {key: value}}`` for the sections announced by banner lines
    (``# === Acquisition Run Metadata ===``, ``# --- Configuration ---``), with values
    kept as written. Parsing stops at the ``Frame Data`` banner.

    Raises:
        FileNotFoundError: the log file does not exist.
        DataValidationError: no header section, or a line outside the ``# Key: value`` form.
    """
    if not os.path.isfile(metadata_log_path):
        raise FileNotFoundError(f"Acquisition metadata log not found: {metadata_log_path}")

    sections: Dict[str, Dict[str, str]] = {}
    current_section = None
    with open(metadata_log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            if not line.startswith("#"):
                break
            content = line[1:].strip()
            banner = _SECTION_BANNER.fullmatch(content)
            if banner:
                current_section = banner.group(1)
                if current_section == _FRAME_DATA_SECTION:
                    break
                sections[current_section] = {}
                continue
            key, separator, value = content.partition(":")
            if current_section is None or not separator:
                raise DataValidationError(
                    f"Malformed metadata header line in {metadata_log_path!r}: {line.rstrip()!r}"
                )
            sections[current_section][key.strip()] = value.strip()

    if not sections:
        raise DataValidationError(f"No metadata header found in {metadata_log_path!r}")
    return sections


def read_frames_lost_at_acquisition(metadata_log_path: str) -> Set[int]:
    """Return the frame indices a run logged but saved no image for.

    The acquisition writes a frame-table row for every focus-pattern entry, also when the
    exposure never reached the PC or the frame loop raised; such a row's
    ``output_filename`` is ``METADATA_MISSING_VALUE`` or a ``METADATA_LOOP_ERROR_PREFIX``
    message instead of a file path.

    Raises:
        FileNotFoundError: the log file does not exist.
        DataValidationError: the frame table lacks the frame-index or output-filename column.
    """
    if not os.path.isfile(metadata_log_path):
        raise FileNotFoundError(f"Acquisition metadata log not found: {metadata_log_path}")
    with open(metadata_log_path, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()
    header_line_count = next((i for i, line in enumerate(lines) if not line.startswith("#")), len(lines))
    frame_table = pd.read_csv(metadata_log_path, skiprows=header_line_count, dtype=str, keep_default_na=False)
    missing_columns = [
        column for column in (_FRAME_INDEX_COLUMN, _OUTPUT_FILENAME_COLUMN) if column not in frame_table.columns
    ]
    if missing_columns:
        raise DataValidationError(f"Frame table of {metadata_log_path!r} lacks columns: {missing_columns}")
    output_filenames = frame_table[_OUTPUT_FILENAME_COLUMN]
    lost_mask = (output_filenames == METADATA_MISSING_VALUE) | output_filenames.str.startswith(
        METADATA_LOOP_ERROR_PREFIX
    )
    return {int(frame) for frame in frame_table.loc[lost_mask, _FRAME_INDEX_COLUMN]}


@dataclass(frozen=True)
class RecordingParameters:
    """How a recording was made, as far as the offline pipeline depends on it."""

    frame_rate_fps: float        # frames per second of the run
    pixel_to_mm_factor: float    # mm per image pixel in the object plane
    optics_source: str           # OPTICS_SOURCE_RECORDING or OPTICS_SOURCE_ACTIVE_PROFILE


def read_recording_parameters(config: Dict[str, Any], image_dir_path: str) -> RecordingParameters:
    """Read frame rate and pixel scale of the recording in ``image_dir_path`` from its metadata log.

    The pipeline analyses recordings made earlier, possibly with another lens or frame rate
    than the config describes now, so both values come from the run's own
    ``metadata_log.csv``. A recording made before the acquisition logged pixel size and
    magnification takes ``acquisition.optics`` of the active hardware profile
    (``optics_source`` says which applied).

    Raises:
        DataValidationError: no metadata log, no or a non-positive frame rate, or only one
            of the two optics fields.
        ConfigError: the recording has no optics fields and the active profile has none either.
    """
    log_filename = config["acquisition"]["paths"]["log_filename"]
    log_path = os.path.join(image_dir_path, log_filename)
    if not os.path.isfile(log_path):
        raise DataValidationError(
            f"Recording {image_dir_path!r} has no {log_filename}: its frame rate and pixel scale are unknown."
        )
    configuration = read_acquisition_metadata_header(log_path).get(_CONFIGURATION_SECTION, {})

    if HEADER_FRAME_RATE_KEY not in configuration:
        raise DataValidationError(f"Metadata header of {log_path!r} has no '{HEADER_FRAME_RATE_KEY}'.")
    try:
        frame_rate_fps = float(configuration[HEADER_FRAME_RATE_KEY])
    except ValueError as error:
        raise DataValidationError(
            f"'{HEADER_FRAME_RATE_KEY}' in {log_path!r} is not a number: {configuration[HEADER_FRAME_RATE_KEY]!r}"
        ) from error
    if not (np.isfinite(frame_rate_fps) and frame_rate_fps > 0):
        raise DataValidationError(f"'{HEADER_FRAME_RATE_KEY}' in {log_path!r} must be positive, got {frame_rate_fps!r}.")

    optics_keys = (HEADER_PIXEL_SIZE_KEY, HEADER_MAGNIFICATION_KEY)
    logged_optics_keys = [key for key in optics_keys if key in configuration]
    if len(logged_optics_keys) == len(optics_keys):
        try:
            optics_config = {
                "pixel_size_um": float(configuration[HEADER_PIXEL_SIZE_KEY]),
                "magnification": float(configuration[HEADER_MAGNIFICATION_KEY]),
            }
        except ValueError as error:
            raise DataValidationError(f"Optics fields in {log_path!r} are not numbers: {error}") from error
        optics_source = OPTICS_SOURCE_RECORDING
    elif logged_optics_keys:
        raise DataValidationError(
            f"Metadata header of {log_path!r} has '{logged_optics_keys[0]}' but not both of {optics_keys}."
        )
    else:
        require_config_sections(config, {"acquisition.optics": ("pixel_size_um", "magnification")}, "config")
        optics_config = config["acquisition"]["optics"]
        optics_source = OPTICS_SOURCE_ACTIVE_PROFILE

    return RecordingParameters(
        frame_rate_fps=frame_rate_fps,
        pixel_to_mm_factor=get_pixel_to_mm_factor(optics_config),
        optics_source=optics_source,
    )


def get_scan_extent_mm(image_dir_path: str, pixel_to_mm_factor: float) -> Tuple[float, float, float]:
    """Return the (x, y, z) extent in mm that the recording in ``image_dir_path`` covers.

    x/y are its frame size (from the header of its first frame) at ``pixel_to_mm_factor``,
    the recording's pixel scale (:func:`read_recording_parameters`) that also converts
    detections to x_mm/y_mm; z is the deepest focal plane
    (fd_mm) in its frame filenames, the token that becomes z_mm. All three count from 0
    (image corner, sweep start plane), so every detected coordinate falls inside this box
    by construction. That makes it the fixed axis range for the tracking plots:
    autoscaling instead zooms each figure onto the particles it happens to contain, so a
    well whose organisms sit in one corner is silently magnified and can no longer be
    compared by eye against a well whose organisms roam.

    Taken from the recording rather than the acquisition config, which describes the rig
    as set up now: a recording made with another camera ROI or scan height would otherwise
    be drawn in a box that does not fit it. The deepest plane is used rather than the
    logged scan height because a scan step that does not divide the scan height ends the
    sweep above or below it.

    Raises:
        DataValidationError: the directory holds no frame named like an acquisition frame.
    """
    frames_df = parse_frame_filenames(image_dir_path)
    if frames_df is None or frames_df.empty:
        raise DataValidationError(f"No acquisition frames in {image_dir_path!r}: the recording's extent is unknown.")
    with Image.open(frames_df["full_path"].iloc[0]) as first_frame:  # reads the header only
        width_px, height_px = first_frame.size
    return (
        width_px * pixel_to_mm_factor,
        height_px * pixel_to_mm_factor,
        float(frames_df["fd_mm"].max()),
    )


def _get_header_field(header: Dict[str, Dict[str, str]], field: tuple, log_path: str) -> str:
    """Return one header value; an absent field takes its ``_HEADER_FIELD_DEFAULTS`` value or raises."""
    section, key = field
    if section not in header or key not in header[section]:
        if field in _HEADER_FIELD_DEFAULTS:
            return _HEADER_FIELD_DEFAULTS[field]
        raise DataValidationError(f"Metadata header of {log_path!r} has no '{key}' in section '{section}'.")
    return header[section][key]


def validate_background_compatibility(fg_metadata_log_path: str, bg_metadata_log_path: str) -> None:
    """Check that an external BG run used the same image-defining settings as its FG run.

    Raises:
        DataValidationError: listing every image-defining header field that differs.
    """
    fg_header = read_acquisition_metadata_header(fg_metadata_log_path)
    bg_header = read_acquisition_metadata_header(bg_metadata_log_path)
    mismatches: List[str] = []
    for field in _IMAGE_DEFINING_HEADER_FIELDS:
        fg_value = _get_header_field(fg_header, field, fg_metadata_log_path)
        bg_value = _get_header_field(bg_header, field, bg_metadata_log_path)
        if fg_value != bg_value:
            mismatches.append(f"'{field[1]}' FG={fg_value!r} BG={bg_value!r}")
    if mismatches:
        raise DataValidationError(
            f"Background run is not compatible with its foreground run "
            f"({bg_metadata_log_path!r} vs {fg_metadata_log_path!r}): {'; '.join(mismatches)}"
        )


def load_experiment_design(design_csv_path: str) -> pd.DataFrame:
    """Load an experiment-design CSV keyed by date, plate and well.

    Raises:
        FileNotFoundError: the file does not exist.
        DataValidationError: required columns are missing, keys are empty or duplicated.
    """
    if not os.path.isfile(design_csv_path):
        raise FileNotFoundError(f"Experiment design file not found: {design_csv_path}")
    design_df = pd.read_csv(design_csv_path)

    required_columns = (*DESIGN_KEY_COLUMNS, DESIGN_SCAN_STEP_COLUMN)
    missing_columns = [column for column in required_columns if column not in design_df.columns]
    if missing_columns:
        raise DataValidationError(f"Experiment design {design_csv_path!r} lacks columns: {missing_columns}")
    if design_df[list(required_columns)].isna().any().any():
        raise DataValidationError(f"Experiment design {design_csv_path!r} has empty key or scan-step cells.")

    design_df["date"] = design_df["date"].astype(str)
    design_df["plate"] = design_df["plate"].astype(int)
    design_df["well"] = design_df["well"].astype(int)
    duplicated = design_df.duplicated(subset=list(DESIGN_KEY_COLUMNS), keep=False)
    if duplicated.any():
        duplicate_keys = design_df.loc[duplicated, list(DESIGN_KEY_COLUMNS)].drop_duplicates()
        raise DataValidationError(
            f"Experiment design {design_csv_path!r} has duplicate rows for: "
            f"{duplicate_keys.to_dict('records')}"
        )
    return design_df


def resolve_sample_design_values(
    design_df: pd.DataFrame,
    sample_id: str,
    metadata_log_paths: List[str],
) -> Dict[str, Any]:
    """Return a sample's experimental-factor values after checking it against its metadata.

    The design row is looked up from the date/plate/well in ``sample_id``; its
    ``scan_step_um`` must equal the scan step logged in every given metadata log (FG,
    and BG in external-background mode). This catches samples whose folders were
    mis-renamed by hand.

    Returns:
        ``{column: value}`` for every design column except the key columns.

    Raises:
        DataValidationError: no design row, or a scan step that disagrees with the metadata.
    """
    date, plate, well = parse_date_plate_well(sample_id)
    if date is None:
        raise DataValidationError(f"Sample folder name {sample_id!r} is not YYYYMMDD_plateX_wellY.")
    rows = design_df[
        (design_df["date"] == date) & (design_df["plate"] == int(plate)) & (design_df["well"] == int(well))
    ]
    if rows.empty:
        raise DataValidationError(
            f"No experiment-design row for sample {sample_id} (date={date}, plate={plate}, well={well})."
        )
    design_row = rows.iloc[0]

    design_scan_step = float(design_row[DESIGN_SCAN_STEP_COLUMN])
    for log_path in metadata_log_paths:
        header = read_acquisition_metadata_header(log_path)
        logged_scan_step = float(_get_header_field(header, _SCAN_STEP_FIELD, log_path))
        if not np.isclose(logged_scan_step, design_scan_step):
            raise DataValidationError(
                f"Sample {sample_id}: experiment design says {DESIGN_SCAN_STEP_COLUMN}={design_scan_step:g} "
                f"but {log_path!r} logged a scan step of {logged_scan_step:g} um."
            )

    return {
        column: design_row[column].item() if hasattr(design_row[column], "item") else design_row[column]
        for column in design_df.columns
        if column not in DESIGN_KEY_COLUMNS
    }
