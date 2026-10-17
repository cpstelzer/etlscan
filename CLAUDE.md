# Well Plate Analyzer — Developer Guide

## What this project does

Automated pipeline for processing well-plate microscopy image sequences.
Raw PNG frame sequences are ingested, organisms are tracked, morphology is
measured, and a curated review package is produced.  A real-time mirror of
the first three stages runs frame-by-frame during live acquisition.

---

## Working approach: diagnose before implementing

When investigating a bug or unexpected behavior, explore breadth-first and confirm the
root cause with evidence **before** writing a fix — don't latch onto the first vivid clue
and start coding:

- Establish the expected ground-truth outcome first (e.g. "1–4 organisms across the
  sequence") and sanity-check results against that magnitude — a wrong order of magnitude
  is a signal, not noise.
- Weigh 2–3 candidate causes by evidence and look for what would *disconfirm* the leading
  one; trace the whole pipeline, not just the first suspicious stage.
- Distinguish symptom from cause, and present the diagnosis before implementing unless the
  fix is trivial.

---

## Repository layout

```
well_plate_analyzer/
├── config/
│   └── config.yml              # Single YAML config for the entire pipeline
├── data/
│   ├── input/<batch>/<sample>/ # Raw PNG sequences (one dir per sample)
│   ├── output/<batch>/         # Stage outputs, organised per sample
│   └── training_data/          # Labelled crops for classifier retraining
├── docs/
│   ├── conventions.md          # Naming and style rules — read this first
│   ├── hpc_slurm_runbook.md
│   ├── random_forest_training.md
│   ├── efficientnet_training.md
│   ├── focussed_feature_columns.md
│   └── realtime_mode_spec.md
├── models/                     # Trained RF + EfficientNet artefacts
├── scripts/
│   ├── run_pipeline.py         # Main entry point (batch mode)
│   ├── run_acquire.py          # Multi-well acquisition entry point
│   ├── run_acquire_single.py   # Single-well acquisition entry point
│   ├── run_acquire_experiment.py # Experiment-sweep acquisition entry point
│   ├── run_acquire_single_sweep.py # Experiment sweep on one well (no XY stage)
│   ├── train_model.py          # Random-Forest retraining
│   └── train_efficientnet_model.py
├── src/
│   ├── acquire/                # Hardware acquisition (Pylon + Zaber)
│   │   ├── workflows.py        # Multi-well, experiment-sweep, single workflows
│   │   ├── focus_pattern.py    # ETL focus cycle (shared with realtime) + plane-visit frame count
│   │   ├── config.py           # Plate-type normalisation and settings merge
│   │   ├── plate_layout.py     # 24/96-well coordinate generation
│   │   ├── stage_controller.py # Zaber XY-stage abstraction
│   │   └── experiment_logging.py
│   ├── core/
│   │   ├── feature_calculators/
│   │   │   ├── morphology_calculator.py
│   │   │   ├── quality_calculator.py
│   │   │   ├── shape_descriptors.py
│   │   │   └── bio_features_calculator.py  # ML-oriented bio features
│   │   ├── image_enhancement.py            # float01_to_uint8 and friends
│   │   ├── image_io_operations.py          # load/save + flat-field correction
│   │   ├── masking_operations.py
│   │   ├── particle_operations.py
│   │   ├── region_props_calculator.py
│   │   ├── segmentation.py
│   │   ├── sharpness_assessment.py
│   │   └── well_analysis.py
│   ├── realtime/
│   │   ├── stage1_rt.py        # Real-time Stage-1 wrapper
│   │   ├── stage2_rt.py        # Real-time Stage-2 wrapper
│   │   ├── orchestrator.py     # Frame-loop and session management
│   │   ├── persistence.py      # Accepted-crop I/O
│   │   └── session.py          # RealtimeSession dataclass
│   ├── stages/
│   │   ├── stage_01_initial_detection.py
│   │   ├── stage_02_characterization.py
│   │   ├── stage_03_filtering_and_classification.py
│   │   └── stage_04_biological_analysis.py
│   ├── utils/
│   │   ├── acquisition_metadata.py # metadata_log header reader; FG/BG run + experiment-design checks
│   │   ├── filename_helpers.py # build_crop_filename()
│   │   ├── common_exceptions.py
│   │   ├── config_loader.py
│   │   ├── logging_setup.py
│   │   ├── path_helpers.py
│   │   └── data_io.py
│   └── visualization/
│       └── feature_visualization.py
└── tests/                      # pytest suite (mirror of src/ layout)
```

---

## Pipeline stages

| # | Module | Input → Output |
|---|--------|----------------|
| 1 | `stage_01_initial_detection` | Raw PNGs → `stage1_tracked_particles.csv` |
| 2 | `stage_02_characterization` | Stage-1 CSV + PNGs → feature CSV + crops |
| 3 | `stage_03_filtering_and_classification` | Stage-2 CSV → `filtered_good.csv` with `ml_label` |
| 4 | `stage_04_biological_analysis` | Stage-3 CSV + crops → `stage4_final_analysis.csv` + plots |

Run all four stages with:
```bash
python scripts/run_pipeline.py \
    --input_batch_dir data/input/<batch> \
    --stages all \
    --config config/config.yml
```

**Stage-1 background source** (`--background_source`, the only difference between the
two modes; every step after FG/BG pairing is shared):
- `time_shift` (default): BG = a later frame of the same recording (offset via
  `BG_frame_offset_mode`), so only *moving* specimens survive the subtraction.
- `external`: BG = the same frame of a separate specimen-free recording of the well, so
  *non-moving* (e.g. fixated) specimens are found too. `--input_batch_dir` is the
  experiment root holding `fg/` and `bg/` with identically named sample folders. Pairs
  match by frame index + focal plane (strict; filename plate/well tokens are not
  trusted). Each BG frame first gets the FG's low-frequency illumination and the well rim
  is excluded; a sample fails if too many frames trip the subtracted-foreground guard.
  Static specimens visible only near their focal plane reappear once per sweep pass, so
  tracking bridges one focus-sweep period (derived per sample from its fd == 0 frames;
  requires `track_in_2d`) and links within `static_search_range_mm`, instead of the movers' `memory_frames`/`search_range_mm`. After tracking, objects whose
  detections are often fibre-shaped (lint) are dropped (conservative, per object).
  Audits per sample: `<sample>_stage1_bg_pairing.csv`, `<sample>_stage1_rejected_fibre_objects.csv`.

`--experiment_design <csv>` (either mode, optional): one row per sample (`date`, `plate`,
`well`, `scan_step_um`, factor columns). Validated against each sample's `metadata_log.csv`
before processing; the non-key columns are prepended to the per-sample rolled-up CSVs and
thus to the batch rollups.

---

## Config organization (`config/config.yml`)

The single config is loaded by `load_config_once` (`src/utils/config_loader.py`),
which applies these transforms at load time:

| Section | Notes |
|---------|-------|
| `paths.pipeline_base` | Offline-pipeline root (output, models, training data). Legacy `project_base_dir` still accepted as a fallback in `get_project_base_dir`. |
| `paths.raw_data_root` | Acquisition output root; **injected** into `acquisition.paths.output_folder_base` (single source of truth). Point `run_pipeline --input_batch_dir` here. |
| `imaging` + `segmentation` | Split for clarity; the loader **assembles** them into the internal `config['system']` view that pipeline code consumes. Don't read `imaging`/`segmentation` directly in stage code — read `system`. |
| `acquisition` | Rig-agnostic runtime. Hardware-specific values come from `hardware_profiles`. |
| `hardware_profiles.<name>` | Deep-merged into `acquisition` per `active_hardware_profile`. Holds camera exposure, ROI, ETL focus, and per-rig plate origin overrides (ETL-lens rig). |
| `stage_hardware_profile` | Dedicated Zaber XY-stage profile, independent of the ETL-lens profiles. **Injected** into `acquisition.stage_controller` (single source — not duplicated per lens profile). |
| `acquisition.plate.type` | Run-level plate selection (`24`/`96`), **decoupled** from the hardware profile. `acquisition.plate.definitions` holds geometry; profiles override only `origin_x`/`origin_y`. Read via `get_plate_settings(acq_conf)`. |
| Particle-area thresholds | The five `min_*_area` knobs are intentionally distinct (see the note atop `config.yml`) — do not consolidate. |
| `acquisition.experiment_sweep` | Scan-step × pattern grid for both sweep scripts, plus `variable_recording_duration` / `plane_revisit_times` (run length = max(`camera.video_duration_sec`, time for every plane to get k visits)). |
| Egg detection | `stage_04_biological_analysis.egg_detection` is only consumed by stage 4; it is **not** merged into `get_unified_segmentation_params`. |
| `stage_01_initial_detection.external_background` | Read only with `run_pipeline --background_source external`: `fg/`/`bg/` subfolder names, low-frequency illumination match (downsample, median kernel), `rim_exclusion_px`, `max_failed_frame_fraction`, lint-fibre object rule (`fibre_min_aspect_ratio`, `fibre_max_solidity`, `fibre_min_detection_share`). Tracking in this mode uses a one-sweep-period memory (derived, not configured) and the static-object linking radius `static_search_range_mm`; the subtracted-foreground gate and all other detection/tracking params are shared with time-shift mode. |

---

## Development setup

```bash
pip install -r requirements.txt
```

Optional heavy dependencies (needed for full test coverage):
```bash
pip install opencv-python scikit-image pandas numpy pillow
```

---

## Running tests

```bash
# Full suite (excludes the tkinter-dependent acquire contract test)
pytest tests/ --ignore=tests/test_acquire_refactor_contracts.py

# Single file
pytest tests/test_filename_helpers.py -v
```

Expected baseline: **109 tests pass**, 6 pre-existing failures in
`test_realtime_config_phase34.py`, `test_stage03_ml_columns.py`, and
`test_stage3_feature_columns.py` (config/signature mismatches unrelated to
active refactoring work).

---

## Key conventions (see `docs/conventions.md` for full detail)

| Rule | Example |
|------|---------|
| Image arrays: `image_{type}` | `image_gray`, `image_float` |
| Config dicts: `{scope}_config` | `mask_config`, `crop_config` |
| Boolean masks: `{name}_mask` (no `_bool`) | `initial_mask`, `filled_mask` |
| Private helpers: `_` prefix | `_require_cv2`, `_load_image_with_pil` |
| Magic numbers: always in `config.yml` | `config['stage_01_initial_detection']['black_plate_well_analysis']['morph_kernel_size']` |
| Strings: f-strings only | `f"loaded {path!r}"` not `"loaded %s" % path` |
| Tests: `tests/` only — no `__main__` blocks in `src/` | |
| Type annotations: all public functions must be fully annotated | |

---

## Exception hierarchy

```
Exception
└── ConfigError                   # bad YAML config

RuntimeError
└── ProcessingError               # pipeline stage failure
    ├── ImageProcessingError      # image-specific
    └── MLModelError              # model load/train/predict

ValueError
└── DataValidationError           # missing columns, wrong dtypes
```

Catch `ProcessingError` to handle any pipeline processing failure (includes
both image and ML errors).

---

## Shared utilities worth knowing

| Function | Location | Purpose |
|----------|----------|---------|
| `build_crop_filename` | `src/utils/filename_helpers.py` | Single canonical crop filename builder used by both batch and realtime pipelines |
| `float01_to_uint8` | `src/core/image_enhancement.py` | Clip-and-scale float [0,1] → uint8 [0,255] |
| `calculate_posterior_attributes` | `src/core/feature_calculators/bio_features_calculator.py` | Annotate RegionProperties with posterior axis vector (used by Stage 4 + visualisation) |
| `load_config_once` | `src/utils/config_loader.py` | Load and cache the YAML config |
| `get_scan_extent_mm` | `src/utils/config_loader.py` | Full (x, y, z) mm extent of the imaged volume — the fixed axis range for the Stage-1/Stage-2 tracking plots |
| `create_well_mask_with_fallback` | `src/core/well_analysis.py` | Well mask with the full-frame bright-mask fallback (brightness matching + external-BG rim exclusion) |
| `read_acquisition_metadata_header` | `src/utils/acquisition_metadata.py` | Parse the `# Key: value` header of a run's `metadata_log.csv` |

---

## Adding a new feature calculator

1. Add a `.py` file under `src/core/feature_calculators/`.
2. Define a public function (fully type-annotated, no `_` prefix).
3. Update the catalogue comment in `src/core/feature_calculators/__init__.py`.
4. Add tests under `tests/test_<module_name>.py`.

---

## Branching convention

Active refactoring work lives on `claude/code-simplifier-3aXsn`.
