# etlscan — Developer Guide

## What this project does

Acquisition and analysis software for depth-resolved imaging with electronically
tunable lenses (ETLs).  An ETL objective steps its focal plane through the sample,
one plane per camera frame; the acquisition records these focal scans as PNG frame
sequences.  The offline pipeline detects and tracks organisms in XY, recovers each
track's depth from its focus peaks (depth-from-focus) to reconstruct 3D trajectories,
measures morphology on the in-focus frames, and produces a curated review package.
A real-time mirror of the first three stages runs frame-by-frame during live
acquisition and stores only sharp crops.  See `README.md` for the user-facing
overview and the accompanying paper.

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
etlscan/
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
│   │   ├── hardware_setup.py   # Camera + ETL controller: connect, configure, close (batch, realtime, flat field, diagnosis)
│   │   ├── focus_pattern.py    # ETL focus cycle (shared with realtime) + plane-visit frame count
│   │   ├── triggered_acquisition.py # Fixed-grid software-trigger loop (ETL/camera sync) + camera-clock helpers
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
│   │   ├── focus_peak_detection.py         # Stage-2 focus peaks against a track's own baseline
│   │   ├── depth_reconstruction.py         # Stage-2 depth knots, speed gate, z_reconstructed spline
│   │   ├── track_columns.py                # Column names of the track tables
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
│   │   ├── acquisition_metadata.py # metadata_log header reader; FG/BG run + experiment-design checks; recording extent
│   │   ├── filename_helpers.py # build_crop_filename()
│   │   ├── common_exceptions.py
│   │   ├── config_loader.py
│   │   ├── logging_setup.py
│   │   ├── path_helpers.py
│   │   └── data_io.py
│   └── visualization/
│       ├── feature_visualization.py  # Stage-4 body/egg overlay
│       ├── focus_diagnostics.py      # Stage-2 sharpness curves with crop collages, diagnostics summary
│       ├── track_maps.py             # Stage-1 and Stage-2 track maps (3D, XY, side views)
│       └── figure_style.py           # Shared plot style and figure sizes from config['figure_style']
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
  trusted). A missing BG frame is accepted only if the BG `metadata_log.csv` records it as
  lost at acquisition (`output_filename` = `Error/NA` / `ERROR_IN_LOOP:`); that FG frame is
  skipped and audited (`bg_lost_at_acquisition`), any other missing BG frame fails the sample.
  Each BG frame first gets the FG's low-frequency illumination; a sample fails if more
  than `max_failed_frame_fraction` of its frames trip the subtracted-foreground guard or
  lost their BG frame.

**Well-rim exclusion** (both modes and the realtime Stage 1, in the shared
`prepare_masked_foreground`): particles are searched only inside the FG and the BG well
mask, `black_plate_well_analysis.rim_exclusion_px` away from either rim
(`create_rim_excluded_well_mask`). Focus breathing moves the rim with the focal plane, so a
time-shift pair shows it at two radii, and its sharp edge near the sweep start otherwise
survives the subtraction, is linked into animal tracks and yields false Stage-2 focus peaks.
  Static specimens visible only near their focal plane reappear once per sweep pass, so
  tracking bridges one focus-sweep period (derived per sample from its fd == 0 frames;
  requires `track_in_2d`) and links within `static_search_range_mm`, instead of the movers'
  `memory_frames`/`search_range_mm`. Lint-shaped detections (`s1_solidity <= lint_max_solidity`)
  are dropped **before** tracking, so an isolated lint object loses its sharp frames and its
  blurred remainder yields no Stage-2 sharpness peak.
  Audits per sample: `<sample>_stage1_bg_pairing.csv`, `<sample>_stage1_rejected_lint_detections.csv`.

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
| `base_config` (variants) | A config with a top-level `base_config: "<file>"` (relative to itself) is a **variant**: it holds only the values it changes and is deep-merged over that base before the transforms below. Use variants instead of copying `config.yml`, so they cannot drift. |
| Stage-2 focus peaks | `stage_02_characterization.focussed_particles.peak_detection`: a peak is measured against the track's own out-of-focus baseline (running median over `baseline_window_mm` of focal travel; frames derived from the sample's scan step), never an absolute sharpness value. Detection runs on `variance_laplacian_per_intensity` (Laplacian variance ÷ crop mean intensity), because out of focus the plain `variance_laplacian` is shot noise that grows with brightness. Peaks at a track's first/last frame are kept and flagged `focus_peak_at_track_edge`. With `one_peak_per_sweep_pass` only the strongest peak of each monotonic sweep pass (derived from z) is kept. `sharpness_min_threshold` in the same block is the **realtime** per-frame gate only. The depth spline (`z_reconstructed`) skips knots that `depth_knot_speed_gate` flags (`depth_knot_speed_outlier`): a knot that disagrees with every neighbouring knot by more than `tolerance_planes` planes + `max_vertical_speed_mm_s` × Δt (Δt from the recording's frame rate, as in Stage 1). |
| No fallbacks in code | `config.yml` is the only source of a parameter value. Code indexes config sections directly, never `.get(key, default)`; a function checks its section once with `require_config_keys`, a stage all of its sections at its start with `require_config_sections` (`validate_stage1_config`, also called by `load_realtime_config`). A missing key raises `ConfigError` naming its path; a key nothing reads is removed. Enforced by `tests/test_config_key_coverage.py`. |
| Recording parameters | The offline pipeline analyses recordings made earlier, so what describes a recording comes from the recording: Stages 1-2 read its frame rate (`Final Acquisition FPS`) and pixel scale (`Pixel Size (um)`, `Magnification`) from the header of its `metadata_log.csv` (`read_recording_parameters`), never from `acquisition.camera.fps`. A recording made before the optics were logged takes `acquisition.optics` of the active hardware profile (logged per stage); a sample without a metadata log fails before the batch starts. `stage_01_initial_detection.bg_offset_min_time_s` is converted to frames with that frame rate. |
| `paths.pipeline_base` | Offline-pipeline root (output, models, training data); required (`get_project_base_dir`). |
| Stage-3 training keys | Everything only the training scripts read (`test_split_size`, `random_state`, `exclude_feature_prefixes`, `rf_param_grid`, `efficient_net_training`) lives under `stage_03_filtering_and_classification.training_params`; inference settings are direct children of the stage. |
| `paths.raw_data_root` | Acquisition output root; **injected** into `acquisition.paths.output_folder_base` (single source of truth). Point `run_pipeline --input_batch_dir` here. |
| `imaging` + `segmentation` | Split for clarity; the loader **assembles** them into the internal `config['system']` view that pipeline code consumes. Don't read `imaging`/`segmentation` directly in stage code — read `system`. |
| `acquisition` | Rig-agnostic runtime. Hardware-specific values come from `hardware_profiles`. |
| `hardware_profiles.<name>` | Deep-merged into `acquisition` per `active_hardware_profile`. Holds camera exposure, ROI, ETL focus, `optics` (`pixel_size_um`, `magnification`) and per-rig plate origin overrides (ETL-lens rig). |
| Acquisition setup and timings | One camera / lens setup for every acquisition path: `src/acquire/hardware_setup.py` (`open_first_camera`, `configure_camera`: first format of `camera.pixel_format_preference` the camera offers, exposure clamped to the camera's range and increment, buffer count, centred ROI; `set_frame_rate`, `connect_lens`, `prefocus_lens`, `close_camera`, `disconnect_lens`). The vendor modules are passed in, so the realtime mode and the tests' simulated devices use the same code. Waits are config values, not literals: `focusing.prefocus_settle_sec`, `workflow.pause_between_wells_sec` / `pause_between_sweep_runs_sec`, `camera.single_frame_timeout_ms` (flat field), the `*_wait_sec` keys of `stage_hardware_profile`, `realtime_mode.runtime.max_consecutive_missing_frames`. An unknown `focusing.pattern` and a frame rate the camera cannot deliver raise instead of being replaced. |
| `stage_hardware_profile` | Dedicated Zaber XY-stage profile (port, devices, microstep, baud rate, serial timeout, waits after each command; `StageController.from_config`), independent of the ETL-lens profiles. **Injected** into `acquisition.stage_controller` (single source — not duplicated per lens profile). |
| `acquisition.plate.type` | Run-level plate selection (`24`/`96`), **decoupled** from the hardware profile. `acquisition.plate.definitions` holds geometry; profiles override only `origin_x`/`origin_y`. Read via `get_plate_settings(acq_conf)`. |
| Particle-area thresholds | The four minimum-area knobs are intentionally distinct (see the note atop `config.yml`) — do not consolidate. |
| `acquisition.experiment_sweep` | Scan-step × pattern grid for both sweep scripts, plus `variable_recording_duration` / `plane_revisit_times` (run length = max(`camera.video_duration_sec`, time for every plane to get k visits)). |
| `acquisition.camera.trigger_mode` | How exposures are timed against the ETL commands (`acquire_and_save_frames`, every `run_acquire*.py`). `software_grid` (default): one software trigger per focus-pattern entry, trigger i due at t0 + i / `fps` on a fixed grid; frame i is exposed only after its own lens command + `focusing.min_settle_ms`, the lens moves only `exposure + post_exposure_margin_ms` after a trigger, `WaitForFrameTriggerReady` runs `trigger_ready_lead_ms` before the due time; a large jump (`large_jump_threshold_dpt`) delays only its own trigger to command end + `large_jump_settle_sec`. The realtime camera source (`src/realtime/acquisition.py`) follows the same setting and runs the same loop on a background thread at `realtime_mode.runtime.target_fps` (adaptive/repeat requests apply from the next lens command; `drop_frame_if_backlogged: false` pauses the loop and restarts the grid instead of dropping). `free_run`: legacy loop (sleep `focusing.lens_settle_time_sec`, take the next free-run frame), kept for comparison — at 25 fps it labels each frame with the previous frame's plane (one-frame lag). `metadata_log.csv` gets `# Trigger Mode` (image-defining: an external-BG run must use the same loop as its FG run; a missing field means `free_run`, i.e. all older data), `# Min Settle (ms)` (grid), `# Lens Temperature Start/End (C)` (if the controller reports it) and, after the legacy columns, `grid_time_s`, `lens_cmd_start_s`, `lens_cmd_end_s`, `trigger_time_s`, `trigger_late_ms`, `block_id`, `camera_timestamp_ns`, `exposure_minus_trigger_ms`, `exposure_minus_lens_cmd_ms`, `lens_temp_c` (times relative to the acquisition start; exposure times from camera timestamps via clock latches). One recording check per run (`camera.clock_sync_samples`, `trigger_late_report_ms`, `timestamp_mismatch_ms`). |
| Egg detection | `stage_04_biological_analysis.egg_detection` is only consumed by stage 4; it is **not** merged into `get_unified_segmentation_params`. |
| `figure_style` | Publication style of **every** figure the pipeline writes (Stage-1/2 track maps, Stage-2 focus diagnostics, Stage-4 overlays, `train_model.py` feature-importance chart): dpi, font family, axis-label/tick/title/in-plot-annotation font sizes, line width, marker size, axes line width. Applied as matplotlib rcParams by the `figure_style(config)` context manager; plot code sets no font sizes, line widths, marker sizes, dpi or figure size itself — new plots must be created *and saved* inside that block. Each figure's size is an entry of `figure_style.figure_size_in`, read inside the block with `figure_size(name)` (a new figure needs a new entry). Figures are ~7–8 in wide so the pt sizes print legibly at single-column width. The track maps (Stage-1 `*_tracks_3d/xy`, Stage-2 `reconstructed_trajectories_3d/xy/xz/yz`) use `track_figure_style(config)` instead: the same style with `track_line_width` / `track_marker_size`, white 3D panes and a faint grid. They draw through the shared `plot_track_line` (15-colour palette by rank), `plot_focus_points` (open rings) and `draw_floor_shadow` (grey copy at z = 0) helpers, so they all look alike; the four Stage-2 views take their tracks from one selection, so a track has the same colour in each. The XZ/YZ side views show depth (`z_reconstructed` line, in-focus `z_mm` rings) at true scale. |
| `stage_01_initial_detection.external_background` | Read only with `run_pipeline --background_source external`: `fg/`/`bg/` subfolder names, low-frequency illumination match (downsample, median kernel), `max_failed_frame_fraction`, per-detection lint filter `lint_max_solidity`. Tracking in this mode uses a one-sweep-period memory (derived, not configured) and the static-object linking radius `static_search_range_mm`; the subtracted-foreground gate and all other detection/tracking params are shared with time-shift mode. |

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

Expected baseline (Windows rig PC, `wellscan24_env`, full suite): **352 tests pass**,
5 pre-existing failures: three in `test_config_refactor_baseline.py` (they assert rig values
the config has since changed), `test_project_base_dir_expansion.py::…expands_tilde` and
`test_stage03_ml_columns.py::test_stage3_writes_ml_columns` (both compare POSIX paths).

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
| `apply_quality_flag_filters` | `src/core/feature_calculators/quality_calculator.py` | Drop particles flagged `has_black_mask_artifact` / `is_on_crop_boundary` per `stage_03_filtering_and_classification.quality_flags_filter`; the one implementation for batch and realtime Stage 3 |
| `calculate_posterior_attributes` | `src/core/feature_calculators/bio_features_calculator.py` | Annotate RegionProperties with posterior axis vector (used by Stage 4 + visualisation) |
| `load_config_once` | `src/utils/config_loader.py` | Load and cache the YAML config |
| `require_config_keys` / `require_config_sections` | `src/utils/config_loader.py` | Check that a config section (or several nested ones) holds the keys a function reads; `ConfigError` names each missing key with its path |
| `read_recording_parameters` | `src/utils/acquisition_metadata.py` | Frame rate and pixel scale (mm per pixel) of a recording, from its `metadata_log.csv` header; what Stages 1-2 use instead of the config's current `camera.fps` / optics |
| `get_scan_extent_mm` | `src/utils/acquisition_metadata.py` | (x, y, z) mm extent a recording covers — its frame size × the recording's pixel scale, and its deepest `fd` plane (not the config's ROI / scan height); the fixed axis range for the Stage-1/Stage-2 tracking plots |
| `parse_frame_filenames` | `src/utils/path_helpers.py` | One row per raw frame of a sample dir (frame, dpt, fd_mm, path), parsed from the acquisition filenames |
| `create_well_mask_with_fallback` | `src/core/well_analysis.py` | Well mask with the full-frame bright-mask fallback (brightness matching + Stage-1 rim exclusion) |
| `create_rim_excluded_well_mask` | `src/core/well_analysis.py` | Intersection of FG/BG well masks inset by `rim_exclusion_px`: the Stage-1 particle search area (batch both modes + realtime) |
| `read_acquisition_metadata_header` | `src/utils/acquisition_metadata.py` | Parse the `# Key: value` header of a run's `metadata_log.csv` |
| `read_frames_lost_at_acquisition` | `src/utils/acquisition_metadata.py` | Frame indices a run logged but saved no image for (markers shared with the acquisition writer) |
| `configure_camera` and friends | `src/acquire/hardware_setup.py` | The one camera / ETL-controller setup, used by batch acquisition, realtime, the flat-field tool and the diagnosis script |
| `figure_style` | `src/visualization/figure_style.py` | Context manager applying `config['figure_style']` to all figures created and saved inside it |
| `figure_size` | `src/visualization/figure_style.py` | Configured (width, height) of a named figure, valid inside a style block |
| `track_figure_style` | `src/visualization/figure_style.py` | The same for the track maps, with their line/marker sizes; draw tracks with `plot_track_line`, `plot_focus_points`, `draw_floor_shadow` |

---

## Adding a new feature calculator

1. Add a `.py` file under `src/core/feature_calculators/`.
2. Define a public function (fully type-annotated, no `_` prefix).
3. Update the catalogue comment in `src/core/feature_calculators/__init__.py`.
4. Add tests under `tests/test_<module_name>.py`.

---

## Branching convention

Changes go to `main` through a feature branch and a pull request. Releases are
tagged `v<major>.<minor>.<patch>`; `v2.0.0` is the version archived with the paper.
