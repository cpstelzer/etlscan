# etlscan

**Depth-resolved imaging with electronically tunable lenses**

`etlscan` is the acquisition and analysis software for imaging systems built around an electronically tunable lens (ETL). An ETL changes its optical power within a few milliseconds in response to an electrical signal, so an objective that contains one can step its focal plane through a sample without moving the sample or the objective. Recording one focal plane per camera frame scans a water column many times deeper than the depth of field, with a single camera on a single optical axis. Because the scan is deterministic, the focal plane of every frame is known, and the frame in which a specimen appears sharpest gives its depth (depth-from-focus).

The package was developed for live plankton (rotifers of 200–600 µm swimming freely in 24-well plates and chamber slides) and accompanies the paper

> Stelzer, C.-P., M. Brunner & M. Pichler. *Depth-resolved imaging of live plankton with electronically tunable lenses.* In revision at Limnology and Oceanography: Methods.

The approach is not specific to rotifers. It applies to small organisms and particles that move in a volume deeper than the depth of field of the optics (see [Citation and data](#citation-and-data)).

## What you can do with it

| Task | What the software does | Entry point |
| --- | --- | --- |
| Record focal scans | Steps the ETL through the Z-range in a zigzag or sawtooth pattern, one focal plane per frame, and triggers each exposure on a fixed time grid after the lens has settled. Single wells, whole well plates on an XY stage, and step-size × pattern experiment sweeps. | `scripts/run_acquire*.py` |
| Track many individuals in 3D | Detects moving specimens, links them into tracks by their XY positions, and recovers the depth of each track from its focus peaks. | `scripts/run_pipeline.py` (Stages 1–2) |
| Measure live specimens non-invasively | Keeps one sharp frame per track in a suitable posture and measures body size and shape on it, without fixation. | `scripts/run_pipeline.py` (Stages 3–4) |
| Collect in-focus images for machine learning | Processes frames during acquisition and stores only sharp crops of the detected specimens, so a few animals yield thousands of images in varied poses. | `scripts/run_realtime.py` |
| Localize non-moving specimens | Detects fixed or resting specimens against a separate recording of the same chamber without specimens. | `scripts/run_pipeline.py --background_source external` |
| Train the classifiers | Retrains the random-forest posture filter or fine-tunes an EfficientNet-B0 on crops. | `scripts/train_model.py`, `scripts/train_efficientnet_model.py` |

## How depth-from-focus works here

1. **Scan.** The acquisition records a repeated focal scan. Each frame's filename carries its frame number, ETL setting (dpt) and focal distance (mm), and `metadata_log.csv` records the acquisition settings and the timing of every frame.
2. **Detect.** Stage 1 subtracts from each frame a background frame of the *same focal plane*, because defocus and image brightness both change with the ETL setting. By default the background is a later frame of the same recording, so only specimens that moved in between remain.
3. **Link.** Detections are linked into tracks by their XY positions only (trackpy). Linking does not depend on the focal plane or on whether a specimen is in focus.
4. **Quantify focus.** Stage 2 crops every tracked specimen from every frame and computes its sharpness as the variance of the Laplacian divided by the mean grey level of the crop.
5. **Find focus peaks.** Along a track, sharpness peaks each time the focal plane passes the specimen. Peaks are detected against the track's own out-of-focus baseline, and the strongest peak of each sweep through the Z-range gives one depth estimate.
6. **Reconstruct.** XY positions come from every frame and depth from the focus peaks. Between peaks, depth is estimated with a smoothing spline (`z_reconstructed`), which gives the 3D trajectory of every individual in the sample.

Lateral positions are therefore sampled at the full frame rate, whereas depth is sampled once per sweep. Step size and Z-range set this trade-off (see [Calibration and limitations](#calibration-and-limitations)).

## Hardware

The acquisition code drives the following reference system. The offline analysis needs no hardware and runs on any recording that follows the frame naming described under [Preparing data](#preparing-data).

| Component | Reference system |
| --- | --- |
| Camera | Basler a2A4504-27g5m (monochrome, 20 MP, 27 fps, 5GigE), controlled through the Basler pylon SDK (`pypylon`) |
| Tunable lens | Optotune EL-16-40-TC (−2 to +3 dpt), integrated in the objective, controlled over USB through the Optotune Python SDK |
| Objectives | Telecentric ETL objectives TS1-0638-70-EL (0.638×) and S5VPJ6415 (1.5×) |
| XY stage (optional) | Zaber stage for well plates. The single-well scripts need no stage. |
| Computer | Windows PC. An NVIDIA GPU is used only for the EfficientNet classifier. |

Rig-specific values live in `config/config.yml` under `hardware_profiles.<name>` and are selected with `active_hardware_profile`: Z-range (`scan_height_mm`), optical leverage (`scan_ratio`, mm per diopter), lens controller port, exposure time and camera ROI. A different ETL objective is added as a further profile.

## Installation

The reference environment is the conda environment `wellscan24_env` (Python 3.9). Step-by-step instructions are in `wellscan24_env_files/`:

- `howto_build_wellscan24_env.txt` – workstation with camera, ETL and XY stage (Windows),
- `howto_build_wellscan24_env_HPC.txt` – analysis only, on a cluster.

In short:

```
conda create -n wellscan24_env --file wellscan24_env_files/conda-packages.txt
conda activate wellscan24_env
pip install -r wellscan24_env_files/pip-requirements.txt
```

PyTorch is installed afterwards as described in the how-to file. The Optotune Python SDK, which only the acquisition needs, is not distributed with this repository; obtain it from Optotune and install it as described there. For the offline analysis alone, `pip install -r requirements.txt` into a Python 3.9 environment is sufficient.

## Quick start

Record one sample (15 s at 25 fps by default, i.e. 375 frames):

```
python scripts/run_acquire_single.py --plate 1 --well 1 --start-dpt -2.0
```

Analyze all recordings of a batch (3D tracks after Stages 1–2, measurements after Stage 4):

```
python scripts/run_pipeline.py --input_batch_dir <raw_data_root> --stages all --config config/config.yml
```

Record and process in real time, storing only sharp crops:

```
python scripts/run_realtime.py
```

## Analysis pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Detection & tracking | `src/stages/stage_01_initial_detection.py` | Pairs every frame with a background frame of the same focal plane, subtracts it, segments the remaining particles inside the well, and links them into tracks by XY position. Writes the table of tracked particles and track maps. |
| 2. Characterization & depth-from-focus | `src/stages/stage_02_characterization.py` | Crops and re-segments every tracked particle, computes sharpness, morphology and quality features, detects the focus peaks of each track, and reconstructs the 3D trajectories (3D, XY, XZ and YZ views). |
| 3. Filtering & classification | `src/stages/stage_03_filtering_and_classification.py` | Applies quality gates to the in-focus particles and scores them with the random-forest feature model or the EfficientNet-B0 crop classifier (selectable in the config). Writes `ml_probability`/`ml_label` and the accepted subset. |
| 4. Biological analysis | `src/stages/stage_04_biological_analysis.py` | Measures the accepted specimens (body and egg metrics) and renders annotated review panels. |

Each stage writes intermediate results (CSV tables, crops and figures), so its performance can be checked on new data.

### Background source for Stage 1

`--background_source` selects where the background frames come from. Everything after the pairing of foreground and background frames is shared between the two modes.

- `time_shift` (default): the background is a later frame of the same recording at the same focal plane, so only *moving* specimens are detected. This assumes a stationary background.
- `external`: the background is the same frame of a separate recording of the chamber without specimens, so *non-moving* specimens (e.g., fixed animals at known depths) are detected too. `--input_batch_dir` is then the experiment root that holds `fg/` and `bg/` with identically named sample folders. Both recordings must have been made with identical acquisition settings, which is checked against their `metadata_log.csv`.

### Realtime mode

`scripts/run_realtime.py` mirrors Stages 1–3 frame by frame during acquisition (`src/realtime/`). It first records one background image per focal plane, then detects specimens by their difference from the background of the matching plane and stores a crop whenever it passes the size, shape and sharpness gates. Tracking and depth-from-focus are omitted. With `adaptive_enabled`, the scan is restricted to the focal planes in which most detections occurred. Settings are under `realtime_mode` in the config (`target_fps` is 5 by default). Crops are written to `<output_root_dir>/<session>_plate<P>_well<W>/accepted_crops/`.

## Acquisition workflows (`src/acquire`)

- `scripts/run_acquire_single.py` → one sample, no XY stage
- `scripts/run_acquire_single_sweep.py` → step-size × pattern sweep on one sample, no XY stage
- `scripts/run_acquire.py` → multi-well workflow with the XY stage (wells chosen in a selector GUI)
- `scripts/run_acquire_experiment.py` → step-size × pattern sweep across wells

Scan settings are under `acquisition` in the config: `focusing.pattern` (`zigzag` or `sawtooth`), `focusing.scan_step_um` (physical step size in the medium), `camera.fps` and `camera.video_duration_sec`. `acquisition.experiment_sweep` defines the grid of the two sweep scripts and can extend each recording until every focal plane has been visited a set number of times (`plane_revisit_times`).

With `camera.trigger_mode: software_grid` (default), exposure *i* is triggered at *t*₀ + *i*/fps, and only after the lens command of its own focal plane plus a settling time (`focusing.min_settle_ms`, longer after large focus changes such as the reset of a sawtooth scan). Each frame therefore shows the focal plane recorded in its filename. The legacy `free_run` loop is kept for comparison. At 25 fps it labels each frame with the focal plane of the previous frame, and `scripts/relabel_frame_lag.py` corrects recordings made that way.

Supporting modules:

- `src/acquire/workflows.py` (workflow orchestration)
- `src/acquire/focus_pattern.py` (focus cycle, shared with the realtime mode)
- `src/acquire/triggered_acquisition.py` (fixed-grid trigger loop and camera-clock checks)
- `src/acquire/config.py`, `src/acquire/plate_layout.py` (plate types and 24/96-well coordinates)
- `src/acquire/stage_controller.py` (Zaber adapter), `src/acquire/well_selector_gui.py` (well selector)
- `src/acquire/ws_flat_field.py` (guided flat-field and dark-frame acquisition)

To copy acquired runs into a pipeline input batch without changing their names, use:

```
python scripts/prepare_acquisition_batch.py <acquisition_output_dir> <batch_name> [--prefix YYYYMMDD_]
```

## Configuration

The YAML file at `config/config.yml` controls acquisition, the offline pipeline and the realtime mode. Key entries:

- `paths.pipeline_base`: offline-pipeline root (writes `data/output/`, hosts `models/` and `data/training_data/`). Required; the former top-level `project_base_dir` key is no longer read.
- `paths.raw_data_root`: where acquisition writes captured frames; point `run_pipeline --input_batch_dir` at it.
- `active_hardware_profile` and `hardware_profiles.<name>`: rig-specific settings (see [Hardware](#hardware)), deep-merged into `acquisition`. Each profile holds the optics (`optics.pixel_size_um`, `optics.magnification`), which the acquisition writes into every run's `metadata_log.csv`.
- The offline pipeline reads a recording's frame rate and pixel scale from its own `metadata_log.csv`, not from the current config, so each sample folder must keep that file. Recordings made before the optics were logged use the optics of the active hardware profile.
- `acquisition`: scan pattern, step size, frame rate, recording duration, trigger mode and plate type (`acquisition.plate.type`, `24` or `96`).
- `imaging` (frame geometry) and `segmentation` (shared thresholding/segmentation) — the loader assembles these into the internal `system` view.
- Stage-specific blocks (`stage_01_initial_detection`, `stage_02_characterization`, `stage_03_filtering_and_classification`, `stage_04_biological_analysis`) that tune detection, tracking, focus-peak detection, filtering, classifier thresholds and visualization. The focus-peak settings are under `stage_02_characterization.focussed_particles.peak_detection`.
- `realtime_mode`: frame rate, background handling, adaptive scan range and crop storage of the realtime mode.
- `figure_style`: fonts, line widths and resolution of every figure the pipeline writes.
- Paths to the trained RF model (`models/rf_fem_classifier.joblib`), scaler, and feature-name list consumed by Stage 3. These artifacts are required so Stage 3 can populate `ml_probability`/`ml_label`, which Stage 4 refuses to run without.

A config file with a top-level `base_config: "<file>"` is a *variant*: it holds only the values it changes and is merged over that base file. Use variants instead of copies of `config.yml`.

## Preparing data

1. **Input batches** – A batch directory contains one directory per sample (recording), named `YYYYMMDD_plate<P>_well<W>` (sweep recordings append the step size and pattern, e.g. `_400um_zigzag`). Frames are PNG files named `YYYYMMDD_plate<P>_well<W>_f<NNNNN>_dpt<D>_fd<F>mm.png` (frame number, ETL setting in diopters, focal distance in mm), next to the recording's `metadata_log.csv`. The acquisition scripts write this layout.
2. **Experiment design (optional)** – A CSV with one row per sample (`date`, `plate`, `well`, `scan_step_um`, and any factor columns), passed with `--experiment_design`. It is validated against each sample's `metadata_log.csv`, and the factor columns are added to the output tables.
3. **Training crops (optional)** – Curated crops in `data/training_data/good/` and `data/training_data/bad/`. Both training scripts consume this layout using the Stage 2 crop outputs.

## Running the pipeline

The entry point is `scripts/run_pipeline.py`. It expects a batch directory containing one or more sample subdirectories.

```
python scripts/run_pipeline.py \
    --input_batch_dir data/input/batch_20240101 \
    --stages all \
    --config config/config.yml
```

CLI options:

- `--input_batch_dir`: **Required.** Path to the parent directory that contains the sample folders (in `external` background mode, the experiment root holding `fg/` and `bg/`).
- `--config`: Optional path to a custom YAML config (defaults to `config/config.yml`).
- `--stages`: Comma-separated list from `{1,2,3,4}` or `all` to control which stage modules run. Dependencies are enforced, so Stage n requires outputs from Stage n−1. When a stage is rerun, its existing output directory is automatically deleted so the new outputs start cleanly.
- `--background_source`: `time_shift` (default) or `external`, see [Background source for Stage 1](#background-source-for-stage-1).
- `--experiment_design`: Optional experiment-design CSV, see [Preparing data](#preparing-data).
- `--resume`: Resume **full pipeline runs only** (`--stages all` or `--stages 1,2,3,4`). The pipeline writes an atomic per-sample completion marker after Stage 4 succeeds, then uses those markers to skip already-complete leading samples after an interruption (timeout, node preemption, power loss).

On the reference workstation, one recording of 375 frames (4512 × 4512 pixels) takes about 7–24 min, depending mainly on the number of detected particles.

### HPC/Slurm execution notes

If you run this pipeline on a cluster, review `docs/hpc_slurm_runbook.md` for a hardened Slurm template and guidance on:

- keeping input, outputs, and logs on node-accessible storage,
- setting `paths.pipeline_base` so batch rollups and per-sample folders end up where you expect,
- staging Stage 3 model artifacts from a separate path via symlink or copy,
- avoiding concurrent jobs writing into the same output root.

## Output layout

For an input batch named `batch_20240101` and sample `sample_A`, outputs land in `data/output/batch_20240101/sample_A/` with stage-specific folders:

- `stage1_detection/`
  - `{sample}_stage1_tracked_particles.csv` – per-frame detections with their track IDs.
  - `{sample}_stage1_tracks_3d.png`, `{sample}_stage1_tracks_xy.png` – maps of the raw tracks.
- `stage2_characterization/`
  - `{sample}_stage2_all_particle_features.csv` – full feature table for every tracked particle in every frame.
  - `filtered_in_focus.csv` – the focus peaks, i.e. one row per in-focus occurrence of a track, with its depth.
  - `reconstructed_trajectories_3d.png`, `…_xy.png`, `…_xz.png`, `…_yz.png` – reconstructed trajectories.
  - Per-track focus diagnostics (sharpness along the track with the detected peaks).
  - `stage2_all_particle_crops/` – normalized crops reused by later stages.
- `stage3_filtering_classification/`
  - `filtered_good.csv` – records of particles that passed quality filters and model scoring (only rows with `ml_label == 'accept'`). Includes `ml_probability` + `ml_label` columns that Stage 4 validates before running.
  - `filtered_good/` – copy of every accepted crop to simplify manual review and Stage 4 feature consumption.
  - `focussed_and_filtered.csv` + `focussed_and_filtered/` – particles that passed Stage 3 focus/quality checks before ML scoring (no `ml_*` columns).
- `stage4_biological_analysis/`
  - `stage4_final_analysis.csv` plus visualization artifacts for downstream review.

Each pipeline run also produces batch-level rollups at the project root: `<batch>.csv` aggregates all Stage 2 `final_particles.csv` tables, `<batch>_stage3_filtered_good.csv` concatenates every Stage 3 `filtered_good.csv`, `<batch>_stage3_focussed_and_filtered.csv` gathers the pre-ML Stage 3 outputs, and `<batch>_stage4_final_analysis.csv` consolidates all Stage 4 analyses.

## ML classification workflow

Stage 3 is split into three segments: (1) deterministic quality filters, (2) ML scoring, and (3) exporting/plotting helpers.

1. **Model artifacts** – `scripts/train_model.py` learns a RandomForest using the Stage 2 feature set, writes `models/rf_fem_classifier.joblib`, `models/rf_fem_scaler.joblib`, and `models/rf_fem_classifier_feature_names.txt`. `scripts/train_efficientnet_model.py` fine-tunes EfficientNet-B0 on the Stage 2 crops and saves `models/efficientnet_b0_best.pt` with the class mapping metadata embedded. Stage 3 loads the appropriate bundle via `get_models_dir()`; missing artifacts raise `MLModelError`.
2. **Runtime scoring** – Stage 3 attaches `ml_probability` (positive-class score) and `ml_label` ("accept"/"reject" decision based on `ml_filter.min_probability_threshold`). Set `stage_03_filtering_and_classification.model_type` to `random_forest_model` or `efficient_net_model` in the YAML to toggle which classifier runs.
3. **Diagnostics** – `plot_probability_histogram` and `plot_roc_curve` in `src/visualization/feature_visualization.py` summarize new runs.

To retrain the random forest from the crops in `data/training_data/good` and `data/training_data/bad`:

```
python scripts/train_model.py --config config/config.yml
```

See `docs/random_forest_training.md` for details and options (including CSV-based training) and `docs/efficientnet_training.md` for the EfficientNet-B0 alternative (including CPU/GPU tips).

The classifier of egg-bearing versus non-egg-bearing rotifers shown in the paper was trained on crops recorded in realtime mode. Its images and results are part of the archived supplementary data (see [Citation and data](#citation-and-data)). `scripts/train_efficientnet_model.py` in this repository trains the Stage 3 crop classifier, which uses the same network but a stronger augmentation scheme.

## Calibration and limitations

- **Depth scale.** Depths follow from the ETL setting and the optical leverage (`scan_ratio`, mm per diopter). When imaging through water the leverage in air must be multiplied by the refractive index (1.333), and it should be measured on the assembled system rather than taken from the data sheet. On the reference system the measured leverage of the TS1-0638-70-EL was 12 % above the specification.
- **Depth resolution.** Stage 2 assigns each focus peak the depth of its focal plane. Interpolation between focal planes (a Gaussian through the peak frame and its two neighbours) is implemented in `scripts/extract_focus_peaks_interp.py` for the axial-localization experiment. It requires step sizes clearly below the depth of field in water.
- **Sequential depth sampling.** A zigzag scan over *N* focal planes revisits a plane every (*N* − 1)/fps seconds on average. Step size and Z-range should be matched to the vertical speed of the organisms.
- **Sweep direction.** Upward and downward sweeps give slightly different depths, depending on the ETL setting. The pipeline does not correct this offset. Averaging the two sweep directions of a zigzag scan removes it.
- **Focus breathing.** The pipeline uses one pixel scale for all focal planes and does not correct the change of magnification with the ETL setting (below 3 % over the focus range of the TS1-0638-70-EL). Where exact sizes matter, rescale measurements by the recorded diopter setting.
- **Brightness.** Background brightness changes with the ETL setting. The pipeline therefore subtracts only frames of the same focal plane, thresholds automatically, and normalizes sharpness by crop brightness. It does not adjust the exposure during acquisition.
- **Moving specimens only, by default.** The `time_shift` background removes everything that does not move. Use the `external` background for non-moving specimens.
- **Dense samples.** All individuals in the water column are projected onto the same field of view. Tracks of individuals that overlap laterally can merge or swap, even when they are separated in depth.

## Project structure

```
etlscan/
├── config/
│   └── config.yml                # Single configuration for acquisition, pipeline and realtime mode
├── docs/                         # Conventions, HPC runbook, ML training guides, realtime spec
├── hpc/                          # Slurm submission scripts
├── scripts/
│   ├── run_acquire*.py           # Acquisition entry points (single, sweep, multi-well)
│   ├── run_pipeline.py           # Offline analysis (Stages 1–4)
│   ├── run_realtime.py           # Realtime acquisition + processing
│   ├── train_model.py            # Random-forest trainer
│   ├── train_efficientnet_model.py
│   └── *.py                      # Helpers and diagnostics (see below)
├── src/
│   ├── acquire/                  # Hardware acquisition (camera, ETL, XY stage)
│   ├── core/
│   │   ├── feature_calculators/  # morphology, quality, shape, bio-feature modules
│   │   └── *.py                  # segmentation, masking, particle ops, sharpness, I/O
│   ├── realtime/                 # Frame-by-frame mirror of Stages 1–3
│   ├── stages/                   # Offline stage implementations (01–04)
│   ├── utils/                    # config loading, acquisition metadata, logging, paths, I/O
│   └── visualization/            # Stage 4 overlays and the shared figure style
├── tests/                        # pytest suite
├── wellscan24_env_files/         # Environment definition and installation notes
├── CITATION.cff
├── CLAUDE.md                     # Developer guide
├── LICENSE
├── requirements.txt
└── README.md
```

`data/` (pipeline input/output, training crops) and `models/` (trained classifiers) are created under `paths.pipeline_base` and are not part of the repository.

Helper scripts:

- `scripts/extract_focus_peaks_interp.py` – focus peaks with interpolation between focal planes for specimens at known XY positions.
- `scripts/relabel_frame_lag.py` – corrects the focal-plane labels of recordings made with the `free_run` loop.
- `scripts/prepare_acquisition_batch.py` – copies acquisition folders into a pipeline input batch.
- `scripts/build_manual_labels_csv.py` – rebuilds labelled training rows from curated crops.
- `scripts/determine_bg_offset_demo.py` – shows the background offset Stage 1 selects for a recording.
- `scripts/detect_black_mask_demo.py` – evaluates the dark-corner check on a directory of images.

## Tests

```
pytest tests/ --ignore=tests/test_acquire_refactor_contracts.py
```

The ignored file needs `tkinter` and the acquisition dependencies.

## Customization

- Adjust detection, segmentation, focus-peak detection and filtering via the stage-specific sections in the YAML config.
- Extend feature extraction by editing `src/core/feature_calculators/` or `src/core/particle_operations.py`. Stage 2 serializes new measurements into the per-particle CSVs for downstream consumption.
- Swap the classification model by modifying `src/stages/stage_03_filtering_and_classification.py` while preserving the contract of reading Stage 2 CSVs and writing the accepted subset CSV.
- Coding conventions are in `docs/conventions.md`.

## Troubleshooting

- Ensure `paths.pipeline_base` in the config points to the intended analysis root; many helpers derive relative locations from it.
- Confirm that sample directories contain PNG frames named according to the acquisition convention (see [Preparing data](#preparing-data)).
- If Stage 3 cannot load the classifier, rerun `scripts/train_model.py` to regenerate the `models/` artifacts.
- Review the log output (`pipeline_run.log`, configured under `logging`) for detailed error messages per stage.

## Citation and data

If you use this software, please cite the paper:

> Stelzer, C.-P., M. Brunner & M. Pichler. *Depth-resolved imaging of live plankton with electronically tunable lenses.* In revision at Limnology and Oceanography: Methods.

The data of the paper, example outputs of every pipeline stage and the training images of the classifiers are archived at Zenodo: <https://doi.org/10.5281/zenodo.20396908>. Citation metadata is in `CITATION.cff`.

## License

MIT, see `LICENSE`.

## Acknowledgements

Development was funded by the Austrian Science Fund (FWF), grant P35916. Software development was assisted by a large language model (Anthropic's Claude Code). The acquisition code was tested on the physical hardware and the analysis pipeline against manually curated datasets.
