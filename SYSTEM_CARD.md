# System Card — Well Plate Analyzer

## Plain-language summary
Well Plate Analyzer is a Python pipeline that processes batches of microscope image sequences into measurements and review-ready outputs. It runs four stages—initial detection, characterization, filtering/classification, and biological analysis—with a batch runner that orchestrates the steps and writes stage-specific CSVs.

## What the system does (end‑to‑end)
1. **Initial detection (Stage 1)** — starts the pipeline on an image directory and prepares a Stage 1 CSV per sample.
2. **Characterization (Stage 2)** — performs detailed characterization for every particle candidate from Stage 1 using a modular feature calculator setup.
3. **Filtering & classification (Stage 3)** — applies quality flag filters to Stage 2 results and then continues with model-based scoring as configured.
4. **Biological analysis & visualization (Stage 4)** — performs final biological analysis and generates visualizations for each curated particle.

The pipeline runner (`scripts/run_pipeline.py`) ties these stages together, resets output folders when a stage is re-run, and builds batch‑level rollup CSVs.

## Intended users and usage
- Designed for analysts or lab staff processing microscopy runs from well‑plate experiments. Input data is resolved under `data/input`, while output data is written under `data/output` based on the configured project base directory.
- A single command runs the pipeline for a batch directory, optionally selecting stages.

## Inputs
- **Images:** Sample directories are discovered from the batch input directory (via the batch runner), and each sample is processed with the Stage 1 entry point.
- **Configuration:** `config/pipeline_config.yml` controls global settings, image dimensions, segmentation thresholds, and stage‑specific parameters like cropping, quality filters, and model selection.
- **Optional training crops:** Training scripts pull labeled crops from the training data directory created under `data/training_data` relative to the project base directory.

## Outputs
- Stage‑specific CSVs and image folders live under `data/output/<batch>/<sample>/`, with standard file names like `filtered_good.csv` and `stage4_final_analysis.csv`.
- Batch‑level rollups are written by the runner after per‑sample results are aggregated.

## Key components (modules and scripts)
- **Pipeline runner:** `scripts/run_pipeline.py` handles stage sequencing, output cleanup, and batch‑level aggregation.
- **Stage modules:**
  - Stage 1: `src/stages/stage_01_initial_detection.py` (initial detection).
  - Stage 2: `src/stages/stage_02_characterization.py` (characterization).
  - Stage 3: `src/stages/stage_03_filtering_and_classification.py` (quality filters + model scoring).
  - Stage 4: `src/stages/stage_04_biological_analysis.py` (biological analysis + visualizations).
- **Training scripts:**
  - `scripts/train_model.py` trains and saves RandomForest artifacts used by Stage 3.
  - `scripts/train_efficientnet_model.py` fine‑tunes EfficientNet‑B0 to produce a checkpoint selectable in the pipeline config.

## Models and decision logic
- **RandomForest option:** training saves model artifacts under the `models/` directory, and Stage 3 loads those artifacts to score particles.
- **EfficientNet option:** the EfficientNet trainer produces a checkpoint that can be selected via the Stage 3 model type setting.
- **Model choice:** set `stage_03_filtering_and_classification.model_type` in the YAML to choose `random_forest_model` or `efficient_net_model`.
- **Stage 4 requirements:** Stage 4 checks for `ml_probability` and `ml_label` in the Stage 3 CSV before proceeding.

## Configuration highlights (plain language)
- **Global image size and segmentation:** image dimensions and segmentation thresholds are centralized under `system` and `system.segmentation`.
- **Stage 1 tracking behavior:** background settings, masking, and tracking parameters like search range and minimum track length are in `stage_01_initial_detection`.
- **Stage 2 cropping and quality checks:** defines crop styles and black-mask checks in `stage_02_characterization`.
- **Stage 3 filtering:** quality and sharpness thresholds plus ML probability cutoffs are controlled in `stage_03_filtering_and_classification`.
- **Stage 4 egg detection:** parameters for egg metrics live under `stage_04_biological_analysis.egg_detection`.

## Data flow and storage
- The batch runner discovers valid sample subdirectories and builds stage‑specific folders for each sample with standard CSV names.
- The runner cleans existing stage output directories before re‑running a stage and aggregates per‑sample CSVs into batch rollups.

## Error handling and logging
- The pipeline uses a logging configuration (log levels, format, and log file) defined in the YAML config.
- Stage 2 and later require earlier stage outputs; missing files or failures raise errors in the runner.

## Operational constraints and limitations
- **File structure is required:** inputs must follow the batch/sample folder layout for the runner to recognize samples.
- **Model artifacts are required for Stage 3:** Stage 3 loads model artifacts before scoring, and Stage 4 requires the `ml_probability` and `ml_label` columns to exist.
- **Configuration is environment‑sensitive:** `project_base_dir` is an absolute path and must match the local machine.

## Privacy and data handling (what the code implies)
- The system reads image data from local input folders and writes outputs under local project directories; the core path helpers resolve paths under `data/input`, `data/output`, and `data/training_data` beneath the configured project base directory.

## Quick start (minimal steps)
1. Set `project_base_dir` in `config/pipeline_config.yml` to the repository path.
2. Run the pipeline on a batch directory, for example:

```
python scripts/run_pipeline.py \
    --input_batch_dir data/input/batch_20240101 \
    --stages all \
    --config config/pipeline_config.yml
```
