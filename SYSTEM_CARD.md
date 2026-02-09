# System Card — Well Plate Analyzer

## Plain-language summary
Well Plate Analyzer is a Python pipeline that turns batches of microscope image sequences from multi-well plates into cleaned measurements and review-ready visuals. It reads folders of PNG frames, tracks organisms over time, measures shape and quality, filters and scores candidates with a machine‑learning model, and produces CSV reports plus image panels for review. The process is split into four stages that can run one by one or as a full pipeline. 【F:README.md†L1-L75】

## What the system does (end‑to‑end)
1. **Detection & tracking (Stage 1)** — scans sample image folders, subtracts background, finds moving particles, and writes the first tracking CSV. 【F:README.md†L8-L17】
2. **Characterization (Stage 2)** — re-crops each tracked particle, re-segments the organism, and collects sharpness/shape/quality measures plus normalized crops. 【F:README.md†L12-L18】
3. **Filtering & classification (Stage 3)** — removes low-quality items and scores the rest with either a RandomForest or EfficientNet‑B0 model, then writes decision columns like `ml_probability` and `ml_label`. 【F:README.md†L13-L20】
4. **Biological analysis & visualization (Stage 4)** — uses the accepted items from Stage 3 to estimate egg metrics and render annotated review panels. 【F:README.md†L14-L18】

The pipeline runner (`scripts/run_pipeline.py`) ties these stages together, resets output folders when a stage is re-run, and builds batch‑level rollup CSVs. 【F:README.md†L63-L83】【F:scripts/run_pipeline.py†L79-L186】

## Intended users and usage
- Designed for analysts or lab staff processing microscopy runs from well‑plate experiments. The data is expected to live under `data/input/<batch>/<sample>/` and outputs are written to `data/output/<batch>/<sample>/` with stage‑specific folders and CSVs. 【F:README.md†L35-L55】【F:README.md†L90-L129】
- A single command runs the pipeline for a batch directory, optionally selecting stages. 【F:README.md†L63-L83】

## Inputs
- **Images:** PNG frame sequences stored in per‑sample folders under an input batch directory. Each immediate child directory of the batch is treated as a sample. 【F:README.md†L51-L55】【F:README.md†L63-L83】
- **Configuration:** `config/pipeline_config.yml` controls global settings, image dimensions, segmentation thresholds, and stage‑specific parameters like cropping, quality filters, and model selection. 【F:config/pipeline_config.yml†L1-L146】
- **Optional training crops:** Labeled “good” and “bad” crops can be provided for model training under `data/training_data/good/` and `data/training_data/bad/`. 【F:README.md†L56-L60】

## Outputs
- Stage‑specific CSVs and image folders live under `data/output/<batch>/<sample>/`, including tracked particles (Stage 1), feature tables and crops (Stage 2), filtered/accepted results (Stage 3), and biological analysis plus visuals (Stage 4). 【F:README.md†L90-L123】
- Batch‑level rollups are written at the project root (for example, `<batch>_stage3_filtered_good.csv`). 【F:README.md†L123-L126】

## Key components (modules and scripts)
- **Pipeline runner:** `scripts/run_pipeline.py` handles stage sequencing, output cleanup, and batch‑level aggregation. 【F:README.md†L35-L44】【F:scripts/run_pipeline.py†L79-L205】
- **Stage modules:**
  - Stage 1: `src/stages/stage_01_initial_detection.py` (detection and tracking). 【F:README.md†L8-L17】
  - Stage 2: `src/stages/stage_02_characterization.py` (recropping and measurements). 【F:README.md†L12-L18】
  - Stage 3: `src/stages/stage_03_filtering_and_classification.py` (quality filters + ML). 【F:README.md†L13-L20】
  - Stage 4: `src/stages/stage_04_biological_analysis.py` (final analysis and visuals). 【F:README.md†L14-L18】
- **Training scripts:**
  - `scripts/train_model.py` retrains the RandomForest using Stage 2 features. 【F:README.md†L21-L28】
  - `scripts/train_efficientnet_model.py` fine‑tunes EfficientNet‑B0 on Stage 2 crops. 【F:README.md†L22-L29】

## Models and decision logic
- **RandomForest option:** stored under `models/` and used in Stage 3. The model and scaler files are required to populate `ml_probability`/`ml_label` and Stage 4 refuses to run without those columns. 【F:README.md†L43-L49】【F:README.md†L41-L49】
- **EfficientNet option:** a fine‑tuned EfficientNet‑B0 checkpoint can be used instead of RandomForest. 【F:README.md†L22-L30】【F:README.md†L70-L79】
- **Model choice:** set `stage_03_filtering_and_classification.model_type` in the YAML to choose `random_forest_model` or `efficient_net_model`. 【F:config/pipeline_config.yml†L97-L114】

## Configuration highlights (plain language)
- **Global image size and segmentation:** image dimensions and segmentation thresholds are centralized under `system` and `system.segmentation`. 【F:config/pipeline_config.yml†L15-L42】
- **Stage 1 tracking behavior:** background settings, masking, and tracking parameters like search range and minimum track length are in `stage_01_initial_detection`. 【F:config/pipeline_config.yml†L43-L92】
- **Stage 2 cropping and quality checks:** defines crop styles and black-mask checks in `stage_02_characterization`. 【F:config/pipeline_config.yml†L94-L137】
- **Stage 3 filtering:** quality and sharpness thresholds plus ML probability cutoffs are controlled in `stage_03_filtering_and_classification`. 【F:config/pipeline_config.yml†L97-L146】
- **Stage 4 egg detection:** parameters for egg metrics live under `stage_04_biological_analysis.egg_detection`. 【F:config/pipeline_config.yml†L148-L159】

## Data flow and storage
- The pipeline expects each batch to have multiple sample subdirectories, and it builds stage‑specific folders for each sample, including standard CSV names like `filtered_good.csv` and `stage4_final_analysis.csv`. 【F:README.md†L90-L123】
- The runner cleans existing stage output directories before re‑running a stage and aggregates per‑sample CSVs into batch rollups. 【F:scripts/run_pipeline.py†L79-L205】

## Error handling and logging
- The pipeline uses a logging configuration (log levels, format, and log file) defined in the YAML config. 【F:config/pipeline_config.yml†L6-L14】
- Stage 2 and later require earlier stage outputs; missing files or failures raise errors in the runner. 【F:scripts/run_pipeline.py†L135-L205】

## Operational constraints and limitations
- **File structure is required:** inputs must follow the batch/sample folder layout for the runner to recognize samples. 【F:README.md†L51-L55】【F:README.md†L63-L83】
- **Model artifacts are required for Stage 3:** if model files are missing, Stage 3 cannot populate ML columns and Stage 4 will not proceed. 【F:README.md†L41-L49】
- **Configuration is environment‑sensitive:** `project_base_dir` is an absolute path and must match the local machine. 【F:config/pipeline_config.yml†L1-L4】

## Privacy and data handling (what the code implies)
- The code processes microscopy imagery and writes derived measurements and crops to disk. It does not describe network uploads or external data sharing, so data handling appears local to the machine running the pipeline. 【F:README.md†L1-L129】

## Quick start (minimal steps)
1. Install Python and dependencies (`pip install -r requirements.txt`). 【F:README.md†L37-L43】
2. Set `project_base_dir` in `config/pipeline_config.yml` to the repository path. 【F:config/pipeline_config.yml†L1-L4】
3. Run the pipeline on a batch directory, for example:

```
python scripts/run_pipeline.py \
    --input_batch_dir data/input/batch_20240101 \
    --stages all \
    --config config/pipeline_config.yml
```

【F:README.md†L63-L83】
