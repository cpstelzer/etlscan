# Well Plate Analyzer

Automated pipeline for processing well-plate microscopy batches. The system ingests raw frame sequences, tracks organisms, measures morphology, and produces curated review packages. The runtime flow is organized into four stages that can be executed independently or sequentially via the pipeline runner.

## Pipeline overview

| Stage | Module | Highlights |
| --- | --- | --- |
| 1. Detection & tracking | `src/stages/stage_01_initial_detection.py` | Scans the input sample directory, performs background subtraction, extracts particle tracks, and writes the initial CSV of candidates. |
| 2. Universal characterization | `src/stages/stage_02_characterization.py` | Re-crops every tracked particle, re-segments the organism, and aggregates sharpness, morphology, and quality metrics along with the normalized particle crops. |
| 3. Filtering & classification | `src/stages/stage_03_filtering_and_classification.py` | Applies configurable quality/sharpness gates, loads the RandomForest model & scaler from `models/`, and emits the list of accepted particles. |
| 4. Biological analysis & visualization | `src/stages/stage_04_biological_analysis.py` | Revisits the Stage 02 crops for the accepted particles, estimates egg metrics, and renders annotated review panels through `src/visualization/feature_visualization.py`. |

Supporting tooling:

- `scripts/train_model.py` retrains the Stage 03 RandomForest classifier using the Stage 02 feature set.
- `scripts/determine_bg_offset_demo.py` surfaces Stage 01 background tuning helpers for diagnostics.
- `scripts/detect_black_mask_demo.py` evaluates every supported image in a directory with `detect_black_mask_artifact` and writes `black_mask_detection_results.csv` beside the inputs. Example:

  ```
  python scripts/detect_black_mask_demo.py data/sanity_checks/masks \
      --config config/pipeline_config.yml
  ```

## Project structure

```
well_plate_analyzer/
├── config/
│   └── pipeline_config.yml       # Global runtime configuration
├── data/
│   ├── input/<batch>/<sample>/   # Raw PNG sequences per sample
│   ├── output/<batch>/<sample>/  # Pipeline outputs organized by stage
│   └── training_data/            # Crops used by scripts/train_model.py
├── models/                       # Trained RF model + scaler artifacts
├── scripts/
│   ├── run_pipeline.py           # Batch-oriented stage runner
│   ├── train_model.py            # Standalone RF trainer
│   └── determine_bg_offset_demo.py
├── src/
│   ├── core/                     # Feature calculators, segmentation, particle ops
│   ├── stages/                   # Stage implementations (01–04)
│   ├── utils/                    # Config loading, logging, path helpers, I/O
│   └── visualization/            # Plotting/overlay helpers used by Stage 04
├── requirements.txt
└── README.md
```

## Prerequisites & installation

1. Install Python 3.8+.
2. Create and activate a virtual environment (recommended).
3. Install dependencies: `pip install -r requirements.txt`.

## Configuration

The YAML file at `config/pipeline_config.yml` controls the entire pipeline. Key entries:

- `project_base_dir`: absolute path to the repository root.
- Stage-specific blocks (`stage_01_initial_detection`, `stage_02_characterization`, `stage_03_filtering_and_classification`, `stage_04_biological_analysis`) that tune detection, cropping, filtering, classifier thresholds, and visualization options.
- `stage_02_characterization.quality_features.black_mask_check` exposes `min_area_ratio`, `max_black_pixel_value`, and `min_edge_thickness_fraction`, which control how aggressively `detect_black_mask_artifact` flags dark corner occlusions.
- Paths to the trained RF model (`models/rf_fem_classifier.joblib`), scaler, and feature-name list consumed by Stage 03.

## Preparing data

1. **Input batches** – Arrange raw PNG frames under a batch directory such as `data/input/batch_20240101/`. Each immediate child directory (e.g., `batch_20240101/sample_A`) is treated as an individual sample run.
2. **Training crops (optional)** – To retrain the classifier, provide curated crops in `data/training_data/good/` and `data/training_data/bad/`. The `train_model.py` script will consume these using the same Stage 02 feature definitions.

## Running the pipeline

The entry point is `scripts/run_pipeline.py`. It expects a batch directory containing one or more sample subdirectories.

```
python scripts/run_pipeline.py \
    --input_batch_dir data/input/batch_20240101 \
    --stages all \
    --config config/pipeline_config.yml
```

Important CLI options:

- `--input_batch_dir`: **Required.** Path to the parent directory that contains the sample folders.
- `--config`: Optional path to a custom YAML config (defaults to `config/pipeline_config.yml`).
- `--stages`: Comma-separated list from `{1,2,3,4}` or `all` to control which stage modules run. Dependencies are enforced, so Stage n requires outputs from Stage n−1.

For classifier retraining run:

```
python scripts/train_model.py --config config/pipeline_config.yml \
    --training_data_dir data/training_data
```

## Output layout

For an input batch named `batch_20240101` and sample `sample_A`, outputs land in `data/output/batch_20240101/sample_A/` with stage-specific folders:

- `stage1_detection/`
  - `{sample}_stage1_tracked_particles.csv` – per-frame detection metadata and particle IDs.
- `stage2_characterization/`
  - `{sample}_stage2_all_particle_features.csv` – full feature table for every tracked particle.
  - `stage2_all_particle_crops/` – normalized crops reused by later stages.
- `stage3_filtering_classification/`
  - `{sample}_stage3_classified_good_particles.csv` – records of particles that passed quality filters and model scoring.
- `stage4_biological_analysis/`
  - `{sample}_stage4_final_analysis.csv` plus visualization artifacts for downstream review.

RF model files updated by `train_model.py` are stored under `models/` and automatically reused by Stage 03.

## Customization

- Adjust detection, segmentation, and filtering logic via the stage-specific sections in the YAML config.
- Extend feature extraction by editing `src/core/feature_calculators/` or `src/core/particle_operations.py`. The Stage 02 module automatically serializes any new measurements into the per-particle CSVs for downstream consumption.
- Swap the classification model by modifying `src/stages/stage_03_filtering_and_classification.py` while preserving the contract of reading Stage 02 CSVs and writing the accepted subset CSV.
- Enhance downstream visuals by updating `src/visualization/feature_visualization.py` and the Stage 04 module that invokes it.

## Troubleshooting

- Ensure `project_base_dir` in the config reflects the absolute repository path; many helpers derive relative locations from it.
- Confirm that sample directories contain PNG frames named according to the acquisition convention expected by Stage 01.
- If Stage 03 cannot load the classifier, rerun `scripts/train_model.py` to regenerate the `models/` artifacts.
- Review the log output (configured via `src/utils/setup_logging`) for detailed error messages per stage.
