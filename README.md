# Well Plate Analyzer: Automated Image Processing Pipeline

This project provides an automated pipeline for processing images from well plate experiments, focusing on detecting, analyzing, and classifying microinvertebrates or other particles of interest. The pipeline is designed to be modular, configurable, and extensible.

It includes stages for:
1.  * Initial Particle Detection: Background subtraction and basic particle identification from series of images.
2.  * Quality Filtering: Cropping, artifact detection (e.g., black masks, boundary-cropped objects), sharpness assessment, and filtering to identify high-quality particle images.
3.  * ML Classification: Training a machine learning model (Random Forest) and using it to classify filtered particles, for instance, to identify "ideal" female specimens carrying eggs (FEMs).

## Project Structure

(BEGIN CODE BLOCK - TREE STRUCTURE)
well_plate_analyzer/
├── config/
│   └── pipeline_config.yml       # Main configuration for all stages
├── data/
│   ├── input/                    # Raw input images per sample (e.g., data/input/SAMPLE_ID/)
│   │   └── SAMPLE_ID/
│   │       └── *.png
│   ├── training_data/            # Data for training the ML model
│   │   ├── good/                 # Images of "good" examples for classification
│   │   └── bad/                  # Images of "bad" examples for classification
│   └── output/                   # Processed outputs per sample (e.g., data/output/SAMPLE_ID/)
│       └── SAMPLE_ID/
│           ├── stage1_initial_detection/
│           ├── stage2_quality_filter/
│           └── stage3_ml_classification/
├── models/                       # Saved machine learning models and scalers
├── scripts/
│   └── run_pipeline.py           # Main script to execute the pipeline
├── src/                          # Source code
│   ├── core/                     # Core image processing, analysis, and ML feature extraction logic
│   ├── stages/                   # Modules for each pipeline stage
│   ├── utils/                    # Utility functions (config, logging, paths, I/O)
│   └── visualization/            # Visualization code
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Specifies intentionally untracked files
└── README.md                     # This file
(END CODE BLOCK - TREE STRUCTURE)

## Prerequisites

* Python 3.7+
* Dependencies listed in `requirements.txt`.

## Installation

1.  * Clone the repository (if applicable):
    ```
    git clone <repository_url>
    cd well_plate_analyzer
    ```

2.  * Create a virtual environment (recommended):
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  * Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Configuration

The main configuration for the pipeline is located in `config/pipeline_config.yml`.
Key parameters to review and adjust:

* `project_base_dir`: **Crucial!** Set this to the absolute path of the `well_plate_analyzer` directory on your system.
* `sample_subdir_name`: Default sample ID, usually overridden by command-line arguments.
* Logging settings.
* Parameters for each stage (`stage_01_initial_detection`, `stage_02_quality_filter`, `stage_03_ml_classification`), including:
    -Thresholds for particle detection and filtering.
    -Image processing parameters (blurring, masking, cropping).
    -Sharpness metrics and criteria.
    -ML model training parameters and feature extraction settings.

## Data Preparation

1.  * Input Images:
    -Place raw images for each sample in a subdirectory under `data/input/`. For example, images for a sample named "MySample01" should go into `data/input/MySample01/`.
    -Images are expected to be PNG files following a specific naming convention (e.g., `YYYYMMDD_plateX_wellY_f<frame_num>_dpt<dpt_val>_fd<fd_val>mm.png`) for Stage 1 processing. This pattern is defined in `src/stages/stage_01_initial_detection.py` and can be adjusted if needed.

2.  * ML Training Data (for Stage 3 training):
    -Place example images for ML model training into `data/training_data/good/` and `data/training_data/bad/` (or as configured in `pipeline_config.yml`). These are typically cropped images of the objects you want to classify.

## Running the Pipeline

The main script to run the pipeline is `scripts/run_pipeline.py`.

**Basic Usage:**

```
python scripts/run_pipeline.py --sample_id "YOUR_SAMPLE_ID"
```
Replace `"YOUR_SAMPLE_ID"` with the name of the sample subdirectory in `data/input/`. This will run all stages of the pipeline.

**Command-line Arguments:**

* `--config FILE_PATH`: Path to the configuration YAML file (defaults to `config/pipeline_config.yml`).
* `--sample_id SAMPLE_NAME`: **Required.** The identifier of the sample to process.
* `--stages STAGES_LIST`: Comma-separated list of stages to run (e.g., "1", "1,2", "3", "all"). Defaults to "all".
    -`1`: Initial Particle Detection
    -`2`: Quality Filtering & Sharp Object Extraction
    -`3`: ML Classification (prediction by default)
    -`train_ml`: Only train the ML model (part of Stage 3).
* `--force_retrain_ml`: If specified with Stage 3 or `train_ml`, forces retraining of the ML model even if a saved model exists.

**Examples:**

* **Run all stages for a sample "SampleA":**
    ```
    python scripts/run_pipeline.py --sample_id "SampleA"
    ```

* **Run only Stage 1 for "SampleB":**
    ```
    python scripts/run_pipeline.py --sample_id "SampleB" --stages "1"
    ```

* **Train the ML model (using data in `data/training_data/`):**
    ```
    # --sample_id is formally required but less critical for train-only mode; provide a dummy or relevant context.
    python scripts/run_pipeline.py --sample_id "ML_Training_Run" --stages "train_ml"
    ```

* **Run Stage 3 prediction for "SampleA" using an existing model, and force retraining first:**
    ```
    python scripts/run_pipeline.py --sample_id "SampleA" --stages "3" --force_retrain_ml
    ```
    (This will first train the model, then run prediction for "SampleA" using the newly trained model).
    Alternatively, to run all stages including retraining:
    ```
    python scripts/run_pipeline.py --sample_id "SampleA" --stages "all" --force_retrain_ml
    ```

## Output

Processed data and results for each sample will be saved in subdirectories under `data/output/SAMPLE_ID/`. Each stage creates its own subdirectory:

* `data/output/SAMPLE_ID/stage1_initial_detection/`:
    -`{SAMPLE_ID}_stage1_particles.csv`: CSV file with initially detected particles and their basic properties.
* `data/output/SAMPLE_ID/stage2_quality_filter/`:
    -`{SAMPLE_ID}_stage2_analyzed_particles.csv`: Comprehensive CSV with detailed analysis (sharpness, artifact flags) for all particles from Stage 1.
    -`{SAMPLE_ID}_stage2_sharply_filtered_particles.csv`: CSV file listing only the particles that passed the final sharpness criteria.
    -`sharp_object_final_crops/`: Directory containing cropped images of these "sharp" particles.
    -`temp_intermediate_particle_crops/`: Temporary directory for intermediate crops used during analysis (can be deleted after run).
* `data/output/SAMPLE_ID/stage3_ml_classification/ideal_fems_ML_classified/`:
    -`{SAMPLE_ID}_stage3_accepted_fem_features.csv`: CSV file with features of particles classified as "ideal" (or positive class).
    -`accepted_images/`: Directory containing copies of the images classified as "ideal".
    -`visualizations/`: Directory containing visualizations (image with overlays) for the "ideal" images.

Trained ML models and scalers are saved in the `models/` directory (e.g., `rf_fem_classifier.joblib`, `rf_fem_scaler.joblib`, `rf_fem_classifier_feature_names.txt`).

## Customization and Extension

* **Configuration:** Modify `config/pipeline_config.yml` to tune parameters for any stage.
* **Core Logic:** The image processing algorithms are in `src/core/`. These can be modified or extended.
* **New Stages:** New processing stages can be added by creating a new module in `src/stages/` and integrating it into `scripts/run_pipeline.py`.
* **ML Model:** The current implementation uses RandomForest. This can be replaced or augmented with other classifiers by modifying `src/stages/stage_03_ml_classification.py`.
* **Feature Extraction:** ML features are defined in `src/core/ml_feature_extractor.py`. New features can be added here.

## Troubleshooting

* Check the log output in the console for detailed error messages. Logging level and format are configurable.
* Ensure `project_base_dir` in `config/pipeline_config.yml` is correctly set to the root of the `well_plate_analyzer` directory.
* Verify that input data paths and file naming conventions match expectations.
* If ML training fails, check the training data quality and quantity in `data/training_data/`.

## Original Modules Source

This refactored pipeline is based on the logic from the following original Python files:
* `wellcounter_imaging_module.py`
* `wellscanner_util_module.py`
* `ws_background_subtraction.txt` (likely `ws_background_subtraction.py`)
* `ws_extract_sharp_objects.txt` (likely `ws_extract_sharp_objects.py`)
* `ws_identify_ideal_fems.txt` (likely `ws_identify_ideal_fems.py`)

The goal of the refactor was to improve modularity, configurability, and maintainability."