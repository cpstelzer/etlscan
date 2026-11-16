# Random Forest training guide

This guide describes how to retrain the Stage 03 RandomForest classifier using curated Stage 02 particle crops. The trainer now extracts features directly from class-labeled image folders by default, so you can invoke it with a single command.

## Dataset layout

Place curated crops under the shared training directory (defaults to `data/training_data/` from `config/config.yml`). Within that root, provide two subfolders named after their class labels:

```
data/training_data/
├── good/  # accepted crops
└── bad/   # rejected crops
```

Each subfolder can contain any number of PNG/JPEG crops. The script will scan both folders, derive the class label from the parent directory name, and generate the feature tables internally—no CSVs are required.

> Tip: If you prefer to use pre-computed feature CSVs, pass `--csv-files` with one or more paths. Image-folder mode is skipped whenever CSVs are explicitly supplied.

## Quick start

Train using the default image-folder mode in a single command:

```
python scripts/train_model.py --config config/config.yml
```

If your training crops live elsewhere, override the base directory:

```
python scripts/train_model.py --config config/config.yml \
    --training-data-dir /path/to/my_crops
```

The script will look for `good/` and `bad/` subdirectories inside the provided root. You can customize the subfolder names via `--positive-class-dir` and `--negative-class-dir` if your data uses different labels.

## Outputs

The training settings are read from `stage_03_filtering_and_classification.training_params` in `config/config.yml`: `test_split_size`, `random_state`, `exclude_feature_prefixes` (feature columns left out of training, e.g. `["lbp_"]`) and `rf_param_grid` (the values `GridSearchCV` searches; one value per parameter trains a single forest). A missing key stops the script.

Successful training writes the refreshed artifacts to `models/` (or the `models_dir` configured in the YAML):

- `rf_fem_classifier.joblib` — trained RandomForest model
- `rf_fem_scaler.joblib` — `StandardScaler` fitted on the feature set
- `rf_fem_classifier_feature_names.txt` — ordered feature list used at inference

Stage 03 automatically loads these artifacts during runtime filtering/classification.
