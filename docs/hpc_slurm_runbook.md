# HPC Slurm runbook

This project can run cleanly on HPC clusters, but a few path and job-isolation details matter.

## 1) Corrected Slurm template

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=well_plate_analyzer
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.tld
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

date_arg="${1:-}"
if [[ -z "$date_arg" ]]; then
  echo "Usage: sbatch $0 <date>"
  exit 1
fi

# Example date input: 20250601
input_path="${SCRATCH}/wellscanner/size_AS1_S1/images/${date_arg}"
repo_root="${SCRATCH}/wellscanner/well_plate_analyzer"

module load Anaconda3/2023.10/miniconda-base-2023.10
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate wellscan_env

cd "$repo_root"

python scripts/run_pipeline.py \
  --input_batch_dir "$input_path" \
  --stages all \
  --config config/pipeline_config.yml
```

Notes:
- The original `#SBATC H --mail-user=...` has a typo and is ignored by Slurm.
- Quote all path variables to avoid whitespace/special-character issues.
- Add `--output/--error` so each job has explicit log files.

## 2) Output placement (important)

`run_pipeline.py` writes outputs relative to `project_base_dir` from your config:

- per-sample outputs: `<project_base_dir>/data/output/<batch>/<sample>/...`
- batch rollups: `<project_base_dir>/<batch>.csv`, `<batch>_stage3_filtered_good.csv`, etc.

For cluster runs, set `project_base_dir` in `config/pipeline_config.yml` to a writable shared filesystem path (e.g., under `$SCRATCH`) so:

- compute nodes can write stage outputs,
- login nodes can read results later,
- you avoid filling small home quotas.

## 3) Stage 3 model artifacts from a separate path

Stage 3 loads model artifacts from `<project_base_dir>/models/`.

RandomForest defaults:
- `rf_fem_classifier.joblib`
- `rf_fem_scaler.joblib`
- `rf_fem_classifier_feature_names.txt` (derived default)

EfficientNet default:
- `efficientnet_b0_best.pt`

If your canonical model/training assets live elsewhere, you currently have two safe options:

1. **Symlink** `<project_base_dir>/models` to your shared model store.
2. **Copy** required model files into `<project_base_dir>/models` before launching jobs.

Also ensure training and inference used compatible feature schemas. For RF mode, Stage 3 aligns data to the persisted feature-name file; missing features are filled with zero, but large schema drift can still hurt predictions.

## 4) Concurrency and reruns

When a stage is selected, the runner deletes and recreates that stage's output folder for each sample before writing new files. Avoid launching two jobs that target the same `project_base_dir + batch + sample`, or they can clobber each other.

Recommended pattern:
- unique batch/date input per job,
- or unique `project_base_dir` per active run,
- or strict scheduler-level mutual exclusion for the same dataset.

## 5) Preflight checklist

Before `sbatch`:

1. `project_base_dir` points to writable shared storage.
2. `--input_batch_dir` exists and contains sample subfolders in the expected naming format (`YYYYMMDD_plateX_wellY`).
3. Stage 3 model artifacts exist under `<project_base_dir>/models` (or symlink resolves).
4. Conda env includes optional dependencies if using EfficientNet (`torch`, `torchvision`, `Pillow`).
5. Job log paths are configured via `#SBATCH --output/--error`.
