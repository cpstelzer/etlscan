# HPC Slurm runbook

This project can run cleanly on HPC clusters, but a few path and job-isolation details matter.

## 1) Data placement
Raw data, i.e., sample directories containing the image series (e.g., 20260212_plate101_well1) are stored in the experiment folders inside wellscanner/ (e.g., size_AS1_S1).

- To prepare the raw data for batch processing use `distribute_parts.sh`. This program splits the folders in a given directory into several “part” folders (e.g., part1–part8)  so each part contains about the same number of folders. It also copies 1_flat_field into every part. If needed, it can undo the split and put the folders back where they were.
- To submit jobs to the cluster (each "part" folder a separate job), execute `submit_batch.sh`, which is a driver Bash script that discovers part1/, part2/, … and submits one Slurm job per subdirectory
- Individual jobs are submitted by `submit_ws_job.slrm` which accepts exactly one argument: `input_dir` (a part*/ directory), validates it, activates the environment, and runs your pipeline on that directory.

## 2) Output placement (important)

`run_pipeline.py` writes outputs relative to `project_base_dir` from your config:

- per-sample outputs: `<project_base_dir>/data/output/<batch>/<sample>/...`
- batch rollups: `<project_base_dir>/<batch>.csv`, `<batch>_stage3_filtered_good.csv`, etc.

For cluster runs, set `project_base_dir` in `config/config.yml` to a writable shared filesystem path (e.g., under `$SCRATCH`) so:

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
