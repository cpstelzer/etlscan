#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit_batch.sh /path/to/data_root
# where /path/to/data_root contains part1/, part2/, ...

DATA_ROOT="${1:-}"

# Path to the Slurm script that runs one job for one subdirectory
SLURM_SCRIPT="${SCRATCH}/wellscanner/well_plate_analyzer/submit_ws_job.slrm"

if [[ -z "$DATA_ROOT" ]]; then
  echo "ERROR: Missing argument."
  echo "Usage: $0 /path/to/data_root"
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT is not a directory: $DATA_ROOT"
  exit 2
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "ERROR: Slurm script not found: $SLURM_SCRIPT"
  exit 2
fi

if [[ ! -x "$(command -v sbatch)" ]]; then
  echo "ERROR: sbatch not found in PATH (Slurm not available in this shell)."
  exit 2
fi

# Discover part* directories (one level below DATA_ROOT)
mapfile -t subdirs < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name "part*" | sort)

if [[ ${#subdirs[@]} -eq 0 ]]; then
  echo "ERROR: No subdirectories matching 'part*' found under: $DATA_ROOT"
  exit 3
fi

echo "Submitting ${#subdirs[@]} jobs found under:"
echo "  $DATA_ROOT"
echo "Using Slurm script:"
echo "  $SLURM_SCRIPT"
echo

for subdir in "${subdirs[@]}"; do
  # Use absolute path to avoid ambiguity about the working directory
  abs_subdir="$(readlink -f "$subdir")"
  echo "sbatch \"$SLURM_SCRIPT\" \"$abs_subdir\""
  sbatch "$SLURM_SCRIPT" "$abs_subdir"
done
