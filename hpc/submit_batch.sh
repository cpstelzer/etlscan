#!/usr/bin/env bash
set -eu
# Enable pipefail if supported (older shells or mis-invocations may not support it)
(set -o pipefail) 2>/dev/null && set -o pipefail

# Usage:
#   ./submit_batch.sh /path/to/data_root
# where /path/to/data_root contains part1/, part2/, ...

DATA_ROOT="${1:-}"

# Path to the Slurm script that runs one job for one subdirectory
SLURM_SCRIPT="${SCRATCH}/wellscanner/basedir/submit_ws_job.slrm"

# Resources to enforce at submission time (command-line flags override script defaults)
CPUS_PER_TASK=4
MEMORY="48G"

if [[ -z "$DATA_ROOT" ]]; then
  echo "ERROR: Missing argument."
  echo "Usage: $0 /path/to/data_root"
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: Not a directory: $DATA_ROOT"
  exit 1
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "ERROR: Slurm script not found: $SLURM_SCRIPT"
  exit 1
fi

# Find immediate subdirectories named part*
mapfile -t subdirs < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name "part*" | sort)

if [[ "${#subdirs[@]}" -eq 0 ]]; then
  echo "ERROR: No subdirectories matching 'part*' found under: $DATA_ROOT"
  exit 1
fi

echo "Submitting ${#subdirs[@]} jobs found under:"
echo "  $DATA_ROOT"
echo "Using Slurm script:"
echo "  $SLURM_SCRIPT"
echo "Forcing resources:"
echo "  --cpus-per-task=${CPUS_PER_TASK}  --mem=${MEMORY}"
echo

for subdir in "${subdirs[@]}"; do
  abs_subdir="$(readlink -f "$subdir")"
  echo "sbatch --cpus-per-task=${CPUS_PER_TASK} --mem=${MEMORY} "$SLURM_SCRIPT" "$abs_subdir""
  sbatch --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" "$SLURM_SCRIPT" "$abs_subdir"
done
