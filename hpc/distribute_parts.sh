#!/usr/bin/env bash
set -euo pipefail
# ----------------------------------------------------------------------------- Partition/revert top-level subdirectories of data_dir into part1..partN.
#
# APPLY mode (default): - Creates part1..partN under data_dir - Copies data_dir/1_flat_field into each part directory - Moves all other top-level directories evenly (round-robin) into the parts - Writes a manifest with every 
#   move: <orig_path>\t<new_path>
#
# REVERT mode: - Reads manifest and moves each directory back to its original location
#
# Usage: distribute_parts.sh --data-dir /path/to/data_dir [--n-targets 8] [--dry-run] distribute_parts.sh --data-dir /path/to/data_dir --revert [--dry-run]
#
# Manifest default: DATA_DIR/.partition_manifest.tsv -----------------------------------------------------------------------------
usage() { cat <<'EOF' Usage: distribute_parts.sh -d DATA_DIR [-n N_TARGETS] [--apply] [--dry-run] [--manifest FILE] distribute_parts.sh -d DATA_DIR --revert [--dry-run] [--manifest FILE] Options: -d, --data-dir Path to directory 
  containing subdirectories (required) -n, --n-targets Number of target parts to create (default: 8) --apply Apply partitioning (default) --revert Revert using manifest --manifest FILE Manifest path (default: 
  DATA_DIR/.partition_manifest.tsv) --dry-run Print actions but do not modify filesystem -h, --help Show help
Manifest format (TSV): <original_absolute_path> <new_absolute_path> Notes: - Only immediate subdirectories of DATA_DIR are considered (depth=1). - Excludes: 1_flat_field and part* directories. - Revert only moves directories that 
  are present at the recorded 'new' paths.
EOF
}
data_dir="" n_targets=8 dry_run=0 mode="apply" manifest=""
# --- argument parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do case "$1" in -d|--data-dir) data_dir="${2:-}"; shift 2 ;; -n|--n-targets) n_targets="${2:-}"; shift 2 ;; --apply) mode="apply"; shift ;; --revert) mode="revert"; shift ;; --manifest) manifest="${2:-}"; 
      shift 2 ;;
    --dry-run) dry_run=1; shift ;; -h|--help) usage; exit 0 ;; *) echo "Error: unknown argument: $1" >&2 usage; exit 2 ;; esac done
# --- validation ---------------------------------------------------------------
if [[ -z "$data_dir" ]]; then echo "Error: --data-dir is required." >&2 usage; exit 2 fi if [[ ! -d "$data_dir" ]]; then echo "Error: data_dir does not exist or is not a directory: $data_dir" >&2 exit 2 fi
# Normalize to absolute path for manifest stability
data_dir="$(cd "$data_dir" && pwd)" if [[ -z "$manifest" ]]; then manifest="${data_dir}/.partition_manifest.tsv" fi run() { if [[ "$dry_run" -eq 1 ]]; then printf '[DRY-RUN] %q' "$1" shift for arg in "$@"; do printf ' %q' "$arg"; 
    done printf '\n'
  else "$@" fi
}
# --- revert ------------------------------------------------------------------
if [[ "$mode" == "revert" ]]; then if [[ ! -f "$manifest" ]]; then echo "Error: manifest not found: $manifest" >&2 exit 2 fi
  # Read manifest lines: orig<TAB>new We require absolute paths in the manifest; apply mode writes those.
  echo "Reverting using manifest: $manifest" while IFS=$'\t' read -r orig new; do [[ -z "${orig:-}" || -z "${new:-}" ]] && continue
    # Only attempt revert if the 'new' path currently exists
    if [[ ! -e "$new" ]]; then echo "Warning: recorded destination missing, skipping: $new" >&2 continue fi
    # Safety: refuse to overwrite existing original path
    if [[ -e "$orig" ]]; then echo "Error: original path already exists; refusing to overwrite: $orig" >&2 echo "Resolve manually (e.g., move/rename it) and rerun revert." >&2 exit 2 fi
    # Ensure original parent exists
    run mkdir -p "$(dirname "$orig")" run mv "$new" "$orig" done < "$manifest" echo "Revert complete." exit 0 fi
# --- apply -------------------------------------------------------------------
if ! [[ "$n_targets" =~ ^[0-9]+$ ]] || [[ "$n_targets" -lt 1 ]]; then echo "Error: n_targets must be a positive integer (got: $n_targets)" >&2 exit 2 fi flat_field="${data_dir}/1_flat_field" if [[ ! -d "$flat_field" ]]; then echo 
  "Error: expected directory not found: $flat_field" >&2 exit 2
fi
# Refuse to overwrite an existing manifest in apply mode (unless user deletes it)
if [[ -e "$manifest" ]]; then echo "Error: manifest already exists: $manifest" >&2 echo "Refusing to overwrite. If this is intentional, delete/rename it first." >&2 exit 2 fi
# Create part directories
for ((i=1; i<=n_targets; i++)); do part_dir="${data_dir}/part${i}" if [[ -e "$part_dir" && ! -d "$part_dir" ]]; then echo "Error: target exists but is not a directory: $part_dir" >&2 exit 2 fi run mkdir -p "$part_dir" done
# Copy 1_flat_field into each part
for ((i=1; i<=n_targets; i++)); do part_dir="${data_dir}/part${i}" dest="${part_dir}/1_flat_field" if [[ -e "$dest" ]]; then echo "Warning: $dest already exists; skipping copy into part${i}." >&2 continue fi run cp -a 
  "$flat_field" "$part_dir/"
done
# Collect top-level dirs to distribute (absolute paths), exclude 1_flat_field and part*
mapfile -d '' dirs_to_move < <( find "$data_dir" -mindepth 1 -maxdepth 1 -type d -print0 \
  | while IFS= read -r -d '' p; do
      base="$(basename "$p")" [[ "$base" == "1_flat_field" ]] && continue [[ "$base" == part[0-9]* ]] && continue printf '%s\0' "$p" done \
  | sort -z
) n_dirs="${#dirs_to_move[@]}" if [[ "$n_dirs" -eq 0 ]]; then echo "No directories to distribute (besides 1_flat_field and part*)." >&2
  # Still create an empty manifest for completeness, or just exit. Here we exit without creating a manifest.
  exit 0 fi
# Create manifest file (atomic write)
tmp_manifest="${manifest}.tmp.$$" if [[ "$dry_run" -eq 1 ]]; then echo "[DRY-RUN] would write manifest to: $manifest" else
  : > "$tmp_manifest"
fi
# Distribute round-robin and record moves
idx=0 for src in "${dirs_to_move[@]}"; do base="$(basename "$src")" target=$(( (idx % n_targets) + 1 )) dest_dir="${data_dir}/part${target}" dest="${dest_dir}/${base}" if [[ -e "$dest" ]]; then echo "Error: destination already 
    exists: $dest" >&2 echo "Refusing to overwrite. Resolve manually or rename one side." >&2 exit 2
  fi if [[ "$dry_run" -eq 1 ]]; then echo "[DRY-RUN] manifest: ${src}	${dest}" else printf '%s\t%s\n' "$src" "$dest" >> "$tmp_manifest" fi run mv "$src" "$dest_dir/" idx=$((idx + 1)) done
# Finalize manifest
if [[ "$dry_run" -ne 1 ]]; then run mv "$tmp_manifest" "$manifest" echo "Manifest written: $manifest" fi echo "Done." echo "data_dir : $data_dir" echo "n_targets : $n_targets" echo "moved_dirs : $n_dirs" if [[ "$dry_run" -eq 1 
]]; then
  echo "mode : DRY-RUN (no changes made)" fi
