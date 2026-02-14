#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  distribute_parts.sh -d DATA_DIR [-n N_TARGETS] [--apply] [--dry-run] [--manifest FILE]"
  echo "  distribute_parts.sh -d DATA_DIR --revert [--dry-run] [--manifest FILE]"
  echo
  echo "Options:"
  echo "  -d, --data-dir     Path to directory containing subdirectories (required)"
  echo "  -n, --n-targets    Number of target parts to create (default: 8)"
  echo "  --apply            Apply partitioning (default)"
  echo "  --revert           Revert using manifest"
  echo "  --manifest FILE    Manifest path (default: DATA_DIR/.partition_manifest.tsv)"
  echo "  --dry-run          Print actions but do not modify filesystem"
  echo "  -h, --help         Show help"
}

data_dir=""
n_targets=8
dry_run=0
mode="apply"
manifest=""

# ------------------------- argument parsing -----------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--data-dir)
      data_dir="${2:-}"; shift 2 ;;
    -n|--n-targets)
      n_targets="${2:-}"; shift 2 ;;
    --apply)
      mode="apply"; shift ;;
    --revert)
      mode="revert"; shift ;;
    --manifest)
      manifest="${2:-}"; shift 2 ;;
    --dry-run)
      dry_run=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

# --------------------------- validation ---------------------------------------
if [[ -z "$data_dir" ]]; then
  echo "Error: --data-dir is required." >&2
  usage
  exit 2
fi

if [[ ! -d "$data_dir" ]]; then
  echo "Error: data_dir does not exist or is not a directory: $data_dir" >&2
  exit 2
fi

if ! [[ "$n_targets" =~ ^[0-9]+$ ]] || [[ "$n_targets" -lt 1 ]]; then
  echo "Error: n_targets must be a positive integer (got: $n_targets)" >&2
  exit 2
fi

# Normalize to absolute path for manifest stability
data_dir="$(cd "$data_dir" && pwd)"

if [[ -z "$manifest" ]]; then
  manifest="${data_dir}/.partition_manifest.tsv"
fi

run() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf '[DRY-RUN] %q' "$1"
    shift
    for arg in "$@"; do printf ' %q' "$arg"; done
    printf '\n'
  else
    "$@"
  fi
}

# ----------------------------- revert -----------------------------------------
if [[ "$mode" == "revert" ]]; then
  if [[ ! -f "$manifest" ]]; then
    echo "Error: manifest not found: $manifest" >&2
    exit 2
  fi

  echo "Reverting using manifest: $manifest"
  while IFS=$'\t' read -r orig new; do
    [[ -z "${orig:-}" || -z "${new:-}" ]] && continue

    if [[ ! -e "$new" ]]; then
      echo "Warning: recorded destination missing, skipping: $new" >&2
      continue
    fi

    if [[ -e "$orig" ]]; then
      echo "Error: original path already exists; refusing to overwrite: $orig" >&2
      exit 2
    fi

    run mkdir -p "$(dirname "$orig")"
    run mv "$new" "$orig"
  done < "$manifest"

  echo "Revert complete."
  exit 0
fi

# ------------------------------ apply -----------------------------------------
flat_field="${data_dir}/1_flat_field"
if [[ ! -d "$flat_field" ]]; then
  echo "Error: expected directory not found: $flat_field" >&2
  exit 2
fi

# Refuse to overwrite an existing manifest in apply mode
if [[ -e "$manifest" ]]; then
  echo "Error: manifest already exists: $manifest" >&2
  echo "Refusing to overwrite. Delete/rename it if you intend to re-partition." >&2
  exit 2
fi

# Create part directories
for ((i=1; i<=n_targets; i++)); do
  part_dir="${data_dir}/part${i}"
  if [[ -e "$part_dir" && ! -d "$part_dir" ]]; then
    echo "Error: target exists but is not a directory: $part_dir" >&2
    exit 2
  fi
  run mkdir -p "$part_dir"
done

# Copy 1_flat_field into each part directory
for ((i=1; i<=n_targets; i++)); do
  part_dir="${data_dir}/part${i}"
  dest="${part_dir}/1_flat_field"
  if [[ -e "$dest" ]]; then
    echo "Warning: $dest already exists; skipping copy into part${i}." >&2
    continue
  fi
  run cp -a "$flat_field" "$part_dir/"
done

# Collect top-level dirs to distribute (exclude 1_flat_field and part*)
mapfile -d '' dirs_to_move < <(
  find "$data_dir" -mindepth 1 -maxdepth 1 -type d -print0 \
  | while IFS= read -r -d '' p; do
      base="$(basename "$p")"
      [[ "$base" == "1_flat_field" ]] && continue
      [[ "$base" == part[0-9]* ]] && continue
      printf '%s\0' "$p"
    done \
  | sort -z
)

n_dirs="${#dirs_to_move[@]}"
if [[ "$n_dirs" -eq 0 ]]; then
  echo "No directories to distribute (besides 1_flat_field and part*)." >&2
  exit 0
fi

tmp_manifest="${manifest}.tmp.$$"
if [[ "$dry_run" -eq 1 ]]; then
  echo "[DRY-RUN] would write manifest to: $manifest"
else
  : > "$tmp_manifest"
fi

# Distribute round-robin and record moves
idx=0
for src in "${dirs_to_move[@]}"; do
  base="$(basename "$src")"
  target=$(( (idx % n_targets) + 1 ))
  dest_dir="${data_dir}/part${target}"
  dest="${dest_dir}/${base}"

  if [[ -e "$dest" ]]; then
    echo "Error: destination already exists: $dest" >&2
    exit 2
  fi

  if [[ "$dry_run" -eq 1 ]]; then
    echo "[DRY-RUN] manifest: ${src}	${dest}"
  else
    printf '%s\t%s\n' "$src" "$dest" >> "$tmp_manifest"
  fi

  run mv "$src" "$dest_dir/"
  idx=$((idx + 1))
done

if [[ "$dry_run" -ne 1 ]]; then
  run mv "$tmp_manifest" "$manifest"
  echo "Manifest written: $manifest"
fi

echo "Done."
echo "data_dir   : $data_dir"
echo "n_targets  : $n_targets"
echo "moved_dirs : $n_dirs"
if [[ "$dry_run" -eq 1 ]]; then
  echo "mode       : DRY-RUN (no changes made)"
fi
