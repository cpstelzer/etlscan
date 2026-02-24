# Refactor Specification: `src/acquire` Integration into Well Plate Analyzer

## 1) Scope and non-negotiable constraints

This specification defines a **structure-only refactor** for `src/acquire` to improve integration with the rest of the repository while preserving behavior.

### Hard constraints

- **No functionality changes** in acquisition behavior (`src/acquire`) or analysis behavior (pipeline under `src/stages`, `src/core`, `scripts/run_pipeline.py`).
- Keep all existing acquisition workflows operational:
  1. multi-well acquisition with XY stage (`ws_acquire.py`),
  2. experiment sweeps (`ws_acquire_experiment.py`),
  3. single-sample acquisition without XY (`ws_acquire_single.py`).
- Preserve existing output naming conventions and folder format used by downstream code (`YYYYMMDD_plateX_wellY`).
- Keep hardware-specific logic (Zaber, Pylon, Optotune) untouched in effect; only move/reorganize code.

## 2) Current-state review (integration findings)

### 2.1 Standalone layout inside package tree

`src/acquire` still behaves like a previously standalone app:

- script-first entry points with duplicated orchestration (`ws_acquire.py`, `ws_acquire_experiment.py`, `ws_acquire_single.py`),
- local-module import style (`from wellscanner_acquisition_module import ...`) rather than package-relative imports,
- acquisition-only config and environment files colocated with runtime modules.

### 2.2 High duplication among runner scripts

`ws_acquire.py` and `ws_acquire_experiment.py` duplicate large blocks:

- plate normalization and coordinate generation,
- GUI well selection,
- Zaber init/homing/movement helpers,
- camera color detection and run-loop orchestration,
- experiment summary logging.

This duplication is the main integration blocker and maintenance risk.

### 2.3 Coupling points with rest of repo

Downstream analysis expects sample folder and naming conventions that acquisition already produces (plate/well/date parsing in utilities). Integration should therefore focus on module boundaries and reuse, not behavior changes.

## 3) Target architecture (no behavior change)

Refactor `src/acquire` into a small internal package with explicit layers:

1. **Domain/config layer** (pure data + validation)
2. **Hardware adapter layer** (Zaber/Pylon/Optotune wrappers)
3. **Workflow orchestration layer** (single, multi-well, parameter sweep)
4. **Thin CLI entrypoints** (current script names kept as compatibility shims)

Suggested module split:

- `src/acquire/config.py` – load/validate acquisition config and plate definitions.
- `src/acquire/plate_layout.py` – coordinate generation for 24/96 layouts.
- `src/acquire/well_selector_gui.py` – Tkinter selection UI.
- `src/acquire/stage_controller.py` – Zaber initialization, homing, movement utilities.
- `src/acquire/camera_capabilities.py` – sensor-mode/capability checks.
- `src/acquire/experiment_logging.py` – summary-log init/append.
- `src/acquire/workflows.py` – reusable workflow functions:
  - `run_multiwell_workflow(...)`,
  - `run_experiment_sweep_workflow(...)`,
  - `run_single_well_workflow(...)`.
- Keep `src/acquire/wellscanner_acquisition_module.py` as the core low-level acquisition engine (behavior preserved).

## 4) Compatibility and migration strategy

### 4.1 Entrypoint compatibility

Keep existing scripts with current names:

- `ws_acquire.py`
- `ws_acquire_experiment.py`
- `ws_acquire_single.py`

Convert them to thin wrappers calling new workflow functions so users do not need to change habits/shortcuts.

### 4.2 Import-path hardening

Normalize imports to package-relative/absolute package paths, e.g. `from src.acquire...` or explicit relative imports within `src/acquire`, to avoid path fragility when invoked from different working directories.

### 4.3 Output contract protection

Treat the following as frozen contracts during refactor:

- sample directory naming pattern (`YYYYMMDD_plateX_wellY`),
- frame filename conventions consumed by analysis utilities,
- per-run metadata/summary CSV field names where currently used.

## 5) Detailed refactor instructions (implementation order)

### Phase A — Baseline capture (before moving code)

1. Capture a reference map of:
   - all CLI arguments/prompts/interactive steps,
   - generated files/folders,
   - log CSV schemas.
2. Record one baseline dry-run trace per workflow mode (single, multi, experiment) using a mock/stub plan where hardware is unavailable.

### Phase B — Extract shared pure logic

1. Move plate-type normalization and settings merge to `config.py`.
2. Move coordinate generation to `plate_layout.py`.
3. Move summary-log helpers to `experiment_logging.py`.
4. Ensure function signatures preserve all current default values and implicit assumptions.

### Phase C — Extract hardware adapters

1. Move Zaber operations into `stage_controller.py` as a small class/function set.
2. Move camera capability checks into `camera_capabilities.py`.
3. Keep exceptions/messages and timing constants semantically identical.

### Phase D — Build workflows module

1. Implement orchestration functions that compose extracted helpers and call existing `run_acquisition(...)`.
2. Keep current wait times, prompts, and loop order exactly unchanged.
3. Encode experiment sweep options as parameters so legacy constants can remain defaults.

### Phase E — Convert legacy scripts to shims

1. Replace script internals with thin calls to workflow functions.
2. Preserve script-level constants and default config path behavior.
3. Preserve script execution UX (`if __name__ == "__main__": main()`).

### Phase F — Integration touchpoints with analysis pipeline

1. Add an optional helper in `scripts/` (non-breaking) documenting/copying acquisition output into `data/input/<batch>/...` expected by pipeline runner.
2. Do not modify stage implementations unless required for import cleanup.
3. If adding docs, keep existing commands valid.

## 6) Verification plan (must pass before merge)

### 6.1 Non-regression matrix

For each workflow mode (single/multi/experiment), verify:

- same config load behavior,
- same coordinate CSV content for same selections,
- same stage movement command sequence for identical selected wells,
- same acquisition call count/order/parameters,
- same output folder names and frame naming format,
- same summary CSV columns and semantics.

### 6.2 Contract tests to add

- Snapshot test for plate coordinate generation (24 and 96).
- Snapshot test for summary-log header and one appended row format.
- Unit test for plate-type normalization aliases.
- Unit test for folder/date/plate/well naming contract compatibility with downstream parser.
- Workflow smoke tests with mocked hardware adapters verifying call order only.

### 6.3 Manual validation checklist

- Run `ws_acquire_single.py` with real camera+lens and compare frame count/timing metadata.
- Run `ws_acquire.py` with at least two wells and confirm stage movement and run folders.
- Run `ws_acquire_experiment.py` and confirm randomized parameter-order behavior still occurs.
- Run analysis pipeline on newly acquired sample(s) and confirm stage outputs unchanged in structure.

## 7) Risks and mitigations

- **Risk:** subtle timing drift from refactor.
  - **Mitigation:** preserve sleep points and constants verbatim; no scheduler/concurrency changes.
- **Risk:** path/import breakage across invocation styles.
  - **Mitigation:** package-consistent imports + thin shims + execution tests from repo root and from script directory.
- **Risk:** accidental naming contract drift breaks stage parsing.
  - **Mitigation:** explicit contract tests for directory/filename formats.

## 8) Definition of done

Refactor is complete when:

1. all three acquisition modes run through same internal shared modules,
2. legacy script names still work unchanged,
3. downstream pipeline consumes acquisition outputs without adjustment,
4. tests/checklists above pass,
5. no behavior differences are observed in output artifacts or acquisition sequencing.
