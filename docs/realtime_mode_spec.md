# Realtime Mode Specification (`src/realtime`)

## 1) Purpose

Define a new realtime acquisition + processing mode that:
- acquires images continuously,
- processes images immediately in memory,
- reuses existing offline pipeline functions where safe,
- does **not** break offline batch analysis (`scripts/run_pipeline.py`) or existing batch acquisition (`src/acquire`).

Execution entrypoint:
- `scripts/run_realtime.py --config <path_to_config.yml> [--plate N --well N]`

## 2) Hard Constraints

1. Realtime acquisition uses only the simple acquisition path (same concept as `ws_acquire_single`).
2. No GUI-based well selector and no stage controller in realtime mode.
3. Plate and well values are manual user inputs and must be embedded in output filenames.
4. Realtime settings live in `config.yml` only.
5. `config.yml` remains untouched and continues serving offline mode only.
6. Realtime defaults to grayscale acquisition.
7. Realtime runs simplified stages 1–3 only.
8. Stage 4 is completely omitted in realtime mode.
9. Keep memory bounded so operation is reliable on mid-level local PCs.

## 3) Scope Boundaries

### In scope
- New module tree under `src/realtime`.
- New script `scripts/run_realtime.py`.
- New realtime config section in `config.yml`.
- Controlled reuse of existing Stage 1/2/3 internals as callable utilities.

### Out of scope
- Any behavior change in offline `scripts/run_pipeline.py`.
- Any required changes to `src/acquire/stage_controller.py` or well selector GUI flow.
- Realtime Stage 4 logic.
- Full track-level temporal analytics (particle tracking remains disabled for now).

## 4) Proposed `src/realtime` Module Layout

Create dedicated orchestration and adapters rather than modifying offline stage wrappers directly.

- `src/realtime/__init__.py`
- `src/realtime/config.py`
  - parse/validate realtime config from `config.yml`
- `src/realtime/session.py`
  - run/session metadata, plate/well tagging, output directory setup
- `src/realtime/acquisition.py`
  - simple camera pull loop (grayscale by default)
- `src/realtime/background_buffer.py`
  - time-aged background image state machine (`BGI_MIN_AGE`, `BGI_MAX_AGE`)
- `src/realtime/stage1_rt.py`
  - in-memory simplified Stage 1 processing
- `src/realtime/stage2_rt.py`
  - in-memory simplified Stage 2 processing
- `src/realtime/stage3_rt.py`
  - lightweight quality filter hooks (artifact/boundary)
- `src/realtime/persistence.py`
  - save accepted particle crops and minimal runtime logs
- `src/realtime/monitoring.py`
  - runtime counters, latency, dropped-frame stats
- `src/realtime/orchestrator.py`
  - end-to-end loop coordinating acquisition, background aging, processing, and cleanup

Design rule: realtime code calls reusable lower-level functions from `src/core` and selected helper functions from stage modules, but does not call offline “full-stage wrapper” functions as black boxes.

## 5) Reuse Strategy (What to Reuse vs Replace)

## Stage 1 realtime simplification

Reuse:
- flat-field loading and correction (`load_flat_field_assets`, `apply_flat_field_correction`)
- subtraction (`subtract_images`)
- candidate detection on subtracted image (`analyze_subtracted_image`)
- local refinement (`refine_with_local_otsu`)

Replace/omit:
- filename parsing/filtering (`_parse_image_filenames`, `_filter_image_df`) → replaced by acquisition metadata created at capture time
- offline FG/BG selector (`select_background_frame`) → replaced by time-aged RAM background buffer
- `prepare_masked_foreground` in realtime → skipped
- fusion step (`_compare_and_fuse_particle_dfs`) → optional bypass; default is single subtraction path output
- particle tracking (`run_particle_tracking`) → disabled

Realtime Stage 1 output contract:
- an in-memory dataframe for the **current frame** containing particle detections and coordinates needed by Stage 2 cropping.

## Stage 2 realtime simplification

Reuse:
- `normalized_crop_config`
- crop generation logic (`crop_particles`) adapted for in-memory source image
- segmentation (`segment_particles`)
- feature calculations using:
  - quality features (`quality_calculator`)
  - morphology features (`morphology_calculator`)

Omit:
- shape descriptor feature extraction (`shape_descriptors`)
- track-centric focus logic (`select_focused_tracks`)
- track-wise “largest by area” picker (`select_final_particles_by_area`)
- Stage 2 table artifact emission (`emit_stage2_artifacts`)

Replace with simple realtime filters (configurable):
- focus threshold: `variance_laplacian > min_focus_score`
- area threshold: `main_area_filled > min_particle_area`

Realtime Stage 2 output contract:
- in-memory rows for current-frame particles that passed simple thresholds.
- crop image arrays + target filenames for persistence.

## Stage 3 realtime simplification

Reuse only quality checks for:
- black mask artifacts
- boundary-touching crops

Placement change:
- run these checks before persistence to avoid writing bad crops.

Realtime Stage 3 output contract:
- list/dataframe of accepted crops to save.

## 6) Background Image Aging Algorithm (RAM-Bounded)

Maintain at most three full-size frames in RAM:
- `bg_image_aged` (used for subtraction)
- `bg_image_new` (candidate background that is aging)
- `current_fg_image` (currently processed frame)

Config:
- `BGI_MIN_AGE_S` default `2.0`
- `BGI_MAX_AGE_S` default `4.0`

Lifecycle:
1. At start, first captured frame becomes `bg_image_new` with timestamp `t_new`.
2. Once age(`bg_image_new`) >= `BGI_MIN_AGE_S`, promote it to `bg_image_aged` if no aged frame exists.
3. Use `bg_image_aged` for subtraction of each current frame while valid.
4. Before `bg_image_aged` expires, preload next candidate:
   - when age(`bg_image_aged`) >= (`BGI_MAX_AGE_S - BGI_MIN_AGE_S`), capture one frame as next `bg_image_new` if not already present.
5. When age(`bg_image_new`) >= `BGI_MIN_AGE_S`, rotate:
   - previous `bg_image_aged` discarded,
   - `bg_image_new` becomes new `bg_image_aged`,
   - clear `bg_image_new` slot.
6. Never use any background older than `BGI_MAX_AGE_S`.

Fallback behavior:
- if no valid `bg_image_aged` exists yet, skip subtraction-based processing for that frame and continue acquiring until a valid aged background is available.

## 7) Realtime Processing Flow (Per Frame)

1. Acquire grayscale frame + timestamp + frame index + manual plate/well context.
2. Update background buffer state.
3. If valid aged background exists:
   - flat-field correct FG and BG,
   - subtract,
   - run Stage 1 candidate detection + local refinement,
   - build current-frame Stage 1 dataframe.
4. Run simplified Stage 2:
   - crop from in-memory FG frame,
   - segment crops,
   - compute quality+morphology features,
   - apply focus and area thresholds.
5. Run simplified Stage 3 quality checks (artifact/boundary rejection).
6. Save accepted crops only.
7. Emit runtime counters/latency metrics.
8. Release per-frame arrays/dataframes from memory immediately after save/metrics update.

## 8) Output and Naming Rules

Saved artifacts in realtime mode:
- accepted particle crop images only (no full stage CSV tables by default).

Filename components must include:
- session timestamp,
- plate and well values provided manually,
- source frame index,
- particle identifier/crop coordinates.

Example naming pattern (spec-level, not code):
- `<session>_plate<plate>_well<well>_frame<idx>_particle<id>_X<x>_Y<y>.png`

## 9) `config.yml` Additions

Add a dedicated section, e.g. `realtime_mode:`

- `enabled` (bool)
- `default_grayscale` (bool, default true)
- `output_root_dir`
- `runtime:`
  - `target_fps`
  - `drop_frame_if_backlogged` (bool)
  - `queue_size`
  - Note: the per-frame processing budget is derived internally as
    `1000 / target_fps`; there is no separate `max_processing_latency_ms`
    knob. When frames exceed the budget (or are dropped), the runtime monitor
    emits a single rate-limited "falling behind" warning per logging interval,
    in both drop and blocking modes.
- `background:`
  - `bgi_min_age_s` (default 2.0)
  - `bgi_max_age_s` (default 4.0)
- `stage1:`
  - minimal threshold options needed by reused detection functions
- `stage2:`
  - crop config
  - `min_focus_score` (default example 100)
  - `min_particle_area` (default example 500)
- `stage3:`
  - enable/disable artifact and boundary filters
- `persistence:`
  - `save_accepted_crops` (bool)
  - optional JPEG/PNG format and compression settings
- `logging:`
  - periodic metric print interval

Requirement:
- all new realtime config keys must have validation and explicit defaults.

## 10) Performance and Reliability Guidance

Given 32 GB RAM + mid-range GPU workstation target:
- Keep processing CPU-first and deterministic; GPU optional and not required.
- Avoid retaining historical full-frame buffers beyond the defined 2–3 frame window.
- Pre-allocate/reuse small working objects where practical.
- Separate acquisition and processing via a bounded queue (size 1–3) to prevent RAM growth.
- If processing falls behind:
  - prefer dropping oldest unprocessed frame (configurable),
  - always keep acquisition responsive.
  - `runtime.drop_frame_if_backlogged: false` (current setting) instead loses no frame: the
    fixed-grid camera loop waits before its next lens command and starts a new time grid,
    so the frame rate drops while processing is behind.

Camera timing (`acquisition.camera.trigger_mode`, shared with the batch acquisition):
- `software_grid`: the camera source runs the batch fixed-grid trigger loop
  (`src/acquire/triggered_acquisition.py`) on a background thread at `runtime.target_fps`;
  every frame's `diopter` is the setting it was exposed under (block-ID mapping, camera
  timestamps in `FramePacket.camera_timestamp_ns`), which keeps the per-diopter background
  library consistent. `free_run`: the legacy loop. Without ETL control the camera free-runs.
- Latency of `submit_adaptive_center_diopter` / `request_repeat_same_diopter_once`: a request
  applies from the next lens command, which follows right after the current exposure. A
  request made while processing frame k therefore applies from frame k + 2 at the earliest
  (one to two frame intervals from the request to the first exposure on the new plan), plus
  however far processing runs behind acquisition.
- The session end calls `AcquisitionControl.request_stop()`, so the loop ends and restores
  free-run.

Runtime telemetry to expose:
- acquired fps,
- processed fps,
- mean/95th percentile per-frame latency,
- dropped frame count,
- accepted crop count.

## 11) Compatibility Guarantees

- No imports from `src/realtime` into offline stage modules required.
- Existing `scripts/run_pipeline.py` behavior remains unchanged.
- Existing `src/acquire` batch flows remain unchanged.
- Realtime entrypoint is isolated: `scripts/run_realtime.py`.

## 12) Implementation Phases

Phase 1 (skeleton + acquisition):
- config parsing, CLI, session naming, simple grayscale acquisition, background buffer.

Phase 2 (simplified Stage 1 in-memory):
- flat-field + subtraction + candidate detection + local refinement output dataframe.

Phase 3 (simplified Stage 2 + Stage 3 quality):
- crop, segment, quality+morphology features, thresholding, artifact/boundary rejection.

Phase 4 (persistence + telemetry + shutdown):
- save accepted crops, metrics, clean cancellation, memory cleanup.

Phase 5 (hardening):
- stress tests at target fps,
- config edge-case validation,
- fault tolerance for camera disconnect/reconnect behavior.

## 13) Acceptance Criteria

1. Realtime command starts and runs with manual plate/well input and grayscale default.
2. Background aging behavior follows min/max age constraints and memory cap.
3. Stage 1–3 simplified flow processes in-memory per frame with no Stage 4 usage.
4. Only accepted crops are saved.
5. Offline and batch acquisition commands behave exactly as before.
6. Runtime remains stable for extended runs without unbounded memory growth.
