# Real-time acquisition + analysis refactor strategy

## Context and current constraints

Today the workflow is batch-oriented:

1. `record/` acquires images and writes them to disk.
2. Analysis is started afterwards with `scripts/run_pipeline.py --input_batch_dir ...`.

This is robust for offline processing, but it introduces end-to-end latency and redundant I/O (write image -> read image again). The main refactor goal should be **streaming data between acquisition and analysis with bounded latency** while preserving fallback to disk when needed.

## Target architecture (incremental)

Build a small real-time orchestration layer instead of rewriting all stage code at once.

### 1) Event-driven pipeline between acquisition and analysis

Use a producer/consumer architecture:

- **Producer**: acquisition loop in `record/` emits `FramePacket` objects (metadata + image array + timestamps).
- **Queue/Buffer**: bounded in-memory queue (or shared-memory ring buffer).
- **Consumers**: stage workers process packets and emit `AnalysisResult` events.
- **Sink**: results logger/UI/CSV writer receives reduced outputs only.

This decouples camera timing from analysis timing and prevents acquisition jitter from analysis spikes.

### 2) Keep existing stage logic, add streaming adapters

Current stages in `src/stages/` are file-path oriented. Instead of rewriting immediately:

- Add thin adapter functions that accept in-memory frames/ROIs and call existing core functions where possible.
- Standardize on a dataclass contract, e.g.:
  - `FramePacket`: `sample_id`, `well_id`, `frame_id`, `timestamp`, `image`, optional `focus_dpt`.
  - `DetectionPacket`: candidate objects + lightweight masks/boxes.
  - `FeaturePacket`: per-object feature rows ready for filtering/classification.

This lets you migrate stage-by-stage.

### 3) Minimize disk writes in the hot path

For real-time mode:

- Write only:
  - final per-well result rows,
  - sparse debug snapshots (every N frames),
  - failure dumps when exceptions occur.
- Keep raw full-frame persistence optional via config flag (for reproducibility sessions).

Disk becomes a debug/trace channel, not the transport layer.

## Practical execution model (works on normal workstation hardware)

### Process split

A good default is **3-process design**:

1. **Acquisition process**
   - talks to camera + stage,
   - normalizes frame format,
   - pushes into bounded queue.
2. **Analysis process pool** (2-6 workers)
   - runs stage 1/2 and lightweight stage 3 decisions,
   - keeps ML models preloaded in worker init.
3. **Coordinator process**
   - merges outputs,
   - applies backpressure policy,
   - publishes incremental results.

Why this works:

- Avoids GIL contention for CPU-heavy OpenCV/skimage steps.
- Keeps acquisition responsive even when a few frames are expensive.

### Backpressure and latency budget

Define explicit target, e.g. **<500 ms median latency per frame** for early-stage metrics.

Implement queue policy:

- bounded queue size (e.g. 16-64 frame packets),
- if queue full:
  - drop oldest *unprocessed* frame for that well (not newest), or
  - switch to ROI-only mode temporarily.

Without backpressure, real-time systems silently become delayed batch systems.

### Prioritize "quick-look" first, "full-quality" second

Use two-tier inference:

- **Tier A (real-time path):** fast detection + core features + quick classification.
- **Tier B (deferred path):** expensive refinements run only for selected wells/objects.

Operators get immediate actionable feedback; high-fidelity output can catch up asynchronously.

## Recommended refactor phases

### Phase 0 (baseline instrumentation)

Before structural changes, measure:

- acquisition FPS,
- per-stage wall-clock times,
- queue wait times,
- end-to-end latency distribution.

Add simple timing probes around current stage entry points and acquisition loop.

### Phase 1 (streaming skeleton)

Add new module, e.g. `src/realtime/`:

- `contracts.py` (packet dataclasses),
- `queues.py` (bounded queue, counters, drop policy),
- `orchestrator.py` (main event loop),
- `adapters.py` (bridge from packet -> existing stage calls).

Add CLI mode:

- `scripts/run_realtime.py --config ... --wells A1,B1 --max-workers 4`.

Keep `run_pipeline.py` unchanged for offline mode.

### Phase 2 (stage-by-stage in-memory conversion)

Convert in this order:

1. Stage 1 detection (highest impact for immediate feedback).
2. Stage 2 feature extraction on detected ROIs only.
3. Stage 3 filtering/classification (batch CSV dependency removal).
4. Stage 4 biological analysis as periodic aggregator.

At each step, retain adapter fallback to file-based implementation to reduce risk.

### Phase 3 (hardware-aware optimization)

- Preallocate numpy buffers and reuse arrays.
- Crop early; avoid full-frame expensive transforms when ROI is enough.
- Use mixed precision / smaller input resolution for ML where accuracy permits.
- Pin worker affinity only if profiling indicates contention.
- Move only the hottest kernels to GPU (if available); avoid full rewrite.

### Phase 4 (operator usability)

Provide a lightweight live dashboard:

- current well,
- queue depth,
- recent detections/classifications,
- latency/FPS health indicators.

This is essential for trust and for tuning field experiments.

## Concrete coding recommendations for this repository

1. **Separate side effects from compute in `record/`**
   - Refactor acquisition functions so one function returns in-memory frames, another handles optional persistence.
2. **Introduce unified config for real-time mode**
   - Add `realtime:` section in `config/pipeline_config.yml` for queue size, worker count, drop policy, debug snapshot interval.
3. **Add model warmup hooks**
   - Initialize ML models once in worker startup to avoid repeated load overhead.
4. **Add deterministic packet IDs**
   - Include `(plate_id, well_id, frame_id)` in every packet and result row for traceability.
5. **Build a replay harness**
   - Feed recorded frame sequences through realtime pipeline for regression testing without hardware attached.

## CPU-first efficiency tactics (no supercomputer required)

- Process every 2nd/3rd frame when scene change is low.
- Trigger deep analysis only when motion/focus/foreground heuristics pass threshold.
- Keep image format compact (grayscale where possible).
- Prefer vectorized ops and avoid repeated conversions between PIL/OpenCV/skimage formats.
- Cache per-well static artifacts (background estimates, masks, calibration terms).

These usually deliver the biggest speed gains before any hardware upgrade.

## Reliability and safety

For lab hardware control, include:

- watchdog for stalled stage/camera operations,
- queue overflow alarms,
- graceful degradation modes (reduced frame rate, reduced stage set),
- atomic periodic checkpointing of summary outputs.

If real-time path fails, the system should continue with minimal offline-compatible logging rather than aborting the experiment.

## Suggested first milestone (2-3 weeks)

Deliverable:

- Real-time pilot for a subset of wells (e.g., 2-4 wells),
- live stage-1 detection + preliminary stage-3 classification,
- median latency <1 s with bounded memory,
- optional raw frame save disabled by default.

Acceptance criteria:

- no dropped camera control commands,
- stable runtime for >30 min acquisition,
- output parity (within tolerance) against offline pipeline on replayed data.

## Risks and mitigations

- **Risk:** Existing stage code tightly coupled to CSV/files.
  - **Mitigation:** adapter layer + dual-mode interfaces until migration complete.
- **Risk:** Latency spikes from occasional hard frames.
  - **Mitigation:** bounded queues + drop/defer policy + tiered analysis.
- **Risk:** Hard to debug without saved images.
  - **Mitigation:** sampled debug snapshots and on-error frame dumps.

## Bottom line

The most efficient path is not a big-bang rewrite. Implement a **streaming orchestrator + adapters**, move one stage at a time to in-memory processing, and enforce explicit latency/backpressure controls. That gives near-real-time feedback on commodity hardware while preserving your proven offline pipeline as a safety net.
