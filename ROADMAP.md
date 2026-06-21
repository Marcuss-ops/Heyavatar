# Heyavatar Roadmap

This file records the **frozen / deprecated / future** paths that
Heyavatar's slimming plan (`docs/REPOSITORY_SLIMMING_PLAN.md`) explicitly
defers. Every entry has a re-introduction gate so we don't keep rebuilding
abstractions without a concrete production requirement.

> Rule for re-introduction: the project may re-engage one of these items
> only when its gate is satisfied **and** a second concrete implementation
> (or a measured production need) exists. Until then the corresponding
> code stays out of the active runtime.

---

## 1 · Frozen / deprecated (re-vivible, not active in MVP)

### Distributed `WorkerPool` + heartbeat tiers
- **Why frozen:** deployment target is `1 API · 1 Redis · 1 GPU worker · 1 encoder path`.
- **What stays:** minimal worker health record `{worker_id, last_seen, engine_id, state}`.
- **What is gone (post Change 3):**
  - `api/app/state.py::lifespan` no longer starts the periodic
    `_start_worker_pool_sync_gatherer` thread.
  - `workers/gpu_worker/worker.py::_start_redis_heartbeat` is gated
    behind `Settings.enable_distributed_heartbeat` (default `False`).
  - `WorkerPool.sync_from_redis` itself is preserved (unit tests
    rely on the wire schema) but no production code path invokes it.
- **Re-introduction gate:** a second API process is wired AND a Redis
  cross-process deploy is in production.
- **Removed in:** Change 3. Re-introduce by setting
  `HEYAVATAR_ENABLE_DISTRIBUTED_HEARTBEAT=1` and re-enabling the
  gatherer call in `lifespan`.

### `EchoMimic` engine
- **Why frozen:** MVP uses LivePortrait+Musetalk; license stays Apache-2.0
  but no real adapter ships in MVP.
- **What stays:** provider directory `providers/echomimic/` kept on
  disk and the `EngineId.ECHO_MIMIC` enum value preserved for
  backwards-compatible engine-id parsing in registry / database
  payloads.
- **What is gone (post Change 3):** the registration in
  `providers/__init__.py::PROVIDERS`. A caller that asks
  `get_provider(EngineId.ECHO_MIMIC)` now raises `KeyError` with a
  freeze message; old callers should branch on the supported engines
  (`LIVE_PORTRAIT`, `MUSE_TALK`).
- **Re-introduction gate:** a real `EchoMimicAdapter` is shipped AND
  it adds visible quality beyond LivePortrait+Musetalk.
- **Removed in:** Change 3 (commit message: `slim(repo): change 3`).
  Re-introduce by re-registering in `PROVIDERS` and shipping a
  real-mode adapter body.

### Full `OpenTelemetry` tracing stack
- **Why frozen:** OTLP exporters + W3C propagation add a runtime
  dependency the MVP doesn't need.
- **What stays:** structured JSON logs with
  `job_id`, `avatar_id`, engine id, stage duration, GPU seconds, terminal
  state, failure reason. **W3C traceparent inject/extract on the queue
  payload** (`propagation.py`) is also kept because it lets a
  re-introduced collector link child spans to the parent FastAPI
  request without changing the API surface; it no-ops when the
  OpenTelemetry SDK is not installed.
- **What is gone (post Change 3):** `setup_tracing` no longer wires
  a `TracerProvider` + `BatchSpanProcessor` + `OTLPSpanExporter` in
  any default path. The module docstring claims the freeze.
- **Re-introduction gate:** an OTLP collector is wired by ops AND
  structured logs prove insufficient for at least one operationally
  critical question.
- **Removed in:** Change 3.

### Multi-quality tiers (`Express` / `Studio` / `Premium`)
- **Why frozen:** no benchmark data distinguishes them. Plan §1 says
  MVP uses one `standard` profile: LivePortrait+Musetalk, 25 FPS,
  256×256 face ROI, H.264 output, prerecorded body template.
- **What stays:** `src/domain/enums.py::Tier` enum is kept for forward
  compatibility but the router collapses to one profile.
- **Re-introduction gate:** real-world benchmark data exists that
  distinguishes at least two tiers.
- **Removed in:** Change 3.

### SadTalker `Audio2Motion` neural audio bridge
- **Why frozen:** MuseTalk is the current lip-sync path; body templates
  already provide head and upper-body motion. SadTalker's weights are
  trained on LRW+VoxCeleb (non-commercial research corpora), so
  `liveportrait-human-v1.commercial_use` stays `false` until a
  commercially-licensed bridge ships.
- **What stays:** `providers/liveportrait/audio_bridge/sadtalker.py`
  exists behind `HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural`, default `dsp`.
- **Re-introduction gate:** DSP bridge proves inadequate on production
  audio mix AND commercially-licensed weights are available.
- **Removed in:** Change 3 (frozen, not deleted).

### S3 / multi-object-store backend
- **Why frozen:** local filesystem storage (`avatar_packs/`, `body_templates/`,
  `captures/`) covers MVP.
- **What stays:** `FsObjectStore`; the `ObjectStore` ABC keeps the
  interface narrow so an `S3ObjectStore` would be a drop-in subclass.
- **What is gone (post Change 3):** `Settings.object_store_backend`
  tightened to `Literal["fs"]`; the `s3_endpoint_url` and
  `s3_bucket` settings are removed. `build_object_store` rejects
  any non-`fs` backend explicitly.
- **Re-introduction gate:** cross-region storage is required AND a
  signed-URL helper for the API is in scope.
- **Removed in:** Change 3.

---

## 2 · Deferred (precondition: deterministic multi-template timeline)

### Semantic gesture planner (LLM-style)
- **Why deferred:** no production 3D rig / universal retargeting
  consumes the planner's output yet. Plan §6 says: prove that
  Heyavatar can concatenate multiple body templates while keeping
  video frames, face transforms, face masks, neck masks, timestamps,
  audio, and final compositing frame-aligned BEFORE adding an LLM.
- **Manual timeline (MVP replacement):**
  ```json
  {
    "segments": [
      {"gesture_id": "idle",          "duration_seconds": 3.0},
      {"gesture_id": "explain_both",  "duration_seconds": 2.0},
      {"gesture_id": "idle",          "duration_seconds": 3.0}
    ]
  }
  ```
- **Re-introduction gate:** a real rigged avatar + a real retargeter +
  a real renderer + at least one end-to-end test using canonical motion
  data exists.
- **Removed in:** Change 1 (deleted `contracts/gesture_planner.py`,
  `src/motion/{composer,resolver,registry,cache_keys}.py`,
  `providers/motion_extraction/mediapipe/{gesture_planner,motion_extractor}.py`,
  `src/application/plan_video.py`).

### Universal 3D motion model (MotionRepository)
- **Why deferred:** prerecorded body templates carry all the motion the
  current MVP needs. A universal MotionClip with
  root translation / body rotations / left-hand rotations /
  right-hand rotations / head rotations / motion phases has no
  production consumer.
- **Re-introduction gate:** a real 3D rig is wired end-to-end with at
  least one canonical motion dataset and one end-to-end test.
- **Removed in:** Change 1.

---

## 3 · Pending (Change 4 of the slimming plan)

### Deterministic multi-template timeline
- **Goal:** at least three body clips concatenated with masks, transforms,
  timestamps, audio, and compositing all frame-aligned.
- **Acceptance:**
  - three+ body clips can be concatenated;
  - video, masks, and transform arrays stay frame-aligned;
  - MuseTalk and compositing operate on the combined timeline;
  - final audio and video durations match.
- **Status:** not started. The `BodyTemplate` dataclass from
  `docs/REPOSITORY_SLIMMING_PLAN.md` §4 will be introduced in this change.

---

## 4 · Pre-existing follow-ups from `CHANGELOG.md [Unreleased]`
- Distributed worker heartbeat via Redis wired by worker-side
  publisher (currently only consumer is wired). Re-introduce with §1.
- Real MediaPipe swap to flip `liveportrait-human-v1.commercial_use`
  to `true` (gated by `tests/smoke/test_real_gpu/test_pipeline.py`).
- Real-mode `EchoMimicAdapter` (currently raises `NotImplementedError`).
