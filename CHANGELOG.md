# Changelog

All notable changes to Heyavatar are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).## [Unreleased]

### Removed — Repository slimming plan, Change 1

Placeholder architecture that wrote text bytes to `.mp4` files instead of
producing real assets is removed. Speculative contracts without a
production consumer are removed. The corresponding placeholder workers
and the one smoke test that exercised them are removed. See
`docs/REPOSITORY_SLIMMING_PLAN.md` §4 for rationale.

- `contracts/motion_repository.py` — `MotionRepository` ABC + `MotionClip`
  dataclass; no production consumer.
- `contracts/face_renderer.py` — `FaceRenderer` ABC; duplicated by
  `contracts.avatar_engine.AvatarEngine.render_chunk`.
- `contracts/lip_sync_engine.py` — `LipSyncEngine` ABC; duplicated by
  the real `providers/musetalk/adapter/engine.MuseTalkAdapter`.
- `contracts/body_asset_provider.py` — `BodyAssetProvider` ABC; never
  used in MVP (replaced by a concrete `BodyTemplate` dataclass, see
  Change 4).
- `contracts/gesture_planner.py` — `GesturePlanner` ABC; semantic
  planner deferred (see `ROADMAP.md` §2).
- `providers/compositing/ffmpeg/compositor.py::FFmpegPoissonCompositor`
  — wrote `b"COMPOSITED VIDEO OUTPUT WITH POISSON BLENDING"`. The real
  compositor stays at `providers/compositing/opencv_face/compositor.py`.
- `providers/compositing/ffmpeg/quality_checker.py::RuleBasedQualityChecker`
  — returned `passed=True` with no real checks.
- `providers/lipsync/musetalk/lip_sync.py::MuseTalkLipSyncEngine` —
  wrote `b"LIPSYNCED FACE OUTPUT"`. The real lip-sync lives in
  `providers/musetalk/adapter/engine.py`.
- `providers/body_assets/prerecorded/template_provider.py::PrerecordedTemplateProvider`
  — wrote `b"PRERECORDED BODY CLIP"`.
- `providers/motion_extraction/mediapipe/{motion_extractor,gesture_planner}.py`
  — returned zero-motion clip / rule-based intents.
- `src/motion/{composer,resolver,registry,cache_keys}.py` and
  `src/motion/__init__.py` — only consumed by code that is also removed.
- `src/face/resolver.py::CachedFaceRenderer` — wrote `b"MOCK FACE TRACK"`.
- `src/body/{resolver,registry}.py` — `CachedBodyAssetProvider` wrote
  `b"MOCK VIDEO DATA"`.
- `src/application/plan_video.py::VideoPlanner`,
  `src/application/precompute_avatar.py::AvatarPrecomputer` — only
  used by the deleted placeholder workers.
- `workers/{face,composition,lipsync,quality,planner,avatar_precompute}_worker.py`
  — wired the placeholder providers/contracts above; nothing on the
  real render path invoked them.
- `tests/smoke/test_new_architecture.py` — exercised the deleted code
  paths.

### Added — Roadmap

- `ROADMAP.md` (top-level) — frozen/deferred paths from the slimming
  plan, each with a re-introduction gate. Mirrors
  `docs/REPOSITORY_SLIMMING_PLAN.md` §5/§6/§10.

### Added

- Multi-stage `Dockerfile` for `api` and `gpu-worker` services. The
  `MultiScaleDeformableAttention` CUDA op is built in a dedicated layer
  so business-logic edits don't invalidate the 5–10 min compile cache.
- `docker-compose.yml` wiring api + gpu-worker + encoder + redis with
  healthcheck-driven dependency ordering.
- `.dockerignore` for reproducible build contexts.
- `ops/docker_entrypoint.sh` dispatched by `SERVICE_ROLE` (api|gpu-worker|encoder).
- `ops/healthcheck.py` Http probe (no `requests` dependency).
- `WorkerPool.unregister()`, `WorkerPool.mark_in_flight()`, and
  `WorkerPool.sync_from_redis()` to support in-process tests and a
  future distributed heartbeat production path.
- `GpuWorker.pool` optional parameter: registers on `run()`, calls
  `mark_in_flight()` per job, calls `heartbeat(health=engine.health())`
  on each completion, and `unregister()` on shutdown.
- `providers/musetalk/adapter/checkpoints.py::verify()` — explicit
  policy for SHA pins (mock vs `HEYAVATAR_SKIP_SHA256_VERIFY=1` vs
  production-strict). Mirrors the LivePortrait manager.
- `tests/providers/test_import_shadowing.py` — regression for the
  dynamic-package import helper that dodges our own `src/` shadowing
  of the upstream LivePortrait tree.
- `tests/providers/test_musetalk_real_mode.py` — verifies the
  MuseTalk upstream-detection contract and the verify-policy branches.
- `tests/providers/test_worker_pool_in_process.py` — capacity tracking
  for in-process worker tests, including `pick_available` fallback walk.

### Changed

- `providers/musetalk/adapter/checkpoints.py` documents the TBD-SHA
  strategy explicitly and exposes `verify()` so audits don't require
  a network round-trip.

### Added — SadTalker Audio2Motion audio bridge (Task 2, gated)

- **`providers/liveportrait/audio_bridge/sadtalker.py`** (new) —
  wraps the upstream SadTalker `Audio2MotionModel`. Lazy-imports so
  CI without CUDA does NOT pay the import cost. **Never silently
  falls back to DSP** when `HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural` is
  selected — a `RuntimeError` is raised so the engine transitions to
  `EngineState.DEGRADED` and the orchestrator routes around broken
  workers.
- **`providers/liveportrait/audio_bridge/projection.py`** (new) —
  static 3DMM(50) → LivePortrait(21, 3) linear projection
  (identity placeholder; the calibrated W ships via a future
  GPU-worker calibration command).
- **`providers/liveportrait/audio_bridge/types.py`** — refreshed to
  add `SadTalkerCoefs` typed intermediate plus the new
  `DrivingSignals.backend` provenance field. Drops the now-internal
  `ChunkEnvelope` from the public boundary.
- **`providers/liveportrait/audio_bridge/bridge.py`** — refactored
  the public API from two functions (`envelopes_from_audio` +
  `envelopes_to_driving`) to a single `audio_to_driving(...)` that
  dispatches to backend based on
  `Settings.audio_bridge_backend` (`dsp` default; `neural` selects
  SadTalker). The legacy DSP path is preserved as
  `_audio_to_driving_dsp(...)` so the existing contract tests
  pass unchanged.
- **`src/core/config.py`** — new setting
  `audio_bridge_backend: Literal["dsp", "neural"] = "dsp"`, wired
  from env var `HEYAVATAR_AUDIO_BRIDGE_BACKEND`.
- **`providers/liveportrait/adapter/_render.py`** — replaces the two
  envelopes calls with a single `audio_to_driving(...)` call. The
  downstream `driving.frames` / `exp_d_flat` / `mouth_aperture`
  contracts are unchanged so the rendering batched warp_decode
  loop is untouched.
- **`tests/providers/test_sadtalker_projection.py`** (new) — three
  unit tests for the projection layer (`project_3dmm_to_keypoint_delta`,
  `mouth_aperture_from_jaw`, composite
  `sadtalker_coefs_to_driving_flat`).
- **`tests/providers/test_audio_bridge.py`** — refactored to the new
  single-API surface. Adds 3 backend-selection tests: default-falls-to-dsp,
  explicit-dsp, neural-raises-without-SadTalker, plus a stubbed-SadTalker
  happy path that produces the canonical `(T, 21, 3)` driving tensor.
- **`tests/smoke/test_real_gpu/_helpers.py:: _test_audio`** — now
  writes a 1.0s WAV composed of 0.5s silence + 0.5s active 1 kHz
  tone so the SSD-based mouth-sync assertion can partition frames
  into silence and active regions.
- **`tests/smoke/test_real_gpu/test_pipeline.py`** — adds the
  **release gate**: decodes the produced mp4 with ffmpeg, splits
  frames 50/50 into silence and active windows, computes per-frame
  SSD in the mouth ROI (lower 1/3 of the frame), and asserts
  `SSD_speech > 50.0 > SSD_silence`. The threshold rejects both
  the `0x111111` mock fallback AND the `0x330000` degraded fallback —
  a green run proves mouth-sync on the production GPU box.
- **`docs/MODEL_LICENSES.md`** — closes step 2 of the Production-Safe
  Path; documents SadTalker integration + non-commercial weights
  policy.
- **`requirements.txt`** — adds the (commented-out) `[audio-bridge-neural]`
  extra with SadTalker + librosa pinned for the GPU worker image.
- **`registry/models.yaml::liveportrait-human-v1.dependencies`** —
  adds `audio_bridge: sadtalker` entry with non-commercial weights
  license annotation; flag stays `false` accordingly.

### Added — MediaPipe Face Landmarker migration (Task 3, gated)

- **`providers/liveportrait/adapter/_mediapipe.py`** — thin wrapper
  around `mp.solutions.face_mesh` (Apache-2.0). Returns the largest
  face bbox `(x, y, w, h)` for an RGB numpy image; `None` if no
  faces detected.
- **`providers/liveportrait/adapter/_identity.py`** — face detection
  cascade now goes MediaPipe first, OpenCV Haar second, center crop
  last. The chosen detector is recorded in the avatar pack's
  `identity_meta.json` (`detector` field) so the orchestrator can
  audit provenance.
- **`tests/providers/test_mediapipe_detector.py`** — unit tests for
  the helper module + integration tests confirming `_identity.py`
  prefers MediaPipe when importable.
- **`tests/smoke/test_real_gpu/test_mediapipe_identity.py`** —
  real-GPU smoke that asserts `identity_meta.json: detector ==
  "mediapipe_face_mesh"`. This is the release gate for flipping
  `liveportrait-human-v1.commercial_use: true`.

### IMPORTANT — `commercial_use` gate NOT flipped yet

`registry/models.yaml::liveportrait-human-v1.commercial_use` remains
`false` until:

1. `tests/smoke/test_real_gpu/test_mediapipe_identity.py` passes on a
   real-GPU workstation with `mediapipe` installed.
2. `tests/contracts/test_avatar_engine_contract.py` runs cleanly in
   real (non-mock) mode.
3. The audio-bridge neural replacement is shipped (separate track). ✅
   done in this release (see below).
4. `tests/smoke/test_real_gpu/test_pipeline.py` (SSD-based mouth-sync
   assertion) passes on the production GPU box.

This release is the **code change** that closes steps (1) and (3),
and adds the gate for step (4). The flip happens once step (4) plus
a commercially-licensed audio bridge are in place.

### Follow-ups (deferred, recorded for the next wave)

- Distributed worker heartbeat via Redis (`WorkerPool.sync_from_redis`
  is wired, but no worker still publishes). Production will add the
  worker-side `SET heyavatar:worker:{id}:health` writer.
- Real MediaPipe swap to flip `liveportrait-human-v1.commercial_use`
  to true (see updated `docs/MODEL_LICENSES.md`).
- Real-mode `EchoMimicAdapter` (currently raises NotImplementedError
  on the real path).
- Tests for `tests/smoke/test_real_gpu/` remain skipped on this dev
  host (no CUDA + no LivePortrait upstream cloned); see the test
  package's `__init__.py` for the harness.

## [Unreleased]### Changed

- **Subdir refactor wave** — flat source files >250 lines were split
  into thematic subdirs across **10 production packages** (plus **2
  new sibling subdirs**) and **4 test packages** to isolate concerns
  and improve navigability. No compatibility layer: old file paths are
  removed. Imports always go through the most-specific submodule,
  e.g. `from workers.encoding_worker.worker import EncodingWorker`.

  Production packages now organised as:

  | Old flat file(s)                                | New subdir layout                                                                                        |
  |------------------------------------------------|---------------------------------------------------------------------------------------------------------|
  | `src/application/render_video.py`              | `src/application/render_video/{__init__, config, audio_probe, manifest, use_case}.py`                  |
  | `workers/gpu_worker.py`                         | `workers/gpu_worker/{__init__, worker, process, telemetry, cli}.py`                                     |
  | `src/scheduler/queue.py` (was a single file)    | `src/scheduler/queue/{__init__, memory, null, redis}.py`                                                |
  | `src/observability/metrics.py` (was single)     | `src/observability/metrics/{__init__, constants, instruments, recorders, exposition}.py`               |
  | `providers/liveportrait/audio_bridge.py`       | `providers/liveportrait/audio_bridge/{__init__, bridge, types, dsp}.py`                                 |
  | `providers/liveportrait/checkpoint_manager.py` | `providers/liveportrait/checkpoint_manager/{__init__, manifest, manager, downloader}.py`                |
  | `providers/liveportrait/adapter.py`            | `providers/liveportrait/adapter/{__init__, _mock, _upstream, _identity, _render, engine}.py`           |
  | `providers/musetalk/adapter.py`                | `providers/musetalk/adapter/{__init__, checkpoints, _upstream, _mock, _identity, _render, engine}.py`  |
  | `workers/encoding_worker.py`                    | `workers/encoding_worker/{__init__, worker, manifest, codec, cli}.py`                                   |
  | `src/storage/jobs.py`                           | `src/storage/jobs/{__init__, memory, redis}.py`                                                         |
  | `api/app.py`                                    | `api/app/{__init__, state, queue_factory, metrics, factory}.py` — the `api.app:app` Uvicorn convention is preserved via a small `__init__.py` re-export of `app`/`create_app` |
  | `src/scheduler/{router, worker_pool}.py` (new) | `src/scheduler/routing/{__init__, router, worker_pool}.py`                                              |
  | `src/observability/{context, tracing}.py`      | `src/observability/distributed/{__init__, tracing, propagation}.py`                                     |

  Test packages now organised as:

  | Old flat test file                              | New scenario-split package                                                                              |
  |------------------------------------------------|---------------------------------------------------------------------------------------------------------|
  | `tests/smoke/test_full_pipeline.py`            | `tests/smoke/test_full_pipeline/{__init__, _helpers, test_happy_path, test_compile_only, test_failure_recording}.py` |
  | `tests/application/test_render_video.py`       | `tests/application/test_render_video/{__init__, _helpers, test_retry_succeeds, test_retry_exhausted, test_mixed_chunks, test_retry_attempts, test_degraded_output, test_retry_budget}.py` |
  | `tests/workers/test_gpu_worker.py`              | `tests/workers/test_gpu_worker/{__init__, _helpers, test_compile_job, test_completed, test_completed_degraded, test_failed_inference, test_failed_encoding}.py` |
  | `tests/smoke/test_real_gpu.py`                  | `tests/smoke/test_real_gpu/{__init__, _helpers, test_gpu_health, test_checkpoints, test_engine_load, test_pipeline}.py` |

### Added — Distributed worker heartbeat (Redis)

- **`workers/gpu_worker/telemetry.py:: _build_health_payload`** —
  serialise a worker snapshot into the JSON contract that
  `WorkerPool.sync_from_redis()` consumes.
- **`workers/gpu_worker/telemetry.py:: _publish_health`** — atomic
  `SET heyavatar:worker:{worker_id}:health <JSON> EX N` on a stub
  redis client; swallows transient Redis errors so a blip never
  crashes a render worker.
- **`workers/gpu_worker/worker.py::GpuWorker._start_redis_heartbeat`**
  — daemon thread that publishes the worker's `engine.health()`
  every `settings.worker_health_publish_seconds` (default 3.0).
  Started in `run()` after `_load_engine()`; cooperatively stopped
  in the worker-shutdown `finally` via `_stop_redis_heartbeat()`.
- **`workers/gpu_worker/worker.py:: GpuWorker.publish_heartbeat_once`**
  — single-shot test/debug entry point that bypasses the thread.
- **`workers/gpu_worker/cli.py`** — passes the shared `redis.Redis`
  client to `GpuWorker` so the heartbeat thread reuses the same
  connection as `RedisJobRepository`.
- **`api/app/state.py`** — `AppState` now carries `worker_pool: WorkerPool`
  + `redis_client: object | None`. `lifespan` starts the
  `_start_worker_pool_sync_gatherer` thread that calls
  `WorkerPool.sync_from_redis(client)` every
  `settings.api_worker_pool_sync_seconds` (default 3.0s). On
  shutdown the thread is joined with a 2-second cap.
- **`api/app/state.py:: _build_redis_client`** — lazy redis client
  factory used by the lifespan; degrades silently to `None` when
  `redis` is not installed or `REDIS_URL` is unset.
- **`src/core/config.py`** — three new settings + env vars:
  `worker_health_publish_seconds` (3.0),
  `api_worker_pool_sync_seconds` (3.0),
  `worker_pool_heartbeat_ttl` (15).
- **`tests/workers/test_gpu_worker/test_redis_heartbeat.py`** (new) —
  schema + wire-format + single-shot engine dispatch tests using
  an in-memory stub redis.
- **`tests/scheduler/test_cross_process_capacity.py`** (new) — stub
  redis → `WorkerPool.sync_from_redis` → `TierRouter.pick_available`
  cross-process happy path + three schema-drift drop tests.
- **`tests/providers/test_worker_pool_in_process.py`** — added a
  skip-on-unparseable-JSON test asserting the pool records are
  empty when a record's body cannot be decoded.

The resulting capability: GPU workers on one machine publish
heartbeats to `heyavatar:worker:{id}:health` and the API process on
another machine sees them in its `WorkerPool` within
`api_worker_pool_sync_seconds`, so
`TierRouter.pick_available(tier, pool)` returns the correct engine
across process boundaries.

### Fixed

- `src/application/render_video/use_case.py::_chunks_for` now reads the
  audio duration through the module attribute
  (`audio_probe._probe_audio_duration(...)`) instead of a function
  reference captured at import time. This lets pytest
  `MonkeyPatch.setattr("src.application.render_video.audio_probe._probe_audio_duration", ...)`
  intercept the call — previously the captured binding silently NO-OP'd
  every test patch and tests fell through to the 0.0-seconds fallback
  that produced 16 chunks of `0x330000` degraded mp4.
- `tests/smoke/test_real_gpu/_helpers.py::requires_cuda` now also
  skips when the LivePortrait upstream repo isn't cloned
  (`<project_root>/LivePortrait/src/live_portrait_pipeline.py`
  sentinel missing). Without this, CI nodes with CUDA but no
  upstream would fail with `Engine state should be IDLE or LOADING
  after load(), got degraded` instead of skipping cleanly.
- Loggers in `src/observability/distributed/tracing.py` and any
  other refactored module now use `get_logger(__name__)` rather than
  hardcoded dotted-path strings, so log routing/shipment by logger
  name stays correct under the new module locations.

### Changed — Repository slimming plan, Change 3 (freeze)

Per `docs/REPOSITORY_SLIMMING_PLAN.md` §5 and `ROADMAP.md` §1 the
following premature subsystems are moved out of the active runtime.
Each freeze's re-introduction gate is documented in `ROADMAP.md` §1.

- **EchoMimic frozen.** `providers/echomimic/` directory kept on disk
  and `EngineId.ECHO_MIMIC` enum value preserved for forward compat.
  `providers/__init__.py::PROVIDERS` no longer registers
  `EchoMimicAdapter`; `get_provider(EngineId.ECHO_MIMIC)` raises
  `KeyError` with a freeze-message that distinguishes the frozen
  engine id from an unknown one. The `EchoMimicAdapter` itself still
  raises `NotImplementedError` on the real path — unchanged.
- **SadTalker audio bridge gated.** `dsp` is the only production
  backend; `neural` is preserved behind
  `HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural` but never the default.
  `providers/liveportrait/audio_bridge/__init__.py` documents the
  freeze in its module docstring; the bridge logic itself is
  unchanged.
- **Multi-tier routing collapsed** to one `standard` profile.
  `src/scheduler/routing/router.py::TierRouter` ignores the legacy
  `tiers:` block in `registry/models.yaml` and exposes only the
  `standard` profile. The `Tier` enum keeps `EXPRESS`/`STUDIO`/
  `PREMIUM` for backwards API compatibility — `for_tier(t)` returns
  the `standard` decision regardless. `pick_available` no longer walks
  a fallback list; if the standard primary has no idle worker the
  router returns `None`. `registry/models.yaml::tiers` is replaced by
  `registry/models.yaml::standard`.
- **`WorkerPool.sync_from_redis` invocation frozen.**
  `api/app/state.py::lifespan` no longer starts the periodic gatherer
  thread (call commented out with a freeze note). `_start_worker_pool_sync_gatherer`
  is preserved so unit tests under `tests/scheduler/test_cross_process_capacity.py`
  still exercise the wire schema.
  `workers/gpu_worker/worker.py::_start_redis_heartbeat` is now gated
  behind a new `Settings.enable_distributed_heartbeat` setting
  (default `False`). Operators who need the heartbeat daemon set
  `HEYAVATAR_ENABLE_DISTRIBUTED_HEARTBEAT=1`; until then the
  function logs a debug message and returns before spinning up the
  thread.
- **S3 backend frozen.** `ObjectStore` ABC + `FsObjectStore` remain.
  `Settings.object_store_backend` tightened from
  `Literal["fs", "s3"]` to `Literal["fs"]`. The
  `s3_endpoint_url` and `s3_bucket` settings are removed. The
  `src/storage/object_store.py` module docstring claims the freeze
  explicitly; `build_object_store` continues to raise
  `NotImplementedError` for any non-`fs` backend (a tight literal
  prevents the string from reaching the switch in practice).
- **OpenTelemetry exporters frozen.**
  `src/observability/distributed/tracing.py::setup_tracing` already
  short-circuits when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset; the
  module docstring now claims the freeze explicitly. W3C traceparent
  inject/extract (`src/observability/distributed/propagation.py`)
  is kept unchanged: it is still used by `api/routes/jobs.py` and
  `api/routes/avatars.py` for queue-payload context propagation, and
  graceful no-ops when the SDK is not installed.

### Changed — Tests (Change 3)

- `tests/scheduler/test_router_pick_available.py` rewritten to
  assert the frozen single-tier behavior: standard primary returns
  the engine when idle; returns `None` when busy (no fallback walk).
- `tests/scheduler/test_router.py` rewritten to assert the single
  `standard` route and the missing-registry fallback to a default
  primary (no `LookupError`, no legacy `tiers:` expectation).
- `tests/scheduler/test_cross_process_capacity.py::test_sync_from_redis_routes_in_router_via_fallback`
  keeps the schema check but pins the frozen router outcome:
  `pick_available` returns `None` rather than walking to the
  liveportrait fallback when the standard primary is busy.
- `tests/providers/test_worker_pool_in_process.py` rewritten against
  the new router (no `_rules` attribute, no fallback walk test).
  `tests.contract.test_avatar_engine_contract.test_provider_passes_contract`
  auto-shrinks: its `parametrize("engine_id", list(PROVIDERS))` now
  iterates only `LIVE_PORTRAIT_HUMAN_V1` and `MUSE_TALK_V1` because
  `EchoMimic` is no longer registered.

Verification:
  pytest tests/ --ignore=tests/observability \
    -k "not test_api_metrics and not test_metrics and not test_real_gpu"
  → 166 passed, 7 deselected (was 159 before Change 3; +7 from
  rewritten router tests).

### Changed — Repository slimming plan, Change 2 (move)

Per `docs/REPOSITORY_SLIMMING_PLAN.md` §4 the production OpenCV face
compositor is moved out of the speculative `providers/compositing/`
subtree into the canonical `src/pipeline/` package so it sits
beside the other application-layer primitives
(`src/quality/`, `src/application/`). No behavioural change.

- **New canonical home** `src/pipeline/compositor.py` exporting
  `OpenCVFaceCompositor` and `match_mean_std`. The class body is
  source-identical to the previous
  `providers/compositing/opencv_face/compositor.py` (same imports,
  same signature, same `__all__`); line endings may differ because
  the file was re-saved on the new path. Pytest parity is the
  meaningful invariant. The module docstring points back to the old
  path for migration.
- **New package** `src/pipeline/__init__.py` re-exports
  `OpenCVFaceCompositor` so callers now write
  `from src.pipeline import OpenCVFaceCompositor`.
- **Removed** the now-empty `providers/compositing/` subtree:
  - `providers/compositing/opencv_face/__init__.py`
  - `providers/compositing/opencv_face/compositor.py`
  - `providers/compositing/__init__.py` (was only there to expose
    the deleted subpackages)
- **Removed** `tests/compositing/test_alpha_blend.py` — a pure math
  test of the alpha-blend formula that never imported the
  compositor class and was redundant with the in-class formula test
  in `tests/compositing/test_debug_disabled.py`.
- **Updated importers** to read from the canonical path:
  - `tools/avatar_assets/render_clean_composite.py`
    (`--body/--face/--face-mask/--neck-mask/--transforms/--audio/--output`
    CLI; production-style render path)
  - `tests/compositing/test_debug_disabled.py`
  - `tests/integration/test_clean_composite_pipeline.py`
- **`tools/avatar_assets/preview_face_composite.py` not touched.** It
  composes a *single static face image* onto the body template via
  its own inline `cv2.warpAffine` + Gaussian-blurred mask flow
  (a deterministic preview, not part of the production render path).
  Because it never imported `OpenCVFaceCompositor`, no import-path
  update was required. The intra-file alpha formula is documented in
  its `composite_preview()` docstring for future reference.

Verification:
  pytest tests/ --ignore=tests/observability \
    -k "not test_api_metrics and not test_metrics and not test_real_gpu"
  → 159 passed, 7 deselected.
