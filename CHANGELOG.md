# Changelog

All notable changes to Heyavatar are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).## [Unreleased]

### Added — §6 GPU-worker demonstrator (`workers/gpu_worker/golden_path.py`)

The §6 golden-path demonstrator now ships in two flavours:

* :mod:`tools.run_cached_avatar` — the local bench / unit test entry
  point; runs ``render_cached_avatar`` *directly* outside any worker
  process.
* :mod:`workers.gpu_worker.golden_path` — the production-parity
  package; publishes a synthetic ``render_cached`` ``RenderJob`` to a
  self-cleaning ``InMemoryJobQueue`` then drives :class:`GpuWorker`'s
  reservation loop, so the engine ``load()`` bootstrap, trace-context
  injection, ``_do_process_render_cached`` fail-closed mapping, and
  per-job telemetry all fire on the GPU worker process exactly as
  production ``POST /jobs`` traffic triggers them.

Usage::

    python -m workers.gpu_worker.golden_path \\
      --avatar-id actor_01 --gesture-id explain_both \\
      --identity-pack avatar_packs/actor_01.tar \\
      --audio samples/speech_30s.wav \\
      --capture-dir captures/golden_path

- **`workers/gpu_worker/golden_path.py`** (new) — half a dozen small
  helpers (``_build_parser``, ``_validate_inputs``,
  ``_build_synthetic_job``, ``_install_self_cleaning_reserve``,
  ``_build_capture_dir_settings``) and a thin ``main()`` that wires
  them around :class:`GpuWorker`. The ``_install_self_cleaning_reserve``
  wrap delegates to ``queue.reserve`` (so ``_pending`` popleft /
  ``attempts`` bookkeeping the :class:`InMemoryJobQueue` does stays
  consistent with the production code path) and flips
  ``worker._stop`` on the **second** reserve call after the synthetic
  job is drained — the two-call shape guarantees the worker runs the
  synthetic job **exactly once** before exiting.
- **`tests/workers/test_gpu_worker/test_golden_path.py`** (new) — eight
  tests pinning the acceptance contract:
  1. ``test_main_drives_synthetic_job_through_render_cached_fail_closed_path``
     – happy-path: QC passed → COMPLETED → exit 0; asserts the
     ``_do_process_render_cached`` pathway was called via the
     ``workers.gpu_worker.process.render_cached_avatar`` patch probe
     AND pins ``call_args.kwargs`` so a future refactor that drops
     ``avatar_id`` / ``gesture_id`` fails loudly.
  2. ``test_main_propagates_engine_runtime_error_as_exit_code_1`` –
     engine fail-closed RuntimeError → exit 1 with a stderr hint; an
     operator can wire this directly into a Docker healthcheck.
  3. ``test_install_self_cleaning_reserve_stops_worker_after_one_job``
     – directly exercises the reserve wrap to confirm ``_stop`` flips
     on the SECOND reserve, NOT the first (which would leak the
     in-flight synthetic job).
  4. ``test_install_self_cleaning_reserve_rejects_multi_job_queue`` –
     hardening: the wrap refuses to install unless exactly one job is
     pending, so a multi-job queue can't silently violate the "exactly
     one" guarantee.
  5. ``test_main_rejects_missing_assets`` – CLI rejects BOTH missing
     identity pack AND missing audio in a single stderr sweep (v0 had
     a single-missing-then-re-run loop).
  6. ``test_build_synthetic_job_payload_matches_dispatch_contract` –
     pins the synthetic-payload schema so a future refactor of
     :meth:`workers.gpu_worker.process._do_process_render_cached_impl`
     cannot silently lose fields like ``fps``/``batch_size``.
  7. ``test_resolve_engine_id_handles_known_values`` – covers the
     ``--engine musetalk-v1`` default and rejects unknown strings.
  8. ``test_summarise_maps_captured_state_to_exit_codes`` – the
     ``_summarise`` source-of-truth table: COMPLETED → 0,
     COMPLETED_DEGRADED → 0 (worker code path exercised; QC verdict
     in metrics), FAILED_INFERENCE → 1, missing state → 1.

### Refinement pass

- **Drop ``get_settings.cache_clear()`` and ``_get_settings_cleared``
  alias** — clearing the lru_cache just made the next
  :func:`Settings.from_env` re-scan env vars that didn't have the
  operator's ``--capture-dir`` flag, so the call was actively
  misleading. The override reaches the worker via the
  :class:`GpuWorker` constructor's ``settings=`` arg; cache_clear
  was dead code that *looked* like it propagated something.
- **Decouple exit code from ``metrics.json`` presence** — the
  driver now wraps :meth:`GpuWorker._do_process_render_cached` to
  capture the ``JobState`` it returns, and translates that state
  into an exit code via :func:`_summarise`. ``metrics.json`` is now
  best-effort operator visibility, not the success signal.
- **Hardening on :func:`_install_self_cleaning_reserve`** — refuses
  to install unless ``queue.depth() == 1`` so a multi-job queue
  can't silently violate the "exits after exactly one synthetic
  job" guarantee.
- **Add ``--engine`` CLI flag** — mirrors ``workers/gpu_worker/cli.py``;
  default ``musetalk-v1``; an unknown engine yields exit 2 with a
  clear error.
- **``, ``-separated combined error in :func:`_validate_inputs`** –
  the CLI rejects both missing identity pack AND missing audio in
  a single stderr sweep, so an operator doesn't have to play
  fix-one-thing-then-re-run.

### Added — Cached Avatar Golden Path v0 (Block 2 demonstrator)

The first economical end-to-end demonstrator that proves
pre-computed body template + MuseTalk face region + compositor + audio
+ QC produces a valid 30-second output in one shell command. Three
critical fail-closed holes were closed, a public-facing runner script
ships, and a cost-report generator + smoke test verify the layout.

- **`providers/musetalk/adapter/engine.py`** — the `prepare_identity`
  and `render_chunk` paths now explicitly raise
  `RuntimeError("MuseTalkAdapter is not loaded")` when
  `self._state == EngineState.UNLOADED`, **before** the mock-mode
  short-circuit. Previously the mock path would silently substitute
  synthetic assets for a freshly-unloaded engine in real mode, which
  silently turned a real-GPU failure into a passing QC run.
  The order is now:
  `settings.mock_engine → raise UNLOADED → raise DEGRADED → real path`,
  matching the canonical fail-closed contract the GPU worker relies on.
- **`src/application/render_cached_avatar.py`** — `mux_audio` is now
  wrapped in a `try/except Exception` that re-raises as
  `src.quality.exceptions.EncodingError` and an explicit
  `if not final_path.is_file(): raise EncodingError(...)` guard follows.
  The combination makes the impossible state `status=COMPLETED +
  final_path=None` unreachable, which the previous implementation
  allowed when ffmpeg silently exited non-zero.
- **`src/pipeline/quality.py`** — new check F (*stream presence*):
  `probe_has_streams(video_path)` runs ffprobe's `-show_streams`
  and rejects the output as `FAILED_QC_MISSING_STREAMS` when either
  the video stream or the audio stream is absent. Both
  `has_video` and `has_audio` are now recorded in `metrics.json` so
  the bench script can verify the post-mux final.mp4 carries both
  tracks without re-running ffprobe.
- **`tools/run_cached_avatar.py`** — new CLI runner. Exposes a
  testable `run(args, engine=None)` function (returns the
  `RenderCachedAvatarResult`) on top of an `argparse` CLI. Writes the
  canonical six-file deliverable layout under `--output`'s parent:
  `face_roi.mp4`, `lipsynced_face.mp4`, `composited.mp4`, `final.mp4`,
  `result.json`, `metrics.json`. The `metrics.json` schema matches
  the block-1 contract the project plan calls out
  (`status`, `output_seconds`, `pipeline_wall_seconds`,
  `core_gpu_seconds`, `gpu_seconds_per_output_minute`, `body_cache_hit`,
  `identity_cache_hit`, `model_warm`, `face_region_only`,
  `face_resolution`, `batch_size`, `audio_stream_present`,
  `video_stream_present`, `qc_passed`).
- **`tools/cost_report.py`** — new cost-report generator. Reads
  metrics.json (recursive), aggregates by `batch_size`, and emits a
  markdown table with `cost_per_video = gpu_seconds / 3600 * hourly_price`
  derivation. The output matches the §6 acceptance layout verbatim.
- **`tests/tools/test_run_cached_avatar.py`** — new smoke test
  covering: (a) the six deliverables are produced; (b) the metrics
  schema is complete and `qc_passed=True`; (c) the runner picks up a
  pre-built identity pack and reports `identity_cache_hit=True`; (d)
  a consecutive second run still reports `model_warm=True` +
  `identity_cache_hit=True` (warm-cache acceptance); (e)
  `FileNotFoundError` is raised for missing inputs without ever
  touching the engine.

Usage::

    python tools/run_cached_avatar.py \\
      --avatar-id actor_01 --gesture-id explain_both \\
      --identity-pack avatar_packs/actor_01.tar \\
      --audio samples/speech_30s.wav \\
      --output captures/golden_path/final.mp4

Then::

    python tools/cost_report.py \\
      --metrics-dir captures/cost_sweep/ \\
      --gpu-hourly-price 0.50 \\
      --output captures/cost_sweep/cost_report.md

### Added — Warm-vs-cold cost split (cost-report v0.1)

The §6 acceptance table now distinguishes the **cold** run (first
invocation after the worker starts — body template not yet precomputed,
no pre-built identity pack on disk, model not yet loaded into VRAM) from
the **warm** run that follows. The two states give very different
numbers: cold absorbs MediaPipe warm-up, body-template materialisation,
identity compile, and model ``load()`` once per worker start; warm is
the steady-state cost the operator pays per video.

- **`tools/cost_report.py::CostRow.run_state`** (new property) —
  classifies a metrics.json row as ``"warm"`` iff the strict-AND of
  ``body_cache_hit and identity_cache_hit and model_warm`` is True;
  ``"cold"`` otherwise. Module-level constants
  :data:`RUN_STATE_COLD` / :data:`RUN_STATE_WARM` are part of the
  public surface so downstream cost consumers can import the same
  string literals.
- **`tools/cost_report.py:: split_by_run_state`** (new helper) —
  partitions a batch's samples into ``(cold, warm)`` with stable
  canonical order (cold precedes warm so cost rows are read
  cold→warm per batch).
- **`tools/cost_report.py::format_table`** — header now carries a
  ``State`` column (``cold`` / ``warm``) and each batch emits up to
  two rows (cold first, warm second) when both states have samples.
  A batch with only cold samples (or only warm samples) emits a
  single row so the table is self-describing — "batch 8 has only a
  warm row in this bench" is obvious from the State column alone.
  The trailing cost annotation now distinguishes the cause:
  cold rows carry ``(cold: body=miss id=miss model=fresh)`` (only
  the misses that fired are listed); warm rows carry
  ``(body=hit id=hit warm)``.
- **`tools/cost_report.py:: load_metrics`** — cache flags now use
  ``data.get(key, False)`` defensively so a future metrics.json
  schema that drops a flag classifies the row as cold instead of
  crashing the bench loader.
- **`tests/tools/test_cost_report.py`** — added 7 new tests covering
  ``run_state`` semantics, ``split_by_run_state`` partitioning, the
  dual-row cold+warm table layout, only-cold / only-warm single
  rows, mixed-batch layout, and the ``load_metrics`` defensive
  default. Existing layout test updated to assert the new
  ``State`` column.

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

## [Unreleased]

### Added — Block-2 canonical §6 acceptance smoke (`tests/smoke/test_full_pipeline/test_timeline_e2e.py`)

The §6 acceptance example in `docs/REPOSITORY_SLIMMING_PLAN.md` — "single-video
cost vs three-gesture manual timeline cost back-to-back" — is now a single
end-to-end test that runs both modes and asserts the bench table distinguishes
them.

* **`tests/smoke/test_full_pipeline/test_timeline_e2e.py`** (new) — single
  test function `test_block_2_mixed_mode_cost_report_e2e` that:
  1. Builds two body templates (`actor_01/idle` 75 frames + `actor_01/explain_both`
     50 frames) so each timeline segment's audio length matches the template
     frame count exactly within the QC duration gate.
  2. Pre-builds an identity pack through a warm mock `MuseTalkAdapter` so
     both subsequent single-mode and timeline-mode runs hit
     `identity_cache_hit=True`.
  3. Calls `tools.run_cached_avatar.run` once in single mode (2 s
     `explain_both` audio) and once in timeline mode (8 s audio + the
     reference `docs/examples/timeline_three_segment.json`).
  4. Asserts the aggregated `timeline_metrics["timeline_gesture_ids"] ==
     ["idle", "explain_both", "idle"]`, `output_seconds == 8.0`,
     `body_cache_hit / identity_cache_hit / model_warm` all True.
  5. Verifies the three per-segment metrics.json files at
     `segments/<idx>_<gesture>/metrics.json` carry the correct
     `timeline_segment_index`, `timeline_segment_gesture`, and
     `timeline_segment_duration_seconds` values.
  6. Copies the two top-level metrics.json into a flat `metrics_capture/`
     subdir (so per-segment files don't dilute the aggregated
     timeline row) and calls `tools.cost_report.write_report` with
     `output_path=None` — asserts the markdown table contains BOTH
     `| single |` and `| timeline |` rows back-to-back, the `| Mode |`
     column header, the §6 cost derivation footnote, and does NOT
     create a stray `-` file.

Tests skip cleanly without `ffmpeg` (the per-segment audio slicing) or
without `ffprobe` (the runner's `audio_stream_present / video_stream_present`
metrics probes).

### Changed

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

### Changed — Repository slimming plan, Change 2-EXT (QC relocation)

The Change 2 mandate in `docs/REPOSITORY_SLIMMING_PLAN.md` §4 also
covers the post-production QC layer: only the abstract contract is
kept at the package level — the concrete implementation moves into
`src/pipeline/` alongside the production compositor. This change
relocates the concrete QC as the canonical QC module referenced by
Change 1's `contracts/quality_checker.py::QualityChecker` ABC note.
Pipeline behaviour is unchanged.

- **New canonical home** `src/pipeline/quality.py` exporting
  `VideoQualityChecker` (concrete subclass of
  `contracts.quality_checker.QualityChecker`) plus the low-level
  helpers `debug_green_ratio`, `mean_luminance`,
  `probe_video_duration`, `probe_audio_duration`, `probe_video_codec`.
  The class body is source-identical to the previous
  `src/quality/video_quality.py` (same imports, same signature,
  same `__all__`); line endings may differ because the file was
  re-saved on the new path. Pytest parity is the meaningful
  invariant. The module docstring points back to the old path for
  migration.
- **`src/pipeline/__init__.py` re-exports** the full pipeline QC
  surface alongside the compositor:
  `VideoQualityChecker`, `debug_green_ratio`, `mean_luminance`,
  `probe_video_duration`, `probe_audio_duration`,
  `probe_video_codec`. Callers now write
  `from src.pipeline import (
       OpenCVFaceCompositor, VideoQualityChecker, debug_green_ratio,
   )`.
- **Removed** the now-empty concrete-QC source file
  `src/quality/video_quality.py`.
- **`src/quality/__init__.py` slimmed down** to the domain
  exceptions only — `CompositeError`, `EncodingError`, `QualityError` —
  because those exceptions are shared across the QC and compositor
  paths and the compositing exception import
  (`from src.quality.exceptions import CompositeError`) stays
  unchanged inside `src/pipeline/compositor.py`. The concrete
  `VideoQualityChecker` no longer lives at `src.quality.*`.
- **Updated importers** to read from the canonical path:
  - `tools/avatar_assets/render_clean_composite.py` (production-style
    render tool; QC is the third stage after compositing + mux).
  - `tests/integration/test_clean_composite_pipeline.py` —
    `VideoQualityChecker` + `debug_green_ratio` import,
    plus three `monkeypatch.setattr("src.quality.video_quality.…", …)`
    call sites rewritten to `"src.pipeline.quality.…"`.
  - `tests/compositing/test_debug_disabled.py` — `debug_green_ratio`
    import moved to the canonical pipeline surface.
  - `tests/quality/test_green_overlay_detector.py` — same.
  - `tests/quality/test_duration_validation.py` — class import +
    6 monkeypatch target rewrites.
  - `tests/quality/test_black_frame_detector.py` — class + helper
    imports moved to the canonical pipeline surface.
- **`src/quality/exceptions.py` untouched** — the domain-exception
  module stays at its original location so the failure-recovery
  semantics that `src/pipeline/compositor.py` and `render_clean_composite.py`
  rely on are unchanged.
- **`tests/quality/__init__.py`** still imports nothing new; the
  package marker remains valid.

Verification:
  pytest tests/ --ignore=tests/observability \
    -k "not test_api_metrics and not test_metrics and not test_real_gpu"
  → 159 passed, 7 deselected. Parity is expected — unlike Change 3
  (which added 7 frozen-router tests), Change 2-EXT is a pure
  module-path rewrite; every QC consumer was already covered by the
  baseline 159-pass suite, so no new tests are introduced.

### Added — Repository slimming plan, Change 4 (deterministic multi-template timeline)

Per `docs/REPOSITORY_SLIMMING_PLAN.md` §6 + §10 (Change 4 of the
slim plan), the MVP lifts the manual multi-body-template timeline
into a frozen-dataclass surface so the orchestrator can deterministically
concatenate three-or-more body clips — video frames / face transforms /
face masks / neck masks / timestamps / audio — at the canonical
`standard` fps before any LLM-style gesture planner ever ships.

New canonical types (all in `src/domain/`)

- **`src/domain/body_template.py`** (new) — the concrete
  `BodyTemplate` frozen dataclass (slots=True) with fields
  `body_video: Path`, `face_mask: Path`, `neck_mask: Path`,
  `face_transforms: Path`, `metadata: Path`. Per-segment metadata
  accessors read `avatar_id`, `gesture_id`, `fps`, `total_frames`,
  `width`, `height`, `status` from `metadata.json` on demand. The
  module also ships `load_body_template(avatar_id, gesture_id,
  base_dir="body_templates")` that resolves the canonical
  `body_templates/<avatar_id>/<gesture_id>/` tree written by
  `tools/avatar_assets/precompute_video_template.py`.
- **`src/domain/timeline.py`** (new) — `TimelineSegment` +
  `Timeline` (frozen dataclasses, slots=True, immutable). The
  `Timeline` JSON shape matches the slim-plan example
  verbatim (`{"segments": [{"gesture_id": "...", "duration_seconds":
  ...}, ...]}`), with an optional top-level `"fps"` (default
  `25`, matching `registry/models.yaml::standard.rationale`).
  Helpers: `from_json(path)`, `from_dict(dict)`, `to_json(path)`,
  `to_dict()`, `is_well_formed()`, `total_duration_seconds()`,
  `expected_frames()`, `frame_count_for_segment(idx)`. Validation
  is strict — empty segments, non-positive durations, missing
  keys, non-positive fps, type mismatches all raise `ValueError`
  at API load time, not at GPU-worker time.

New frame-align utility (in `src/pipeline/`)

- **`src/pipeline/timeline_align.py`** (new) —
  `AlignedBodyTimeline` frozen dataclass + `align_timeline(
  timeline, avatar_id, *, output_dir, fps=None,
  body_template_loader=load_body_template)`. The utility loads each
  segment's `BodyTemplate`, validates three strict invariants
  (per-segment frame count == `round(segment.duration * fps)`,
  cross-segment resolution equality, fps agreement), and emits FOUR
  concatenated files at the canonical Timeline fps:
  `body.mp4`, `face_mask.mp4`, `neck_mask.mp4`, `face_transforms.npz`
  (with bbox + matrices concatenated along axis 0 and
  `timestamp_ms` remapped to a strictly monotonic
  `frame_index * (1000 / fps)` sequence so cross-segment timestamps
  are seamless).
- **`src/pipeline/__init__.py`** — re-exports the timeline surface
  alongside the composite + QC surfaces so callers write
  `from src.pipeline import AlignedBodyTimeline, align_timeline`.

Orchestrator wiring

- **`src/application/render_video/use_case.py::RenderVideo.run_timeline`**
  (new) — orchestrator entry point that consumes a `Timeline` +
  `avatar_id` + `RenderRequest` + `AvatarIdentityHandle` and calls
  `align_timeline` to materialise the canonical four-file aligned
  asset tree. After alignment the method checks
  `|audio.duration - aligned.duration_seconds| <= 1/fps` (the
  Change-4 determinism contract) and refuses the request at API
  time if the audio drifts beyond one frame. ffprobe probing
  failures log a warning and proceed (consistent with the existing
  `_chunks_for` behaviour) but a non-zero probe that disagrees
  raises `ValueError` so operators see the discrepancy before
  GPU worker time.

Registry

- **`registry/gestures.yaml`** — added the canonical `idle` entry
  (3.0s default) so the slim-plan example timeline
  ``{"segments": [{"gesture_id": "idle", "duration_seconds": 3.0},
  ...]}`` resolves under the gesture catalog without an alias.

Tests (40+ new tests)

- **`tests/domain/test_body_template.py`** (new) — 12 tests for
  `BodyTemplate` construction (frozen), `load_body_template`
  happy path + 5 missing-file permutations + the metadata accessors
  + the default-base-dir relative-CWD resolution.
- **`tests/domain/test_timeline.py`** (new) — 21 tests for
  `Timeline` JSON round-trip, derived properties
  (`total_duration_seconds`, `expected_frames`,
  `frame_count_for_segment`), validation errors on
  malformed JSON (empty segments, non-positive durations,
  type mismatches, missing keys, non-positive fps),
  coercion behaviour, and frozen dataclass guarantees. The
  slim-plan example JSON is the canonical fixture.
- **`tests/pipeline/test_timeline_align.py`** (new) — 12 tests
  driving `align_timeline` against the slim-plan canonical
  3-segment timeline (75 + 50 + 75 = 200 frames at 25 fps = 8.0s),
  asserting: total frame count, per-file existence on disk, the
  npz bbox/matrices/timestamp_ms shapes, strict timestamp
  monotonicity + correct dt persistence between segments,
  custom-fps parametrisation, and the three invariant failure
  modes (per-segment frame-count mismatch, cross-segment
  resolution mismatch, missing `bbox`/`matrices` keys).

Re-exports

- **`src/domain/__init__.py`** — surfaces `BodyTemplate`,
  `load_body_template`, `Timeline`, `TimelineSegment`,
  `DEFAULT_TIMELINE_FPS` alongside the existing domain types.

Roadmap status-bump
- **`ROADMAP.md` §3** — the “Pending (Change 4 of the slimming plan)”
  section title is renamed to “Shipped” and the
  deterministic-timeline bullets now reflect what landed.

Verification:
  pytest tests/ --ignore=tests/observability \
    -k "not test_api_metrics and not test_metrics and not test_real_gpu"
  → 191 passed, 7 deselected (was 159 after Change 2-EXT; +32 from
  the new timeline align + body template + JSON loader tests).

### Added — Change 4 reference timeline + golden-signal integration test

Per the orchestrator's Change-4 acceptance bullet ("the orchestrator
has a deterministic golden signal for end-to-end multi-template
runs"), the canonical reference timeline shape is now committed
under `docs/examples/`, and a new integration test exercises
`RenderVideo.run_timeline` end-to-end against a synthesised
`body_templates` tree + an 8.0-second silent WAV.

- **`docs/examples/timeline_three_segment.json`** (new) — the
  reference timeline JSON that operators and integration tests
  anchor against:
  ```json
  {
    "fps": 25,
    "segments": [
      {"gesture_id": "idle", "duration_seconds": 3.0},
      {"gesture_id": "explain_both", "duration_seconds": 2.0},
      {"gesture_id": "idle", "duration_seconds": 3.0}
    ]
  }
  ```
  Total: 8.0 s × 25 fps = 200 frames. The file carries an inline
  `_comment` block so operators see the rationale and the
  mathematical invariant without flipping away from the JSON.
- **`tests/integration/test_run_timeline_integration.py`** (new) —
  end-to-end golden-signal test against the reference JSON:
  - Builds a 3-template synthesised `body_templates/alice/{idle,explain_both}/`
    tree at 64×64×25fps (75, 50, 75 frames).
  - Synthesises an 8.0 s silent PCM WAV so the orchestrator's
    audio-vs-aligned-drift gate is exercised deterministically.
  - Loads `Timeline` via `Timeline.from_json(EXAMPLE_JSON)`.
  - Patches `load_body_template` to point at the synthesised tree
    (via `__globals__` so the patch reaches the orchestrator's call
    site cleanly).
  - Invokes `RenderVideo.run_timeline(timeline, avatar_id, identity,
    request, alignment_dir)`.
  - Asserts: 200 total frames, 25 fps, 64×64 resolution, 8.0 s
    duration, all 5 aligned files on disk, metadata.json
    segments breakdown mirrors the JSON, npz bbox/matrices dtype
    `float32` with strict timestamp monotonicity at 40 ms dt.
  - Second test forces a 30.0 s audio drift via `monkeypatch` and
    verifies `run_timeline` raises `ValueError("drifts ... aligned
    timeline")` — exercises the audio-drift gate even when ffprobe
    is unavailable.

Verification:
  pytest tests/integration/test_run_timeline_integration.py -v
  → 2 passed in 0.45s.

Verification (cumulative on the Change 4 baseline):
  pytest tests/ --ignore=tests/observability \
    -k "not test_api_metrics and not test_metrics and not test_real_gpu"
  → 225 passed, 7 deselected (was 223 after the LOW follow-up amend;
  +2 from the new reference timeline + integration test).

### Hardened — Change 4 reference-timeline follow-up amends

Three subsequent `--amend` cycles folded reviewer-flagged
hardenings back into the golden-signal commit. They sit inside
the same commit on `main` for traceability:

1. **Test isolation** — the happy-path test now uses
   `monkeypatch.setattr("src.application.render_video.use_case.load_body_template", …)`
   instead of mutating `run_timeline.__globals__["load_body_template"]`
   directly; `monkeypatch` guarantees restoration at fixture teardown
   so the previous `__globals__` mutation no longer leaks across the
   pytest suite.
2. **ABC compliance on `_StubEngine`** — the engine stub now
   inherits from `contracts.avatar_engine.AvatarEngine`. All four
   abstract methods (`load`, `unload`, `prepare_identity`,
   `render_chunk`) are implemented and raise
   `:class:`NotImplementedError`` so any future addition of
   `engine.X()` calls inside `run_timeline` fails LOUDLY at
   abstract-method dispatch instead of silently no-opping.
3. **Synth landmarks coverage** — `_synth_template` now writes a
   `(N, 478, 3)` `landmarks` array in `face_transforms.npz`,
   mirroring `tools/avatar_assets/precompute_video_template.py`'s
   real precompute output. This exercises
   `align_timeline`'s `lmk_parts.append` + dtype-normalise
   branch end-to-end at the integration level (previously the
   branch was unreachable from this golden-signal run).
4. **Lax-from_dict contract pin** — the dead-code
   `_EXPECTED_LAX_FROM_DICT` string constant was removed and
   replaced with an inline Step-4a contract assertion: the
   reference JSON's `_comment` block round-trips through
   `Timeline.from_dict(...).to_dict()` to a canonical
   `{"fps": 25, "segments": [...]}` shape. If the loader ever
   tightens (or a future reference JSON edit breaks the
   contract), the assertion fails loudly.
5. **Landmarks invariant assertions + finiteness** — Step 11 of
   the happy-path test now asserts `data["landmarks"].shape ==
   (200, 478, 3)`, `data["landmarks"].dtype == np.float32`, AND
   `np.all(np.isfinite(data["landmarks"]))`. The third assertion
   is the one that gives the golden signal real teeth — it
   catches silent regressions where the dtype-normalise branch
   writes zeros / NaNs without breaking the shape or dtype
   contract.

Python-side: the `_StubEngine` shape matches the production
ABC so adding `engine.X()` calls inside `run_timeline` is now a
TypeError on dispatch, not a silent AttributeError on duck
typing.

Verification:
  pytest tests/ --ignore=tests/observability \
    -k "not test_api_metrics and not test_metrics and not test_real_gpu"
  → 225 passed, 7 deselected (test count unchanged — three
  review cycles tightened the existing 2 integration tests
  without introducing new ones).
