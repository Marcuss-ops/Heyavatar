# Model Licenses

Every entry in `registry/models.yaml` MUST be reviewed before any
commercial deployment. The table below summarises the current state.

| Engine                | Code      | Weights    | Commercial | Notes                                                     |
|-----------------------|-----------|-----------|------------|-----------------------------------------------------------|
| `liveportrait-human-v1` | MIT       | MIT       | conditional | InsightFace detector is non-commercial. Replace with      |
|                       |           |           |            | MediaPipe Face Landmarker before commercial release.      |
| `musetalk-v1`         | MIT       | MIT       | yes        | Verify VAE + Whisper versions on install.                  |
| `echomimic-v1`        | Apache-2.0 | Apache-2.0| yes        | Apache-2.0 carries an explicit patent grant.               |

## Replacing InsightFace (LivePortrait)

The default LivePortrait pipeline uses `insightface` for
face detection and landmark extraction, which is restricted to
**non-commercial research use only**. Before using
`liveportrait-human-v1` commercially:

1. Install `mediapipe` (Apache-2.0).
2. Replace the detector invocation in the LivePortrait adapter with
   the MediaPipe Face Landmarker.
3. Re-run the contract tests under `tests/contracts/`.

## Verifying installations

```bash
python -c "from src.application.compile_avatar import AvatarCompiler; print('ok')"
pytest tests/contracts -q
pytest tests/providers -q
```

A green contract run means every provider still satisfies the
`AvatarEngine` contract. A green `tests/providers` run means the
adapter-specific DSP and checkpoint code paths still work in mock mode.

## LivePortrait — adapter obligations (added v0.2)

The new `providers/liveportrait/adapter.py` wraps the upstream
code with the following contractual behaviours:

1. **Checkpoint pinning.** Real-mode weights are downloaded from the
   upstream release page and verified against SHA256 pins in
   `providers/liveportrait/checkpoint_manager.py`. Pins are
   currently placeholder values (`TBD`); replace them with the
   values from the official release before turning off
   `HEYAVATAR_LOCK_ENGINE=1` in production.
2. **Custom CUDA op.** The MultiScaleDeformableAttention extension
   MUST be compiled in the worker image (use upstream
   `tools/prepare_env.sh`). When the import fails, the adapter
   transitions to `EngineState.DEGRADED` so the orchestrator can
   route around the broken worker instead of crashing mid-job.
3. **Audio-to-expression bridge.** The audio chunk the contract
   hands us is mapped to per-frame LivePortrait keypoint deltas by
   `providers/liveportrait/audio_bridge.py`. This DSP-only bridge
   is a *thin honest fallback*, not a neural audio-to-motion
   model; replace it with SadTalker's audio-to-motion module or a
   blendshape predictor before commercial deployment.
4. **Pack layout.** `prepare_identity` writes the standard
   `src/domain/avatar_pack.py` entries plus three extras
   (`transform_matrix.bin`, `inference_config.json`,
   `crop_config.json`). Older packs without these remain readable
   — the hybrid reader in the adapter falls back to identity
   keypoints when the entry is absent.

## Production-Safe Path for LivePortrait (`commercial_use: true`)

To unblock commercial deployment, the following must all hold:

1. **MediaPipe replaces InsightFace.** ✅ DONE — see
   `providers/liveportrait/adapter/_mediapipe.py` and the gate test
   `tests/smoke/test_real_gpu/test_mediapipe_identity.py`. The
   cascade is MediaPipe first → OpenCV Haar fallback → center crop
   last; the chosen detector is recorded in `identity_meta.json`.
2. **Audio-bridge neural replacement.** ✅ DONE — the
   `providers/liveportrait/audio_bridge/` package now dispatches
   between the legacy DSP bridge (`HEYAVATAR_AUDIO_BRIDGE_BACKEND=dsp`)
   and SadTalker Audio2Motion (`HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural`).
   See `providers/liveportrait/audio_bridge/sadtalker.py` for the
   import contract and `providers/liveportrait/audio_bridge/projection.py`
   for the 3DMM(50) → LP(21, 3) projection (calibrated placeholder;
   real W matrix ships via the GPU-worker calibration step). SadTalker
   weights are trained on LRW + VoxCeleb (non-commercial research
   corpora), so the **weights** themselves are non-commercial —
   `liveportrait-human-v1.commercial_use` stays `false` for now.
3. **Pin every checkpoint SHA256** in `registry/models.yaml` — the
   current `TBD` markers block supply-chain trust. Run the new
   `tests/providers/test_musetalk_real_mode.py::test_musetalk_checkpoint_verify_policy`
   branch with the SHA pins populated.
4. **Pass `tests/contracts/` AND `tests/smoke/test_real_gpu/test_pipeline.py`
   in real (non-mock) mode.** The new
   `test_pipeline.py::_lower_face_ssd` check partitions the rendered
   mp4 into silence/speech windows and asserts the per-frame pixel
   SSD in the mouth ROI is significantly higher in the active-speech
   window than in the silence window. This is the **release gate**
   that proves the audio bridge drives the mouth in time with audio,
   not the static `0x111111`/`0x330000` dummy. A green run validates
   the end-to-end adapter contract on the target production stack.

When 1 + 2 + 4 pass on the production stack, the technical
pre-conditions for `commercial_use: true` are met. The flag does NOT
flip automatically because SadTalker's audio2motion weights are
non-commercial; to release commercially you must either:

* (preferred) replace SadTalker Audio2Motion with a
  commercially-licensed model (e.g. WHSP → ARKit-blendshape, or a
  Studio-grade in-house model), or
* accept that the audio bridge ships non-commercial weights and
  document the limitation in customer contracts.

Until those items close, **live mode of `liveportrait-human-v1` is for
research and pilot users only**.

See `providers/liveportrait/inference_config.py` for the schema
version (`LIVE_PORTRAIT_PACK_VERSION`); bumps must come with a
migration path in `_load_source_bundle`.

## SadTalker Audio2Motion integration

Production-grade lip-sync is now wired through
`providers/liveportrait/audio_bridge/sadtalker.py`. The bridge
follows a deliberately constrained policy:

* `HEYAVATAR_AUDIO_BRIDGE_BACKEND=neural` requires SadTalker's
  `audio2motion.models.Audio2MotionModel` to import on the GPU
  worker. If the import fails the bridge raises `RuntimeError` and
  the engine transitions to `EngineState.DEGRADED` so the
  orchestrator routes around the broken worker.
* The neon^functional policy is "never silent fallback" — a worker
  with broken CUDA extensions cannot ship DSP-tier mouth motion to
  paying customers. The dsp backend stays the default for CI
  (`HEYAVATAR_AUDIO_BRIDGE_BACKEND=dsp`) so mock-mode tests stay
  green.
