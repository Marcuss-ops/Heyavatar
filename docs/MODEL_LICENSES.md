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

See `providers/liveportrait/inference_config.py` for the schema
version (`LIVE_PORTRAIT_PACK_VERSION`); bumps must come with a
migration path in `_load_source_bundle`.
