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
```

A green contract run means every provider still satisfies the
`AvatarEngine` contract.
