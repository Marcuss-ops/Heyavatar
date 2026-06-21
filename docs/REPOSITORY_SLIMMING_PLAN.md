# Heyavatar Repository Slimming Plan

## Status

This document defines the immediate repository-reduction plan for Heyavatar.

The goal is not to redesign the platform. The goal is to remove speculative architecture, duplicate abstractions, placeholder providers, and premature infrastructure so the repository can focus on one production path:

```text
body template
→ face generation / reenactment
→ lip-sync
→ face compositing
→ audio mux
→ quality checks
→ final MP4
```

Until this path is stable on real inputs, every new feature must support it directly.

---

## 1. Product boundary for the current MVP

Heyavatar currently needs to do only this:

1. Receive an avatar identity, a precomputed body template, and audio.
2. Load one persistent GPU worker.
3. Generate or adapt the face.
4. Apply real lip-sync.
5. Composite the generated face over the body template.
6. Encode and mux the final video.
7. Fail explicitly when inference, compositing, encoding, or QC fails.
8. Return a real playable MP4 path or URL.

The MVP does **not** need:

- universal 3D body retargeting;
- SMPL-X motion infrastructure;
- NeRF avatars;
- EchoMimic;
- three quality tiers;
- multi-engine fallback routing;
- a distributed warm pool;
- dynamic GPU routing;
- S3 support;
- full OpenTelemetry tracing;
- a semantic gesture planner;
- automatic motion capture;
- multiple implementations behind every contract.

---

## 2. Keep: canonical production path

The following areas are part of the current product and must remain.

### API and job execution

Keep:

```text
api/
contracts/avatar_engine.py
contracts/job_queue.py
src/core/
src/domain/
src/storage/jobs/
workers/gpu_worker/
workers/encoding_worker/
```

Required capabilities:

- FastAPI request validation;
- Redis-backed jobs;
- persistent worker process;
- explicit terminal job states;
- cancellation;
- result metadata;
- one real render pipeline;
- deterministic mock mode for tests only.

### Avatar and rendering components

Keep:

```text
providers/liveportrait/
providers/musetalk/
tools/avatar_assets/
src/storage/avatar_packs/
```

Required capabilities:

- identity preparation;
- body-template preprocessing;
- face transforms;
- face and neck masks;
- real face generation or reenactment;
- real MuseTalk integration;
- final compositing;
- encoding;
- QC.

### Tests

Keep tests that validate:

- API → queue → worker → output;
- real-mode strictness;
- no random or coloured fallback in production;
- valid MP4 output;
- duration and frame consistency;
- compositing without debug overlays;
- job-state correctness.

---

## 3. Remove now: placeholder architecture

Any provider that writes text bytes instead of producing a valid asset must be removed.

Examples of forbidden placeholder behaviour:

```python
output_path.write_bytes(b"LIPSYNCED FACE OUTPUT")
output_path.write_bytes(b"COMPOSITED VIDEO OUTPUT WITH POISSON BLENDING")
body_video.write_bytes(b"PRERECORDED BODY CLIP")
```

A missing production asset must raise an explicit error.

Correct behaviour:

```python
if not body_video.is_file():
    raise FileNotFoundError(body_video)
```

No fake `.mp4` file may be treated as a completed result.

---

## 4. Remove now: speculative contracts

Do not keep an abstraction solely because it may be useful later.

### Remove `MotionRepository` and `MotionClip`

Remove the current universal motion model containing:

- root translation;
- body rotations;
- left-hand rotations;
- right-hand rotations;
- head rotations;
- motion phases.

Reason:

The current runtime uses prerecorded body templates. There is no production 3D rig or universal retargeting implementation consuming this contract.

Reintroduce it only when Heyavatar has:

1. a real rigged avatar;
2. a real retargeter;
3. a real renderer;
4. at least one end-to-end test using canonical motion data.

### Remove `FaceRenderer`

`AvatarEngine` already owns face rendering.

Do not maintain both:

```text
AvatarEngine.render_chunk()
FaceRenderer.render_face()
```

Keep one canonical rendering boundary: `AvatarEngine`.

### Remove duplicate `LipSyncEngine` scaffolding

Do not maintain both:

```text
providers/musetalk/
providers/lipsync/musetalk/
```

Keep one real MuseTalk implementation under `providers/musetalk/`.

Extract a generic `LipSyncEngine` only after a second real lip-sync backend exists.

### Remove generic `BodyAssetProvider`

The current MVP supports one body mode:

```text
prerecorded_template
```

Replace the generic provider with a concrete data model and resolver:

```python
@dataclass(frozen=True)
class BodyTemplate:
    body_video: Path
    face_mask: Path
    neck_mask: Path
    face_transforms: Path
    metadata: Path
```

```python
def load_body_template(avatar_id: str, gesture_id: str) -> BodyTemplate:
    ...
```

Do not include unused dimensions such as:

- outfit ID;
- camera ID;
- lighting ID;
- renderer ID;
- body-generation fallback.

Add them only when they are real runtime inputs.

### Remove generic `QualityChecker` ABC

Replace it with one concrete module:

```text
src/pipeline/quality.py
```

It must perform actual checks and return real failures.

Required checks:

- output readable by ffprobe;
- video duration versus audio duration;
- expected frame count;
- black-frame ratio;
- debug-green detection;
- invalid or missing transforms;
- missing audio stream;
- empty output.

A quality checker that always returns `passed=True` must not exist.

### Remove placeholder compositor provider

Do not keep a compositor class that only creates a fake file.

Move the real face-compositing implementation to one canonical module:

```text
src/pipeline/compositor.py
```

This module should own:

- affine warp;
- face mask;
- neck mask;
- feathering;
- colour matching;
- optional seamless cloning;
- alpha composition;
- temporal smoothing.

All tools and runtime paths must use this same implementation.

---

## 5. Freeze now: valid future work that is premature

The following areas should be removed from the active runtime or placed behind disabled experimental flags.

### SadTalker audio bridge

Freeze:

```text
providers/liveportrait/audio_bridge/sadtalker.py
providers/liveportrait/audio_bridge/projection.py
SadTalker coefficient projection
neural audio bridge selection
```

Reason:

MuseTalk is the current lip-sync path, while body templates already provide head and upper-body motion.

The MVP does not need a second audio-to-motion model.

Re-enable only after:

- the main LivePortrait + MuseTalk pipeline is stable;
- commercial licensing is confirmed;
- the neural bridge visibly improves output;
- a real GPU benchmark justifies the extra dependency.

### EchoMimic

Remove EchoMimic from:

- provider registration;
- runtime routing;
- tier fallback chains;
- worker selection;
- documentation describing current capability.

It may remain only in `ROADMAP.md`.

### Quality tiers

Remove the active distinction between:

```text
Express
Studio
Premium
```

Use one profile:

```text
standard
```

Initial standard profile:

```text
LivePortrait + MuseTalk
25 FPS
face ROI 256×256
H.264 output
prerecorded body template
```

Add tiers only after real benchmark data exists.

### Distributed WorkerPool and complex heartbeat

Current deployment target:

```text
1 API
1 Redis
1 GPU worker
1 encoder path
```

Keep only a minimal worker health record:

```text
worker_id
last_seen
engine_id
state
```

Freeze:

- cross-process capacity routing;
- fallback routing by VRAM;
- warm pool;
- burst pool;
- multi-engine capacity scoring;
- frequent WorkerPool sync threads.

### Full tracing stack

Freeze:

- OpenTelemetry exporters;
- W3C trace propagation;
- Grafana dashboards;
- OTLP collector assumptions;
- tracing-specific package structure.

Keep:

- structured logs;
- `job_id`;
- `avatar_id`;
- engine ID;
- stage duration;
- GPU seconds;
- terminal state;
- failure reason.

### S3 and multiple object stores

Use local filesystem storage for the MVP:

```text
avatar_packs/
body_templates/
captures/
```

Freeze:

- S3 backend;
- aioboto3;
- signed URLs;
- storage provider selection;
- multi-store resolvers.

---

## 6. Defer: gesture intelligence

Do not implement the semantic gesture planner yet.

For the next integration test, use a manual timeline:

```json
{
  "segments": [
    {"gesture_id": "idle", "duration_seconds": 3.0},
    {"gesture_id": "explain_both", "duration_seconds": 2.0},
    {"gesture_id": "idle", "duration_seconds": 3.0}
  ]
}
```

Before adding an LLM, Heyavatar must prove that it can concatenate multiple body templates while keeping all associated data aligned:

- video frames;
- face transforms;
- face masks;
- neck masks;
- timestamps;
- audio;
- final face compositing.

The semantic planner comes after this deterministic timeline works.

---

## 7. Target repository structure

The active repository should converge toward:

```text
Heyavatar/
├── api/
│   ├── app.py
│   ├── routes/
│   │   ├── avatars.py
│   │   ├── jobs.py
│   │   └── health.py
│   └── schemas.py
│
├── contracts/
│   ├── avatar_engine.py
│   └── job_queue.py
│
├── src/
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── domain/
│   │   ├── avatars.py
│   │   ├── jobs.py
│   │   └── render.py
│   ├── pipeline/
│   │   ├── compile_avatar.py
│   │   ├── body_templates.py
│   │   ├── compositor.py
│   │   ├── quality.py
│   │   └── render_video.py
│   └── storage/
│       ├── avatar_packs.py
│       └── jobs.py
│
├── providers/
│   ├── liveportrait/
│   └── musetalk/
│
├── workers/
│   ├── render_worker.py
│   └── encoding.py
│
├── tools/
│   └── avatar_assets/
│       ├── precompute_video_template.py
│       └── preview_face_composite.py
│
├── registry/
│   └── gestures.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
│
├── README.md
├── ARCHITECTURE.md
└── pyproject.toml
```

This is a target, not permission for one giant refactor.

---

## 8. One canonical application use case

The repository should expose one primary application operation:

```python
render_avatar_video(
    avatar_id: str,
    body_template_id: str,
    audio_path: Path,
) -> RenderResult
```

Internal flow:

```text
load body template
→ prepare / render face
→ MuseTalk lip-sync
→ composite face
→ encode and mux audio
→ run QC
→ store result
```

No alternative path may report success without generating a valid video.

---

## 9. Mandatory deletion rules

A file or subsystem should be removed when all of the following are true:

1. It is not called by the active end-to-end pipeline.
2. It is not required by an executable offline tool.
3. It has no real implementation.
4. It produces placeholders or simulated success.
5. It duplicates an existing responsibility.
6. It exists only for a future engine, tier, or deployment model.
7. Git history already preserves it.

A file should not be removed when:

1. It is used by the real render path.
2. It is required for job-state correctness.
3. It is part of real-mode failure handling.
4. It has a focused test proving current behaviour.
5. Removing it would mix unrelated refactoring into the slimming work.

---

## 10. Execution plan

Do not perform a single massive deletion commit.

### Change 1 — Remove placeholder architecture

Remove:

- `MotionRepository` and unused motion-domain models;
- duplicate `FaceRenderer` contract;
- duplicate `LipSyncEngine` scaffolding;
- generic `BodyAssetProvider` scaffolding;
- fake compositor provider;
- fake QC provider;
- every provider that creates invalid placeholder assets.

Acceptance criteria:

- no runtime code writes textual placeholder bytes to `.mp4` files;
- imports and tests remain green;
- the existing real render path remains unchanged.

### Change 2 — Unify compositing

Create one real canonical compositor.

Acceptance criteria:

- offline preview and runtime use the same module;
- masks never become visible debug overlays in production;
- output is a valid video;
- debug output is stored separately;
- QC catches green overlays and unreadable output.

### Change 3 — Slim the runtime

Disable or remove active runtime use of:

- EchoMimic;
- SadTalker;
- tier routing;
- cross-engine fallback;
- distributed capacity scoring;
- S3;
- full tracing stack.

Acceptance criteria:

- API + Redis + one render worker can complete a real job;
- the configuration surface is smaller;
- the worker loads only required models;
- README documents only working capability.

### Change 4 — Deterministic multi-template timeline

Add a manual body-template timeline before semantic planning.

Acceptance criteria:

- at least three body clips can be concatenated;
- video, masks, and transform arrays remain frame-aligned;
- MuseTalk and compositing operate on the combined timeline;
- final audio and video durations match.

---

## 11. Definition of done for the slim MVP

The repository is considered successfully reduced when this command path works reliably:

```text
avatar identity
+ body template
+ WAV audio
→ final MP4
```

Required result:

- real face generation or reenactment;
- real lip-sync;
- stable face placement;
- no green mask or debug overlay;
- no random frames;
- no text-placeholder files;
- valid audio stream;
- correct duration;
- correct terminal state;
- explicit error on failure;
- one documented production path.

The MVP is not complete merely because unit tests pass. It must produce a visually valid video from real assets.

---

## 12. Rule for all future additions

A new feature may enter the active repository only if at least one of these is true:

1. It is called by the production end-to-end pipeline.
2. It is used by a real executable offline tool.
3. It replaces an existing implementation with a tested canonical implementation.
4. It has a measurable production requirement.

Do not add a registry, resolver, provider, contract, worker, or backend solely because it may be useful later.

Prefer this progression:

```text
concrete implementation
→ real output
→ focused tests
→ second implementation
→ shared contract / registry
```

Not this:

```text
contract
→ provider hierarchy
→ placeholder implementation
→ future plan
```

---

## Final direction

Heyavatar's current mission is:

> Take a prerecorded body template, an avatar identity, and audio, then produce one clean, synchronized, playable MP4.

Everything that does not directly support that mission should be removed, frozen, or moved to the roadmap until the core pipeline is proven.