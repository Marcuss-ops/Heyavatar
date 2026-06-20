# Architecture

## Process boundaries

```text
            ┌─────────────────────────────────────────────┐
            │           API gateway (api/app.py)         │
            │   FastAPI · Auth · Pydantic validation      │
            │   No CUDA · No torch import · Stateless    │
            └────────────────┬────────────────────────────┘
                             │ publish() / cancel() / depth()
                             ▼
            ┌─────────────────────────────────────────────┐
            │          Job Queue (src/scheduler)          │
            │   Redis Streams prod  |  InMemory tests     │
            └────────────────┬────────────────────────────┘
                             │ reserve(handle) → ack() / fail()
                             ▼
            ┌─────────────────────────────────────────────┐
            │     GPU Worker (workers/gpu_worker.py)      │
            │   Exactly one AvatarEngine loaded per proc  │
            │   One JobQueue connection, owns its VRAM     │
            │   Persists Avatar Packs in object store     │
            └────────────────┬────────────────────────────┘
                             │ RenderResult
                             ▼
            ┌─────────────────────────────────────────────┐
            │   Encoding Worker (workers/encoding_worker) │
            │   ffmpeg concat demuxer · NVENC if present  │
            └─────────────────────────────────────────────┘
                             │
                             ▼
                  Object store + ingestion
```

## Engine contract

Every provider implements :class:`contracts.avatar_engine.AvatarEngine`:

```python
def load(self) -> None: ...
def unload(self) -> None: ...
def prepare_identity(self, source_image: Path) -> dict[str, bytes]: ...
def render_chunk(self, request: RenderChunkRequest, identity: AvatarIdentityHandle) -> RenderChunkResult: ...
def health(self) -> EngineHealth: ...
```

Application code never imports providers directly; it imports the
contract. The provider registry (`providers/__init__.py`) maps
:class:`src.domain.enums.EngineId` values to concrete adapter classes.

## Domain types

```text
domain/
├── enums.py        → Tier, EngineId
├── types.py        → RenderRequest, RenderResult, RenderChunkRequest,
│                     RenderChunkResult, AvatarPackManifest,
│                     AvatarIdentityHandle, IdentitySpec, RenderSpec
└── avatar_pack.py  → write_pack / read_pack / read_pack_asset / verify_pack
```

Every type is `dataclass(slots=True, frozen=True)` so the GPU worker
hot path can move thousands per second without GC pressure.

## Application services

- `AvatarCompiler.compile(spec)` runs the on-boarding pipeline
  (segmentation, keypoints, identity embedding, latent) and writes an
  immutable `Avatar Pack` tar on disk.
- `RenderVideo.run(request, identity)` splits the audio into
  `chunk_seconds` windows, dispatches each to the engine, then hands
  the manifest off to the encoding worker.
- `TelemetryRecorder` aggregates per-engine GPU-seconds and per-span
  latencies. The single business metric
  is **GPU-seconds consumed per minute of useful avatar output**.

## Scheduler

- `JobQueue` ABC + three implementations: `InMemoryJobQueue` (single
  process, used for tests and CI), `NullJobQueue` (queueing disabled),
  `RedisJobQueue` (production — Redis Streams + consumer groups).
- `WorkerPool` tracks heartbeat, in-flight count, and `EngineHealth`
  per worker process, keyed by `worker_id`.
- `TierRouter` reads `tiers:` from `registry/models.yaml` and returns
  the primary + fallback engine ids for each tier; `pick_available`
  walks the list until it finds an engine with idle capacity.

## Storage

- `AvatarPackRepository` keeps one pack per identity on disk; pack is
  verified at load time and the digest is content-addressed.
- `InMemoryJobRepository` is the v1 metadata store. Swap is a
  one-class change for PostgreSQL later.
- `ObjectStore` ABC + `FsObjectStore` v1; `S3ObjectStore` is added by
  implementing the same interface.

## Provider manifests

Every provider directory contains:

- `adapter.py` — concrete :class:`AvatarEngine` implementation.
- `manifest.yaml` — license, runtime, supported knobs.
  Kept in sync with `registry/models.yaml` (model-level info: weights
  SHA256, dependencies).

## Mock mode

`HEYAVATAR_MOCK_ENGINE=1` flips every adapter into a deterministic
CPU-only implementation that produces synthetic assets and a black
`.mp4` of the right duration. This is what CI uses.

## Cancellation

Each chunk polls a Redis cancel flag every
`HEYAVATAR_CANCEL_CHECK_EVERY` frames; if set, the worker stops
issuing GPU work and ack-fails the job.
