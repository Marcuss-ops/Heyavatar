# Heyavatar 🗣️
### Avatar Engine Platform — multi-process, multi-engine, real-time-capable.

Heyavatar is a **proprietary avatar rendering engine**, not a CLI wrapper.
It splits the platform into four independent processes:

```
┌──────────────┐      ┌──────────────┐      ┌────────────────┐
│ FastAPI      │ ───▶ │  Job Queue   │ ───▶ │  GPU Worker    │
│  Gateway     │      │  (Redis      │      │  (LivePortrait│
│  (stateless) │      │   Streams)   │      │   / MuseTalk /│
└──────────────┘      └──────────────┘      │   EchoMimic)   │
                                            └──────┬─────────┘
                                                   ▼
                                            ┌────────────────┐
                                            │ Encoder Worker │
                                            │ (NVENC/ffmpeg) │
                                            └────────────────┘
```

The gateway **never** owns a GPU. Each engine runs in its **own container**
with its specific Python/CUDA/pytorch version. Per-identity work is
amortised across many videos via the persistent **`Avatar Pack`**.

## ✨ Why this shape?

- **Cost**: pre-computed avatar packs + face-region-only animation.
- **Throughput**: persistent GPU workers batch jobs; warm pool + burst pool.
- **Resilience**: VRAM fragmented → restart worker; request cancelled →
  Redis cancel flag consulted every N frames.
- **Licensing**: every engine + dependency tracked in `registry/models.yaml`.

## 📦 Quickstart (mock mode — runs anywhere, no GPU required)

```bash
git clone https://github.com/Marcuss-ops/Heyavatar.git
cd Heyavatar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Compile a synthetic identity, then render a 2-second chunk:
HEYAVATAR_MOCK_ENGINE=1 pytest tests/smoke/test_end_to_end.py -q
```

The mock adapters produce a deterministic black `.mp4` for every chunk
and write an in-memory avatar pack to `HEYAVATAR_PACK_DIR`.

## 🧠 Architecture

| Layer            | Lives in           | Why                                            |
|------------------|--------------------|------------------------------------------------|
| `api/`           | one process        | HTTP, auth, request validation, job publishing |
| `contracts/`     | importable ABC     | Stable interface; no engine code               |
| `src/domain/`    | pure dataclasses   | No IO — flows between layers without coupling  |
| `src/application/` | use cases        | compile, render, telemetry                     |
| `src/scheduler/` | scheduler host     | Queue + worker pool + tier router              |
| `src/storage/`   | repositories       | Avatar packs / jobs / blobs (FS or S3)         |
| `providers/`     | one dir per engine | Adapter implementation behind the contract     |
| `workers/`       | entrypoints        | GPU worker + encoding worker                   |
| `registry/`      | YAML               | Model + license source of truth                |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full breakdown.

## 🗺️ Where to find it

Following the *one concern, one module* rule, every package is split
into thematic subdirs. Jump straight to the row that names the
behaviour you're hunting. **Imports always go through the most-specific
submodule** — e.g. `from workers.encoding_worker.worker import
EncodingWorker`, not `from workers.encoding_worker import …`.

### Production orchestration

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| `RenderVideo` orchestrator + chunking            | `src/application/render_video/use_case.py`               |
| `ChunkConfig` (size + retries + max-chunks cap)  | `src/application/render_video/config.py`                 |
| Audio duration probe (`ffprobe` wrapper)         | `src/application/render_video/audio_probe.py`            |
| Chunk-list manifest writer                       | `src/application/render_video/manifest.py`               |

### Workers

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| GPU worker class + reservation loop              | `workers/gpu_worker/worker.py`                           |
| Worker per-job processing (`_do_process`)        | `workers/gpu_worker/process.py`                          |
| Worker Prometheus + pack-archive reader          | `workers/gpu_worker/telemetry.py`                        |
| Worker entry point (`main()`, `build_queue`)     | `workers/gpu_worker/cli.py`                              |
| `EncodingWorker` (chunk-list assembler)          | `workers/encoding_worker/worker.py`                      |
| Encoding manifest parser                         | `workers/encoding_worker/manifest.py`                    |
| Codec picker (`h264_nvenc` vs `libx264`)         | `workers/encoding_worker/codec.py`                       |
| Encoding CLI (`main()`)                          | `workers/encoding_worker/cli.py`                         |

### Scheduler & queue

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| In-process queue                                 | `src/scheduler/queue/memory.py`                          |
| Drop-everything queue (smoke + tests)           | `src/scheduler/queue/null.py`                            |
| Redis Streams queue (production)                | `src/scheduler/queue/redis.py`                           |
| `WorkerPool` + live worker records               | `src/scheduler/routing/worker_pool.py`                   |
| `TierRouter` (registry-driven primary+fallback) | `src/scheduler/routing/router.py`                        |
| Capacity-aware engine picker                     | `TierRouter.pick_available(tier, pool)`                  |

### Storage

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| In-memory job repository                         | `src/storage/jobs/memory.py`                             |
| Redis-backed job repository                      | `src/storage/jobs/redis.py`                              |
| Avatar pack repository (`.tar` round-trip)       | `src/storage/avatar_packs.py`                            |
| Object store (FS today, S3 tomorrow)             | `src/storage/object_store.py`                            |

### API gateway

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| `create_app()` + module-level `app` ASGI instance | `api/app/factory.py` — also re-exported via `api/app/__init__.py` so Uvicorn's `api.app:app` convention still resolves |
| `AppState` dataclass + `lifespan` async context | `api/app/state.py`                                       |
| Queue-backend picker (`_build_queue`)            | `api/app/queue_factory.py`                               |
| Prometheus `/metrics` mount + middleware         | `api/app/metrics.py`                                     |

### Observability

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| OpenTelemetry tracer + OTLP provider lifecycle   | `src/observability/distributed/tracing.py`               |
| W3C `traceparent` inject + extract across the queue boundary | `src/observability/distributed/propagation.py` |
| Prometheus metric constants + label cardinality | `src/observability/metrics/constants.py`                |
| Prometheus instruments (counters/gauges/histograms) | `src/observability/metrics/instruments.py`            |
| Recording helpers + helpers for terminal/job state | `src/observability/metrics/recorders.py`               |
| `/metrics` text exposition                       | `src/observability/metrics/exposition.py`               |

### LivePortrait engine

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| `LivePortraitAdapter` dataclass + facade         | `providers/liveportrait/adapter/engine.py`               |
| Real-mode method implementations                  | `providers/liveportrait/adapter/{_upstream, _identity, _render}.py` (attached to the class at module load) |
| Mock fallback (used under `HEYAVATAR_MOCK_ENGINE=1`) | `providers/liveportrait/adapter/_mock.py`            |
| Checkpoint manifest + SHA256 pinning             | `providers/liveportrait/checkpoint_manager/manifest.py`  |
| Checkpoint CLI / verifier                        | `providers/liveportrait/checkpoint_manager/manager.py`   |
| HuggingFace + urllib download with smart cache   | `providers/liveportrait/checkpoint_manager/downloader.py` |
| Audio → driving envelopes (21-keypoint tensors)  | `providers/liveportrait/audio_bridge/{bridge, types, dsp}.py` |
| Typed mirrors of upstream `InferenceConfig`/`CropConfig` | `providers/liveportrait/inference_config.py` (kept flat; 3 dataclasses, low file size) |

### MuseTalk engine

| Concern                                          | File                                                     |
|--------------------------------------------------|----------------------------------------------------------|
| `MuseTalkAdapter` dataclass + facade             | `providers/musetalk/adapter/engine.py`                   |
| Real-mode method implementations                  | `providers/musetalk/adapter/{_upstream, _identity, _render}.py` (attached at module load) |
| Mock fallback                                    | `providers/musetalk/adapter/_mock.py`                    |
| Per-adapter checkpoint manager                   | `providers/musetalk/adapter/checkpoints.py`              |

### Navigation rules of thumb

- **Cross-sibling coupling is rare.** Most refactored subdirs share
  no imports between siblings and compose only through their parent's
  composition root (or by attaching free functions to the dataclass at
  module load — the adapter pattern in `providers/{liveportrait,
  musetalk}/adapter/`). The exception today is
  `src/scheduler/routing/`, where `router.py` reaches its sibling
  via `from .worker_pool import WorkerPool`.
- **Cross-package** imports are absolute — e.g.
  `from src.scheduler.routing.router import TierRouter` from a route
  handler.
- **No compat layer (almost everywhere).** Old paths are gone; if
  something imports `from src.application.render_video import
  RenderVideo`, that import is stale and will fail — use `from
  src.application.render_video.use_case import RenderVideo` instead.
  The one deliberate exception is `api/app/__init__.py`, which
  re-exports `app` and `create_app` from `factory.py` so Uvicorn's
  documented `api.app:app` launch convention keeps resolving.

## 🎚️ Three quality tiers

| Tier     | Primary engine       | Use case                                  | VRAM  | Cost/GPU-s |
|----------|----------------------|-------------------------------------------|-------|------------|
| Express  | `musetalk-v1`        | News, courses, dubbing, bulk content      | 4 GB  | ★★         |
| Studio   | `liveportrait-human-v1` | Half-body shots, marketing clips       | 6 GB  | ★★★       |
| Premium  | `echomimic-v1`       | Premium ads, full-body, anchor control    | 8 GB+ | ★★★★      |

The scheduler picks automatically based on the request's `tier` and on
current `WorkerPool` capacity. See `src/scheduler/router.py`.

## ⚙️ Configuration

Every knob is an environment variable; see `src/core/config.py`.

```text
HEYAVATAR_LOG_LEVEL                  # DEBUG | INFO | WARNING | ERROR
HEYAVATAR_MOCK_ENGINE=1              # run all adapters in mock mode
HEYAVATAR_QUEUE_BACKEND              # null | memory | redis
HEYAVATAR_REGISTRY                   # path to registry/models.yaml
HEYAVATAR_PACK_DIR                   # where avatar packs live
HEYAVATAR_CAPTURE_DIR                # where rendered chunks live
HEYAVATAR_OBJECT_STORE               # root for the FS backend
REDIS_URL                            # redis://... for production
HEYAVATAR_API_KEY                    # optional X-API-Key check
```

## 🚀 Running the gateway

```bash
HEYAVATAR_QUEUE_BACKEND=memory HEYAVATAR_MOCK_ENGINE=1 \
    uvicorn api.app:app --reload --port 8000
```

Then probe it:

```bash
curl http://localhost:8000/ping             # {"status":"pong"}
curl http://localhost:8000/healthz          # queue / backend / mock-mode stats
curl -X POST http://localhost:8000/avatars/compile \
     -H "Content-Type: application/json" \
     -d '{"source_image": "examples/face.png"}'
```

## 🛠️ Development

```bash
pip install -e ".[dev]"
ruff check src api providers workers contracts
pytest -q
```

## 📜 License

MIT — see [`LICENSE`](LICENSE). Engine-specific licenses (especially
model weights and bundled detectors) are tracked in `registry/models.yaml`
and reviewed before any commercial deployment.
