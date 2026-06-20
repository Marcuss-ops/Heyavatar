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
