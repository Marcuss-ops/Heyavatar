#!/usr/bin/env bash
# =============================================================================
#  Docker entrypoint — dispatches by SERVICE_ROLE or first positional arg
#
#  Modes:
#    api         →  uvicorn api.app:app --host 0.0.0.0 --port 8000
#    gpu-worker  →  python -m workers.gpu_worker.cli
#                   --engine ${ENGINE_ID:-liveportrait-human-v1}
#                   --worker-id ${HEYAVATAR_WORKER_ID:-gpu-1}
#    encoder     →  python -m workers.encoding_worker.cli
#
#  Default: gpu-worker
#
#  All other args are passed through (so `--help`, etc. work).
# =============================================================================
set -euo pipefail

ROLE="${1:-${SERVICE_ROLE:-gpu-worker}}"
shift || true

case "$ROLE" in
  api)
    echo "[entrypoint] starting FastAPI gateway"
    exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  gpu-worker)
    echo "[entrypoint] starting GPU worker for engine=${ENGINE_ID:-liveportrait-human-v1}"
    exec python -m workers.gpu_worker.cli \
      --engine "${ENGINE_ID:-liveportrait-human-v1}" \
      --worker-id "${HEYAVATAR_WORKER_ID:-gpu-1}" "$@"
    ;;
  encoder)
    echo "[entrypoint] starting Encoding worker"
    # Encoding worker is event-driven: it polls Redis for finished renders
    # whose manifest was published by the gpu-worker. If your topology
    # uses a separate encoder process, wire it via workers/encoding_worker/cli.py.
    exec python -m workers.encoding_worker.cli "$@"
    ;;
  --help|-h|help)
    cat <<EOF
heyavatar docker entrypoint

Roles (positional or SERVICE_ROLE env):
  api         FastAPI gateway (port 8000)
  gpu-worker  GPU render worker (port 9100 metrics)
  encoder     ffmpeg/NVENC chunk-list assembler

Environment variables honoured:
  ENGINE_ID              liveportrait-human-v1 | musetalk-v1 | echomimic-v1
  HEYAVATAR_WORKER_ID    Stable identifier for queue + heartbeat
  HEYAVATAR_QUEUE_BACKEND  redis (default here) | memory | null
  REDIS_URL              e.g. redis://redis:6379/0
  OTEL_EXPORTER_OTLP_ENDPOINT  empty disables OTLP export
EOF
    exit 0
    ;;
  *)
    echo "[entrypoint] unknown role: $ROLE (use api|gpu-worker|encoder|--help)" >&2
    exit 64
    ;;
esac
