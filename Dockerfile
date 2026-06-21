# =============================================================================
#  Heyavatar — multi-stage production image
#  -----------------------------------------------------------------------------
#  Two stages share the dependency-install layer so we cache torch + base
#  deps once. The `MultiScaleDeformableAttention` CUDA op is compiled in a
#  dedicated layer BEFORE COPY . so that business-logic changes do not
#  invalidate a 5-10 min custion build.
#
#  Build modes (set via docker build --target=):
#    --target=api         →  lightweight image for the FastAPI gateway
#    --target=gpu-worker  →  image ready for the GPU worker process
#    (no target)          →  default small `runtime` image with no service
# =============================================================================

# Pin Python + CUDA. Bumping these is a deliberate, reviewable change.
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04

# -----------------------------------------------------------------------------
# Stage 1 — base dependencies shared by both API and GPU worker
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps for HDF5/tarfile wheels, ffmpeg, ffprobe (encoder worker needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer; rebuild only when pyproject.toml changes)
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install -e ".[dev,observability]"

# -----------------------------------------------------------------------------
# Stage 2 — API service (no CUDA, no torch)
# -----------------------------------------------------------------------------
FROM base AS api

COPY api ./api
COPY src ./src
COPY contracts ./contracts
COPY providers/_ffmpeg.py ./providers/_ffmpeg.py
COPY providers/__init__.py ./providers/__init__.py
COPY providers/liveportrait/__init__.py ./providers/liveportrait/__init__.py
COPY providers/musetalk/__init__.py ./providers/musetalk/__init__.py
COPY providers/echomimic/__init__.py ./providers/echomimic/__init__.py
COPY providers/echomimic/adapter.py ./providers/echomimic/adapter.py
COPY registry ./registry
COPY workers/encoding_worker ./workers/encoding_worker
COPY workers/__init__.py ./workers/__init__.py

# Install the package deps without the GPU extras (no torch in this image).
RUN pip install -e ".[dev,observability,redis]"

# Health probe port (FastAPI runs on 8000).
EXPOSE 8000
RUN chmod +x ops/docker_entrypoint.sh
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python ops/healthcheck.py api || exit 1

ENV HEYAVATAR_PROCESS_ROLE=api \
    HEYAVATAR_QUEUE_BACKEND=redis \
    HEYAVATAR_MOCK_ENGINE=0

ENTRYPOINT ["ops/docker_entrypoint.sh"]
CMD ["api"]

# -----------------------------------------------------------------------------
# Stage 3 — GPU-worker service (CUDA + torch + MultiScaleDeformableAttention)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS cuda-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        pkg-config \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------------------
# Dedicated CUDA-op build layer. Keeps 5-10 min compile out of the fast
# rebuild path. Build only when upstream LivePortrait changes.
# -----------------------------------------------------------------------------
FROM cuda-base AS cuda-op-builder

ARG LIVEPORTRAIT_REF=main
RUN git clone --depth 1 --branch ${LIVEPORTRAIT_REF} \
        https://github.com/KlingAIResearch/LivePortrait.git /tmp/LivePortrait && \
    cd /tmp/LivePortrait && \
    bash tools/prepare_env.sh 2>&1 | tail -50 && \
    pip install -e /tmp/LivePortrait 2>&1 | tail -10

# -----------------------------------------------------------------------------
# Stage 4 — operational GPU-worker image
# -----------------------------------------------------------------------------
FROM cuda-base AS gpu-worker

COPY --from=cuda-op-builder /tmp/LivePortrait /opt/LivePortrait
# Re-install LivePortrait's Python package in this stage so imports of the
# upstream outside the dynamic-package loader (rare but possible) work
# too. The build cache above means this is essentially free.
RUN pip install -e /opt/LivePortrait 2>&1 | tail -3
RUN chmod +x ops/docker_entrypoint.sh
ENV HEYAVATAR_LIVE_PORTRAIT_SRC=/opt/LivePortrait \
    PYTHONPATH=/opt/LivePortrait:${PYTHONPATH}

# Install Python deps (this layer is invalidated only on pyproject changes).
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -e ".[dev,observability,gpu,redis]"

# Application source
COPY api ./api
COPY src ./src
COPY contracts ./contracts
COPY providers ./providers
COPY workers ./workers
COPY registry ./registry

EXPOSE 9100
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python ops/healthcheck.py gpu-worker || exit 1

ENV HEYAVATAR_PROCESS_ROLE=worker \
    HEYAVATAR_QUEUE_BACKEND=redis \
    HEYAVATAR_MOCK_ENGINE=0 \
    HEYAVATAR_API_METRICS_ENABLED=1

ENTRYPOINT ["ops/docker_entrypoint.sh"]
CMD ["gpu-worker"]

# -----------------------------------------------------------------------------
# Default small runtime image (used by ops scripts / linting stages)
# -----------------------------------------------------------------------------
FROM base AS runtime
COPY ops ./ops
CMD ["python", "ops/healthcheck.py", "--version"]
