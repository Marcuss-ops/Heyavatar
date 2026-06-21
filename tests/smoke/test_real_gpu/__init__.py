"""Real-GPU smoke tests split by scenario.

Verifies the full pipeline with ``HEYAVATAR_MOCK_ENGINE=0`` and a real
NVIDIA GPU. Intended to be run on a dedicated GPU worker node (with
CUDA toolkit, MSVC/Linux build tools, and the LivePortrait upstream
repo cloned and built).

Pre-requisites (before running these tests)
-------------------------------------------
1. ``pip install torch --index-url https://download.pytorch.org/whl/cu124``
2. ``pip install huggingface_hub``
3. Clone LivePortrait: ``git clone https://github.com/KlingAIResearch/LivePortrait``
4. Build the CUDA op: ``cd LivePortrait && bash tools/prepare_env.sh``
   (on Windows, install MSVC Build Tools first)
5. Add the repo root to PYTHONPATH (so ``src.live_portrait_pipeline`` resolves):
   ``set PYTHONPATH=%CD%\LivePortrait;%PYTHONPATH%``
6. Set env vars:
   ``set HEYAVATAR_MOCK_ENGINE=0``
   ``set HEYAVATAR_SKIP_SHA256_VERIFY=1``
   ``set HEYAVATAR_LIVE_PORTRAIT_SRC=%CD%\LivePortrait``
7. Run: ``pytest tests/smoke/test_real_gpu -v -s``

Scenario files
--------------
* :mod:`test_gpu_health` — CUDA reachable + smoke op.
* :mod:`test_checkpoints` — checkpoint files present + SHA256 pinning.
* :mod:`test_engine_load` — engine loads + prepare_identity non-mock.
* :mod:`test_full_pipeline` — compile → render → encode end-to-end.

The :data:`requires_cuda` marker, LivePortrait repo-sys.path
bootstrap, and the test PNG/WAV factories live in :mod:`_helpers`.
Pytest discovers each ``test_*.py`` file under this package
automatically (the subdir is itself a Python package).
"""
