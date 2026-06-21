"""GPU + CUDA health check.

Verifies that PyTorch can see an NVIDIA device with enough VRAM and
that a trivial op runs end-to-end. This is the prerequisite for every
other scenario in this package — if it skips, so do all the rest.
"""

from __future__ import annotations

import pytest

from tests.smoke.test_real_gpu._helpers import requires_cuda

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - requires_cuda marker also guards
    torch = None  # type: ignore


@requires_cuda
def test_gpu_cuda_healthy():
    """Verify the GPU is reachable and has sufficient VRAM."""
    assert torch.cuda.is_available(), "CUDA must be available"
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory // (1 << 30)
    assert vram_gb >= 4, f"Need >= 4 GB VRAM, got {vram_gb} GB"
    # Smoke: allocate a small tensor and run an op.
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    assert y.shape == (100, 100)
    print(f"\nGPU: {props.name} ({vram_gb} GB VRAM)")
