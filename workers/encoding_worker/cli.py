"""Command-line entry point for the encoding worker.

Drives :class:`EncodingWorker.encode` from a shell invocation. The primary
use case is running the worker out-of-process during manual integration
testing or via a fork-style supervisor; production deployments use
:class:`GpuWorker` which inlines the encoding step inside the worker
loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.config import get_settings
from src.core.logging import configure_logging

from workers.encoding_worker.worker import EncodingWorker


def main() -> int:  # pragma: no cover - manual integration
    parser = argparse.ArgumentParser(description="Run the encoding worker.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--codec", default="h264")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    worker = EncodingWorker(settings=settings)
    out = worker.encode(args.job_id, args.manifest, audio_path=args.audio, codec=args.codec)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
