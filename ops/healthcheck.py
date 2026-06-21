"""Container healthcheck for the Heyavatar services.

Picks the right probe based on the role:

* ``api``         — HTTP ``GET /ping`` against the FastAPI gateway.
* ``gpu-worker``  — HTTP ``GET /metrics`` against the worker's exposition
                    port (9100 default). Even a single ``# HELP`` line in
                    the Prometheus text output proves the worker booted
                    far enough to publish metrics.
* ``encoder``     — Same shape as gpu-worker: confirms the encoding
                    process is responsive.

Exit codes follow Docker convention:

* ``0`` — healthy.
* ``1`` — probe failed (Docker will mark the container unhealthy).

The script does NOT import ``requests``; using ``urllib.request`` keeps
the image slim and avoids dependency drift between CI and runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from typing import Tuple


def _probe(url: str, *, timeout: float = 3.0) -> Tuple[int, str]:
    """Return ``(status_code, body_prefix)`` for ``url``.

    A failed connection raises an ``OSError`` that is mapped to exit 1
    by the caller.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "heyavatar-healthcheck/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(1024).decode("utf-8", errors="ignore")
        return resp.status, body


def _probe_api(port: int) -> int:
    """Hit ``GET /ping`` on the API gateway."""
    status, _ = _probe(f"http://127.0.0.1:{port}/ping")
    if status != 200:
        print(f"api healthcheck: unexpected status {status}", file=sys.stderr)
        return 1
    return 0


def _probe_worker(port: int) -> int:
    """Hit ``GET /metrics`` on the worker Prometheus port.

    The contract is just "the exposition server is up" — we accept a
    200 with or without metrics samples yet, because a freshly-booted
    worker may still be warming its instruments.
    """
    status, _ = _probe(f"http://127.0.0.1:{port}/metrics")
    if status != 200:
        print(f"worker healthcheck: unexpected status {status}", file=sys.stderr)
        return 1
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Container health probe.")
    parser.add_argument(
        "role",
        nargs="?",
        default=os.environ.get("SERVICE_ROLE", "gpu-worker"),
        choices=["api", "gpu-worker", "encoder"],
        help="Which service to probe (default: SERVICE_ROLE env or gpu-worker).",
    )
    parser.add_argument("--port", type=int, default=None, help="Override probe port.")
    args = parser.parse_args(argv)

    if args.role == "api":
        port = args.port or int(os.environ.get("HEALTHCHECK_API_PORT", "8000"))
        return _probe_api(port)

    if args.role in ("gpu-worker", "encoder"):
        port = args.port or int(os.environ.get("HEALTHCHECK_WORKER_PORT", "9100"))
        return _probe_worker(port)

    print(f"unknown role: {args.role}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
