"""End-to-end pipeline tests split by scenario.

Subpackages under this one each cover one flow:

* :mod:`test_happy_path` — full API → Worker → Encoder → Status walk.
* :mod:`test_compile_only` — `POST /avatars/compile` → job → pack ready.
* :mod:`test_failure_recording` — failed jobs persisted in queue + repo.

Shared helpers (the ``requires_ffmpeg`` pytest marker and the
``_publish_render_job`` / ``_simulate_worker_reserve`` plumbing) live
in :mod:`_helpers`. Pytest discovers each ``test_*.py`` file under
this package automatically.
"""
