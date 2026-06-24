# TODO — 24 Giugno 2026

## Stato attuale
- ✅ **70 test passanti** (commit `60eee63` su `origin/main`)
- ✅ Checkpoint LivePortrait scaricati e SHA256 pinnati (498 MB)
- ✅ `huggingface_hub` integrato nel `CheckpointManager`
- ✅ Adapter aggiornato: bypassa `Cropper`, costruisce `LivePortraitWrapper` direttamente
- ✅ GPU NVIDIA RTX 4060 Ti (7 GB VRAM), `torch` 2.6.0+cu124 funzionante
- ✅ Engine load in real mode **PASSED** in 7.70s — Task 1 retroattivamente risolto (fix già implementata in `_upstream.py`)

---

## Cleanup legacy (June 2026)

- ✅ Rimossi 16 file + 2 directory vuote (fake-byte / speculativi / worker orfani, in due passate) nel commit legacy-cleanup-batch-a-b:
  - workers: `lipsync_worker.py`, `composition_worker.py`, `avatar_precompute_worker.py`.
  - providers fake-byte: `lipsync/musetalk/lip_sync.py`, `compositing/ffmpeg/compositor.py`.
  - contratti orfani: `contracts/lip_sync_engine.py`, `contracts/motion_repository.py`, `contracts/body_asset_provider.py`.
  - precompute chain non wired nel canonical: `src/application/precompute_avatar.py`, `src/body/resolver.py`, `src/motion/resolver.py`, `providers/body_assets/` (intero).
  - directory vuote rimosse: `providers/lipsync/`, `providers/body_assets/`.
- ✅ Test rimosso
- ✅ Seconda passata: rimossi 3 worker orfani (`planner_worker.py`, `face_worker.py`, `quality_worker.py`) la cui unica reference era nel test cancellato. `workers/` ora contiene solo `__init__.py`, `encoding_worker/`, `gpu_worker/`.
: `tests/smoke/test_new_architecture.py` (testava la worker chain fake-byte).
- ✅ Aggiunta dipendenza `opencv-python>=4.8` a `pyproject.toml` + `requirements.txt` (canonical `OpenCVFaceCompositor` usa `cv2`).

---


### Cross-architettura (June 2026) -- Pipelinegen wave 18 cross-project reference

Per il record canonico di questo cleanup lato infrastructure-as-code (ratchet
tracker + ownership map), vedi:
  - `architecture/migration.yaml` (repo Pipelinegen, **Wave 18 -- Heyavatar
    legacy cleanup**) -- entry di tipo cross-project reference, status `done`,
    verified_zero true, file_count 16, directory_count 2.
  - `architecture/ownership.yaml` (repo Pipelinegen, sezione
    `cross_project_refs.heyavatar`) -- canonical owners post-cleanup + status
    dei contratti rimossi/kept.

Il cleanup e' solo documentale lato Pipelinegen (zero file modificati:
`no_pipelinegen_files_changed: true` su Wave 18). I file effettivamente
rimossi vivono solo sotto `C:/Users/pater/Pyt/Heyavatar/`.
## Task 1 ✅ — Dynamic package mounting per `live_portrait_pipeline`

**Stato**: ✅ **RISOLTO**. La fix di dynamic package mounting è già implementata in `providers/liveportrait/adapter/_upstream.py::_import_upstream_live_portrait()` (62 righe, PEP-style con `importlib.util.spec_from_file_location`).
**Problema (storico)**: Il nostro progetto ha `src/` che faceva shadowing su `LivePortrait/src/` in `sys.modules`. Soluzione adottata: registrare `LivePortrait/src/` come pacchetto con nome unico `liveportrait_upstream` via `spec_from_file_location` + `submodule_search_locations`. Le relative imports interne (`from .config ...`) risolvono correttamente perché il nuovo package `liveportrait_upstream` è isolato da `sys.modules['src']` del nostro progetto.

**Soluzione proposta dal thinker**: Dynamic package mounting — registrare `LivePortrait/src/`
come pacchetto con nome unico (`liveportrait_upstream`) usando `importlib.util.spec_from_file_location`:

```python
def _import_upstream_live_portrait() -> Any:
    import importlib.util, sys
    from pathlib import Path

    extra = os.environ.get("HEYAVATAR_LIVE_PORTRAIT_SRC", "./LivePortrait/src")
    src_path = Path(extra).resolve()
    pkg_name = "liveportrait_upstream"

    if pkg_name in sys.modules:
        return importlib.import_module(f"{pkg_name}.live_portrait_pipeline")

    init_path = src_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(init_path),
        submodule_search_locations=[str(src_path)]
    )
    if spec is None or spec.loader is None:
        return None

    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)

    try:
        return importlib.import_module(f"{pkg_name}.live_portrait_pipeline")
    except ImportError as exc:
        LOG.warning("Failed to import from dynamic package: %s", exc)
        return None
```

**Verifica**: `pytest tests/smoke/test_real_gpu.py::test_engine_loads_in_real_mode -v -s` deve passare.

**Verifica (24 Giugno 2026)**:
```
$ python -m pytest tests/smoke/test_real_gpu/test_engine_load.py::test_engine_loads_in_real_mode -v -s
... Engine state after load: idle
PASSED
============================== 1 passed in 7.70s ===============================
```

Il test supera la `real_mode_env` fixture in `_helpers`. End-to-end funzionante con HEYAVATAR_MOCK_ENGINE=0 + CUDA, RTX 4060 Ti 7 GB VRAM, torch 2.6.0+cu124. I 5 checkpoint scaricati vengono letti dal `CheckpointManager` locale e la CUDA warmup completa in ~3 s.


---

## Task 2 — Primo render end-to-end su GPU reale
Una volta che `test_engine_loads_in_real_mode` passa:

1. **`test_prepare_identity_real_mode`**: verifica che `prepare_identity()` produca `source_features.bin` > 1 KB (non mock)
2. **`test_full_pipeline_real_mode`**: compile identity → render chunk → encode mp4 → verifica output non vuoto

**Comando**: `pytest tests/smoke/test_real_gpu.py -v -s`

**Risultato atteso**: un `.mp4` reale nella directory `captures/`, non il dummy nero del mock mode.

---

## Task 3 — InsightFace buffalo_l (opzionale, per Cropper)
Se vogliamo riattivare il `Cropper` (face detection / crop), servono i modelli InsightFace:
- `git lfs pull` da `deepinsight/insightface` (buffalo_l)
- Copiare in `LivePortrait/pretrained_weights/insightface/`
- Aggiornare `_to_upstream_crop_config` con `insightface_root` corretto
- **Nota**: InsightFace ha licenza non-commercial. Per produzione serve MediaPipe.

---

## Task 4 — MuseTalk engine (secondo provider)
- Scaricare checkpoint MuseTalk da HuggingFace (`TMElyralab/MuseTalk`)
- Creare `providers/musetalk/adapter.py` (se non già completo)
- Test `HEYAVATAR_MOCK_ENGINE=0` con MuseTalk
- Aggiungere a `tests/smoke/test_real_gpu.py`

---

## Task 5 — Docker GPU worker
- `Dockerfile` con `nvidia/cuda:12.4-runtime-ubuntu22.04`
- `docker-compose.yml` con Redis + GPU worker
- Bind mount per checkpoint e avatar packs
- `HEYAVATAR_MOCK_ENGINE=0` nel container

---

## Quick reference

| Comando | Cosa fa |
|---------|---------|
| `pytest tests/ -v --ignore=tests/observability -k "not test_api_metrics and not test_metrics and not test_real_gpu"` | Suite completa (70 test) |
| `pytest tests/smoke/test_real_gpu.py -v -s` | Test GPU reali |
| `python -c "import torch; print(torch.cuda.is_available())"` | Verifica CUDA |
| `nvidia-smi` | Stato GPU |

---

## File modificati di recente
| File | Commit |
|------|--------|
| `providers/liveportrait/adapter.py` | `60eee63` |
| `providers/liveportrait/checkpoint_manager.py` | `60eee63` |
| `tests/smoke/test_real_gpu.py` | `60eee63` |
| `.gitignore` | `60eee63` |
| `LivePortrait/src/__init__.py` | (locale, non committato) |
