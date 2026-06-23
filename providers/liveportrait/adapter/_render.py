"""Real-mode per-chunk rendering for LivePortrait.

* :func:`_real_render_chunk_impl` — the ``render_chunk`` body, attached
  to :class:`LivePortraitAdapter` at module-end. Reads audio, builds
  driving keypoints, runs the batched ``warp_decode`` loop, writes the
  mp4.
* :func:`_load_source_bundle` — free function: deserialise the source
  features + canonical keypoints from the pack.
* :func:`_build_driving_keypoints` — free function: batched lip /
  eye / head-pose retargeting that produces a per-frame
  ``[N_frames, 21, 3]`` driving tensor.

All three are CPU/GPU-aware; real-mode callers require a torch.cuda
device, otherwise the upstream ``warp_decode`` will fail.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from providers._ffmpeg import (
    FACE_REGION_RESOLUTION,
    _read_pack_entry,
    _to_uint8_hwc,
    _write_frames_to_mp4,
)
from providers.liveportrait.audio_bridge.bridge import audio_to_driving
from providers.liveportrait.audio_bridge.types import (
    DrivingSignals,
    EXPRESSION_DIM,
    N_KEYPOINTS,
)
from providers.liveportrait.adapter._upstream import (
    LIVE_PORTRAIT_UPSTREAM_PKG_NAME as _UPSTREAM_PKG,
    _import_torch,
)
from src.core.logging import get_logger
from src.domain.types import (
    AvatarIdentityHandle,
    RenderChunkRequest,
    RenderChunkResult,
)


def _motion_style_profile(style: str, intensity: float, *, eye_lock: bool = False) -> Dict[str, float]:
    """Return a small motion profile for the speaking avatar.

    ``balanced`` preserves the current feel, ``subtle`` reduces the
    amplitude for calmer content, and ``expressive`` boosts classic
    speaking-head movement: nods, sway, blink frequency, and brow lift.
    """
    style_key = (style or "balanced").strip().lower()
    presets: Dict[str, Dict[str, float]] = {
        "subtle": {
            "head_pitch": 0.70,
            "head_yaw": 0.55,
            "head_roll": 0.45,
            "sway_x": 0.55,
            "sway_y": 0.45,
            "blink_rate": 0.80,
            "brow_lift": 0.60,
            "mouth_boost": 0.95,
            "speech_nod": 0.80,
        },
        "balanced": {
            "head_pitch": 1.00,
            "head_yaw": 1.00,
            "head_roll": 0.90,
            "sway_x": 1.00,
            "sway_y": 1.00,
            "blink_rate": 1.00,
            "brow_lift": 1.00,
            "mouth_boost": 1.00,
            "speech_nod": 1.00,
        },
        "expressive": {
            "head_pitch": 1.85,
            "head_yaw": 1.55,
            "head_roll": 1.30,
            "sway_x": 2.20,
            "sway_y": 1.95,
            "blink_rate": 1.70,
            "brow_lift": 1.80,
            "mouth_boost": 1.25,
            "speech_nod": 2.15,
        },
    }
    profile = dict(presets.get(style_key, presets["balanced"]))
    scale = max(0.1, float(intensity))
    for key in profile:
        profile[key] *= scale
    if eye_lock:
        profile["head_yaw"] *= 0.35
        profile["head_roll"] *= 0.35
        profile["sway_x"] *= 0.20
        profile["sway_y"] *= 0.45
        profile["blink_rate"] *= 0.90
        profile["brow_lift"] *= 0.85
    return profile


def _upstream_crop_module():
    """Lazily import the upstream LivePortrait ``utils.crop`` module.

    Routes through :data:`LIVE_PORTRAIT_UPSTREAM_PKG_NAME` so a future
    upstream rename stays in one place — call sites MUST use this
    helper rather than ``from liveportrait_upstream.utils.crop import …``
    (``from … import`` has no Python-syntax way to be parametric on
    the prefix, so we use ``importlib.import_module`` here).
    """
    import importlib as _importlib
    return _importlib.import_module(f"{_UPSTREAM_PKG}.utils.crop")


def _paste_back_seamless(img_crop: np.ndarray, M_c2o: np.ndarray, img_ori: np.ndarray, mask_ori: np.ndarray) -> np.ndarray:
    """Seamless clone-based pasteback to avoid neck boundary seams/ghosting."""
    import cv2
    dsize = (img_ori.shape[1], img_ori.shape[0])
    
    # 1. Warp the crop frame to the original image coordinates
    warped_crop = cv2.warpAffine(img_crop, M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
    
    # 2. Get binary mask from float mask
    mask_binary = (mask_ori * 255).astype(np.uint8)
    if len(mask_binary.shape) == 3:
        mask_binary = mask_binary[:, :, 0]
        
    _, mask_binary = cv2.threshold(mask_binary, 1, 255, cv2.THRESH_BINARY)
    
    # 3. Find bounding box/center of mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.clip(mask_ori * warped_crop + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
        
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    center = (x + w // 2, y + h // 2)
    
    try:
        # cv2.seamlessClone matches boundaries and matches background illumination
        cloned = cv2.seamlessClone(warped_crop, img_ori, mask_binary, center, cv2.NORMAL_CLONE)
        return cloned
    except Exception:
        # Fallback in case of boundary violations or cv2 exceptions
        return np.clip(mask_ori * warped_crop + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)


def _real_render_chunk_impl(
    self,
    request: RenderChunkRequest,
    identity: AvatarIdentityHandle,
    clipped_end: float,
) -> RenderChunkResult:
    """Real-mode render: read audio, drive LivePortrait, encode."""
    torch = _import_torch()
    if torch is None or self._wrapper is None or self._torch_device is None:
        raise RuntimeError("render_chunk called without a healthy real-mode load")

    start, end = request.audio_window
    # Single canonical entry point. Dispatches between the DSP and
    # the SadTalker neural backend based on
    # ``Settings.audio_bridge_backend``; on missing-import failures
    # in neural mode we surface the RuntimeError so the engine
    # transitions to EngineState.DEGRADED.
    driving: DrivingSignals = audio_to_driving(
        request.audio_path,
        start_seconds=start,
        end_seconds=end,
        fps=request.fps,
    )
    motion_profile = _motion_style_profile(
        getattr(self.settings, "motion_style", "balanced"),
        getattr(self.settings, "motion_intensity", 1.0),
        eye_lock=bool(
            self.inf_cfg.extra.get("eye_lock", False)
            or getattr(self.settings, "eye_lock", False)
        ),
    )
    f_s, kp_s, _exp_s = _load_source_bundle(identity.pack_path, torch, self._torch_device)
    
    # Load original background image and pasteback assets if not face_region_only
    img_ori = None
    mask_ori = None
    M_c2o = None
    if not request.face_region_only:
        try:
            from PIL import Image
            import io
            # Late-bind the upstream pasteback helpers via the package-name
            # constant so the source of truth stays in one place.
            _crop_mod = _upstream_crop_module()
            prepare_paste_back = _crop_mod.prepare_paste_back
            paste_back = _crop_mod.paste_back

            orig_bytes = _read_pack_entry(identity.pack_path, "source_image_original.png")
            orig_img = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
            img_ori = np.asarray(orig_img, dtype=np.uint8)
            
            mask_bytes = _read_pack_entry(identity.pack_path, "face_mask.png")
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            if mask_img.size != (512, 512):
                mask_img = mask_img.resize((512, 512), Image.Resampling.BILINEAR)
            mask_crop = np.asarray(mask_img, dtype=np.uint8)
            
            m_bytes = _read_pack_entry(identity.pack_path, "transform_matrix.bin")
            M_c2o_2x3 = np.frombuffer(m_bytes, dtype=np.float32).reshape(2, 3)
            M_c2o = np.vstack([M_c2o_2x3, np.array([0, 0, 1], dtype=np.float32)])
            
            dsize_ori = (img_ori.shape[1], img_ori.shape[0])
        except Exception as exc:
            LOG.warning("Failed to setup pasteback, falling back to cropped output: %s", exc)
            img_ori = None
            M_c2o = None

    static_head = self.inf_cfg.extra.get("static_head", False)
    eye_lock = bool(
        self.inf_cfg.extra.get("eye_lock", False)
        or getattr(self.settings, "eye_lock", False)
    )
    # If the wrapper's stitching retargeting module is exposed, use the
    # full batched retarget path; otherwise fall back to the simpler
    # expression-delta mock-form.
    kp_d = _build_driving_keypoints(
        driving,
        kp_s,
        torch,
        self._torch_device,
        self._wrapper,
        static_head=static_head,
        eye_lock=eye_lock,
        motion_profile=motion_profile,
    )

    warped_frames = []
    per_frame_seconds = 0.0
    t_start = time.monotonic()

    warping = getattr(self._wrapper, "warping_module", None)
    stitching = None
    if warping is None:
        raise RuntimeError(
            "LivePortrait wrapper does not expose warping_module; "
            "check upstream version."
        )

    exp_d = np.asarray(
        driving.exp_d_flat, dtype=np.float32
    ).reshape(driving.frames, N_KEYPOINTS, EXPRESSION_DIM)
    kp_d_np = np.asarray(kp_d, dtype=np.float32)

    # ── batched render loop ─────────────────────────────────
    # Stack driving keypoints into batches to saturate Tensor
    # Cores. Each batch passes through warp_decode in one GPU
    # kernel launch, reducing driver overhead by ~batch_size×.
    batch = self.render_batch_size
    for batch_start in range(0, driving.frames, batch):
        batch_end = min(batch_start + batch, driving.frames)
        batch_slice = slice(batch_start, batch_end)

        # Stack driving keypoints: [batch, 1, 21, 3]
        kp_d_batch = torch.as_tensor(
            kp_d_np[batch_slice],
            dtype=torch.float32,
            device=self._torch_device,
        )

        # Stitching refines driving keypoints per-frame; apply
        # per-frame then stack if upstream supports it.
        if stitching is not None:
            refined = []
            for j in range(batch_end - batch_start):
                kp_d_single = kp_d_batch[j : j + 1]
                refined.append(self._wrapper.stitching(kp_s, kp_d_single))
            kp_d_batch = torch.cat(refined, dim=0)

        # Repeat source features AND source keypoints across batch
        # dimension so upstream warp_decode sees consistent shapes.
        batch_n = batch_end - batch_start
        f_s_batch = f_s.expand(batch_n, -1, -1, -1, -1)
        kp_s_batch = kp_s.expand(batch_n, -1, -1)

        # Single GPU kernel launch for the entire batch.
        batch_output = self._wrapper.warp_decode(f_s_batch, kp_s_batch, kp_d_batch)['out']

        # Collect frames back.
        for j in range(batch_end - batch_start):
            frame = batch_output[j : j + 1]
            frame_rgb = _to_uint8_hwc(frame)
            
            if not request.face_region_only and img_ori is not None:
                # Scope note: ``prepare_paste_back``, ``paste_back``,
                # ``mask_crop``, ``M_c2o`` and ``dsize_ori`` are only
                # safe to dereference when ``img_ori is not None``.
                # Any exception inside the upstream setup ``try``
                # (including partial-success paths where the bind
                # landed but an asset read failed) drops us into the
                # ``except`` branch which wipes ``img_ori``/``M_c2o``
                # and short-circuits the per-frame path.
                M_c2o_frame = M_c2o.copy()
                frame_idx = batch_start + j
                t_sec = frame_idx / request.fps
                tx_sway = 0.0 if static_head else 4.0 * math.sin(2 * math.pi * 0.15 * t_sec)
                ty_sway = 0.0 if static_head else 1.5 * math.cos(2 * math.pi * 0.10 * t_sec)
                M_c2o_frame[0, 2] += tx_sway
                M_c2o_frame[1, 2] += ty_sway
                
                # Dynamically project the 512x512 mask for this frame's transformation
                mask_ori_frame = prepare_paste_back(mask_crop, M_c2o_frame, dsize_ori)
                mask_ori_frame = mask_ori_frame[..., None]
                
                pasted = _paste_back_seamless(frame_rgb, M_c2o_frame, img_ori, mask_ori_frame)
                warped_frames.append(pasted)
            else:
                warped_frames.append(frame_rgb)

    per_frame_seconds = time.monotonic() - t_start

    out_dir = self.settings.capture_dir / request.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
    # Face-region-only: skip any pasteback/upscale, output at 256×256.
    output_resolution = FACE_REGION_RESOLUTION if request.face_region_only else request.resolution
    _write_frames_to_mp4(
        warped_frames,
        out_path,
        fps=request.fps,
        target_resolution=output_resolution,
    )
    duration = max(0.5, clipped_end - request.audio_window[0])
    gpu_seconds = max(per_frame_seconds, 0.012 * duration)
    return RenderChunkResult(
        chunk_index=request.chunk_index,
        output_path=out_path,
        duration_seconds=duration,
        frames_rendered=len(warped_frames),
        gpu_seconds=gpu_seconds,
        engine_id=self.engine_id,
    )


def _load_source_bundle(
    pack_path: Path,
    torch: Any,
    device: Any,
) -> Tuple[Any, Any, Any]:
    """Read the source features / canonical keypoints from the pack."""
    f_s_bytes = _read_pack_entry(pack_path, "source_features.bin")
    kp_bytes = _read_pack_entry(pack_path, "canonical_keypoints.bin")
    # LivePortrait ships 32*16*64*64 = 2,097,152 element feature volume
    # stored as float16. We accept slight mismatch (e.g. cached mock
    # features) by reshaping to whatever the bytes count implies.
    elem_count = len(f_s_bytes) // 2
    f_s = torch.frombuffer(f_s_bytes, dtype=torch.float16).reshape(
        1, 32, max(1, elem_count // (32 * 64 * 64)), 64, 64
    ).to(device)
    # The canonical keypoints entry concatenates kp[1,21,3] (252 B) +
    # exp[1,21,3] (252 B). Mock-mode prepares a 68*2*4 = 544 B
    # array — best-effort split is 252 B for kp, the rest for exp.
    kp_d_total = np.frombuffer(kp_bytes, dtype=np.float32)
    if kp_d_total.size >= N_KEYPOINTS * EXPRESSION_DIM * 2:
        kp_s_arr = kp_d_total[: N_KEYPOINTS * EXPRESSION_DIM].reshape(
            1, N_KEYPOINTS, EXPRESSION_DIM
        )
        exp_s_arr = kp_d_total[
            N_KEYPOINTS * EXPRESSION_DIM : 2 * N_KEYPOINTS * EXPRESSION_DIM
        ].reshape(1, N_KEYPOINTS, EXPRESSION_DIM)
    else:
        # Mock-pack fallback: build a fake 21*3 identity keypoint array.
        kp_s_arr = np.zeros((1, N_KEYPOINTS, EXPRESSION_DIM), dtype=np.float32)
        exp_s_arr = np.zeros((1, N_KEYPOINTS, EXPRESSION_DIM), dtype=np.float32)
    kp_s = torch.as_tensor(kp_s_arr, device=device)
    exp_s = torch.as_tensor(exp_s_arr, device=device)
    return f_s, kp_s, exp_s


def _build_driving_keypoints(
    driving: DrivingSignals,
    kp_s: Any,
    torch: Any,
    device: Any,
    wrapper: Any = None,
    static_head: bool = False,
    eye_lock: bool = False,
    motion_profile: Dict[str, float] | None = None,
) -> np.ndarray:
    """Combine canonical source keypoints with the expression deltas.

    Returns an ``[N_frames, 21, 3]`` numpy array.

    When the upstream wrapper exposes ``stitching_retargeting_module``,
    uses the fully-batched retargeting path: lip retargeting via
    ``wrapper.retarget_lip``, organic-interval blinking via
    ``wrapper.retarget_eye`` (2.4-4.8 s random inter-blink gap), and
    audio-modulated head-pose micro-movements via batched 3×3 rotation
    matrices. Falls back to the simpler expression-delta form when
    the wrapper doesn't expose retargeting modules.
    """
    motion_profile = motion_profile or _motion_style_profile("balanced", 1.0, eye_lock=eye_lock)
    if wrapper is not None and getattr(wrapper, "stitching_retargeting_module", None) is not None:
        fps = 25

        # 1. Lip retargeting (batched)
        apertures = torch.tensor(driving.mouth_aperture, dtype=torch.float32, device=device)
        lip_close_ratios = torch.zeros((driving.frames, 2), dtype=torch.float32, device=device)
        lip_close_ratios[:, 0] = 0.15
        lip_close_ratios[:, 1] = 0.15 + apertures * 0.55

        kp_s_expanded = kp_s.expand(driving.frames, -1, -1)
        lip_deltas = wrapper.retarget_lip(kp_s_expanded, lip_close_ratios)

        # 2. Eye retargeting (blinking) with organic random intervals
        rng = random.Random(42)
        blink_val_list = []
        blink_frames = -1
        blink_floor = max(24, int(60 / max(0.5, motion_profile["blink_rate"])))
        blink_ceil = max(blink_floor + 1, int(120 / max(0.5, motion_profile["blink_rate"])))
        next_blink_delay = rng.randint(blink_floor, blink_ceil)
        frames_since_blink = 0

        for i in range(driving.frames):
            if frames_since_blink >= next_blink_delay and blink_frames < 0:
                blink_frames = 0
                frames_since_blink = 0
                next_blink_delay = rng.randint(blink_floor, blink_ceil)

            if blink_frames >= 0 and blink_frames <= 5:
                blink_weights = [0.6, 0.2, 0.0, 0.0, 0.4, 0.8]
                blink_val = blink_weights[blink_frames]
                blink_frames += 1
            else:
                blink_val = 1.0
                blink_frames = -1
                frames_since_blink += 1

            blink_val_list.append(blink_val)

        blink_vals = torch.tensor(blink_val_list, dtype=torch.float32, device=device)
        target_eyes = 0.12 + blink_vals * (0.23 * motion_profile["blink_rate"])
        if eye_lock:
            target_eyes = torch.clamp(0.15 + (target_eyes - 0.15) * 0.72, 0.08, 0.42)
        eye_close_ratios = torch.zeros((driving.frames, 3), dtype=torch.float32, device=device)
        eye_close_ratios[:, 0] = 0.35
        eye_close_ratios[:, 1] = 0.35
        eye_close_ratios[:, 2] = target_eyes

        eye_deltas = wrapper.retarget_eye(kp_s_expanded, eye_close_ratios)

        # Combine deltas
        kp_d_batched = kp_s_expanded + lip_deltas + eye_deltas

        # 3. Dynamic audio-modulated head pose micro-movements (completely batched)
        t = torch.arange(driving.frames, dtype=torch.float32, device=device)

        # Scale movement based on audio volume
        scales = 0.8 + 0.5 * apertures

        if static_head:
            p_deg = torch.zeros_like(apertures)
            y_deg = torch.zeros_like(apertures)
            r_deg = torch.zeros_like(apertures)
        else:
            speech_nod = 0.55 + apertures * motion_profile["speech_nod"]
            p_deg = (
                scales
                * motion_profile["head_pitch"]
                * torch.sin(2 * math.pi * 0.35 * t / fps)
                + 0.4 * apertures * motion_profile["speech_nod"]
                + 0.25 * speech_nod * torch.sin(2 * math.pi * 0.72 * t / fps)
            )
            y_deg = scales * motion_profile["head_yaw"] * torch.cos(2 * math.pi * 0.25 * t / fps)
            r_deg = (
                scales
                * 0.6
                * motion_profile["head_roll"]
                * torch.sin(2 * math.pi * 0.45 * t / fps)
            )
            if eye_lock:
                y_deg = y_deg * 0.35
                r_deg = r_deg * 0.40

        p = torch.deg2rad(p_deg)
        y = torch.deg2rad(y_deg)
        r = torch.deg2rad(r_deg)

        cos_p, sin_p = torch.cos(p), torch.sin(p)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_r, sin_r = torch.cos(r), torch.sin(r)

        # Build batch of rotation matrices
        Rx = torch.zeros((driving.frames, 3, 3), dtype=torch.float32, device=device)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = cos_p
        Rx[:, 1, 2] = -sin_p
        Rx[:, 2, 1] = sin_p
        Rx[:, 2, 2] = cos_p

        Ry = torch.zeros((driving.frames, 3, 3), dtype=torch.float32, device=device)
        Ry[:, 0, 0] = cos_y
        Ry[:, 0, 2] = sin_y
        Ry[:, 1, 1] = 1.0
        Ry[:, 2, 0] = -sin_y
        Ry[:, 2, 2] = cos_y

        Rz = torch.zeros((driving.frames, 3, 3), dtype=torch.float32, device=device)
        Rz[:, 0, 0] = cos_r
        Rz[:, 0, 1] = -sin_r
        Rz[:, 1, 0] = sin_r
        Rz[:, 1, 1] = cos_r
        Rz[:, 2, 2] = 1.0

        R = torch.bmm(torch.bmm(Rx, Ry), Rz)

        # Rotate keypoints around centroid
        centroid = torch.mean(kp_d_batched, dim=1, keepdim=True)
        kp_d_rotated = torch.bmm(kp_d_batched - centroid, R.transpose(1, 2)) + centroid

        # Small translation/sway batch-wise
        tx = (
            torch.zeros_like(t)
            if static_head
            else 0.005 * motion_profile["sway_x"] * torch.sin(2 * math.pi * 0.2 * t / fps)
        )
        ty = (
            torch.zeros_like(t)
            if static_head
            else 0.005 * motion_profile["sway_y"] * torch.cos(2 * math.pi * 0.15 * t / fps)
        )
        if eye_lock:
            tx = tx * 0.25
            ty = ty * 0.60

        kp_d_rotated[:, :, 0] += tx[:, None]
        kp_d_rotated[:, :, 1] += ty[:, None]
        # Classic speaking-face motion: lightly lift the brows / upper
        # face when the mouth opens so the avatar looks more alive.
        brow_lift = torch.clamp(apertures * motion_profile["brow_lift"], 0.0, 1.0)
        kp_d_rotated[:, :4, 1] -= 0.012 * brow_lift[:, None]
        kp_d_rotated[:, 17:, 1] -= 0.006 * brow_lift[:, None]

        return kp_d_rotated.detach().cpu().numpy()

    # Mock/legacy fallback
    src = kp_s.detach().cpu().numpy()[0]  # [21, 3]
    base = np.tile(src[None, ...], (driving.frames, 1, 1))  # [N, 21, 3]
    delta = np.asarray(driving.exp_d_flat, dtype=np.float32).reshape(
        driving.frames, N_KEYPOINTS, EXPRESSION_DIM
    )
    mouth = np.asarray(driving.mouth_aperture, dtype=np.float32)
    if mouth.shape[0] == driving.frames:
        delta[:, 14:18, 1] *= motion_profile["mouth_boost"]
        delta[:, :4, 1] -= 0.008 * mouth[:, None] * motion_profile["brow_lift"]
        delta[:, 17:, 1] -= 0.004 * mouth[:, None] * motion_profile["brow_lift"]
    return base + delta


# ── attach to LivePortraitAdapter ──────────────────────────────────
def _attach_render_methods():
    from providers.liveportrait.adapter.engine import LivePortraitAdapter
    LivePortraitAdapter._real_render_chunk = _real_render_chunk_impl


_attach_render_methods()
