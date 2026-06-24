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
import cv2

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
from src.motion.face_bias import load_face_motion_timeline, sample_face_motion_biases
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

LOG = get_logger(__name__)


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
    import os
    import cv2
    dsize = (img_ori.shape[1], img_ori.shape[0])
    
    # 1. Warp the crop frame to the original image coordinates
    warped_crop = cv2.warpAffine(img_crop, M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
    
    # Fast linear blend bypass
    if os.environ.get("HEYAVATAR_FAST_BLEND", "1") == "1":
        return np.clip(mask_ori * warped_crop + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
    
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
    
_DRIVING_CACHE: Dict[Tuple[str, int], DrivingSignals] = {}


def _get_sliced_driving_signals(audio_path: Path, start: float, end: float, fps: int) -> DrivingSignals:
    key = (str(audio_path.resolve()), fps)
    if key not in _DRIVING_CACHE:
        from src.application.render_video.audio_probe import _probe_audio_duration
        duration = _probe_audio_duration(audio_path)
        if duration <= 0:
            duration = end
        _DRIVING_CACHE[key] = audio_to_driving(
            audio_path,
            start_seconds=0.0,
            end_seconds=duration,
            fps=fps,
        )
    
    full_ds = _DRIVING_CACHE[key]
    
    start_frame = int(round(start * fps))
    end_frame = int(round(end * fps))
    start_frame = max(0, min(start_frame, full_ds.frames))
    end_frame = max(0, min(end_frame, full_ds.frames))
    if end_frame <= start_frame:
        end_frame = start_frame + 1
        
    expected_frames = end_frame - start_frame
    
    flat_per_frame = 21 * 3
    flat_start = start_frame * flat_per_frame
    flat_end = end_frame * flat_per_frame
    
    sliced_exp_d_flat = full_ds.exp_d_flat[flat_start:flat_end]
    sliced_blink_mask = full_ds.blink_mask[start_frame:end_frame]
    sliced_mouth_aperture = full_ds.mouth_aperture[start_frame:end_frame]
    
    return DrivingSignals(
        frames=expected_frames,
        exp_d_flat=sliced_exp_d_flat,
        blink_mask=sliced_blink_mask,
        mouth_aperture=sliced_mouth_aperture,
        backend=full_ds.backend,
    )


def torch_warp_affine(img: Any, M_c2o: Any, dsize: Tuple[int, int]) -> Any:
    """Warp a torch tensor img of shape [C, H_crop, W_crop] using 3x3 affine matrix M_c2o.
    
    Returns warped tensor of shape [C, dsize[1], dsize[0]].
    """
    import torch
    C, H_crop, W_crop = img.shape
    W_ori, H_ori = dsize
    
    # Cast matrix to float32 since linalg.inv does not support float16 (Half)
    M_c2o_f32 = M_c2o.to(torch.float32)
    M_o2c = torch.linalg.inv(M_c2o_f32)
    
    # Construct N_crop and N_ori matrices in float32
    N_crop = torch.tensor([
        [(W_crop - 1) / 2.0, 0.0, (W_crop - 1) / 2.0],
        [0.0, (H_crop - 1) / 2.0, (H_crop - 1) / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=img.device)
    
    N_ori = torch.tensor([
        [(W_ori - 1) / 2.0, 0.0, (W_ori - 1) / 2.0],
        [0.0, (H_ori - 1) / 2.0, (H_ori - 1) / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=img.device)
    
    # Compute theta: N_crop_inv * M_o2c * N_ori
    N_crop_inv = torch.linalg.inv(N_crop)
    theta_3x3 = N_crop_inv @ M_o2c @ N_ori
    
    # Extract the top 2x3 part and cast back to the image dtype (e.g. float16)
    theta = theta_3x3[:2, :].unsqueeze(0).to(img.dtype)  # Shape [1, 2, 3]
    
    # Generate grid and sample
    grid = torch.nn.functional.affine_grid(theta, [1, C, H_ori, W_ori], align_corners=True)
    warped = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped.squeeze(0)


def _real_render_chunk_impl(
    self,
    request: RenderChunkRequest,
    identity: AvatarIdentityHandle,
    clipped_end: float,
) -> RenderChunkResult:
    """Real-mode render: read audio, drive LivePortrait, encode."""
    import cv2
    torch = _import_torch()
    if torch is None or self._wrapper is None or self._torch_device is None:
        raise RuntimeError("render_chunk called without a healthy real-mode load")

    start, end = request.audio_window
    driving: DrivingSignals = _get_sliced_driving_signals(
        request.audio_path,
        start,
        end,
        request.fps,
    )
    # Load face biases from timeline if available
    face_timeline = load_face_motion_timeline(request.face_motion_timeline_path)
    face_biases = sample_face_motion_biases(
        face_timeline,
        frames=driving.frames,
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
    bg_image = None
    bg_image_rgb = None
    person_mask = None
    mask_ori = None
    M_c2o = None
    downscale_factor = 1.0
    img_ori_down = None
    M_c2o_down = None
    mask_crop = None
    w_ori, h_ori = 512, 512
    dsize_ori = (512, 512)
    
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
            
            bg_path = Path("assets/tv_studio_background.png")
            if bg_path.is_file():
                bg_image = cv2.imread(str(bg_path))
                if bg_image is not None:
                    LOG.info("Loaded TV studio background image for frame-by-frame keying: %s", bg_path)
            
            mask_bytes = _read_pack_entry(identity.pack_path, "face_mask.png")
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            if mask_img.size != (512, 512):
                mask_img = mask_img.resize((512, 512), Image.Resampling.BILINEAR)
            mask_crop = np.asarray(mask_img, dtype=np.uint8)
            
            m_bytes = _read_pack_entry(identity.pack_path, "transform_matrix.bin")
            M_c2o_2x3 = np.frombuffer(m_bytes, dtype=np.float32).reshape(2, 3)
            M_c2o = np.vstack([M_c2o_2x3, np.array([0, 0, 1], dtype=np.float32)])
            
            w_ori, h_ori = img_ori.shape[1], img_ori.shape[0]
            # Downscale by 50% for rendering if target resolution is large (e.g. > 1000px) to speed up seamlessClone
            if max(w_ori, h_ori) > 1000:
                downscale_factor = 0.5
                w_down = int(w_ori * downscale_factor)
                h_down = int(h_ori * downscale_factor)
                img_ori_down = cv2.resize(img_ori, (w_down, h_down), interpolation=cv2.INTER_AREA)
                M_c2o_down = M_c2o.copy()
                M_c2o_down[:2, :] *= downscale_factor
                dsize_ori = (w_down, h_down)
            else:
                downscale_factor = 1.0
                img_ori_down = img_ori
                M_c2o_down = M_c2o
                dsize_ori = (w_ori, h_ori)
                
            # Extract green screen mask of the person at final resolution
            img_ori_down_bgr = cv2.cvtColor(img_ori_down, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_ori_down_bgr, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40], dtype=np.uint8)
            upper_green = np.array([90, 255, 255], dtype=np.uint8)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_mask = cv2.GaussianBlur(green_mask, (15, 15), 0)
            person_mask = 1.0 - (green_mask.astype(np.float32) / 255.0)
            person_mask = np.clip(person_mask, 0.0, 1.0)
            
            # Prepare background image in RGB format at original full resolution
            if bg_image is not None:
                bg_resized = cv2.resize(bg_image, (w_ori, h_ori))
                bg_image_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
            else:
                bg_image_rgb = np.zeros((h_ori, w_ori, 3), dtype=np.uint8)
        except Exception as exc:
            LOG.warning("Failed to setup pasteback, falling back to cropped output: %s", exc)
            img_ori = None
            M_c2o = None
            downscale_factor = 1.0
            img_ori_down = None
            M_c2o_down = None

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
        face_biases=face_biases,
    )

    warped_frames = []
    t_start = time.monotonic()

    warping = getattr(self._wrapper, "warping_module", None)
    stitching = getattr(self._wrapper, "stitching_retargeting_module", None)
    if warping is None:
        raise RuntimeError(
            "LivePortrait wrapper does not expose warping_module; "
            "check upstream version."
        )

    exp_d = np.asarray(
        driving.exp_d_flat, dtype=np.float32
    ).reshape(driving.frames, N_KEYPOINTS, EXPRESSION_DIM)
    kp_d_np = np.asarray(kp_d, dtype=np.float32)

    # Render every single frame for maximum fluid quality and natural lip synchronization
    render_indices = list(range(driving.frames))
    kp_d_np_rendered = kp_d_np[render_indices]
    rendered_frames_rgb = []

    # ── batched render loop ─────────────────────────────────
    batch = self.render_batch_size
    dtype = torch.float16 if self.inf_cfg.flag_use_half_precision else torch.float32
    
    for batch_start in range(0, len(render_indices), batch):
        batch_end = min(batch_start + batch, len(render_indices))
        batch_slice = slice(batch_start, batch_end)

        # Stack driving keypoints: [batch, 1, 21, 3]
        kp_d_batch = torch.as_tensor(
            kp_d_np_rendered[batch_slice],
            dtype=dtype,
            device=self._torch_device,
        )

        if stitching is not None:
            refined = []
            for j in range(batch_end - batch_start):
                kp_d_single = kp_d_batch[j : j + 1]
                refined.append(self._wrapper.stitching(kp_s, kp_d_single))
            kp_d_batch = torch.cat(refined, dim=0)

        # Repeat source features AND source keypoints across batch
        batch_n = batch_end - batch_start
        f_s_batch = f_s.expand(batch_n, -1, -1, -1, -1)
        kp_s_batch = kp_s.expand(batch_n, -1, -1)

        # Single GPU kernel launch for the entire batch.
        batch_output = self._wrapper.warp_decode(f_s_batch, kp_s_batch, kp_d_batch)['out']

        # Collect frames back.
        for j in range(batch_end - batch_start):
            frame = batch_output[j]
            rendered_frames_rgb.append(frame)

    face_frames = rendered_frames_rgb

    # Setup GPU composting tensors
    img_ori_gpu = None
    mask_crop_gpu = None
    M_c2o_down_gpu = None
    person_mask_gpu = None
    bg_image_gpu = None
    if not request.face_region_only and img_ori_down is not None:
        img_ori_gpu = torch.as_tensor(img_ori_down, dtype=dtype, device=self._torch_device).permute(2, 0, 1) / 255.0
        mask_crop_gpu = torch.as_tensor(mask_crop, dtype=dtype, device=self._torch_device).unsqueeze(0) / 255.0
        M_c2o_down_gpu = torch.as_tensor(M_c2o_down, dtype=dtype, device=self._torch_device)
        if person_mask is not None:
            person_mask_gpu = torch.as_tensor(person_mask, dtype=dtype, device=self._torch_device).unsqueeze(0)
        if bg_image_rgb is not None:
            bg_image_gpu = torch.as_tensor(bg_image_rgb, dtype=dtype, device=self._torch_device).permute(2, 0, 1) / 255.0

    output_resolution = FACE_REGION_RESOLUTION if request.face_region_only else request.resolution
    w_out, h_out = output_resolution

    # ── pasteback loop on GPU (optimized to prevent VRAM accumulation) ──
    warped_frames = []
    for frame_idx, frame_gpu in enumerate(face_frames):
        if not request.face_region_only and img_ori_gpu is not None and M_c2o_down_gpu is not None:
            M_c2o_frame = M_c2o_down_gpu.clone()
            t_sec = frame_idx / request.fps
            tx_sway = 0.0 if static_head else 1.5 * math.sin(2 * math.pi * 0.15 * t_sec)
            ty_sway = 0.0 if static_head else 0.6 * math.cos(2 * math.pi * 0.10 * t_sec)
            M_c2o_frame[0, 2] += tx_sway * downscale_factor
            M_c2o_frame[1, 2] += ty_sway * downscale_factor
            
            warped_crop = torch_warp_affine(frame_gpu, M_c2o_frame, dsize_ori)
            warped_mask = torch_warp_affine(mask_crop_gpu, M_c2o_frame, dsize_ori)
            
            # Apply 2D box blur/feathering on the GPU to soften the mask boundary
            feathered_mask = torch.nn.functional.avg_pool2d(
                warped_mask.unsqueeze(0),
                kernel_size=5,
                stride=1,
                padding=2
            ).squeeze(0)
            pasted_gpu = feathered_mask * warped_crop + (1.0 - feathered_mask) * img_ori_gpu
            
            # Get default person mask if not set
            p_mask = person_mask_gpu if person_mask_gpu is not None else torch.ones((1, dsize_ori[1], dsize_ori[0]), dtype=dtype, device=self._torch_device)
            
            # Apply coordinated waist-pivot rotation and translation to the person (pasted_gpu)
            # and the person's green screen mask (p_mask) on the GPU.
            if not static_head and bg_image_gpu is not None:
                hb = face_biases["head"][frame_idx] if face_biases else 0.0
                px = dsize_ori[0] / 2.0
                py = float(dsize_ori[1])
                
                # Coordinated shoulder rotation that responds dynamically to speech emphasis/cenni (very slow, smooth)
                angle_deg = (0.6 + 0.9 * hb) * math.sin(2 * math.pi * 0.15 * t_sec)
                theta = math.radians(angle_deg)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                global_tx = tx_sway * (0.35 + 0.25 * hb) * downscale_factor
                global_ty = ty_sway * (0.25 + 0.15 * hb) * downscale_factor
                
                M_global = torch.tensor([
                    [cos_t, -sin_t, px * (1.0 - cos_t) + py * sin_t + global_tx],
                    [sin_t,  cos_t, py * (1.0 - cos_t) - px * sin_t + global_ty],
                    [0.0,    0.0,   1.0]
                ], dtype=dtype, device=self._torch_device)
                
                pasted_gpu = torch_warp_affine(pasted_gpu, M_global, dsize_ori)
                p_mask = torch_warp_affine(p_mask, M_global, dsize_ori)

            # Feather the warped person mask on GPU
            p_mask_feathered = torch.nn.functional.avg_pool2d(
                p_mask.unsqueeze(0),
                kernel_size=5,
                stride=1,
                padding=2
            ).squeeze(0)
            p_mask_feathered = torch.clamp(p_mask_feathered, 0.0, 1.0)

            # Upscale the warped person frame and feathered mask back to original high-res size (h_ori, w_ori)
            if downscale_factor != 1.0:
                pasted_gpu = torch.nn.functional.interpolate(
                    pasted_gpu.unsqueeze(0),
                    size=(h_ori, w_ori),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
                p_mask_feathered = torch.nn.functional.interpolate(
                    p_mask_feathered.unsqueeze(0),
                    size=(h_ori, w_ori),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)

            # Blend the high-res warped person over the static high-res background image on GPU
            if bg_image_gpu is not None:
                pasted_gpu = p_mask_feathered * pasted_gpu + (1.0 - p_mask_feathered) * bg_image_gpu
            
            # Move to CPU immediately to free VRAM
            pasted_cpu = (pasted_gpu.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            warped_frames.append(pasted_cpu)
        else:
            if frame_gpu.shape[1:] != (h_out, w_out):
                frame_gpu = torch.nn.functional.interpolate(
                    frame_gpu.unsqueeze(0),
                    size=(h_out, w_out),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
            
            # Move to CPU immediately to free VRAM
            frame_cpu = (frame_gpu.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            warped_frames.append(frame_cpu)

    per_frame_seconds = time.monotonic() - t_start

    out_dir = self.settings.capture_dir / request.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{request.chunk_index:04d}.mp4"
    output_resolution = FACE_REGION_RESOLUTION if request.face_region_only else request.resolution
    _write_frames_to_mp4(
        warped_frames,
        out_path,
        fps=request.fps,
        target_resolution=output_resolution,
        audio_path=request.audio_path if not request.face_region_only else None,
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
    face_biases: Dict[str, list[float]] | None = None,
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
    # Generate eye micro-saccades for both paths to keep the gaze natural
    saccade_dx = np.zeros(driving.frames, dtype=np.float32)
    saccade_dy = np.zeros(driving.frames, dtype=np.float32)
    rng = random.Random(12345)
    curr_dx, curr_dy = 0.0, 0.0
    next_change = 0
    idx_s = 0
    fps_val = 25
    while idx_s < driving.frames:
        if idx_s >= next_change:
            target_dx = rng.uniform(-0.0035, 0.0035)
            target_dy = rng.uniform(-0.0018, 0.0018)
            duration = rng.randint(int(fps_val * 0.5), int(fps_val * 1.5))
            next_change = idx_s + duration
            transition_len = rng.randint(2, 4)
            for t_sacc in range(transition_len):
                if idx_s + t_sacc < driving.frames:
                    alpha = (t_sacc + 1) / transition_len
                    saccade_dx[idx_s + t_sacc] = curr_dx + alpha * (target_dx - curr_dx)
                    saccade_dy[idx_s + t_sacc] = curr_dy + alpha * (target_dy - curr_dy)
            curr_dx, curr_dy = target_dx, target_dy
            idx_s += transition_len
        else:
            saccade_dx[idx_s] = curr_dx
            saccade_dy[idx_s] = curr_dy
            idx_s += 1

    motion_profile = motion_profile or _motion_style_profile("balanced", 1.0, eye_lock=eye_lock)
    if face_biases is None:
        face_biases = {
            "blink": [0.0] * driving.frames,
            "brow": [0.0] * driving.frames,
            "mouth": [0.0] * driving.frames,
            "head": [0.0] * driving.frames,
        }
    blink_bias = torch.tensor(face_biases.get("blink", [0.0] * driving.frames), dtype=torch.float32, device=device)
    brow_bias = torch.tensor(face_biases.get("brow", [0.0] * driving.frames), dtype=torch.float32, device=device)
    mouth_bias = torch.tensor(face_biases.get("mouth", [0.0] * driving.frames), dtype=torch.float32, device=device)
    head_bias = torch.tensor(face_biases.get("head", [0.0] * driving.frames), dtype=torch.float32, device=device)

    if wrapper is not None and getattr(wrapper, "stitching_retargeting_module", None) is not None:
        fps = 25

        # 1. Lip retargeting (batched) - Target ratio goes to 0.0 (fully closed) when apertures is 0.0
        apertures = torch.tensor(driving.mouth_aperture, dtype=torch.float32, device=device)
        lip_close_ratios = torch.zeros((driving.frames, 2), dtype=torch.float32, device=device)
        lip_close_ratios[:, 0] = 0.15
        # Scale the mouth movements to make them more natural (scaled by mouth_boost presets)
        # Note: No offset here so lips close fully on silent frames (P/B/M phonemes)
        lip_close_ratios[:, 1] = apertures * 0.75 * motion_profile.get("mouth_boost", 1.0)

        kp_s_expanded = kp_s.expand(driving.frames, -1, -1)
        lip_deltas = wrapper.retarget_lip(kp_s_expanded, lip_close_ratios)

        # 2. Eye retargeting (blinking) with organic random intervals
        rng_blink = random.Random(42)
        blink_val_list = []
        blink_frames = -1
        blink_floor = max(24, int(60 / max(0.5, motion_profile["blink_rate"])))
        blink_ceil = max(blink_floor + 1, int(120 / max(0.5, motion_profile["blink_rate"])))
        next_blink_delay = rng_blink.randint(blink_floor, blink_ceil)
        frames_since_blink = 0

        for i in range(driving.frames):
            if frames_since_blink >= next_blink_delay and blink_frames < 0:
                blink_frames = 0
                frames_since_blink = 0
                next_blink_delay = rng_blink.randint(blink_floor, blink_ceil)

            if blink_frames >= 0 and blink_frames <= 5:
                # Add organic depth variation to the blink so it is not perfectly rigid
                blink_depth_factor = rng_blink.uniform(0.85, 1.15)
                blink_weights = [0.6, 0.2, 0.0, 0.0, 0.4, 0.8]
                blink_val = 1.0 - (1.0 - blink_weights[blink_frames]) * blink_depth_factor
                blink_frames += 1
            else:
                blink_val = 1.0
                blink_frames = -1
                frames_since_blink += 1

            # Blend organic blinking with timeline-driven blink_bias
            timeline_blink = face_biases.get("blink", [0.0] * driving.frames)[i]
            if timeline_blink > 0.5:
                blink_val = min(blink_val, 0.0)

            blink_val_list.append(blink_val)

        blink_vals = torch.tensor(blink_val_list, dtype=torch.float32, device=device)
        # Widen the eyes slightly during peaks of speech emphasis (apertures)
        emphasis = torch.clamp(apertures, 0.0, 1.0)
        target_eyes = 0.12 + blink_vals * (0.23 * motion_profile["blink_rate"]) + 0.06 * emphasis
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
            # Add multi-frequency organic micro-pose jitters to avoid freeze/robotic look
            micro_p = 0.15 * torch.sin(2 * math.pi * 1.8 * t / fps) + 0.1 * torch.cos(2 * math.pi * 3.1 * t / fps)
            micro_y = 0.15 * torch.cos(2 * math.pi * 1.4 * t / fps) + 0.1 * torch.sin(2 * math.pi * 2.7 * t / fps)
            micro_r = 0.12 * torch.sin(2 * math.pi * 1.6 * t / fps) + 0.08 * torch.cos(2 * math.pi * 2.9 * t / fps)

            # Pitch nod reacts to head_bias
            p_deg = (
                scales
                * motion_profile["head_pitch"]
                * torch.sin(2 * math.pi * 0.35 * t / fps)
                + 0.55 * apertures * motion_profile["speech_nod"]
                + 0.25 * speech_nod * torch.sin(2 * math.pi * 0.72 * t / fps)
                + 3.5 * head_bias * torch.sin(2 * math.pi * 1.5 * t / fps)
                + micro_p
            )
            # Yaw shake reacts to head_bias
            y_deg = (
                scales * motion_profile["head_yaw"] * torch.cos(2 * math.pi * 0.25 * t / fps)
                + 1.5 * head_bias * torch.sin(2 * math.pi * 0.8 * t / fps)
                + micro_y
            )
            # Roll tilt reacts to head_bias
            r_deg = (
                scales
                * 0.6
                * motion_profile["head_roll"]
                * torch.sin(2 * math.pi * 0.45 * t / fps)
                + 2.0 * head_bias * torch.cos(2 * math.pi * 1.0 * t / fps)
                + micro_r
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
        # Classic speaking-face motion: lift the brows / upper face and widen eyes when speaking emphatically
        # We blend apertures with brow_bias to boost expression on words like fantastic/optimized
        brow_lift = torch.clamp((apertures * 1.2 + brow_bias * 1.8) * motion_profile["brow_lift"], 0.0, 1.5)
        kp_d_rotated[:, :4, 1] -= 0.022 * brow_lift[:, None]
        kp_d_rotated[:, 17:, 1] -= 0.011 * brow_lift[:, None]

        # Apply eye micro-saccades to keypoints [11, 13, 15, 16, 18]
        saccade_dx_t = torch.as_tensor(saccade_dx, dtype=torch.float32, device=device)
        saccade_dy_t = torch.as_tensor(saccade_dy, dtype=torch.float32, device=device)
        for eye_idx in [11, 13, 15, 16, 18]:
            kp_d_rotated[:, eye_idx, 0] += saccade_dx_t
            kp_d_rotated[:, eye_idx, 1] += saccade_dy_t

        # Add 3D depth to the mouth cavity by pushing the inner lip keypoints backward along the z-axis
        # We shift the z-coordinate of lip keypoints backward proportional to the mouth aperture
        kp_d_rotated[:, [14, 17, 19, 20], 2] -= 0.035 * apertures[:, None]

        return kp_d_rotated.detach().cpu().numpy()

    # Mock/legacy fallback
    src = kp_s.detach().cpu().numpy()[0]  # [21, 3]
    base = np.tile(src[None, ...], (driving.frames, 1, 1))  # [N, 21, 3]
    delta = np.asarray(driving.exp_d_flat, dtype=np.float32).reshape(
        driving.frames, N_KEYPOINTS, EXPRESSION_DIM
    )
    mouth = np.asarray(driving.mouth_aperture, dtype=np.float32)
    np_brow_bias = np.asarray(face_biases.get("brow", [0.0] * driving.frames), dtype=np.float32)
    np_head_bias = np.asarray(face_biases.get("head", [0.0] * driving.frames), dtype=np.float32)
    if mouth.shape[0] == driving.frames:
        delta[:, 14:18, 1] *= motion_profile["mouth_boost"]
        # Boost fallback brow lift
        delta[:, :4, 1] -= 0.012 * (mouth[:, None] * 1.2 + np_brow_bias[:, None] * 1.8) * motion_profile["brow_lift"]
        delta[:, 17:, 1] -= 0.006 * (mouth[:, None] * 1.2 + np_brow_bias[:, None] * 1.8) * motion_profile["brow_lift"]

    # Apply eye micro-saccades to fallback path as well
    for eye_idx in [11, 13, 15, 16, 18]:
        delta[:, eye_idx, 0] += saccade_dx
        delta[:, eye_idx, 1] += saccade_dy

    # Apply 3D mouth z-depth to fallback path as well
    delta[:, [14, 17, 19, 20], 2] -= 0.035 * mouth[:, None]

    return base + delta


# ── attach to LivePortraitAdapter ──────────────────────────────────
def _attach_render_methods():
    from providers.liveportrait.adapter.engine import LivePortraitAdapter
    LivePortraitAdapter._real_render_chunk = _real_render_chunk_impl


_attach_render_methods()
