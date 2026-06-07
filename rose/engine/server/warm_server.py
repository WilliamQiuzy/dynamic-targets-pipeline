"""ROSE warm model server.

Keeps DA3, FastSAM, and SAM3 loaded in GPU memory across inference
requests, eliminating the ~15 s model-loading overhead per run.
Optionally enables ``torch.compile`` (20-40 % faster GPU inference)
whose ~30 s warmup cost is paid once at server startup.

Usage::

    # Terminal 1 — start server (loads models, warms up torch.compile)
    python scripts/start_warm_server.py --compile --port 5050

    # Terminal 2 — send requests (no model loading, compiled kernels)
    from rose.engine.server.warm_client import ROSEClient
    client = ROSEClient(port=5050)
    client.wait_ready()
    result = client.build_4dsg("/path/to/video.mp4")
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.pipeline.rose_e2e import (
    _any_mask_iou_above,
    _crop_is_uniform,
    _crop_object_from_mask,
    _mask_centroid,
    _mask_iou,
    _mask_is_fragmented,
    _mask_is_low_texture,
    _merge_chunk_caches,
    _stitch_chunk_ids,
)
from rose.engine.pipeline.rose_pipeline import (
    FastFrameInput,
    FastLocalDetection,
    ROSEPipeline,
)
from rose.vision.perception.da3_wrapper import DA3Wrapper, compute_chunks
from rose.vision.perception.fastsam_wrapper import FastSAMWrapper
from rose.vision.perception.sam3_shared_session_wrapper import (
    SAM3SharedMask,
    SAM3SharedSessionManager,
)

logger = logging.getLogger(__name__)

_SENTINEL = None  # poison pill for CPU worker queue


def _dedup_masks_by_iou(masks, iou_threshold: float = 0.85):
    """D2: greedy frame-level mask dedup — GPU-accelerated for K>=6.

    Strategy:
      • Precompute the K×K mask-IoU matrix in a single GEMM on GPU
        (K=#masks at this frame, typically 10-40).
      • Greedy-select on CPU using the precomputed matrix (microseconds).

    For small K (< 6) the CPU path is faster; we fall back to it.
    """
    n = len(masks)
    if n <= 1:
        return masks
    # CPU fast-path for tiny K
    if n < 6:
        sorted_masks = sorted(masks, key=lambda m: m.score, reverse=True)
        kept = []
        for m in sorted_masks:
            keep = True
            for k in kept:
                inter = np.logical_and(m.mask, k.mask).sum()
                if inter == 0:
                    continue
                union = np.logical_or(m.mask, k.mask).sum()
                if union > 0 and inter / union >= iou_threshold:
                    keep = False
                    break
            if keep:
                kept.append(m)
        return kept

    # GPU path
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idx_by_score = sorted(range(n), key=lambda i: masks[i].score, reverse=True)
        sorted_arr = np.stack([masks[i].mask for i in idx_by_score]).astype(np.float32, copy=False)
        H, W = sorted_arr.shape[1], sorted_arr.shape[2]
        with torch.autocast(device_type="cuda", enabled=False):
            stack = torch.from_numpy(sorted_arr).to(device).view(n, H * W)
            area = stack.sum(dim=1)
            inter = stack @ stack.t()
            union = area[:, None] + area[None, :] - inter
            iou = (inter / union.clamp(min=1.0)).cpu().numpy()
        kept_local: List[int] = []
        for i in range(n):
            ok = True
            for k in kept_local:
                if iou[i, k] >= iou_threshold:
                    ok = False
                    break
            if ok:
                kept_local.append(i)
        return [masks[idx_by_score[i]] for i in kept_local]
    except Exception:
        # If anything goes wrong, fall back to CPU
        sorted_masks = sorted(masks, key=lambda m: m.score, reverse=True)
        kept = []
        for m in sorted_masks:
            keep = True
            for k in kept:
                inter = np.logical_and(m.mask, k.mask).sum()
                if inter == 0:
                    continue
                union = np.logical_or(m.mask, k.mask).sum()
                if union > 0 and inter / union >= iou_threshold:
                    keep = False
                    break
            if keep:
                kept.append(m)
        return kept


# =====================================================================
# Pydantic request/response models
# =====================================================================

class InferenceRequest(BaseModel):
    video_path: str
    question: Optional[str] = None  # None = 4DSG only, skip VLM


class InferenceResponse(BaseModel):
    status: str  # "ok" | "error"
    answer: Optional[str] = None
    four_dsg_dict: Optional[Dict[str, Any]] = None
    scene_json: Optional[str] = None
    keyframe_dir: Optional[str] = None
    error_message: Optional[str] = None
    inference_time_s: Optional[float] = None
    # Dynamic-targets export (动态目标管线 ALL_FRAMES schema). Populated only when
    # config.dynamic_targets.enabled; also written to <frame_dir>/dynamic_targets.json.
    dynamic_targets: Optional[Dict[str, Any]] = None
    dynamic_targets_path: Optional[str] = None


class ServerStatusResponse(BaseModel):
    status: str  # "ready" | "warming_up" | "busy" | "loading"
    models_loaded: Dict[str, bool]
    compile_enabled: bool
    compile_warmed: bool
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    requests_served: int


# =====================================================================
# WarmModelPool — core model lifecycle manager
# =====================================================================

class WarmModelPool:
    """Keep DA3/FastSAM/SAM3 loaded in GPU memory between requests."""

    def __init__(self, config: ROSEConfig):
        self.config = config
        self._da3 = DA3Wrapper(config.da3)
        self._fastsam = FastSAMWrapper(config.fastsam)
        # Switch SAM3 wrapper based on config.use_multiplex.  Both wrappers
        # expose the same interface (set_video_frames/dir, create_run_with_initial_bboxes,
        # add_object_point, propagate_all, propagate_new_objects, end_all_runs).
        if getattr(config.sam3, "use_multiplex", False):
            from rose.vision.perception.sam3_multiplex_wrapper import (
                SAM3MultiplexSharedSessionManager,
            )
            self._sam3 = SAM3MultiplexSharedSessionManager(config.sam3)
            logger.info("Using SAM 3.1 multiplex predictor (Object Multiplex)")
        else:
            self._sam3 = SAM3SharedSessionManager(config.sam3)
        self._lock = threading.Lock()
        self._status = "loading"
        self._compile_warmed = False
        self._requests_served = 0
        # VLM client (lazily created per request if question is provided)
        self._vlm_client = None
        self._bf16_ctx = None  # Scoped bf16 autocast for SAM3 Phase 2
        self._pending_cleanup_thread = None  # deferred _cleanup_sam3_session (off critical path)
        self._da3_warmed_sizes: List[int] = []  # Pre-compiled DA3 batch sizes
        self._cudagraph_enc = None  # CUDAGraphedModule handle when cuda_graph_memory_encoder on
        self._namer = None  # lazy GemmaInstanceNamer (dynamic-targets export only)
        self._last_dynamic_targets = None
        self._last_dynamic_targets_path = None

    def _get_namer(self):
        """Lazily load the Gemma instance namer (dynamic-targets export only)."""
        if self._namer is None:
            from rose.reasoning.vlm.gemma_namer import make_namer
            dtc = self.config.dynamic_targets
            provider = getattr(dtc, "namer_provider", "qwen")
            model_path = getattr(dtc, "namer_model_path", None) or (
                dtc.qwen_model_path if provider == "qwen" else dtc.gemma_model_path)
            self._namer = make_namer(
                provider=provider,
                model_path=model_path,
                device=self.config.device,
                max_crops_per_object=dtc.max_crops_per_object,
                max_new_tokens=dtc.namer_max_new_tokens,
            )
            self._namer.load()
        return self._namer

    def _safe_empty_cache(self) -> None:
        """torch.cuda.empty_cache(), but SKIP it while CUDA graphs are live.

        empty_cache() returns cached blocks to the driver via cudaFree — which
        invalidates memory baked into captured CUDA graphs → illegal memory
        access on the next replay (the crash we hit across videos).  On a
        latency-bound H200 with GPU 50% idle and 14-31GB/141GB used, empty_cache
        buys us nothing anyway, so skipping it when graphs are live is free.
        """
        import torch
        if self._cudagraph_enc is not None:
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all models into GPU memory. Call once at server start."""
        import torch

        # Auto low-VRAM mode: on GPUs with < ~40GB (e.g. V100/T4/most cards), enable
        # SAM3 CPU offload of state+video (big VRAM saving during propagation) and
        # DON'T preload the 8GB instance namer (load it lazily at naming time, after
        # the SAM3/DA3 peak). Without this, all models resident (~30GB) leave too
        # little headroom and SAM3 propagation OOMs. H200/A100-80G keep full-resident
        # speed. Also benefits from PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        self._low_vram = False
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_gb < 40.0:
                self._low_vram = True
                self.config.sam3.offload_state_to_cpu = True
                self.config.sam3.offload_video_to_cpu = True
                logger.info("Low-VRAM GPU (%.0fGB < 40): SAM3 CPU offload ON + namer lazy-loaded.", total_gb)
        except Exception:
            pass

        logger.info("Loading DA3 model...")
        self._da3.load()
        logger.info("Loading FastSAM model...")
        self._fastsam.load()
        logger.info("Loading SAM3 model...")
        self._sam3.load()

        # Dynamic-targets export: preload the instance-naming VLM (one-time) so the
        # first video doesn't pay the load on its critical path — but ONLY on
        # high-VRAM GPUs; on low-VRAM cards keep it lazy so it isn't resident during
        # the SAM3 propagation peak.
        if getattr(self.config, "dynamic_targets", None) is not None and \
                self.config.dynamic_targets.enabled and self.config.dynamic_targets.name_objects \
                and not self._low_vram:
            try:
                logger.info("Loading dynamic-targets instance namer...")
                self._get_namer()
            except Exception as e:
                logger.warning("namer preload failed (non-fatal, will retry lazily): %s", e)

        # Flash-Attention policy.
        # DEFAULT = OFF: force every attention module to use_fa3=False so the model
        # runs on ANY CUDA GPU via PyTorch SDPA (torch auto-picks flash on A100+/
        # Hopper, mem-efficient/math on older cards like V100). The SAM3 checkpoint
        # otherwise loads with ~59 modules at use_fa3=True, which require the FA3
        # (Hopper sm_90) kernel and CRASH on non-Hopper GPUs. Set
        # config.sam3.enable_fa3=True (Hopper H100/H200 only, with FA3 compiled in)
        # to flip them all ON for max speed.
        try:
            root = self._sam3._predictor.model
            enable_fa3 = getattr(self.config.sam3, "enable_fa3", False) or \
                getattr(self.config.sam3, "fa3_everywhere", False)
            if os.environ.get("ROSE_DISABLE_FA3"):
                enable_fa3 = False
            target = bool(enable_fa3)
            n = 0
            for _name, mod in root.named_modules():
                if hasattr(mod, "use_fa3") and bool(getattr(mod, "use_fa3")) != target:
                    mod.use_fa3 = target
                    n += 1
            logger.info("Flash-Attention: use_fa3=%s on all attention modules (%d changed)%s",
                        target, n, "" if target else " — SDPA fallback, runs on any GPU")
            # When FA is OFF, also disable the Hopper-tuned torch.compile / CUDA-graph
            # paths below: the mask-decoder compile re-enables use_fa3=True (a compile-
            # correctness workaround that REQUIRES the FA3 kernel) and compile/bf16 on
            # older GPUs (e.g. V100) is risky. Eager SDPA is correct and runs anywhere.
            if not target:
                for _f in ("compile_memory_encoder", "compile_mask_decoder_transformer",
                           "compile_tracker_strong", "cuda_graph_memory_encoder", "enable_compile"):
                    try:
                        setattr(self.config.sam3, _f, False)
                    except Exception:
                        pass
                logger.info("FA off → torch.compile/cudagraph disabled (pure eager SDPA, any-GPU safe)")
        except Exception as e:
            logger.warning("FA flag flip failed (non-fatal): %s", e)

        # Optional: torch.compile the SAM 3.1 memory-attention encoder. The
        # encoder.forward has a stable input shape ((HW=5184, num_buckets, C)
        # for queries, varying-length memory bank as keys/values).  Compile
        # gives a measured ~4% steady-state speedup on Easy1/Easy2 with no
        # quality loss; the cost is a ~3-4 min compile on the first call.
        try:
            tracker_inner = self._sam3._predictor.model.tracker.model
            _compile_dynamic = getattr(self.config.sam3, "compile_dynamic", True)
            _use_cudagraph_enc = getattr(self.config.sam3, "cuda_graph_memory_encoder", False)
            _strong = getattr(self.config.sam3, "compile_tracker_strong", False)
            if _strong:
                # USER'S "compile SAM2 modules separately": the full native compile
                # ABORTS on torch 2.4 in the SAM3 DETECTOR (backbone.vision_backbone.trunk).
                # So apply the native STRONG recipe to ONLY the 3 SAM2-derived tracker
                # modules, skipping the detector. fullgraph max-autotune; the fixed-shape
                # maskmem/decoder get CUDA graphs (dynamic=False), the growing-memory
                # encoder uses max-autotune-no-cudagraphs (dynamic=True).
                import os as _os, torch._dynamo as _dynamo
                # Parallel Inductor autotuning (user's "compile N kernels at once").
                _os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "32")
                _dynamo.config.cache_size_limit = 128
                _dynamo.config.accumulated_cache_size_limit = 2048
                _dynamo.config.capture_scalar_outputs = True
                _dynamo.config.suppress_errors = True   # any module that won't fullgraph → eager, never wrong
                # Keep mask-decoder numerics correct: force FA3 on its attn modules so
                # AOTAutograd treats SDPA as opaque (naive compile → all-0 masks otherwise).
                dec = getattr(tracker_inner, "sam_mask_decoder", None)
                if dec is not None and hasattr(dec, "transformer"):
                    for layer in getattr(dec.transformer, "layers", []):
                        for nm in ("self_attn", "cross_attn_token_to_image",
                                   "cross_attn_image_to_token"):
                            mod = getattr(layer, nm, None)
                            if mod is not None and hasattr(mod, "use_fa3"):
                                mod.use_fa3 = True
                    fa = getattr(dec.transformer, "final_attn_token_to_image", None)
                    if fa is not None and hasattr(fa, "use_fa3"):
                        fa.use_fa3 = True
                _compiled = []
                # ROSE torch-2.8: mode configurable. "default" (dynamic=True) avoids the
                # max-autotune storm on variable real-video shapes while still compiling
                # the FULL tracker (the part that's stable + benefits), detector left eager.
                _ts_mode = getattr(self.config.sam3, "tracker_strong_mode", "max-autotune")
                _ts_dyn = (_ts_mode != "max-autotune")  # default mode → dynamic ok; max-autotune → static
                mmb = getattr(tracker_inner, "maskmem_backbone", None)
                if mmb is not None:
                    mmb.forward = torch.compile(mmb.forward, mode=_ts_mode,
                                                fullgraph=True, dynamic=_ts_dyn)
                    _compiled.append("maskmem_backbone")
                enc = tracker_inner.transformer.encoder
                _enc_mode = "max-autotune-no-cudagraphs" if _ts_mode == "max-autotune" else _ts_mode
                enc.forward = torch.compile(enc.forward, mode=_enc_mode,
                                            fullgraph=True, dynamic=True)
                _compiled.append("memory_attn_encoder")
                if dec is not None:
                    dec.forward = torch.compile(dec.forward, mode=_ts_mode,
                                                fullgraph=True, dynamic=_ts_dyn)
                    _compiled.append("sam_mask_decoder")
                logger.info(
                    "STRONG tracker compile (SAM2 modules only, detector skipped): %s. "
                    "fullgraph max-autotune, parallel Inductor threads=%s. First inference "
                    "autotunes (slow); warm videos absorb it.",
                    _compiled, _os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
                )
            elif _use_cudagraph_enc:
                # MANUAL CUDA graph (shape-keyed) on the memory-attention encoder.
                # Takes priority over torch.compile of the same module (mutually
                # exclusive — double-wrapping would conflict).
                from rose.vision.perception.cuda_graph_module import CUDAGraphedModule
                enc = tracker_inner.transformer.encoder
                enc.forward = CUDAGraphedModule(enc, name="mem_encoder")
                self._cudagraph_enc = enc.forward  # keep handle for stats
                # GLOBALLY neuter torch.cuda.empty_cache while CUDA graphs are
                # live. SAM3's own close_session() (sam3_base_predictor.py) calls
                # empty_cache() between videos, which frees the graph pools and
                # corrupts the next video's replay (illegal memory access). Our
                # local _safe_empty_cache guard can't catch that internal call.
                # empty_cache buys nothing on a 141GB H200 using 14-31GB, so a
                # process-wide no-op is safe and is the only robust catch-all.
                if not getattr(self, "_empty_cache_neutered", False):
                    torch.cuda.empty_cache = lambda *a, **k: None
                    self._empty_cache_neutered = True
                logger.info(
                    "Memory encoder wrapped with MANUAL CUDA graph "
                    "(shape-keyed capture/replay; first hit of each shape captures). "
                    "torch.cuda.empty_cache globally neutered to protect graph pools."
                )
            elif getattr(self.config.sam3, "compile_memory_encoder", False):
                enc = tracker_inner.transformer.encoder
                _hr_mode = getattr(self.config.sam3, "hand_roll_compile_mode", "reduce-overhead")
                enc.forward = torch.compile(
                    enc.forward, mode=_hr_mode, dynamic=_compile_dynamic,
                )
                logger.info(
                    "Memory encoder forward marked for torch.compile "
                    "(first inference call will take ~3-4 min to compile)."
                )
            # Compile the mask-decoder's TwoWayTransformer. Naive compile breaks
            # quality (AOTAutograd decomposes SDPA → different bf16 numerics →
            # all 0 tracks). Setting use_fa3=True on the decoder Attention
            # modules forces FA3 which AOTAutograd treats as opaque, preserving
            # numerics. Measured combined speedup vs encoder-compile-only:
            # 4.64s → 4.23s (-9%), full quality.
            if (not _strong) and getattr(self.config.sam3, "compile_mask_decoder_transformer", False):
                dec = getattr(tracker_inner, "sam_mask_decoder", None)
                if dec is not None and hasattr(dec, "transformer"):
                    tfm = dec.transformer
                    attn_modules = []
                    for layer in getattr(tfm, "layers", []):
                        for name in ("self_attn", "cross_attn_token_to_image",
                                     "cross_attn_image_to_token"):
                            mod = getattr(layer, name, None)
                            if mod is not None and hasattr(mod, "use_fa3"):
                                attn_modules.append(mod)
                    fa = getattr(tfm, "final_attn_token_to_image", None)
                    if fa is not None and hasattr(fa, "use_fa3"):
                        attn_modules.append(fa)
                    for a in attn_modules:
                        a.use_fa3 = True
                    _hr_mode = getattr(self.config.sam3, "hand_roll_compile_mode", "reduce-overhead")
                    tfm.forward = torch.compile(
                        tfm.forward, mode=_hr_mode, dynamic=_compile_dynamic,
                    )
                    logger.info(
                        "Mask decoder transformer compiled (use_fa3=True on "
                        "%d attention modules to keep AOTAutograd from "
                        "decomposing SDPA).", len(attn_modules),
                    )
        except Exception as e:
            logger.warning("Failed to compile SAM 3.1 sub-modules (non-fatal): %s", e)

        logger.info("All models loaded.")

    # DA3's DINOv2 ViT JIT-compiles CUDA kernels per (batch, H, W).
    # First forward pass with a new batch size costs ~31 s; subsequent
    # calls are ~0.15 s.  We pre-compile multiple batch sizes at startup
    # (step=10) so real inference pads at most 9 frames (< 2% depth error).
    DA3_WARMUP_STEP = 8
    DA3_WARMUP_MIN = 16    # Pre-compile small sizes too — short videos (1-2s)
                           # avoid padding to batch=40 which wastes 25 frames of compute.
    DA3_WARMUP_MAX = 120   # Longest expected video @ 10fps ≈ 12s

    def warmup_cuda(self) -> float:
        """Warm up CUDA kernels for FastSAM and DA3.

        Pre-compiles DA3 CUDA kernels for batch sizes
        [DA3_WARMUP_MIN, min+step, ..., DA3_WARMUP_MAX].
        During inference each video's frame count is rounded up to the
        nearest pre-warmed size (max 9 padding frames → < 2% depth error).

        Returns warmup duration in seconds.
        """
        import torch

        self._status = "warming_up"
        t0 = time.time()

        # FastSAM warmup (quick, ~0.5s)
        rng = np.random.RandomState(42)
        synth = rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            self._fastsam.detect(synth)
        except Exception as e:
            logger.debug("FastSAM warmup error (non-fatal): %s", e)

        # DA3 warmup: pre-compile a range of batch sizes at step=10.
        # Synthetic frames at 1920x1080 → DA3 processes to (280, 504),
        # matching real HD/FHD video resolution.
        sizes = list(range(
            self.DA3_WARMUP_MIN,
            self.DA3_WARMUP_MAX + 1,
            self.DA3_WARMUP_STEP,
        ))
        logger.info(
            "DA3 warmup: pre-compiling CUDA kernels for %d batch sizes %s...",
            len(sizes), sizes,
        )

        # Allocate max-size synthetic frames once, slice for each batch size
        max_sz = sizes[-1]
        synth_frames = [
            rng.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            for _ in range(max_sz)
        ]

        for bs in sizes:
            t_bs = time.time()
            try:
                self._da3._infer_batch_core(synth_frames[:bs])
                self._da3_warmed_sizes.append(bs)
                logger.info("  batch=%d: %.1fs", bs, time.time() - t_bs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("  batch=%d: OOM — stopping warmup here", bs)
                break
            except Exception as e:
                logger.warning("  batch=%d: error %s", bs, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not self._da3_warmed_sizes:
            logger.warning("DA3 warmup produced no valid sizes — inference will JIT on demand")

        elapsed = time.time() - t0
        logger.info(
            "CUDA kernel warmup complete: %.1fs (DA3 sizes=%s)",
            elapsed, self._da3_warmed_sizes,
        )
        return elapsed

    def _pick_da3_batch_size(self, n_frames: int) -> int:
        """Round *n_frames* up to the nearest pre-warmed DA3 batch size.

        Returns *n_frames* unchanged if no pre-warmed size is large enough
        (will trigger a one-time JIT compilation during inference).
        """
        for bs in self._da3_warmed_sizes:
            if bs >= n_frames:
                return bs
        # No pre-warmed size covers this video — use exact count (JIT ~31s)
        return n_frames

    def warmup_compile(self) -> float:
        """Run SAM3's built-in compilation warmup.

        SAM3's ``warm_up_compilation()`` uses a dummy 30-frame video and
        iterates over 0 … ``num_obj_for_compile`` (default 16) objects
        with multiple detection thresholds.  This exercises every tensor
        shape that ``torch.compile(mode='max-autotune')`` will encounter
        during real inference, preventing costly re-compilation later.

        Returns warmup duration in seconds.
        """
        import torch

        self._status = "warming_up"
        logger.info(
            "Starting SAM3 built-in compile warmup "
            "(num_obj_for_compile=%d, ~2-4 min)...",
            self._sam3._predictor.model.num_obj_for_compile,
        )
        t0 = time.time()

        # SAM3's own warmup: dummy video, all object counts, all thresholds
        try:
            self._sam3._predictor.model.warm_up_compilation()
        except Exception as e:
            logger.warning("SAM3 compile warmup error (non-fatal): %s", e)
        torch.cuda.empty_cache()

        # Also warm FastSAM and DA3 CUDA kernels
        rng = np.random.RandomState(42)
        synth = rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            self._fastsam.detect(synth)
        except Exception as e:
            logger.warning("FastSAM warmup error (non-fatal): %s", e)
        try:
            self._da3.infer_batch([synth, synth])
        except Exception as e:
            logger.warning("DA3 warmup error (non-fatal): %s", e)

        elapsed = time.time() - t0
        self._compile_warmed = True
        logger.info("torch.compile warmup complete: %.1fs", elapsed)
        return elapsed

    # ------------------------------------------------------------------
    # GPU memory info
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_mem() -> Tuple[float, float]:
        """Return (used_gb, total_gb) for default CUDA device."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                return (round(used, 2), round(total, 2))
        except Exception:
            pass
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> ServerStatusResponse:
        used, total = self._gpu_mem()
        return ServerStatusResponse(
            status=self._status,
            models_loaded={
                "da3": self._da3._model is not None,
                "fastsam": self._fastsam._model is not None,
                "sam3": self._sam3._predictor is not None,
            },
            compile_enabled=self.config.sam3.enable_compile,
            compile_warmed=self._compile_warmed,
            gpu_memory_used_gb=used,
            gpu_memory_total_gb=total,
            requests_served=self._requests_served,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Thread-safe inference. Acquires lock, processes video, returns result."""
        with self._lock:
            prev_status = self._status
            self._status = "busy"
            try:
                return self._run_inference_impl(request)
            finally:
                self._status = prev_status if prev_status == "ready" else "ready"

    def run_inference_pipelined(self, requests):
        """Process a stream of videos with cross-video overlap.

        Pattern (for each video):
          - Main thread: GPU phases (extract → DA3 ‖ FastSAM/SAM3 → propagate)
          - Background thread: build_4dsg + (optional) VLM on previous video's
            mask_cache + da3_results.  Runs in parallel with the next video's
            GPU work.

        Throughput per video ≈ max(GPU_phase_time, build+VLM_time).  Since
        build_4dsg + VLM (~0.6s) is much shorter than GPU phases (~2.6s),
        overlap effectively hides the entire CPU/VLM tail.
        """
        results = []
        prev_state: Dict[str, Any] = {}

        def _finalize_worker(state):
            """Bg thread: build_4dsg (CPU only).  VLM is deferred to main
            thread because Qwen-VL .generate() running concurrently with
            the next video's SAM3.1 GPU phase on the bg thread corrupts
            SAM3.1 multiplex tracker state — track count drops (5 → 2)
            and per-track obs collapses to 4 on some videos (tested v53).
            With VLM in main thread, both run sequentially and quality is
            stable; the ~50-150ms VLM cost stays in the critical path."""
            try:
                fdsg, sjson = self._finalize_4dsg(
                    state["pipeline"], state["best_crops"], state["frame_dir"],
                )
                state["fdsg"] = fdsg
                state["sjson"] = sjson
            except BaseException as exc:
                state["error"] = exc

        for req in requests:
            with self._lock:
                t0 = time.time()
                # GPU-phase only: extract, DA3 thread, FastSAM+SAM3 setup+propagate.
                # Returns the per-frame mask cache + da3 results without
                # building 4DSG / calling VLM.
                gpu_state = self._run_gpu_phase(req)
                gpu_time = time.time() - t0
            # Now the lock is RELEASED — launch finalize for this video in bg
            # thread, then immediately move on to next video's GPU phase.
            if "error" in gpu_state:
                results.append(InferenceResponse(
                    status="error", error_message=str(gpu_state["error"]),
                    inference_time_s=round(gpu_time, 2),
                ))
                continue
            gpu_state["pipeline_time"] = gpu_time
            gpu_state["question"] = req.question
            cur_thread = threading.Thread(target=_finalize_worker, args=(gpu_state,), daemon=True)
            cur_thread.start()

            # Wait on PREVIOUS video's bg thread (build_4dsg).  Then run VLM
            # in MAIN thread (Qwen-VL bg-thread .generate() crashes with
            # CUDA index errors).  VLM is fast (~0.15s) so the overhead is
            # small relative to the GPU phase already in flight.
            if "thread" in prev_state:
                prev_state["thread"].join()
                results.append(self._finalize_response(prev_state["state"]))

            prev_state = {"thread": cur_thread, "state": gpu_state}

        # Drain final pending thread
        if "thread" in prev_state:
            prev_state["thread"].join()
            results.append(self._finalize_response(prev_state["state"]))
        return results

    def _finalize_response(self, state: Dict[str, Any]) -> InferenceResponse:
        """Run VLM in main thread on bg-built 4DSG, return response."""
        if "error" in state:
            return InferenceResponse(
                status="error", error_message=str(state["error"]),
                inference_time_s=round(state.get("pipeline_time", 0), 2),
            )
        fdsg = state["fdsg"]; sjson = state["sjson"]
        answer = None
        if state.get("question"):
            try:
                answer = self._query_vlm(fdsg, state["question"])
            except Exception as exc:
                logger.warning("VLM call failed: %s", exc)
                answer = f"[VLM error: {exc}]"
        return InferenceResponse(
            status="ok", answer=answer,
            four_dsg_dict=fdsg, scene_json=sjson,
            keyframe_dir=str(state["frame_dir"]),
            inference_time_s=round(state["pipeline_time"], 2),
        )

    def _run_gpu_phase(self, request: InferenceRequest) -> Dict[str, Any]:
        """GPU-only phase for cross-video pipelining.

        Runs frame extraction, DA3, FastSAM, SAM3 multiplex propagation and
        builds the per-frame `pipeline` state by streaming.  Returns a state
        dict with everything `_finalize_4dsg` needs.  Does NOT call VLM.

        Used by ``run_inference_pipelined`` to start finalize for video N
        in a bg thread while main starts video N+1's GPU work.
        """
        import torch
        phase_t: Dict[str, float] = {}
        t_overall = time.time()
        try:
            video_path = Path(request.video_path)
            if not video_path.exists():
                return {"error": f"Video not found: {video_path}"}

            t_extract = time.time()
            frames, frame_dir, source_indices, keyframe_paths, timestamps_s = (
                self._extract_frames(video_path)
            )
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            phase_t["extract"] = time.time() - t_extract

            bs = self._pick_da3_batch_size(len(frames))
            da3_holder: Dict[str, Any] = {}
            da3_stream = torch.cuda.Stream()
            t_da3_start = time.time()
            def _da3_worker():
                try:
                    with torch.cuda.stream(da3_stream):
                        da3_holder["results"] = self._da3.infer_batch_chunked(
                            frames, chunk_size=bs,
                        )
                    da3_stream.synchronize()
                    da3_holder["time"] = time.time() - t_da3_start
                except BaseException as exc:
                    da3_holder["error"] = exc
            da3_thread = threading.Thread(target=_da3_worker, daemon=True)
            da3_thread.start()

            t_p2_start = time.time()
            mask_cache = self._run_phase2(frames, pil_frames)
            phase_t["sam3_total"] = time.time() - t_p2_start

            da3_thread.join()
            if "error" in da3_holder:
                raise da3_holder["error"]
            da3_results = da3_holder["results"]
            phase_t["da3"] = float(da3_holder.get("time", 0.0))

            # Sub-phase times stashed by _run_phase2_multiplex_anchor.
            phase_t["fastsam"] = float(getattr(self, "_last_fastsam_t", 0.0))

            # Build pipeline state via the same per-frame helper used for
            # streaming, but call it sequentially here (still on main thread,
            # very fast since SAM3 propagate is the slow part).
            t_lift = time.time()
            pipeline = ROSEPipeline(self.config)
            best_crops: Dict[tuple, Tuple[np.ndarray, float, int, float]] = {}
            crop_pad = self.config.vlm.object_crop_padding
            crop_sz = self.config.vlm.object_crop_size
            for fidx in range(len(frames)):
                frame_masks = mask_cache.get(fidx, [])
                self._process_frame_for_4dsg(
                    fidx, frame_masks, frames, da3_results,
                    source_indices, timestamps_s,
                    pipeline, best_crops, crop_pad, crop_sz,
                )
            phase_t["lifting"] = time.time() - t_lift
            phase_t["gpu_phase_wall"] = time.time() - t_overall

            # Phase timing breakdown (silent in production; enable with DEBUG logging).
            logger.debug(
                "[PHASE_T] extract=%.2fs  da3=%.2fs  fastsam=%.2fs  "
                "sam3_total=%.2fs  lifting=%.2fs  wall=%.2fs",
                phase_t.get("extract", 0), phase_t.get("da3", 0),
                phase_t.get("fastsam", 0), phase_t.get("sam3_total", 0),
                phase_t.get("lifting", 0), phase_t.get("gpu_phase_wall", 0),
            )

            self._requests_served += 1
            return {
                "pipeline": pipeline,
                "best_crops": best_crops,
                "frame_dir": frame_dir,
                "phase_t": phase_t,
                "pil_frames": pil_frames,
            }
        except BaseException as exc:
            return {"error": exc}
        finally:
            # End SAM3 session and exit any leaked bf16 autocast before
            # returning.  Without this, propagation state from previous
            # videos leaks into the next call and SAM3 only emits masks
            # on a few frames, truncating F_k observations to ~3 entries.
            try:
                self._cleanup_sam3_session()
            except Exception:
                pass
            if self._bf16_ctx is not None:
                try:
                    self._bf16_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                self._bf16_ctx = None
            self._safe_empty_cache()

    def _run_inference_impl(self, request: InferenceRequest) -> InferenceResponse:
        """Core inference logic using warm models."""
        import torch

        # If the previous video's session cleanup was deferred to a bg thread,
        # finish it now — before we open a new session.  This was off the previous
        # caller's critical path; here it overlaps this video's frame extraction.
        if self._pending_cleanup_thread is not None:
            self._pending_cleanup_thread.join()
            self._pending_cleanup_thread = None

        t0 = time.time()
        # Reset per-request dynamic-targets output (avoid leaking previous video's).
        self._last_dynamic_targets = None
        self._last_dynamic_targets_path = None

        try:
            video_path = Path(request.video_path)
            if not video_path.exists():
                return InferenceResponse(
                    status="error",
                    error_message=f"Video not found: {video_path}",
                )

            # Step 0: Extract frames
            frames, frame_dir, source_indices, keyframe_paths, timestamps_s = (
                self._extract_frames(video_path)
            )
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]

            try:
                # Two-way overlap:
                #   Thread A:  DA3 batch infer (own CUDA stream, parallel)
                #   Main:      FastSAM at anchors + SAM3 setup + add_prompt + propagate
                #
                # 4DSG streaming via callback was tried (v15) but hurt Hz by
                # ~5% due to Python GIL/queue overhead exceeding the CPU work
                # being hidden — _build_4dsg is < 1s and CUDA already releases
                # GIL during ViT forward.
                bs = self._pick_da3_batch_size(len(frames))
                logger.info("Phase 1+2 OVERLAPPED: DA3 parallel with SAM3+FastSAM")
                t_overlap_start = time.time()

                da3_holder: Dict[str, Any] = {}
                da3_stream = torch.cuda.Stream()
                def _da3_worker():
                    try:
                        _td = time.time()
                        with torch.cuda.stream(da3_stream):
                            da3_holder["results"] = self._da3.infer_batch_chunked(
                                frames, chunk_size=bs,
                            )
                        da3_stream.synchronize()
                        da3_holder["time"] = time.time() - _td
                    except BaseException as exc:
                        da3_holder["error"] = exc
                da3_thread = threading.Thread(target=_da3_worker, daemon=True)
                da3_thread.start()

                # Phase 2 runs concurrently with DA3.  Optional TEMPORAL
                # SUBSAMPLING: propagate SAM3 on every Nth frame, then copy each
                # mask to the skipped frames.  Propagation (incl. the dominant
                # B-8 late-discovery) scales ~1/N; DA3 + the 4DSG stay at the
                # full 10fps (depth/centroids use each frame's own DA3 depth).
                _stride = max(1, int(getattr(self.config.sam3, "sam3_propagate_stride", 1)))
                if getattr(self.config.sam3, "use_cutie_propagation", False):
                    # Cutie propagates every frame natively (fast) → no stride/flow/dual.
                    mask_cache = self._run_phase2(frames, pil_frames)
                elif getattr(self.config.sam3, "interleaved_dual_stream", False):
                    # GPU-async: even frames on main SAM3 / odd frames on shadow
                    # SAM3, concurrently on two CUDA streams.  Full per-frame
                    # quality (no mask copy) at ~2x.
                    mask_cache = self._run_phase2_dual(frames, pil_frames)
                elif _stride > 1 and len(frames) > 2 * _stride:
                    sub_idx = list(range(0, len(frames), _stride))
                    sub_cache = self._run_phase2(
                        [frames[i] for i in sub_idx],
                        [pil_frames[i] for i in sub_idx],
                    )
                    if getattr(self.config.sam3, "flow_warp_skipped", False):
                        # Tier-4: motion-compensate skipped frames via optical flow.
                        mask_cache = self._flow_warp_skipped(frames, sub_idx, sub_cache)
                    else:
                        # Nearest-keyframe copy (fast, but 3D centroid stair-steps).
                        mask_cache = {}
                        last = len(sub_idx) - 1
                        for fidx in range(len(frames)):
                            mask_cache[fidx] = sub_cache.get(min(round(fidx / _stride), last), [])
                else:
                    mask_cache = self._run_phase2(frames, pil_frames)

                da3_thread.join()
                if "error" in da3_holder:
                    raise da3_holder["error"]
                da3_results = da3_holder["results"]
                streaming_4dsg = False  # disabled — see comment above
                build_state = {}
                logger.info("Phase 1+2 (overlapped) done: %.2fs", time.time() - t_overlap_start)

                # Phase 2d + Phase 3: Build 4DSG
                if streaming_4dsg and build_state.get("pipeline") is not None:
                    # 4DSG already built incrementally during SAM3 propagation.
                    # Just finalize crops + serialize.
                    four_dsg_dict, scene_json = self._finalize_4dsg(
                        build_state["pipeline"], build_state["best_crops"],
                        frame_dir,
                    )
                else:
                    four_dsg_dict, scene_json = self._build_4dsg(
                        frames, mask_cache, da3_results,
                        source_indices, timestamps_s, frame_dir,
                    )

                # Optional: VLM query
                answer = None
                if request.question:
                    answer = self._query_vlm(four_dsg_dict, request.question)

                elapsed = time.time() - t0
                self._requests_served += 1

                return InferenceResponse(
                    status="ok",
                    answer=answer,
                    four_dsg_dict=four_dsg_dict,
                    scene_json=scene_json,
                    keyframe_dir=str(frame_dir),
                    inference_time_s=round(elapsed, 2),
                    dynamic_targets=self._last_dynamic_targets,
                    dynamic_targets_path=self._last_dynamic_targets_path,
                )

            except Exception as e:
                logger.error("Inference failed: %s", e, exc_info=True)
                shutil.rmtree(frame_dir, ignore_errors=True)
                return InferenceResponse(
                    status="error",
                    error_message=str(e),
                    inference_time_s=round(time.time() - t0, 2),
                )
            finally:
                # Clean up SAM3 session state (keep model loaded).  The 4DSG result
                # is already built (masks copied to CPU) before this finally, so the
                # session GPU tensors are safe to free now.  When deferred, run it in
                # a daemon thread so it doesn't block the caller's return (~0.46s);
                # the NEXT inference joins it before opening a new session.
                if getattr(self.config.sam3, "defer_session_cleanup", False):
                    def _bg_cleanup():
                        try:
                            self._cleanup_sam3_session()
                        except Exception:
                            pass
                    _th = threading.Thread(target=_bg_cleanup, daemon=True)
                    _th.start()
                    self._pending_cleanup_thread = _th
                else:
                    try:
                        self._cleanup_sam3_session()
                    except Exception:
                        pass
                if self._bf16_ctx is not None:
                    try:
                        self._bf16_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                    self._bf16_ctx = None
                self._safe_empty_cache()

        except Exception as e:
            logger.error("Inference request failed: %s", e, exc_info=True)
            return InferenceResponse(
                status="error",
                error_message=str(e),
                inference_time_s=round(time.time() - t0, 2),
            )

    # ------------------------------------------------------------------
    # Phase 2: FastSAM + SAM3 two-pass
    # ------------------------------------------------------------------

    def _ensure_cutie(self):
        """Lazily load the pretrained Cutie VOS model (shared across videos)."""
        c = getattr(self, "_cutie_model", None)
        if c is not None:
            return c
        logger.info("Loading Cutie (pretrained real-time VOS propagator)...")
        from cutie.utils.get_default_model import get_default_model
        c = get_default_model()
        self._cutie_model = c
        return c

    def _run_phase2_cutie(
        self, frames: List[np.ndarray], pil_frames: list,
    ) -> Dict[int, List[SAM3SharedMask]]:
        """Cutie propagation path: FastSAM+dedup discovers objects, then the
        pretrained Cutie VOS model propagates their masks across ALL frames
        (mid-video object additions supported).  Replaces the SAM3 per-frame
        tracking (~73% of runtime) with a purpose-built real-time propagator
        (~3.7-4.3x faster), no training.  Returns the same mask_cache format.
        """
        import torch
        from cutie.inference.inference_core import InferenceCore
        sc = self.config.sam3
        n = len(frames); H, W = frames[0].shape[:2]
        # B-1/B-2: FastSAM at anchors + dedup (reuse the existing discovery)
        stride = int(getattr(sc, "anchor_stride", 4))
        anchors = list(range(0, n, stride))
        if anchors and anchors[-1] != n - 1:
            anchors.append(n - 1)
        max_frac = float(getattr(self.config.fastsam, "max_mask_frac", 0.0))
        anchor_dets = {}
        for a in anchors:
            raw = self._fastsam.detect(frames[a])
            anchor_dets[a] = [
                d for d in raw
                if not _mask_is_fragmented(d.mask)
                and not _mask_is_low_texture(frames[a], d.mask)
                and not (max_frac > 0 and float(d.mask.sum()) > max_frac * d.mask.size
                         and _mask_is_low_texture(frames[a], d.mask))
            ]
        uniq = self._dedup_anchor_detections(
            anchors, anchor_dets,
            iou_thresh=float(getattr(sc, "cross_anchor_iou_thresh", 0.4)),
            containment_iom=float(getattr(sc, "seed_containment_iom", 0.7)),
            use_gpu=bool(getattr(sc, "gpu_anchor_dedup", True)),
        )
        cap = sc.max_active_tracks if sc.max_active_tracks > 0 else 50
        if len(uniq) > cap:
            uniq = sorted(uniq, key=lambda o: -o["mask_area"])[:cap]
        if not uniq:
            return {}
        by_frame: Dict[int, list] = {}
        for i, o in enumerate(uniq):
            by_frame.setdefault(int(o["first_anchor"]), []).append((i + 1, o["mask"]))

        proc = InferenceCore(self._ensure_cutie(), cfg=self._ensure_cutie().cfg)
        mask_cache: Dict[int, List[SAM3SharedMask]] = {}
        with torch.inference_mode(), torch.autocast("cuda"):
            for f in range(n):
                img = torch.from_numpy(frames[f]).permute(2, 0, 1).float().cuda() / 255.0
                new = by_frame.get(f, [])
                if new:
                    idxm = torch.zeros(H, W, dtype=torch.long, device="cuda")
                    for oid, m in new:
                        idxm[torch.from_numpy(np.ascontiguousarray(m.astype(bool))).cuda()] = oid
                    prob = proc.step(img, idxm, objects=[oid for oid, _ in new])
                else:
                    prob = proc.step(img)
                labels = proc.output_prob_to_mask(prob)  # (H,W) tmp ids
                fmasks = []
                for tmp_id, obj in proc.object_manager.tmp_id_to_obj.items():
                    bm = (labels == tmp_id)
                    if bool(bm.any()):
                        fmasks.append(SAM3SharedMask(
                            run_id="cutie", obj_id_local=int(obj.id),
                            mask=bm.cpu().numpy(), score=1.0))
                mask_cache[f] = fmasks
        return mask_cache

    def _flow_warp_skipped(
        self, frames: List[np.ndarray], sub_idx: List[int],
        sub_cache: Dict[int, List[SAM3SharedMask]],
    ) -> Dict[int, List[SAM3SharedMask]]:
        """Tier-4: fill skipped (non-keyframe) frames by WARPING the nearest
        keyframe's masks via dense optical flow (Farneback), instead of copying
        them.  This motion-compensates the mask to the object's new position, so
        the per-frame 3D centroid follows the real motion (no stair-step) — while
        still only running SAM3 on the keyframes (the launch-reducing win).
        Flow is computed at half resolution (~1s/video) ≪ the SAM3 frames saved.
        """
        import cv2 as _cv2
        from rose.vision.perception.sam3_multiplex_wrapper import SAM3SharedMask
        n = len(frames)
        H, W = frames[0].shape[:2]
        gw, gh = W // 2, H // 2
        grays = [_cv2.cvtColor(_cv2.resize(f, (gw, gh)), _cv2.COLOR_RGB2GRAY) for f in frames]
        kf_of = {orig: i for i, orig in enumerate(sub_idx)}
        sub_set = set(sub_idx)
        gx = np.arange(gw, dtype=np.float32)[None, :]
        gy = np.arange(gh, dtype=np.float32)[:, None]
        out: Dict[int, List[SAM3SharedMask]] = {}

        def _warp_one(fidx):
            # Per-frame Farneback flow + remap (independent across frames).
            kf = max((s for s in sub_idx if s <= fidx), default=sub_idx[0])
            src = sub_cache.get(kf_of[kf], [])
            if not src:
                return fidx, []
            flow = _cv2.calcOpticalFlowFarneback(
                grays[kf], grays[fidx], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mapx = (gx + flow[..., 0])
            mapy = (gy + flow[..., 1])
            warped = []
            for m in src:
                small = _cv2.resize(m.mask.astype(np.uint8), (gw, gh),
                                    interpolation=_cv2.INTER_NEAREST)
                w = _cv2.remap(small, mapx, mapy, _cv2.INTER_NEAREST)
                wf = _cv2.resize(w, (W, H), interpolation=_cv2.INTER_NEAREST).astype(bool)
                warped.append(SAM3SharedMask(run_id=m.run_id, obj_id_local=m.obj_id_local,
                                             mask=wf, score=m.score))
            return fidx, warped

        for fidx in sub_set:
            out[fidx] = sub_cache.get(kf_of[fidx], [])
        skipped = [f for f in range(n) if f not in sub_set]
        nw = int(getattr(self.config.sam3, "flow_warp_workers", 1) or 1)
        if nw > 1 and len(skipped) > 1:
            # Parallelize the independent per-frame flow (cv2 releases the GIL).
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(nw, len(skipped))) as _ex:
                for fidx, warped in _ex.map(_warp_one, skipped):
                    out[fidx] = warped
        else:
            for fidx in skipped:
                _, out[fidx] = _warp_one(fidx)
        return out

    def _ensure_shadow(self) -> "WarmModelPool":
        """Lazily build a 2nd, independent SAM3+FastSAM pool (its own session
        state) so two threads can propagate on two CUDA streams concurrently
        without sharing any mutable state.  DA3 is NOT loaded (phase 2 only needs
        SAM3 + FastSAM)."""
        sh = getattr(self, "_shadow", None)
        if sh is not None:
            return sh
        logger.info("Building shadow SAM3 pool for interleaved dual-stream...")
        sh = WarmModelPool(self.config)
        sh._fastsam.load()
        sh._sam3.load()
        # mirror the tracker knobs load_all() applies to the main SAM3 so both
        # halves track identically.
        try:
            tk = sh._sam3._predictor.model.tracker
            sc = self.config.sam3
            tk.max_cond_frames_in_attn = sc.max_cond_frames_in_attn
            tk.keep_first_cond_frame = sc.keep_first_cond_frame
            tk.use_memory_selection = sc.use_memory_selection
            tk.mf_threshold = sc.mf_threshold
            if sc.num_maskmem != 7:
                tk.num_maskmem = sc.num_maskmem
        except Exception as e:
            logger.warning("shadow tracker-knob mirror failed (non-fatal): %s", e)
        sh._status = "ready"
        self._shadow = sh
        return sh

    def _run_phase2_dual(
        self, frames: List[np.ndarray], pil_frames: list,
    ) -> Dict[int, List[SAM3SharedMask]]:
        """Interleaved dual-stream phase 2: EVEN frames on the main SAM3 (stream
        A), ODD frames on the shadow SAM3 (stream B), CONCURRENTLY.  Each half
        propagates a real mask for every one of its frames (no copying), and the
        two fill each other's GPU idle → ~2x at full per-frame quality.  The two
        sessions assign object ids independently, so the shadow's run_ids are
        namespaced apart; the downstream re-ID / mask-IoU dedup then unifies the
        same physical object's even-half and odd-half tracks.
        """
        import torch, threading
        from rose.vision.perception.sam3_multiplex_wrapper import SAM3SharedMask
        n = len(frames)
        if n < 4:
            return self._run_phase2(frames, pil_frames)
        shadow = self._ensure_shadow()
        even = list(range(0, n, 2))
        odd = list(range(1, n, 2))
        out: Dict[str, Any] = {}
        sa, sb = torch.cuda.Stream(), torch.cuda.Stream()

        def _work(pool, idxs, key, stream):
            try:
                with torch.cuda.stream(stream):
                    out[key] = pool._run_phase2(
                        [frames[i] for i in idxs], [pil_frames[i] for i in idxs],
                    )
                stream.synchronize()
            except BaseException as exc:
                out[key + "_err"] = exc

        ta = threading.Thread(target=_work, args=(self, even, "a", sa), daemon=True)
        tb = threading.Thread(target=_work, args=(shadow, odd, "b", sb), daemon=True)
        ta.start(); tb.start(); ta.join(); tb.join()
        if "a_err" in out:
            raise out["a_err"]
        if "b_err" in out:
            raise out["b_err"]
        mc_a, mc_b = out.get("a", {}), out.get("b", {})

        # Cross-session matching: the even (A) and odd (B) sessions assign object
        # ids independently, so the SAME physical object is a different id in each
        # and (being on alternating frames) the mask-IoU dedup can't unify them.
        # Match each B-track to the A-track it overlaps on ADJACENT frames (B's
        # odd frame f vs A's even frame f±1) and relabel B onto A's id; unmatched
        # B-tracks (objects only seen on odd frames) keep a namespaced id.
        import cv2 as _cv2

        def _track_masks(mc, idxs):
            d: Dict[tuple, Dict[int, np.ndarray]] = {}
            for k, fidx in enumerate(idxs):
                for m in mc.get(k, []):
                    mk = m.mask
                    if mk is None or not mk.any():
                        continue
                    small = _cv2.resize(mk.astype(np.uint8), (96, 96),
                                        interpolation=_cv2.INTER_NEAREST).astype(bool)
                    d.setdefault((m.run_id, m.obj_id_local), {})[fidx] = small
            return d

        A = _track_masks(mc_a, even)
        B = _track_masks(mc_b, odd)

        def _iou(a, b):
            inter = np.logical_and(a, b).sum()
            uni = np.logical_or(a, b).sum()
            return float(inter / uni) if uni > 0 else 0.0

        b_to_a: Dict[tuple, tuple] = {}
        for bk, bm in B.items():
            best_a, best_iou = None, 0.0
            for ak, am in A.items():
                ious = []
                for of in bm:
                    for adj in (of - 1, of + 1):
                        if adj in am:
                            ious.append(_iou(bm[of], am[adj]))
                if ious:
                    miou = float(np.mean(ious))
                    if miou > best_iou:
                        best_iou, best_a = miou, ak
            if best_a is not None and best_iou >= 0.4:
                b_to_a[bk] = best_a

        merged: Dict[int, List[SAM3SharedMask]] = {}
        for i, fidx in enumerate(even):
            merged[fidx] = mc_a.get(i, [])
        for j, fidx in enumerate(odd):
            lst = []
            for m in mc_b.get(j, []):
                tgt = b_to_a.get((m.run_id, m.obj_id_local))
                if tgt is not None:   # same object as an A-track → adopt its id
                    lst.append(SAM3SharedMask(run_id=tgt[0], obj_id_local=tgt[1],
                                              mask=m.mask, score=m.score))
                else:                 # odd-only object → namespaced id
                    lst.append(SAM3SharedMask(run_id=f"B::{m.run_id}",
                                              obj_id_local=m.obj_id_local,
                                              mask=m.mask, score=m.score))
            merged[fidx] = lst
        return merged

    def _run_phase2(
        self,
        frames: List[np.ndarray],
        pil_frames: list,
    ) -> Dict[int, List[SAM3SharedMask]]:
        """Run Phases 2a-2c: FastSAM detection + SAM3 tracking.

        Dispatches based on config.sam3.use_multiplex:
          - True  → _run_phase2_multiplex_anchor (design B: K-anchor batched init,
                    all prompts upfront, ONE propagate_in_video).
          - False → original two-pass discovery flow (base SAM3 wrapper).
        use_cutie_propagation overrides both: FastSAM discovery + Cutie tracking.
        """
        if getattr(self.config.sam3, "use_cutie_propagation", False):
            return self._run_phase2_cutie(frames, pil_frames)
        if getattr(self.config.sam3, "use_multiplex", False):
            return self._run_phase2_multiplex_anchor(frames, pil_frames)

        import torch

        n = len(frames)

        # ── Quality fix 1: background-blob filter (Issue: "发现物体是否合格") ──
        # FastSAM happily segments whole-frame background regions (floor, sky,
        # building facades) as "objects".  config.fastsam.max_mask_frac exists
        # to reject masks covering more than a fraction of the frame, but it was
        # only ever wired into the multiplex path.  Wire it in here too so the
        # non-multiplex discovery path stops promoting background blobs to tracks.
        max_frac = float(getattr(self.config.fastsam, "max_mask_frac", 0.0))
        def _too_large(mask, image) -> bool:
            # Close-up fix: only reject a LARGE mask when it is ALSO background-
            # like (low texture).  A close-up foreground subject (an animal/object
            # filling most of the frame) is large but textured and must be KEPT;
            # only sky / wall / floor blobs (large AND low-texture) are dropped.
            if max_frac <= 0.0:
                return False
            h, w = mask.shape[:2]
            if float(mask.sum()) <= max_frac * h * w:
                return False
            return _mask_is_low_texture(image, mask)

        # SAM3 tracker requires bf16 autocast on the calling thread.
        # Scope it to Phase 2 only — leaving it active globally causes
        # DA3 to run under bf16, triggering CUDA kernel JIT recompilation
        # (30s instead of 0.75s).
        self._bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self._bf16_ctx.__enter__()
        sam3 = self._sam3
        sam3_cfg = self.config.sam3

        # Set video frames (resets session)
        sam3.set_video_frames(pil_frames)

        # -- Phase 2a: FastSAM init frame → SAM3 init + full propagation --
        logger.info("Phase 2a: FastSAM frame 0 + SAM3 init + propagation...")
        t_2a = time.time()

        # Try frame 0 first; if FastSAM finds nothing, scan subsequent
        # frames and pick the one with the most detections.
        MAX_INIT_FRAME_ATTEMPTS = min(n, 10)
        _raw0 = self._fastsam.detect(frames[0])
        # P1-1 + P1-2: drop fragmented & low-texture init detections so the
        # initial SAM3 run isn't seeded with ground / sky / multi-object blobs.
        dets_0 = [
            d for d in _raw0
            if not _mask_is_fragmented(d.mask)
            and not _mask_is_low_texture(frames[0], d.mask)
            and not _too_large(d.mask, frames[0])
        ]
        init_frame = 0

        if not dets_0:
            # Frame 0 failed — scan candidates and pick the richest frame
            best_frame = -1
            best_dets = None
            best_count = 0
            for try_idx in range(1, MAX_INIT_FRAME_ATTEMPTS):
                raw = self._fastsam.detect(frames[try_idx])
                dets = [
                    d for d in raw
                    if not _mask_is_fragmented(d.mask)
                    and not _mask_is_low_texture(frames[try_idx], d.mask)
                    and not _too_large(d.mask, frames[try_idx])
                ]
                if len(dets) > best_count:
                    best_frame = try_idx
                    best_dets = dets
                    best_count = len(dets)
            if best_dets is None:
                logger.warning(
                    "FastSAM detected 0 objects on first %d frames.",
                    MAX_INIT_FRAME_ATTEMPTS,
                )
                self._bf16_ctx.__exit__(None, None, None)
                self._bf16_ctx = None
                return {}
            init_frame = best_frame
            dets_0 = best_dets
            logger.info(
                "FastSAM fallback: using frame %d as init (%d detections; frame 0 had none)",
                init_frame, len(dets_0),
            )

        bboxes = [list(d.bbox_xywh_norm) for d in dets_0]

        _, init_masks = sam3.create_run_with_initial_bboxes(
            boxes_xywh=bboxes,
            box_labels=[1] * len(bboxes),
            frame_idx=init_frame,
            tag="fastsam_bbox",
        )
        logger.info("SAM3 init: %d masks from %d bboxes (frame %d)", len(init_masks), len(bboxes), init_frame)

        # Apply acceleration settings
        sam3._predictor.model.retain_feature_cache = True
        if sam3_cfg.num_maskmem != 7:
            sam3._predictor.model.tracker.num_maskmem = sam3_cfg.num_maskmem
        if sam3_cfg.memory_temporal_stride != 1:
            sam3._predictor.model.tracker.memory_temporal_stride_for_eval = (
                sam3_cfg.memory_temporal_stride
            )
        # Unlock SAM 3.1 native re-tracking memory (the multiplex checkpoint
        # ships with max_cond_frames_in_attn=4 and keep_first_cond_frame=False
        # which evicts the initial frame-0 prompt and kills long-horizon re-
        # acquisition).  Also turn on SAM's native memory-frame selection.
        if hasattr(sam3._predictor.model, "tracker"):
            tk = sam3._predictor.model.tracker
            tk.max_cond_frames_in_attn = sam3_cfg.max_cond_frames_in_attn
            tk.keep_first_cond_frame = sam3_cfg.keep_first_cond_frame
            tk.use_memory_selection = sam3_cfg.use_memory_selection
            tk.mf_threshold = sam3_cfg.mf_threshold

        # Pre-compute backbone features
        if sam3_cfg.retain_backbone_cache:
            sam3.precompute_backbone_features(vg_stride=sam3_cfg.vg_stride)

        # Full propagation — single call triggers internal loop over all frames;
        # subsequent calls for other frame indices return from cache.
        sam3.propagate_all(n - 1)

        # Add remaining init-frame bboxes as point prompts
        MAX_FRAME0_POINTS = 8
        frame0_point_count = 0
        if len(bboxes) > 1:
            remaining = list(range(1, len(dets_0)))
            remaining.sort(key=lambda i: dets_0[i].mask.sum(), reverse=True)
            for i in remaining:
                if frame0_point_count >= MAX_FRAME0_POINTS:
                    break
                if init_masks and _any_mask_iou_above(
                    dets_0[i].mask, init_masks, 0.5,
                ):
                    continue
                bx, by, bw, bh = bboxes[i]
                sam3.add_object_point(init_frame, (bx + bw / 2.0, by + bh / 2.0))
                frame0_point_count += 1

            if frame0_point_count > 0:
                sam3.propagate_new_objects()

        logger.info("Phase 2a done: %.2fs", time.time() - t_2a)

        # -- Phase 2b: Discovery --
        t_2b = time.time()
        new_obj_count = 0
        discovery_thresh = self.config.fastsam.discovery_iou_thresh
        max_disc = sam3_cfg.max_discovery_per_frame
        min_mask_frac = sam3_cfg.discovery_min_mask_frac
        max_total = sam3_cfg.max_discovery_total
        discovery_stride = sam3_cfg.full_propagation_stride

        # Cap by max_active_tracks
        max_active = sam3_cfg.max_active_tracks
        if max_active > 0:
            current_active = len(sam3._obj_id_to_run_id)
            headroom = max(0, max_active - current_active)
            if max_total > 0:
                max_total = min(max_total, headroom)
            else:
                max_total = headroom

        if sam3.active_runs:
            for fidx in range(init_frame + 1, n):
                if max_total > 0 and new_obj_count >= max_total:
                    break
                if discovery_stride > 1 and fidx % discovery_stride != 0:
                    continue
                dets = self._fastsam.detect(frames[fidx])
                cached = sam3.propagate_all(fidx)
                frame_disc = 0
                for det in dets:
                    if max_disc > 0 and frame_disc >= max_disc:
                        break
                    if max_total > 0 and new_obj_count >= max_total:
                        break
                    if min_mask_frac > 0:
                        h, w = det.mask.shape[:2]
                        if det.mask.sum() < min_mask_frac * h * w:
                            continue
                    # Quality fix 1: skip whole-frame background blobs.
                    if _too_large(det.mask, frames[fidx]):
                        continue
                    # P1-1: skip fragmented FastSAM masks (multiple comparable blobs)
                    if _mask_is_fragmented(det.mask):
                        continue
                    # P1-2: skip low-texture ground / sky / wall segments
                    if _mask_is_low_texture(frames[fidx], det.mask):
                        continue
                    if not _any_mask_iou_above(
                        det.mask, cached, discovery_thresh
                    ):
                        cy_px, cx_px = _mask_centroid(det.mask)
                        h, w = det.mask.shape[:2]
                        sam3.add_object_point(fidx, (cx_px / w, cy_px / h))
                        new_obj_count += 1
                        frame_disc += 1

        logger.info("Phase 2b: %d discoveries in %.2fs", new_obj_count, time.time() - t_2b)

        # -- Phase 2c: Partial propagation for discovery objects --
        t_2c = time.time()
        if new_obj_count > 0:
            sam3.propagate_new_objects()
        logger.info("Phase 2c: %.2fs", time.time() - t_2c)

        # Collect mask cache
        mask_cache: Dict[int, List[SAM3SharedMask]] = {}
        for fidx in range(n):
            mask_cache[fidx] = list(sam3.propagate_all(fidx))

        # Exit bf16 autocast so subsequent phases (DA3 etc.) are not affected
        self._bf16_ctx.__exit__(None, None, None)
        self._bf16_ctx = None

        # ── Quality fix 2: merge fragmented tracks (Issue: "物体被切成多个 track / 断帧") ──
        # FastSAM late-discovery + SAM3 loss/re-acquire produce several obj_ids
        # for one physical object.  The multiplex path dedups these by per-track
        # mask-IoU across shared frames (robust to DA3 relative-depth scale,
        # unlike 3D-centroid dedup); the non-multiplex path never called it.
        # Apply the SAME tested dedup here so identity duplicates collapse to one
        # track before the 4DSG ingests the masks.
        mask_cache = self._dedup_mask_cache_by_trajectory(
            mask_cache,
            iou_thresh=float(getattr(self.config.fusion, "mask_traj_iou_thresh", 0.4)),
        )

        return mask_cache

    # ------------------------------------------------------------------
    # Phase 2 (multiplex / design B): K-anchor batched init
    # ------------------------------------------------------------------

    def _run_phase2_multiplex_anchor(
        self,
        frames: List[np.ndarray],
        pil_frames: list,
    ) -> Dict[int, List[SAM3SharedMask]]:
        """Design B for SAM 3.1 multiplex.

        Pattern:
          1. Run FastSAM at K anchor frames spaced by ``anchor_stride``
             (default 4 → 8 anchors over 32 frames).
          2. Cross-anchor dedup: bbox center distance + mask IoU heuristic.
          3. ONE add_prompt loop: each unique object registered at its
             FIRST-SEEN anchor frame.
          4. ONE propagate_in_video → multiplex jointly tracks all objects
             across all frames in a single batched joint propagation.
          5. Collect masks per frame.

        Multiplex's win is in step 4 (joint propagation across all objects
        in one pass).  Steps 1-3 keep FastSAM-based discovery so we don't
        rely on text grounding.
        """
        import torch

        n = len(frames)
        sam3 = self._sam3
        sam3_cfg = self.config.sam3
        anchor_stride = int(getattr(sam3_cfg, "anchor_stride", 4))

        self._bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self._bf16_ctx.__enter__()

        sam3.set_video_frames(pil_frames)

        # ---- Phase B-1: FastSAM at all K anchor frames ---------------
        # NOTE: tried to thread FastSAM in parallel with SAM3 setup (v19),
        # but it caused 3% Hz regression — likely due to ultralytics' default
        # stream contending with main-thread torch ops + GIL.  Sequential
        # FastSAM inside main is faster.
        anchor_idxs = list(range(0, n, anchor_stride))
        if anchor_idxs[-1] != n - 1 and (n - 1) - anchor_idxs[-1] >= anchor_stride // 2:
            anchor_idxs.append(n - 1)

        t_b1 = time.time()
        # H200 win: batch all anchors in a SINGLE FastSAM forward pass instead
        # of 8 sequential per-frame calls (saves ~1-1.5s).
        anchor_frames = [frames[fidx] for fidx in anchor_idxs]
        if hasattr(self._fastsam, "detect_batch"):
            batch_results = self._fastsam.detect_batch(anchor_frames)
            anchor_dets: Dict[int, list] = {
                fidx: batch_results[i] for i, fidx in enumerate(anchor_idxs)
            }
        else:
            anchor_dets = {fidx: self._fastsam.detect(frames[fidx]) for fidx in anchor_idxs}
        n_total_dets = sum(len(d) for d in anchor_dets.values())
        self._last_fastsam_t = time.time() - t_b1
        logger.info(
            "Phase B-1: FastSAM batched on %d anchors → %d total detections in %.2fs",
            len(anchor_idxs), n_total_dets, self._last_fastsam_t,
        )

        # ---- Phase B-2: cross-anchor dedup ---------------------------
        t_b2 = time.time()
        unique_objects = self._dedup_anchor_detections(
            anchor_idxs, anchor_dets,
            iou_thresh=getattr(sam3_cfg, "cross_anchor_iou_thresh", 0.4),
            containment_iom=getattr(sam3_cfg, "seed_containment_iom", 0.7),
            use_gpu=bool(getattr(sam3_cfg, "gpu_anchor_dedup", True)),
        )
        logger.info(
            "Phase B-2: dedup → %d unique objects in %.2fs",
            len(unique_objects), time.time() - t_b2,
        )

        if not unique_objects:
            logger.warning("No objects detected across anchors — empty mask cache")
            self._bf16_ctx.__exit__(None, None, None)
            self._bf16_ctx = None
            return {}

        # ---- Phase B-2.5: build FastSAM saliency masks for token pruning -----
        if getattr(sam3_cfg, "use_token_pruning", False):
            t_b25 = time.time()
            try:
                import torch as _torch
                # Build everything outside bf16 autocast so .float() conversions
                # of large bool masks stay on the CPU path quickly.
                with _torch.autocast(device_type="cuda", enabled=False):
                    from rose.vision.sam3.token_pruning import (
                        build_per_frame_saliency_masks,
                        install_token_pruning,
                        set_saliency_on_model,
                    )
                    feat_h = feat_w = int(getattr(sam3_cfg, "token_prune_feat_size", 32))
                    dilate = int(getattr(sam3_cfg, "token_prune_dilate_cells", 2))
                    # Downsample FastSAM masks per-anchor directly to feature scale
                    # (avoid the union step on full-res bool tensors — let the
                    # downsample collapse all detections into one small bool).
                    import torch.nn.functional as _F
                    anchor_unions = {}
                    for fidx, dets in anchor_dets.items():
                        if not dets:
                            continue
                        # Concatenate masks (N, H, W) → max-pool over N then downsample.
                        # This is much cheaper than N pairwise ORs on full-res.
                        stack = _torch.from_numpy(
                            __import__("numpy").stack([d.mask for d in dets], axis=0)
                        ).float()  # (N, H_img, W_img)
                        # Union via max over N
                        union = stack.amax(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
                        small = _F.interpolate(union, size=(feat_h, feat_w), mode="nearest")
                        anchor_unions[fidx] = small.squeeze().bool()
                    if anchor_unions:
                        masks = build_per_frame_saliency_masks(
                            anchor_unions, n_frames=n,
                            feat_h=feat_h, feat_w=feat_w,
                            dilate_cells=dilate,
                        )
                        install_token_pruning(sam3._predictor)
                        set_saliency_on_model(sam3._predictor, masks)
                        avg_keep = sum(
                            m.float().mean().item() for m in masks.values()
                        ) / max(1, len(masks))
                        logger.info(
                            "Phase B-2.5: token pruning enabled — %d frames, avg keep_ratio=%.2f",
                            len(masks), avg_keep,
                        )
                    else:
                        set_saliency_on_model(sam3._predictor, None)
            except Exception as e:
                if not getattr(self, "_tp_warned", False):
                    logger.warning("Phase B-2.5 (token pruning) skipped: %s", e, exc_info=True)
                    self._tp_warned = True
                else:
                    logger.warning("Phase B-2.5 (token pruning) skipped: %s", e)
            logger.debug("[TIMING] Phase B-2.5 token-pruning setup: %.3fs", time.time() - t_b25)

        # ---- Phase B-3: register ALL objects in ONE multi-bbox add_prompt -----
        # Multiplex's bbox-grounding path accepts a multi-bbox tensor and
        # processes them jointly (~0.2s for 16 bboxes vs 1s each per-call).
        # The state RESETS per add_prompt with bboxes — so we must do exactly
        # one call.  All bboxes go to the FIRST anchor frame.
        t_b3 = time.time()
        max_objects = sam3_cfg.max_active_tracks if sam3_cfg.max_active_tracks > 0 else 50
        if len(unique_objects) > max_objects:
            unique_objects.sort(key=lambda o: -o["mask_area"])
            unique_objects = unique_objects[:max_objects]
            logger.info("Capped to top-%d objects by mask area", max_objects)

        # Save late-anchor objects (first_anchor != 0) BEFORE the frame-0
        # filter so we can run a second multiplex session for them in B-8.
        # Two-session design: session 1 handles frame-0 batch (stable K=3
        # baseline), session 2 handles the late discoveries in a fresh state.
        # This avoids the multiplex mid-session-add instability we observed
        # in v3-v12 attempts (each session is "clean": one prompt type only).
        # Close-up / late-entry fix: session 1 was hardcoded to seed at frame 0,
        # so a video whose subject is NOT detected on frame 0 (e.g. a dog that
        # enters at frame 12) seeded NOTHING → 0 tracks.  Seed from the EARLIEST
        # anchor that actually has detections (still frame 0 in the common case).
        prompt_frame = 0
        if unique_objects and not any(o["first_anchor"] == 0 for o in unique_objects):
            prompt_frame = min(o["first_anchor"] for o in unique_objects)
            logger.info("Frame 0 had no detections — seeding session 1 at anchor %d", prompt_frame)

        objs_late = [o for o in unique_objects if o["first_anchor"] != prompt_frame]
        unique_objects = [o for o in unique_objects if o["first_anchor"] == prompt_frame]
        if not unique_objects:
            logger.warning("No objects detected at any anchor — empty mask cache")
            self._bf16_ctx.__exit__(None, None, None)
            self._bf16_ctx = None
            return {}
        all_bboxes = [o["bbox_xywh_norm"] for o in unique_objects]
        obj_ids: List[int] = []
        batch_obj_ids: List[int] = []
        if hasattr(sam3, "add_bboxes_batch"):
            batch_obj_ids = list(sam3.add_bboxes_batch(all_bboxes, frame_idx=prompt_frame))
            obj_ids = list(batch_obj_ids)
            logger.info("Phase B-3: ONE batched add_prompt for %d bboxes → %d obj_ids in %.2fs",
                        len(all_bboxes), len(batch_obj_ids), time.time() - t_b3)
        else:
            for obj in unique_objects:
                sam3.create_run_with_initial_bboxes(
                    boxes_xywh=[obj["bbox_xywh_norm"]],
                    box_labels=[1],
                    frame_idx=obj["first_anchor"],
                    tag=f"anchor{obj['first_anchor']}",
                )
            logger.info("Phase B-3 (slow fallback): registered %d objects in %.2fs",
                        len(unique_objects), time.time() - t_b3)

        # Apply acceleration toggles (matches base path)
        try:
            if hasattr(sam3._predictor, "model"):
                if hasattr(sam3._predictor.model, "retain_feature_cache"):
                    sam3._predictor.model.retain_feature_cache = True
                # Gate the B-6 re-detection fix: when on, per-frame backbone features are
                # NOT evicted and _prepare_backbone_feats reuses them (skips ~1.5s of
                # redundant FA detection). See ROSEConfig.sam3.skip_b6_redetect.
                sam3._predictor.model._rose_skip_b6_redetect = bool(
                    getattr(sam3_cfg, "skip_b6_redetect", False))
                if hasattr(sam3._predictor.model, "tracker"):
                    tk = sam3._predictor.model.tracker
                    if sam3_cfg.num_maskmem != 7:
                        tk.num_maskmem = sam3_cfg.num_maskmem
                    if sam3_cfg.memory_temporal_stride != 1:
                        tk.memory_temporal_stride_for_eval = sam3_cfg.memory_temporal_stride
                    # Unlock SAM 3.1 native re-tracking (multiplex defaults are
                    # max_cond_frames_in_attn=4, keep_first_cond_frame=False
                    # which evicts the original prompt — kills long-horizon
                    # re-acquisition).  Also enable native memory selection.
                    tk.max_cond_frames_in_attn = sam3_cfg.max_cond_frames_in_attn
                    tk.keep_first_cond_frame = sam3_cfg.keep_first_cond_frame
                    tk.use_memory_selection = sam3_cfg.use_memory_selection
                    tk.mf_threshold = sam3_cfg.mf_threshold
        except Exception:
            pass

        def _t(name, t0):
            # Per-phase timing diagnostics (silent in production; enable with DEBUG logging).
            logger.debug("[TIMING] %s: %.3fs", name, time.time() - t0)

        # ---- Phase B-4: SAM3.1 grounding propagate (sets prior-stage flag) ----
        # B-3 (add_bboxes_batch) already grounded prompt_frame and set the model's
        # cached_frame_outputs[prompt_frame]. When skip_b4_frame0_reground is on, start
        # the VG propagation at prompt_frame+1 so we don't RE-detect that frame (~0.6s).
        # B-6's partial propagation re-walks [prompt_frame..N] afterwards, so the mask
        # cache is unaffected.  Only safe when B-6 will run (batch_obj_ids non-empty).
        t_b4 = time.time()
        if (getattr(self.config.sam3, "skip_b4_frame0_reground", False)
                and batch_obj_ids
                and hasattr(sam3, "refine_object_with_point")):
            sam3.propagate_new_objects(start_frame=prompt_frame + 1)
        else:
            sam3.propagate_new_objects()
        _t("Phase B-4 grounding propagate", t_b4)

        # ---- Phase B-5: refine each obj with bbox-center point -------
        if hasattr(sam3, "refine_object_with_point") and batch_obj_ids:
            t_b5 = time.time()
            n_refined = 0
            for obj, oid in zip(unique_objects, batch_obj_ids):
                bx, by, bw, bh = obj["bbox_xywh_norm"]
                cx, cy = bx + bw / 2.0, by + bh / 2.0
                sam3.refine_object_with_point(
                    obj_id=int(oid), frame_idx=prompt_frame,
                    point_xy=(cx, cy),
                )
                n_refined += 1
            _t(f"Phase B-5 refine {n_refined} objects", t_b5)

            # ---- Phase B-6: SAM3.1 memory-tracker propagate (full traj) ----
            t_b6 = time.time()
            sam3._propagation_cache.clear()
            sam3.propagate_new_objects()
            _t("Phase B-6 memory-tracker propagate", t_b6)

        # ---- Phase B-7: collect session-1 masks ----------------------
        mask_cache: Dict[int, List[SAM3SharedMask]] = {}
        for fidx in range(n):
            mask_cache[fidx] = list(sam3.propagate_all(fidx))

        # ---- Phase B-7.5: re-track via SAM3.1 ------------------------
        # Leverage SAM3.1's memory-based re-tracking: for each late-anchor
        # candidate, try to match it against an EXISTING session-1 track at
        # its first_anchor frame.  Matches get refined into session 1 with
        # refine_object_with_point (no new obj_id allocated) — SAM3.1's
        # memory bank then re-acquires the identity.  This eliminates the
        # main duplicate source (same physical object tracked separately by
        # session 1 and session 2) BEFORE session 2 runs.
        if objs_late and hasattr(sam3, "refine_object_with_point"):
            t_b75 = time.time()
            # Pre-compute color histograms for session-1 tracks and for each
            # late candidate (Tier-3 visual matching).  Built once, reused per
            # candidate — O(K + L) work, not O(K·L).
            _use_dino = bool(getattr(self.config.sam3, "late_match_use_dino", False))
            _dino_thr = float(getattr(self.config.sam3, "late_match_dino_thresh", 0.62))
            _feat_fn = self._mask_dino_embed if _use_dino else self._mask_color_hist
            # DINOv2 Tier-3 gates (cosine on L2-normalised embeddings); HSV keeps its own.
            _vis_kw = ({"visual_sim_thresh": _dino_thr,
                        "visual_sim_with_proximity": max(0.0, _dino_thr - 0.08)}
                       if _use_dino else {})
            session1_features = self._build_session1_features(mask_cache, frames, use_dino=_use_dino)
            already_matched: set = set()
            refines: List[Tuple[int, int, Tuple[float, float]]] = []
            objs_truly_late: List[Dict] = []
            _tw = int(getattr(self.config.sam3, "late_match_temporal_window", 2))
            # Precompute the mask-IoU inputs (m1 area/centroid once, intersection batched
            # on GPU) for the whole pass — bit-identical, replaces the per-candidate
            # full-res numpy IoU loop.  See _precompute_late_match.
            _pm_meta = _pm_inter = None
            if bool(getattr(self.config.sam3, "gpu_late_match", True)):
                _pm_meta, _pm_inter, _pm_gpu = self._precompute_late_match(
                    objs_late, mask_cache, n, _tw,
                    use_gpu=True,
                )
            for _ci, cand in enumerate(objs_late):
                fidx_c = int(cand["first_anchor"])
                cand_mask = cand.get("mask")
                cand_hist = (
                    _feat_fn(frames[fidx_c], cand_mask)
                    if cand_mask is not None else None
                )
                match_oid = self._match_late_to_session1(
                    cand,
                    mask_cache,
                    n_frames=n,
                    cand_hist=cand_hist,
                    session1_features=session1_features,
                    already_matched=already_matched,
                    temporal_window=_tw,
                    precomp_meta=_pm_meta,
                    precomp_inter=(_pm_inter.get(_ci) if _pm_inter is not None else None),
                    **_vis_kw,
                )
                if match_oid is not None:
                    bx, by, bw, bh = cand["bbox_xywh_norm"]
                    cx = max(0.0, min(1.0, float(bx + bw / 2.0)))
                    cy = max(0.0, min(1.0, float(by + bh / 2.0)))
                    refines.append((match_oid, fidx_c, (cx, cy)))
                    already_matched.add(match_oid)
                else:
                    objs_truly_late.append(cand)

            n_rematched = len(refines)
            applied = 0
            # OPT: apply refines to the model state but DEFER the propagate
            # — the final Phase B-8 propagate will pick up these refines
            # together with the new-obj prompts in ONE pass (saves a full-
            # video SAM2 propagation, ~5-8s).  Only edge case: if there are
            # NO truly-new candidates, we still need to propagate here so
            # the refines actually take effect.
            for obj_id, fidx_r, pt in refines:
                try:
                    sam3.refine_object_with_point(
                        obj_id=obj_id, frame_idx=fidx_r, point_xy=pt,
                    )
                    applied += 1
                except Exception as e:
                    logger.warning(
                        "Phase B-7.5 refine_object_with_point failed (obj=%d, frame=%d): %s",
                        obj_id, fidx_r, e,
                    )
            # Whether we need to propagate NOW depends on whether Phase B-8
            # will run a propagate afterwards.
            late_mode_for_fusion = getattr(self.config.sam3, "late_discovery_mode", "in_session")
            b8_will_propagate = (
                bool(objs_truly_late)
                and late_mode_for_fusion in ("in_session", "session_reset")
            )
            if applied > 0 and not b8_will_propagate:
                # No B-8 propagate coming → must propagate now to materialise refines.
                sam3._propagation_cache.clear()
                sam3.propagate_new_objects()
                for fidx_u in range(n):
                    mask_cache[fidx_u] = list(sam3.propagate_all(fidx_u))
                logger.info(
                    "Phase B-7.5: re-prompted %d late candidates (no B-8 follow-up) in %.2fs",
                    applied, time.time() - t_b75,
                )
            else:
                logger.info(
                    "Phase B-7.5: %d refines applied + deferred to B-8 propagate "
                    "(%d truly new remain) in %.2fs",
                    applied, len(objs_truly_late), time.time() - t_b75,
                )
            _t(f"Phase B-7.5 re-prompt ({n_rematched} matched, defer={b8_will_propagate})", t_b75)
            objs_late = objs_truly_late

        # ---- Phase B-7.6: EARLY DEDUP (same-frame mask containment) --------
        # Quality-neutral B-8 cut: a late candidate whose discovery-frame mask is
        # largely CONTAINED in (or contains) an EXISTING track's mask at that SAME
        # frame is spatially the same physical object — the existing track already
        # covers it, so propagating it as a NEW B-8 object across all frames is pure
        # redundant work (post-hoc reid_merge would collapse it anyway). Drop it.
        # Direct same-frame containment (IoM) — distinct from B-7.5's temporal-window
        # IoU+visual match (which routes to refine); catches dups B-7.5's
        # already_matched fall-through and alternating-frame cases miss BEFORE we pay
        # the propagation. Config-gated; 0.0 = off.
        _ed_iom = float(getattr(self.config.sam3, "early_dedup_iom", 0.0))
        if objs_late and _ed_iom > 0.0:
            t_ed = time.time()
            kept: List[Dict] = []
            n_dropped = 0
            for cand in objs_late:
                fidx_c = int(cand["first_anchor"])
                cm = cand.get("mask")
                existing = mask_cache.get(fidx_c, []) if cm is not None else []
                ca = float(cm.sum()) if cm is not None else 0.0
                is_dup = False
                if ca > 0.0:
                    for em in existing:
                        emask = getattr(em, "mask", None)
                        if emask is None:
                            continue
                        inter = float(np.logical_and(cm, emask).sum())
                        if inter <= 0.0:
                            continue
                        iom = inter / min(ca, float(emask.sum()) or 1.0)
                        if iom >= _ed_iom:
                            is_dup = True
                            break
                if is_dup:
                    n_dropped += 1
                else:
                    kept.append(cand)
            objs_late = kept
            logger.info(
                "Phase B-7.6 early-dedup (IoM>=%.2f): dropped %d same-frame-duplicate "
                "late candidates, %d remain, in %.3fs",
                _ed_iom, n_dropped, len(objs_late), time.time() - t_ed,
            )
            _t(f"Phase B-7.6 early-dedup (dropped {n_dropped})", t_ed)

        # Save the session-1 max run_id BEFORE we reset (used to offset
        # session-2 run_ids so they don't collide with session-1's).
        session1_max_run_id = sam3._next_run_id

        # ---- Phase B-8: late-anchor discovery -------------------------
        # Two modes (config.sam3.late_discovery_mode):
        #
        #   "in_session"  (recommended) — keep session 1 active.  Add each
        #     late candidate as a NEW obj_id via add_object_point at its
        #     first-anchor frame.  Re-propagate session 1 once.  Every
        #     object shares the same memory bank → continuous propagation
        #     across all frames.
        #
        #   "session_reset" (legacy) — fresh session 2.  Observed pathology:
        #     session-2 propagation is sparse (masks emit only at anchor
        #     frames, ----X---X---X pattern).  Kept for fallback / ablation.
        late_mode = getattr(self.config.sam3, "late_discovery_mode", "in_session")
        if objs_late and late_mode == "in_session":
            t_b8 = time.time()
            try:
                n_added = 0
                # Track obj_ids we just added so we can later assign them a
                # distinct run_id (so downstream cross-run fusion still sees
                # them as a separate group from the frame-0 anchors).
                added_obj_ids: List[int] = []
                for obj in objs_late:
                    fidx = int(obj["first_anchor"])
                    bx, by, bw, bh = obj["bbox_xywh_norm"]
                    cx = max(0.0, min(1.0, float(bx + bw / 2.0)))
                    cy = max(0.0, min(1.0, float(by + bh / 2.0)))
                    try:
                        new_oid = sam3.add_object_point(fidx, (cx, cy))
                        added_obj_ids.append(int(new_oid))
                        n_added += 1
                    except Exception as e:
                        logger.warning(
                            "Phase B-8 (in_session) add_object_point failed at frame %d: %s",
                            fidx, e,
                        )
                if n_added > 0:
                    # Incremental late-prop: only re-propagate from the earliest
                    # late-discovery frame onward (frames before it have no late
                    # object), preserving the existing cache for earlier frames.
                    _inc = getattr(self.config.sam3, "incremental_late_prop", False)
                    _start = (min(int(o["first_anchor"]) for o in objs_late)
                              if (_inc and objs_late) else 0)
                    if _start > 0:
                        for _fk in list(sam3._propagation_cache.keys()):
                            if _fk >= _start:
                                del sam3._propagation_cache[_fk]
                    else:
                        sam3._propagation_cache.clear()
                    added_set = set(added_obj_ids)

                    # ── STREAMING CALLBACK ────────────────────────────────
                    # Use the wrapper's _per_frame_callback to push each
                    # frame's masks into a queue AS THEY ARE PRODUCED by
                    # B-8 propagation.  A background worker thread does
                    # fragment-check + per-frame IoU dedup + crop extract
                    # IN PARALLEL with the GPU's next-frame propagation.
                    #
                    # Hides ~3-5s of CPU work behind the 27s GPU B-8 phase.
                    self._stream_pre = {}   # fidx -> {frame_masks, best_crops_local}
                    self._stream_lock = threading.Lock()
                    stream_q: "queue.Queue" = queue.Queue()
                    stream_done = threading.Event()

                    def _stream_worker():
                        while True:
                            item = stream_q.get()
                            if item is None:
                                stream_done.set()
                                return
                            fidx_l, masks_l = item
                            # Re-stamp late-added obj_ids' run_id (cross-run group).
                            for m in masks_l:
                                if int(m.obj_id_local) in added_set:
                                    m.run_id = session1_max_run_id
                            # Pre-filter: fragment check + per-frame dedup.
                            # These touch only this frame's masks → safe in worker.
                            fm = [m for m in masks_l if not _mask_is_fragmented(m.mask)]
                            fm = _dedup_masks_by_iou(fm, iou_threshold=0.95)
                            with self._stream_lock:
                                self._stream_pre[fidx_l] = fm

                    stream_thread = threading.Thread(target=_stream_worker, daemon=True)
                    stream_thread.start()

                    def _cb(fid, frame_masks):
                        stream_q.put((int(fid), list(frame_masks)))

                    sam3._per_frame_callback = _cb
                    try:
                        sam3.propagate_new_objects(start_frame=_start)
                    finally:
                        sam3._per_frame_callback = None
                    # Signal worker, then wait for it to drain.
                    stream_q.put(None)
                    stream_done.wait(timeout=10.0)

                    n_late_masks = 0
                    new_cache: Dict[int, List[SAM3SharedMask]] = {}
                    for fidx in range(n):
                        fm = self._stream_pre.get(fidx)
                        if fm is None:
                            # Fallback if stream missed a frame (shouldn't happen).
                            fm = list(sam3.propagate_all(fidx))
                            for m in fm:
                                if int(m.obj_id_local) in added_set:
                                    m.run_id = session1_max_run_id
                        new_cache[fidx] = fm
                        n_late_masks += sum(
                            1 for m in fm if int(m.obj_id_local) in added_set
                        )
                    mask_cache = new_cache
                    self._stream_pre = None
                    logger.info(
                        "Phase B-8 (in_session): added %d late objects to session 1, "
                        "%d masks emitted across %d frames",
                        n_added, n_late_masks, n,
                    )
                _t(f"Phase B-8 in_session ({n_added} objs)", t_b8)
            except Exception as e:
                logger.warning(
                    "Phase B-8 (in_session) failed (non-fatal, falling back): %s",
                    e, exc_info=True,
                )

        elif objs_late and late_mode == "session_reset":
            t_b8 = time.time()
            try:
                # Reset multiplex session (closes old, opens new, reloads frames).
                sam3.set_video_frames(pil_frames)
                n_added = 0
                for obj in objs_late:
                    fidx = int(obj["first_anchor"])
                    bx, by, bw, bh = obj["bbox_xywh_norm"]
                    cx = max(0.0, min(1.0, float(bx + bw / 2.0)))
                    cy = max(0.0, min(1.0, float(by + bh / 2.0)))
                    try:
                        sam3.add_object_point(fidx, (cx, cy))
                        n_added += 1
                    except Exception as e:
                        logger.warning("Phase B-8 add_object_point failed at frame %d: %s", fidx, e)
                if n_added > 0:
                    sam3.propagate_new_objects()
                    n_late_masks = 0
                    for fidx in range(n):
                        s2_masks = list(sam3.propagate_all(fidx))
                        for m in s2_masks:
                            m.run_id += session1_max_run_id
                        mask_cache[fidx].extend(s2_masks)
                        n_late_masks += len(s2_masks)
                    logger.info(
                        "Phase B-8 (session_reset): added %d objects, %d masks merged",
                        n_added, n_late_masks,
                    )
                _t(f"Phase B-8 session_reset ({n_added} objs)", t_b8)
            except Exception as e:
                logger.warning("Phase B-8 (session_reset) failed (non-fatal): %s",
                               e, exc_info=True)

        self._bf16_ctx.__exit__(None, None, None)
        self._bf16_ctx = None

        # Trajectory-aware mask-IoU dedup BEFORE the 4DSG pipeline ingests the
        # masks.  This catches the same physical object tracked from multiple
        # anchors (boat at frame 0 and frame 12 → both tracked by SAM3 as
        # separate obj_ids; their masks have ~0.6 IoU at every shared frame
        # → merge into one track).  Operates on full per-frame masks so it's
        # robust to relative-depth scale issues that broke 3D-centroid dedup.
        mask_cache = self._dedup_mask_cache_by_trajectory(
            mask_cache,
            iou_thresh=float(getattr(self.config.fusion, "mask_traj_iou_thresh", 0.4)),
        )
        return mask_cache

    @staticmethod
    def _dedup_mask_cache_by_trajectory(
        mask_cache: Dict[int, List["SAM3SharedMask"]],
        iou_thresh: float = 0.4,
    ) -> Dict[int, List["SAM3SharedMask"]]:
        """Per-trajectory mask-IoU dedup — GPU-accelerated.

        H200 win: instead of looping pairwise (T choose 2) × F mask-IoU on
        the CPU (~11s on a typical video), we batch all per-frame masks as
        a single (T, F, H*W) bool tensor and compute per-frame T×T IoU + IoM
        matrices via one matmul each on GPU.  Aggregate across frames in
        bool→float arithmetic.  ~50-100× faster than the prior CPU path.

        Logic preserved exactly:
          • mean_iou ≥ iou_thresh  → merge
          • mean_iom ≥ 0.6          → merge (containment case)
          • require ≥ MIN_SHARED shared frames
          • longer track wins as union root
        """
        import torch

        # Collect per-track per-frame masks
        track_masks: Dict[Tuple, Dict[int, np.ndarray]] = {}
        for fidx, masks in mask_cache.items():
            for m in masks:
                key = (m.run_id, m.obj_id_local)
                if m.mask is None or m.mask.size == 0:
                    continue
                track_masks.setdefault(key, {})[int(fidx)] = m.mask

        keys = list(track_masks.keys())
        if len(keys) < 2:
            return mask_cache

        parent: Dict[Tuple, Tuple] = {k: k for k in keys}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        MIN_SHARED = 2
        IOM_THRESH = 0.6

        # Build (T, F) presence matrix + flat mask tensors per frame
        T = len(keys)
        all_frames = sorted({f for k in keys for f in track_masks[k]})
        if not all_frames:
            return mask_cache
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Accumulators across frames
        inter_sum = torch.zeros(T, T, device=device, dtype=torch.float32)
        union_sum = torch.zeros(T, T, device=device, dtype=torch.float32)
        min_area_sum = torch.zeros(T, T, device=device, dtype=torch.float32)
        shared_count = torch.zeros(T, T, device=device, dtype=torch.int32)

        # Disable any leaked autocast (bf16 from Phase 2) for this dedup —
        # the matmul accumulator needs float32 to avoid bf16 mask-IoU rounding.
        import torch
        with torch.autocast(device_type="cuda", enabled=False):
            for fidx in all_frames:
                # Track which keys have a mask at this frame.
                present_idx = [i for i, k in enumerate(keys) if fidx in track_masks[k]]
                if len(present_idx) < 2:
                    continue
                # Stack the K present masks as (K, H*W) float32 on GPU.
                present_masks = [track_masks[keys[i]][fidx] for i in present_idx]
                H, W = present_masks[0].shape
                stack_np = np.stack(present_masks).astype(np.float32, copy=False)
                stack = torch.from_numpy(stack_np).to(device).view(len(present_idx), H * W)
                # Per-mask area.
                area = stack.sum(dim=1)                       # (K,)
                # Pairwise intersection via single GEMM.  (K,HW)@(HW,K) = (K,K).
                inter = stack @ stack.t()                     # (K, K), float
                # union = area_i + area_j - inter
                union = area[:, None] + area[None, :] - inter
                # min_area = pairwise minimum
                min_area = torch.minimum(area[:, None], area[None, :])
                # Scatter to global T×T accumulator
                idx = torch.tensor(present_idx, device=device, dtype=torch.long)
                grid_i = idx[:, None].expand(-1, len(present_idx))
                grid_j = idx[None, :].expand(len(present_idx), -1)
                inter_sum.index_put_((grid_i, grid_j), inter, accumulate=True)
                union_sum.index_put_((grid_i, grid_j), union, accumulate=True)
                min_area_sum.index_put_((grid_i, grid_j), min_area, accumulate=True)
                ones = torch.ones_like(inter, dtype=torch.int32)
                shared_count.index_put_((grid_i, grid_j), ones, accumulate=True)

        # Compute per-pair mean IoU / mean IoM (divisor = shared_count).
        sc = shared_count.float().clamp(min=1.0)
        mean_iou = (inter_sum / union_sum.clamp(min=1.0)) * (union_sum > 0).float()
        mean_iom = (inter_sum / min_area_sum.clamp(min=1.0)) * (min_area_sum > 0).float()
        # Average over shared frames (we accumulated per-frame ratios summed):
        # Actually we accumulated raw inter/union/min_area sums, so divide once.
        mean_iou = (inter_sum / union_sum.clamp(min=1.0))
        mean_iom = (inter_sum / min_area_sum.clamp(min=1.0))

        # Apply gates and read pairs back to CPU for Union-Find.
        valid_mask = (
            (shared_count >= MIN_SHARED)
            & ((mean_iou >= iou_thresh) | (mean_iom >= IOM_THRESH))
        )
        # Upper triangle (i < j) only.
        ut = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        merge_mask = (valid_mask & ut).cpu().numpy()
        pair_idx = np.argwhere(merge_mask)  # rows of (i, j)

        for i, j in pair_idx:
            ki, kj = keys[int(i)], keys[int(j)]
            ri, rj = _find(ki), _find(kj)
            if ri == rj:
                continue
            ni = len(track_masks[ri])
            nj = len(track_masks[rj])
            if nj > ni:
                ri, rj = rj, ri
            parent[rj] = ri

        # Relabel masks per-frame to the surviving track key.
        survivor_for_key: Dict[Tuple, Tuple] = {k: _find(k) for k in keys}

        from rose.vision.perception.sam3_multiplex_wrapper import SAM3SharedMask
        new_cache: Dict[int, List[SAM3SharedMask]] = {}
        for fidx, masks in mask_cache.items():
            seen_survivors: Dict[Tuple, "SAM3SharedMask"] = {}
            for m in masks:
                key = (m.run_id, m.obj_id_local)
                survivor = survivor_for_key.get(key, key)
                prev = seen_survivors.get(survivor)
                # If the surviving slot already has a mask at this frame, keep
                # whichever has higher score (more confident detection wins).
                if prev is None or m.score > prev.score:
                    seen_survivors[survivor] = SAM3SharedMask(
                        run_id=survivor[0], obj_id_local=survivor[1],
                        mask=m.mask, score=m.score,
                    )
            new_cache[fidx] = list(seen_survivors.values())
        return new_cache

    @staticmethod
    def _mask_color_hist(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """HSV color histogram (8H × 8S × 4V = 256 bins) of pixels inside mask.

        Returns L1-normalised float32 histogram, or None when mask is empty.
        Cheap (~0.2 ms on a 480×910 mask); robust to scale/translation since
        we ignore spatial layout.
        """
        if mask is None or not mask.any():
            return None
        import cv2 as _cv2
        hsv = _cv2.cvtColor(image, _cv2.COLOR_RGB2HSV)
        masked = hsv[mask]
        if masked.size == 0:
            return None
        # OpenCV calcHist needs an image, so build per-channel histograms.
        h_hist = np.histogram(masked[:, 0], bins=8, range=(0, 180))[0].astype(np.float32)
        s_hist = np.histogram(masked[:, 1], bins=8, range=(0, 256))[0].astype(np.float32)
        v_hist = np.histogram(masked[:, 2], bins=4, range=(0, 256))[0].astype(np.float32)
        hist = np.concatenate([h_hist, s_hist, v_hist])
        s = float(hist.sum())
        if s <= 0:
            return None
        return hist / s

    @staticmethod
    def _mask_dino_embed(image: np.ndarray, mask: np.ndarray,
                         model_name: str = "vit_base_patch14_dinov2.lvd142m"):
        """L2-normalised DINOv2 embedding of the masked object's bbox crop.

        Same descriptor the post-hoc reid_merge uses (0 false merges at cos>=0.72).
        Returns a float32 vector or None on empty mask / failure (caller falls back).
        ``image`` is RGB (frames are RGB), matching the embedder's expected input.
        """
        if mask is None or not mask.any():
            return None
        try:
            import torch, cv2 as _cv2
            from rose.engine.pipeline.rose_pipeline import _get_reid_embedder, _REID_STATE
            ys, xs = np.where(mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                return None
            model, mean, std = _get_reid_embedder(model_name)
            size = int(_REID_STATE.get("size", 518))
            crop = _cv2.resize(crop, (size, size))   # image already RGB
            dev = mean.device
            t = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().to(dev) / 255.0
            t = ((t - mean) / std).unsqueeze(0)
            with torch.no_grad():
                f = torch.nn.functional.normalize(model(t), dim=1)[0].detach().cpu().numpy()
            return f.astype(np.float32)
        except Exception as e:
            logger.debug("dino embed failed: %s", e)
            return None

    @staticmethod
    def _hist_cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised histograms."""
        if a is None or b is None:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _build_session1_features(
        mask_cache: Dict[int, List["SAM3SharedMask"]],
        frames: List[np.ndarray],
        use_dino: bool = False,
    ) -> Dict[int, np.ndarray]:
        """Per session-1 obj_id, compute an appearance feature on the frame with
        the highest mask score (most reliable mask).  Returns obj_id → feature.
        ``use_dino`` swaps the HSV histogram for a DINOv2 embedding (stronger)."""
        _feat = (WarmModelPool._mask_dino_embed if use_dino
                 else WarmModelPool._mask_color_hist)
        # Pick best (score, area) frame per obj_id.
        best_per_oid: Dict[int, Tuple[float, int, int]] = {}  # oid -> (score, area, fidx)
        for fidx, masks in mask_cache.items():
            for m in masks:
                if m.mask is None or not m.mask.any():
                    continue
                oid = int(m.obj_id_local)
                area = int(m.mask.sum())
                # Some wrappers don't fill score per-frame — fall back to area.
                score = float(getattr(m, "score", 0.0) or 0.0)
                key = (score, area)
                prev = best_per_oid.get(oid)
                if prev is None or key > (prev[0], prev[1]):
                    best_per_oid[oid] = (score, area, fidx)
        features: Dict[int, np.ndarray] = {}
        for oid, (_s, _a, fidx) in best_per_oid.items():
            # Re-locate the mask object at that frame.
            for m in mask_cache.get(fidx, []):
                if int(m.obj_id_local) == oid and m.mask is not None and m.mask.any():
                    feat = _feat(frames[fidx], m.mask)
                    if feat is not None:
                        features[oid] = feat
                    break
        return features

    @staticmethod
    def _precompute_late_match(
        objs_late: List[Dict],
        mask_cache: Dict[int, List["SAM3SharedMask"]],
        n_frames: int,
        temporal_window: int,
        use_gpu: bool,
    ) -> Tuple[Dict[int, list], Dict[int, Optional[dict]], bool]:
        """Precompute the per-(candidate, frame, session-1-mask) intersection and
        the per-session-1-mask area/centroid that `_match_late_to_session1` needs,
        ONCE for the whole B-7.5 pass instead of O(candidates) times.

        Returns ``(meta, inter, used_gpu)`` where:
          * ``meta[frame]``  = list of ``(oid, area, cy, cx)`` for the NON-EMPTY masks
            in ``mask_cache[frame]``, in iteration order (centroid normalised by H,W).
          * ``inter[cand_idx][frame]`` = list of intersection pixel counts aligned 1:1
            with ``meta[frame]`` (or ``inter[cand_idx] is None`` if the candidate mask
            is empty — matching the original early-return).

        The intersection is exact: a float32 matmul of 0/1 masks sums to the same
        integer as ``np.logical_and(...).sum()`` (HW ≪ 2**24, float32-exact).  Union
        is then recovered downstream via |A∪B| = |A|+|B|−|A∩B| (set identity), so the
        IoU — and every threshold decision — is bit-identical to the CPU path.
        """
        # Frames actually consulted = union of every candidate's ±window.
        needed = set()
        for cand in objs_late:
            a = int(cand["first_anchor"])
            lo = max(0, a - temporal_window)
            hi = min(n_frames - 1, a + temporal_window)
            needed.update(range(lo, hi + 1))

        meta: Dict[int, list] = {}
        m1_masks: List[np.ndarray] = []      # flattened-later refs, aligned to global index
        m1_index: Dict[int, List[int]] = {}  # frame -> indices into m1_masks (aligned to meta[frame])
        for f in sorted(needed):
            lst, idxs = [], []
            for m1 in mask_cache.get(f, []):
                mm = m1.mask
                if mm is None or not mm.any():
                    continue
                ys, xs = np.where(mm)
                H, W = mm.shape
                lst.append((int(m1.obj_id_local), int(mm.sum()),
                            float(ys.mean()) / H, float(xs.mean()) / W))
                idxs.append(len(m1_masks))
                m1_masks.append(mm)
            meta[f] = lst
            m1_index[f] = idxs

        cand_masks = [c.get("mask") for c in objs_late]
        valid_ci = [ci for ci, cm in enumerate(cand_masks)
                    if cm is not None and cm.any()]
        inter: Dict[int, Optional[dict]] = {ci: None for ci in range(len(cand_masks))}

        def _distribute(ci: int, flat_counts) -> None:
            inter[ci] = {f: [int(flat_counts[j]) for j in m1_index[f]] for f in meta}

        used_gpu = False
        if use_gpu and m1_masks and valid_ci:
            try:
                import torch as _t
                m1_mat = _t.stack([
                    _t.from_numpy(np.ascontiguousarray(mm.reshape(-1))).float()
                    for mm in m1_masks
                ]).cuda()                                            # [S, HW]
                cand_mat = _t.stack([
                    _t.from_numpy(np.ascontiguousarray(cand_masks[ci].reshape(-1))).float()
                    for ci in valid_ci
                ]).cuda()                                            # [Cv, HW]
                inter_mat = (cand_mat @ m1_mat.t()).round().long().cpu()  # [Cv, S], exact ints
                for row, ci in enumerate(valid_ci):
                    _distribute(ci, inter_mat[row].tolist())
                used_gpu = True
                return meta, inter, used_gpu
            except Exception as e:
                logger.warning("B-7.5 GPU match precompute failed, CPU fallback: %s", e)

        # CPU fallback — same exact integer intersections.
        for ci in valid_ci:
            cm = cand_masks[ci]
            _distribute(ci, [int(np.logical_and(cm, mm).sum()) for mm in m1_masks])
        return meta, inter, used_gpu

    @staticmethod
    def _match_late_to_session1(
        obj_late: Dict,
        mask_cache: Dict[int, List["SAM3SharedMask"]],
        n_frames: int,
        cand_hist: Optional[np.ndarray] = None,
        session1_features: Optional[Dict[int, np.ndarray]] = None,
        iou_thresh: float = 0.20,
        centroid_dist_thresh: float = 0.08,
        size_ratio_thresh: float = 0.40,
        temporal_window: int = 2,
        # Tier-3 (visual) gates
        visual_sim_thresh: float = 0.85,        # cosine sim required on its own (strong evidence)
        visual_sim_with_proximity: float = 0.75, # required when paired with a relaxed spatial gate
        relaxed_dist_thresh: float = 0.20,       # used only when visual sim above visual_sim_with_proximity
        already_matched: Optional[set] = None,
        precomp_meta: Optional[dict] = None,     # frame -> [(oid, area, cy, cx)] (from _precompute_late_match)
        precomp_inter: Optional[dict] = None,    # frame -> [intersection pixel counts] aligned to precomp_meta
    ) -> Optional[int]:
        """Return the session-1 obj_id that a late-anchor candidate likely re-tracks.

        Searches a temporal window around the candidate's ``first_anchor``
        frame (default ±``temporal_window`` frames) — session 1's mask may
        have drifted at the exact anchor but still be intact nearby.

        Per session-1 obj_id we keep the BEST (max IoU, min centroid dist)
        across the window.  An obj_id is a match if either:
          Tier 1: max mask IoU across the window >= ``iou_thresh``
          Tier 2: min centroid distance < ``centroid_dist_thresh`` AND
                  area ratio at that frame >= ``size_ratio_thresh``

        ``already_matched`` is an optional set of session-1 obj_ids that have
        already absorbed a refine in this pass; passing it prevents two
        candidates from being routed to the same obj_id (the second one
        falls through to Phase B-8 as a genuinely new object).
        """
        cand_mask = obj_late.get("mask")
        if cand_mask is None or not cand_mask.any():
            return None
        cand_area = float(obj_late.get("mask_area", cand_mask.sum()))
        H, W = cand_mask.shape
        cand_ys, cand_xs = np.where(cand_mask)
        cand_cy = float(cand_ys.mean()) / H
        cand_cx = float(cand_xs.mean()) / W
        anchor = int(obj_late["first_anchor"])

        # Aggregate best score per obj_id across the temporal window.
        per_oid_iou: Dict[int, float] = {}
        per_oid_dist: Dict[int, Tuple[float, float]] = {}  # dist, size_ratio

        f_lo = max(0, anchor - temporal_window)
        f_hi = min(n_frames - 1, anchor + temporal_window)
        if precomp_meta is not None:
            # Precomputed path: m1 area/centroid computed ONCE; intersection from the
            # batched (GPU or CPU) matmul.  union via |A∪B| = |A|+|B|−|A∩B| (set identity)
            # → identical integer IoU as the per-mask numpy path below.
            cand_pixels = int(cand_mask.sum())
            pinter = precomp_inter or {}
            for f in range(f_lo, f_hi + 1):
                metas = precomp_meta.get(f, [])
                inters = pinter.get(f, [])
                for (oid, m1_area, m1_cy, m1_cx), inter in zip(metas, inters):
                    if already_matched is not None and oid in already_matched:
                        continue
                    union = cand_pixels + m1_area - inter
                    iou = (inter / union) if union > 0 else 0.0
                    if iou > per_oid_iou.get(oid, 0.0):
                        per_oid_iou[oid] = iou
                    dist = ((cand_cy - m1_cy) ** 2 + (cand_cx - m1_cx) ** 2) ** 0.5
                    size_ratio = min(cand_area, m1_area) / max(cand_area, m1_area, 1.0)
                    prev = per_oid_dist.get(oid)
                    if prev is None or dist < prev[0]:
                        per_oid_dist[oid] = (dist, size_ratio)
        else:
            for f in range(f_lo, f_hi + 1):
                for m1 in mask_cache.get(f, []):
                    m1_mask = m1.mask
                    if m1_mask is None or not m1_mask.any():
                        continue
                    oid = int(m1.obj_id_local)
                    if already_matched is not None and oid in already_matched:
                        continue

                    m1_area = int(m1_mask.sum())
                    inter = int(np.logical_and(cand_mask, m1_mask).sum())
                    union = int(np.logical_or(cand_mask, m1_mask).sum())
                    iou = (inter / union) if union > 0 else 0.0
                    if iou > per_oid_iou.get(oid, 0.0):
                        per_oid_iou[oid] = iou

                    m1_ys, m1_xs = np.where(m1_mask)
                    m1_cy = float(m1_ys.mean()) / H
                    m1_cx = float(m1_xs.mean()) / W
                    dist = ((cand_cy - m1_cy) ** 2 + (cand_cx - m1_cx) ** 2) ** 0.5
                    size_ratio = min(cand_area, m1_area) / max(cand_area, m1_area, 1.0)
                    prev = per_oid_dist.get(oid)
                    if prev is None or dist < prev[0]:
                        per_oid_dist[oid] = (dist, size_ratio)

        # Tier 1: best IoU above threshold wins.
        best_iou_id, best_iou = None, 0.0
        for oid, iou in per_oid_iou.items():
            if iou > best_iou:
                best_iou = iou
                best_iou_id = oid
        if best_iou >= iou_thresh and best_iou_id is not None:
            return best_iou_id

        # Tier 2: tightest centroid match that also passes size-ratio gate.
        best_dist_id, best_dist = None, float("inf")
        for oid, (dist, sr) in per_oid_dist.items():
            if sr < size_ratio_thresh:
                continue
            if dist < centroid_dist_thresh and dist < best_dist:
                best_dist = dist
                best_dist_id = oid
        if best_dist_id is not None:
            return best_dist_id

        # Tier 3: visual-feature (HSV color histogram) match.  Required when
        # SAM 3.1 has completely lost the track and its session-1 mask drifted
        # far from the candidate's position.  Two routes:
        #   • Strong-only: cosine sim >= visual_sim_thresh  →  match
        #   • Combined: cosine sim >= visual_sim_with_proximity AND mask-
        #     centroid distance < relaxed_dist_thresh        →  match
        # The visual gate keeps us from refining onto an unrelated object.
        if cand_hist is None or session1_features is None or not session1_features:
            return None
        best_vis_id, best_vis = None, 0.0
        # Compute best per-oid centroid distance again (cheap, already have
        # per_oid_dist from above) to use the combined gate.
        for oid, hist in session1_features.items():
            if already_matched is not None and oid in already_matched:
                continue
            sim = WarmModelPool._hist_cosine(cand_hist, hist)
            if sim > best_vis:
                best_vis = sim
                best_vis_id = oid
        if best_vis_id is None:
            return None
        if best_vis >= visual_sim_thresh:
            return best_vis_id
        if best_vis >= visual_sim_with_proximity:
            dist_pair = per_oid_dist.get(best_vis_id)
            if dist_pair is not None and dist_pair[0] < relaxed_dist_thresh:
                return best_vis_id
        return None

    @staticmethod
    def _dedup_anchor_detections(
        anchor_idxs: List[int],
        anchor_dets: Dict[int, list],
        iou_thresh: float = 0.4,
        containment_iom: float = 0.7,
        use_gpu: bool = True,
    ) -> List[Dict]:
        """Greedy cross-anchor dedup.

        For each anchor in chronological order, keep its detections that don't
        match (bbox IoU + center proximity + mask containment) any already-kept
        object's most-recent appearance.  Crude but effective for typical videos
        with small inter-anchor camera motion.

        The mask-containment (IoM) check is the ROOT-CAUSE fix for fragmentation:
        FastSAM proposes one object as several nested part-masks; bbox-IoU misses
        the nesting so each part used to be seeded as a separate SAM3 object.
        """
        # Flatten detections in chronological (anchor) order, keep global index.
        # ROOT-CAUSE of the B-2 cost: the per-pair mask-containment IoM below is a
        # CPU np.logical_and over HxW, O(M·K) → blows up when FastSAM proposes many
        # detections (30→81 dets = 0.3s→1.7s). Pre-compute the pairwise mask-IoM
        # matrix ONCE on GPU (intersection via mf@mf.T, like _fuse_candidates), then
        # the greedy loop just looks it up — bit-identical dedup, no HxW CPU work.
        _flat = []  # (fidx, det)
        for fidx in anchor_idxs:
            for det in anchor_dets.get(fidx, []):
                if float(det.mask.sum()) >= 100:
                    _flat.append((fidx, det))
        # Pairwise INTERSECTION matrix only (exact integers via float32 matmul of
        # 0/1 masks, HW ≪ 2**24).  The IoM ratio + 0.7 threshold are computed in
        # float64 Python in the loop below — identical to the CPU path, so there is
        # NO float32-division boundary flip (unlike the old `_iom = _inter/_mina`).
        _inter_mat = None
        if use_gpu and containment_iom > 0 and len(_flat) > 1:
            try:
                import torch as _t
                _m = _t.stack([_t.from_numpy(np.ascontiguousarray(d.mask).astype(np.bool_))
                               for _, d in _flat]).cuda()
                _mf = _m.reshape(len(_flat), -1).float()
                _inter_mat = (_mf @ _mf.t()).round().long().cpu().numpy()  # exact int [M, M]
                del _m, _mf
            except Exception as _e:
                logger.warning("B-2 GPU IoM precompute failed (CPU fallback): %s", _e)
                _inter_mat = None

        kept: List[Dict] = []  # each entry: {bbox_xywh_norm, mask, first_anchor, mask_area, last_seen, idx}

        def bbox_iou(a, b):
            ax1, ay1 = a[0], a[1]
            ax2, ay2 = a[0] + a[2], a[1] + a[3]
            bx1, by1 = b[0], b[1]
            bx2, by2 = b[0] + b[2], b[1] + b[3]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / ua if ua > 0 else 0.0

        def center_dist(a, b):
            ac = (a[0] + a[2] / 2, a[1] + a[3] / 2)
            bc = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5

        for gi, (fidx, det) in enumerate(_flat):
            bbox = list(det.bbox_xywh_norm)
            area = float(det.mask.sum())  # area >= 100 already filtered in _flat
            # match against existing kept objects.  We use the LATEST bbox
            # of each kept object (not the first-seen one) to handle camera
            # motion: across 4-frame anchors, an object's bbox may drift
            # but should still be close to its position at the previous anchor.
            matched = False
            new_mask = (det.mask.astype(bool)
                        if (containment_iom > 0 and _inter_mat is None) else None)
            for k in kept:
                if bbox_iou(bbox, k["last_bbox"]) >= iou_thresh:
                    k["last_seen"] = fidx
                    k["last_bbox"] = bbox  # update for next-anchor comparison
                    matched = True
                    break
                # Backup: small center distance also counts as same object
                # (handles cases where bbox aspect changed but center similar)
                if center_dist(bbox, k["last_bbox"]) < 0.05:  # 5% of normalized image
                    # additional check: similar mask area (within 2x)
                    if 0.5 <= (area / max(k["mask_area"], 1)) <= 2.0:
                        k["last_seen"] = fidx
                        k["last_bbox"] = bbox
                        matched = True
                        break
                # ROOT-CAUSE FIX: mask containment — a nested part-proposal of
                # an already-kept object (one mask ≥containment_iom inside the
                # other) is the SAME physical object.  Keep the LARGER mask as
                # the representative so SAM3 is seeded on the whole object.
                if containment_iom > 0:
                    if _inter_mat is not None:
                        # GPU exact-integer intersection; ratio in float64 → identical
                        # to the CPU branch (no float32 boundary flip).
                        inter = float(_inter_mat[gi, k["idx"]])
                    else:
                        km = k["mask"].astype(bool)
                        inter = float(np.logical_and(new_mask, km).sum())
                    iom_val = (inter / max(min(area, k["mask_area"]), 1.0)
                               if inter > 0 else 0.0)
                    if iom_val >= containment_iom:
                        if area > k["mask_area"]:
                            k["bbox_xywh_norm"] = bbox; k["last_bbox"] = bbox
                            k["mask"] = det.mask; k["mask_area"] = area
                            k["idx"] = gi  # representative det changed → update lookup index
                        k["last_seen"] = fidx
                        matched = True
                        break
            if not matched:
                kept.append({
                    "bbox_xywh_norm": bbox,
                    "last_bbox": bbox,
                    "mask": det.mask,
                    "first_anchor": fidx,
                    "mask_area": area,
                    "last_seen": fidx,
                    "idx": gi,
                })
        return kept

    # ------------------------------------------------------------------
    # 4DSG construction (incremental + finalize for streaming overlap)
    # ------------------------------------------------------------------

    def _process_frame_for_4dsg(
        self,
        fidx: int,
        frame_masks: list,
        frames: List[np.ndarray],
        da3_results: list,
        source_indices: List[int],
        timestamps_s: List[float],
        pipeline: "ROSEPipeline",
        best_crops: Dict[tuple, Tuple[np.ndarray, float, int, float]],
        crop_pad: float,
        crop_sz: int,
    ) -> None:
        """Streaming entry point: process ONE frame's masks into 4DSG state."""
        if fidx >= len(frames) or fidx >= len(da3_results):
            return
        image = frames[fidx]

        # Dedup by (run_id, obj_id_local) — keep highest score
        best_by_key: Dict[tuple, SAM3SharedMask] = {}
        for m in frame_masks:
            key = (m.run_id, m.obj_id_local)
            prev = best_by_key.get(key)
            if prev is None or m.score > prev.score:
                best_by_key[key] = m
        frame_masks = list(best_by_key.values())
        # P1-1: drop fragmented SAM3 masks (two distinct objects bundled together).
        frame_masks = [m for m in frame_masks if not _mask_is_fragmented(m.mask)]
        # D2 cross-run mask IoU dedup
        frame_masks = _dedup_masks_by_iou(frame_masks, iou_threshold=0.95)

        for m in frame_masks:
            key = (m.run_id, m.obj_id_local)
            prev = best_crops.get(key)
            if prev is not None and m.score < prev[1]:
                continue
            crop = _crop_object_from_mask(image, m.mask, padding=crop_pad, size=crop_sz)
            brightness = float(crop.mean())
            if prev is None or (m.score, brightness) > (prev[1], prev[3]):
                best_crops[key] = (crop, m.score, source_indices[fidx], brightness)

        detections = [
            FastLocalDetection(run_id=m.run_id, local_obj_id=m.obj_id_local,
                                mask=m.mask, score=m.score)
            for m in frame_masks
        ]
        fi = FastFrameInput(
            frame_idx=source_indices[fidx],
            depth_t=da3_results[fidx].depth,
            K_t=da3_results[fidx].K,
            T_wc_t=da3_results[fidx].T_wc,
            detections=detections,
            depth_conf_t=da3_results[fidx].depth_conf,
            depth_is_metric=da3_results[fidx].is_metric,
            timestamp_s=timestamps_s[fidx],
        )
        pipeline.process_frame(fi)

    def _finalize_4dsg(
        self,
        pipeline: "ROSEPipeline",
        best_crops: Dict[tuple, Tuple[np.ndarray, float, int, float]],
        frame_dir: Path,
    ) -> Tuple[Dict, str]:
        """Finalize: write crops, run merge_duplicate_tracks, serialize."""
        t_save = time.time()
        crops_dir = frame_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        object_crops: Dict[int, Dict[str, object]] = {}
        for key, gid in pipeline._local_to_global.items():
            if key in best_crops and gid not in object_crops:
                crop_rgb, _score, src_idx, _bright = best_crops[key]
                if _crop_is_uniform(crop_rgb):
                    continue
                crop_path = crops_dir / f"obj_{gid:04d}.jpg"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                object_crops[gid] = {"path": str(crop_path), "source_frame_idx": src_idx}
        t_save_done = time.time() - t_save
        t_merge = time.time()
        object_crops = pipeline.reid_merge_tracks(object_crops)
        object_crops = pipeline.merge_duplicate_tracks(object_crops)
        t_merge_done = time.time() - t_merge
        t_build = time.time()
        four_dsg_dict = pipeline.build_4dsg_dict(object_crops=object_crops)
        scene_json = json.dumps(four_dsg_dict, separators=(",", ":"), sort_keys=False)
        t_build_done = time.time() - t_build
        # Stash sub-phase times for instrumented bench scripts.
        self._last_finalize_t = {
            "crop_save": t_save_done,
            "merge_dup": t_merge_done,
            "build_dict_json": t_build_done,
            "total": t_save_done + t_merge_done + t_build_done,
        }
        return four_dsg_dict, scene_json

    def _build_4dsg(
        self,
        frames: List[np.ndarray],
        mask_cache: Dict[int, List[SAM3SharedMask]],
        da3_results: list,
        source_indices: List[int],
        timestamps_s: List[float],
        frame_dir: Path,
    ) -> Tuple[Dict, str]:
        """Phase 2d + Phase 3: Build 4DSG from masks + depth."""
        pipeline = ROSEPipeline(self.config)
        crop_pad = self.config.vlm.object_crop_padding
        crop_sz = self.config.vlm.object_crop_size
        best_crops: Dict[tuple, Tuple[np.ndarray, float, int, float]] = {}

        # Pre-compute per-frame filtered masks + crops in a thread pool —
        # these per-frame steps are INDEPENDENT across frames (no shared
        # state).  Threading them on the CPU while the GIL releases for
        # numpy/cv2 ops hides ~half the wall time.
        from concurrent.futures import ThreadPoolExecutor

        def _preprocess(fidx_l):
            frame_masks = list(mask_cache.get(fidx_l, []))
            best_by_key: Dict[tuple, SAM3SharedMask] = {}
            for m in frame_masks:
                key = (m.run_id, m.obj_id_local)
                prev = best_by_key.get(key)
                if prev is None or m.score > prev.score:
                    best_by_key[key] = m
            frame_masks = list(best_by_key.values())
            frame_masks = [m for m in frame_masks if not _mask_is_fragmented(m.mask)]
            frame_masks = _dedup_masks_by_iou(frame_masks, iou_threshold=0.95)
            # Local best-crop dict for this frame
            local_crops = {}
            image_l = frames[fidx_l]
            for m in frame_masks:
                key = (m.run_id, m.obj_id_local)
                crop = _crop_object_from_mask(
                    image_l, m.mask, padding=crop_pad, size=crop_sz,
                )
                brightness = float(crop.mean())
                local_crops[key] = (crop, m.score, source_indices[fidx_l], brightness)
            return fidx_l, frame_masks, local_crops

        # Run preprocessing in parallel.  H200 has plenty of CPU cores;
        # 8 workers are ample for 32 frames of independent work.
        per_frame_data: Dict[int, Tuple[list, dict]] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_preprocess, f) for f in range(len(frames))]
            for fut in futures:
                fidx_l, fm, lc = fut.result()
                per_frame_data[fidx_l] = (fm, lc)

        # Merge per-frame best_crops into global best_crops (sequential, fast).
        for fidx_l, (_fm, lc) in per_frame_data.items():
            for key, val in lc.items():
                prev = best_crops.get(key)
                if prev is None or (val[1], val[3]) > (prev[1], prev[3]):
                    best_crops[key] = val

        # Build all per-frame inputs (cheap).
        frame_inputs = []
        for fidx in range(len(frames)):
            frame_masks, _ = per_frame_data.get(fidx, ([], {}))
            detections = [
                FastLocalDetection(
                    run_id=m.run_id,
                    local_obj_id=m.obj_id_local,
                    mask=m.mask,
                    score=m.score,
                )
                for m in frame_masks
            ]
            frame_inputs.append(FastFrameInput(
                frame_idx=source_indices[fidx],
                depth_t=da3_results[fidx].depth,
                K_t=da3_results[fidx].K,
                T_wc_t=da3_results[fidx].T_wc,
                detections=detections,
                depth_conf_t=da3_results[fidx].depth_conf,
                depth_is_metric=da3_results[fidx].is_metric,
                timestamp_s=timestamps_s[fidx],
            ))

        # ── PARALLEL LIFT (2026-05-31) ────────────────────────────────────
        # The per-frame backprojection (compute_candidates) is the dominant 4DSG
        # cost (~90% of build_4dsg, ~30% of e2e) and runs at full 10fps (does NOT
        # scale with sam3_propagate_stride). It is PURE per frame (no shared-state
        # mutation), so compute it across frames on the 192-core box (numpy releases
        # the GIL on the heavy nonzero/gather/matmul/percentile/patch-token ops).
        # The track-state update (fuse + observe) stays SEQUENTIAL in frame order →
        # output is numerically IDENTICAL to the serial path. config-gated.
        _parallel = getattr(self.config.sam3, "parallel_lift", True)
        if _parallel and len(frame_inputs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            _w = min(len(frame_inputs), int(getattr(self.config.sam3, "parallel_lift_workers", 16)))
            with ThreadPoolExecutor(max_workers=_w) as _ex:
                precomp = list(_ex.map(pipeline.compute_candidates, frame_inputs))
        else:
            precomp = [None] * len(frame_inputs)
        # Sequential fuse + observe (deterministic track state).
        for _fi, _pc in zip(frame_inputs, precomp):
            pipeline.process_frame(_fi, precomputed=_pc)

        # Save crops
        crops_dir = frame_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        object_crops: Dict[int, Dict[str, object]] = {}
        for key, gid in pipeline._local_to_global.items():
            if key in best_crops and gid not in object_crops:
                crop_rgb, _score, src_idx, _bright = best_crops[key]
                if _crop_is_uniform(crop_rgb):
                    continue
                crop_path = crops_dir / f"obj_{gid:04d}.jpg"
                cv2.imwrite(
                    str(crop_path),
                    cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                )
                object_crops[gid] = {
                    "path": str(crop_path),
                    "source_frame_idx": src_idx,
                }

        # Post-hoc dedup
        object_crops = pipeline.reid_merge_tracks(object_crops)
        object_crops = pipeline.merge_duplicate_tracks(object_crops)
        four_dsg_dict = pipeline.build_4dsg_dict(object_crops=object_crops)
        scene_json = json.dumps(
            four_dsg_dict, separators=(",", ":"), sort_keys=False
        )

        # ── Dynamic-targets export (动态目标管线 ALL_FRAMES) ──────────────────
        # Additive + gated: only runs when config.dynamic_targets.enabled. The
        # default 5.56 path never reaches this (enabled=False).
        if getattr(self.config, "dynamic_targets", None) is not None and \
                self.config.dynamic_targets.enabled:
            try:
                dtc = self.config.dynamic_targets
                # Collect up to K best crops (by SAM3 score) per FINAL surviving gid,
                # fusing multi-frame evidence for the VLM namer.
                gid_crops: Dict[int, list] = {}
                if dtc.name_objects:
                    K = int(dtc.max_crops_per_object)

                    def _sharpness(rgb):
                        # variance of Laplacian — higher = sharper (less motion blur)
                        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                        return float(cv2.Laplacian(g, cv2.CV_64F).var())

                    per_gid: Dict[int, list] = {}
                    for _fl, (_fm, _lc) in per_frame_data.items():
                        for key, val in _lc.items():
                            gid = pipeline._local_to_global.get(key)
                            if gid is None or gid not in object_crops:
                                continue
                            crop = val[0]
                            # rank crops for naming by mask-score * sharpness so the VLM
                            # sees the CLEAREST, most-confident views (blurry crops mis-name).
                            rank = float(val[1]) * (1.0 + _sharpness(crop))
                            per_gid.setdefault(gid, []).append((rank, crop))
                    for gid, lst in per_gid.items():
                        lst.sort(key=lambda x: x[0], reverse=True)
                        gid_crops[gid] = [c for _s, c in lst[:K]]
                    names = self._get_namer().name_objects(
                        gid_crops, vote=getattr(dtc, "namer_vote", True))
                else:
                    names = {}
                image_hw = (int(frames[0].shape[0]), int(frames[0].shape[1])) if frames else None
                src_fps = None
                if len(timestamps_s) >= 2:
                    diffs = np.diff(np.asarray(timestamps_s, dtype=float))
                    diffs = diffs[diffs > 1e-6]
                    if diffs.size:
                        src_fps = 1.0 / float(np.median(diffs))
                dt_dict = pipeline.build_dynamic_targets_dict(
                    object_crops=object_crops,
                    instance_names=names,
                    image_hw=image_hw,
                    source_fps=src_fps,
                )
                out_path = frame_dir / dtc.output_filename
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(dt_dict, f, ensure_ascii=False, indent=2)
                self._last_dynamic_targets = dt_dict
                self._last_dynamic_targets_path = str(out_path)
                logger.info("Dynamic-targets export: %d instances, %d frames → %s",
                            dt_dict["metadata"]["num_instances"],
                            dt_dict["metadata"]["num_frames"], out_path)
            except Exception as e:
                logger.error("Dynamic-targets export failed: %s", e, exc_info=True)

        return four_dsg_dict, scene_json

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: Path,
    ) -> Tuple[List[np.ndarray], Path, List[int], list, List[float]]:
        """Extract sampled frames from video, save as JPEGs."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_dir = Path(tempfile.mkdtemp(prefix="rose_warm_"))
        frames: List[np.ndarray] = []
        source_indices: List[int] = []
        timestamps_s: List[float] = []
        keyframe_paths: list = []
        save_idx = 0
        src_idx = 0
        target_fps = float(self.config.sampling.target_fps)
        max_frames = self.config.sampling.max_frames
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            source_fps = target_fps
        sample_interval_s = 1.0 / target_fps if target_fps > 0 else 0.0
        next_sample_s = 0.0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            src_t_s = src_idx / source_fps
            take = sample_interval_s == 0.0 or (src_t_s + 1e-9 >= next_sample_s)
            src_idx += 1
            if not take:
                continue
            if sample_interval_s > 0.0:
                while src_t_s + 1e-9 >= next_sample_s:
                    next_sample_s += sample_interval_s
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            source_indices.append(src_idx - 1)
            timestamps_s.append(src_t_s)
            # Skip JPEG writes during extraction — SAM3 receives PIL images
            # directly via set_video_frames(), no file I/O needed.  The
            # frame_dir is only used for crop images later.
            keyframe_paths.append((src_idx - 1, None))
            save_idx += 1
            if max_frames is not None and save_idx >= max_frames:
                break

        cap.release()
        logger.info(
            "Extracted %d frames (target_fps=%.1f) from %s",
            len(frames), target_fps, video_path.name,
        )
        return frames, frame_dir, source_indices, keyframe_paths, timestamps_s

    # ------------------------------------------------------------------
    # SAM3 session cleanup (keep model, clear per-video state)
    # ------------------------------------------------------------------

    def _cleanup_sam3_session(self) -> None:
        """Clean up SAM3 per-video state while keeping the model loaded.

        ``end_all_runs()`` calls ``_close_session()`` which releases the
        predictor session and all associated GPU tensors (including
        feature_cache and inference states).
        """
        self._sam3.end_all_runs()

    # ------------------------------------------------------------------
    # VLM (thin delegation to ROSEEndToEnd)
    # ------------------------------------------------------------------

    def _query_vlm(self, four_dsg_dict: Dict, question: str) -> str:
        """Query VLM with 4DSG context."""
        from rose.engine.pipeline.rose_e2e import ROSEEndToEnd

        # Create a temporary e2e instance just for VLM (no model loading)
        e2e = ROSEEndToEnd.__new__(ROSEEndToEnd)
        e2e.config = self.config
        e2e._vlm_client = self._vlm_client
        answer = e2e._query_vlm(four_dsg_dict, question)
        # Cache the VLM client for reuse
        self._vlm_client = e2e._vlm_client
        return answer

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def unload_all(self) -> None:
        """Clean shutdown: release all GPU resources."""
        import torch

        logger.info("Unloading all models...")
        if self._pending_cleanup_thread is not None:
            try:
                self._pending_cleanup_thread.join()
            except Exception:
                pass
            self._pending_cleanup_thread = None
        try:
            self._sam3.end_all_runs()
        except Exception:
            pass
        try:
            self._da3.unload()
        except Exception:
            pass
        try:
            self._fastsam.unload()
        except Exception:
            pass
        if self._sam3._predictor is not None:
            del self._sam3._predictor
            self._sam3._predictor = None
        torch.cuda.empty_cache()
        self._status = "stopped"
        logger.info("All models unloaded.")


# =====================================================================
# FastAPI application factory
# =====================================================================

def create_app(config: ROSEConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    pool = WarmModelPool(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load models and warm up
        pool.load_all()
        if config.sam3.enable_compile:
            pool.warmup_compile()
        pool._status = "ready"
        logger.info("Warm server ready. Accepting requests.")
        yield
        # Shutdown
        pool.unload_all()

    app = FastAPI(
        title="ROSE Warm Server",
        description="Keep vision models warm in GPU memory for fast inference.",
        lifespan=lifespan,
    )

    @app.post("/infer", response_model=InferenceResponse)
    def infer(request: InferenceRequest):
        return pool.run_inference(request)

    @app.get("/status", response_model=ServerStatusResponse)
    def status():
        return pool.get_status()

    @app.post("/shutdown")
    def shutdown():
        """Gracefully shut down the server."""
        pool.unload_all()
        os.kill(os.getpid(), signal.SIGTERM)
        return {"status": "shutting_down"}

    return app
