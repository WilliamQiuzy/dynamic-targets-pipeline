"""SAM 3.1 multiplex predictor wrapper — drop-in replacement for SAM3SharedSessionManager.

Key architectural difference from base SAM3:
    Base SAM3 processes objects one at a time during propagation, so adding
    discovery objects (via add_object_point) requires a separate
    propagate_new_objects() call that re-scans all frames per object.

    SAM 3.1 multiplex processes up to ``multiplex_count`` objects JOINTLY in
    a single ViT/tracker forward pass.  All prompts (initial bboxes + mid-video
    discoveries) are accumulated, then ONE propagate_in_video call yields
    masks for all objects across all frames simultaneously.  Meta reports
    ~7× speedup at 128 objects vs base SAM3.

API mapping (current wrapper → multiplex):
    create_run_with_initial_bboxes(boxes, ...)  → add_prompt(boxes)
    add_object_point(frame, xy)                  → add_prompt(points, obj_id)
    propagate_all(frame_idx)                      → cached output of one propagation
    propagate_new_objects()                       → re-call propagate_in_video
    end_all_runs()                                → close_session
"""
from __future__ import annotations

import inspect
import logging
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

from rose.engine.config.rose_config import SAM3Config

logger = logging.getLogger(__name__)


@dataclass
class SAM3SharedMask:
    """Drop-in compatible SAM3SharedMask for multiplex outputs."""
    run_id: int
    obj_id_local: int
    mask: np.ndarray
    score: float


@dataclass
class SAM3SharedTagRun:
    """Drop-in compatible run record."""
    run_id: int
    tag: str
    start_frame: int
    status: str = "active"
    last_propagated_frame: int = -1
    obj_ids: Set[int] = field(default_factory=set)


def _patch_init_state_kwargs_filter(predictor):
    """SAM3.1 multiplex's init_state doesn't accept offload_state_to_cpu but
    base predictor's start_session always passes it.  Filter unsupported kwargs."""
    _orig_init_state = predictor.model.init_state
    sig = inspect.signature(_orig_init_state)
    def _filtered(*args, **kwargs):
        kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return _orig_init_state(*args, **kw)
    predictor.model.init_state = _filtered


def _patch_fast_video_loader():
    """Monkey-patch SAM3's load_resource_as_video_frames so when called with
    the special string ``"__FAST_LOADED_FRAMES__"``, it returns the pre-built
    GPU tensor stashed by ``set_video_frames``.  Saves ~9s/video on the CPU
    PIL+JPEG roundtrip.  No-op if the marker isn't set."""
    import sam3.model.io_utils as _io_utils
    if getattr(_io_utils, "_fast_loader_patched", False):
        return
    _orig_load = _io_utils.load_resource_as_video_frames

    def _patched(resource_path, *args, **kwargs):
        if (
            isinstance(resource_path, str)
            and resource_path == "__FAST_LOADED_FRAMES__"
            and SAM3MultiplexSharedSessionManager._FAST_LOADED_FRAMES is not None
        ):
            return SAM3MultiplexSharedSessionManager._FAST_LOADED_FRAMES
        return _orig_load(resource_path, *args, **kwargs)

    _io_utils.load_resource_as_video_frames = _patched
    # Also patch the symbol imported into sam3_video_inference (init_state uses it directly)
    import sam3.model.sam3_video_inference as _vi
    _vi.load_resource_as_video_frames = _patched
    try:
        import sam3.model.sam3_multiplex_tracking as _mt
        _mt.load_resource_as_video_frames = _patched
    except Exception:
        pass
    _io_utils._fast_loader_patched = True


class SAM3MultiplexSharedSessionManager:
    """Drop-in replacement for SAM3SharedSessionManager using SAM 3.1 multiplex."""

    def __init__(self, config: Optional[SAM3Config] = None):
        self.config = config or SAM3Config()
        self._predictor = None
        self._session_id: Optional[str] = None
        self._video_dir: Optional[Path] = None
        self._tmp_video_dir: Optional[Path] = None
        self._n_frames: int = 0

        # Mirror the base wrapper's bookkeeping so callers don't change.
        self._runs: Dict[int, SAM3SharedTagRun] = {}
        self._next_run_id: int = 0
        self._obj_id_to_run_id: Dict[int, int] = {}
        self._next_obj_id: int = 0

        # Per-frame mask cache populated after each propagation pass.
        self._propagation_cache: Dict[int, List[SAM3SharedMask]] = {}
        self._propagation_dirty: bool = True  # True ⇒ need to re-propagate before reading

        # D4 bookkeeping (no-op for multiplex but keeps API compat)
        self._obj_id_to_discovery_frame: Dict[int, int] = {}
        self._discovery_backward_window: int = 5

    @property
    def num_runs(self) -> int:
        return len(self._runs)

    @property
    def active_runs(self) -> List[SAM3SharedTagRun]:
        return [r for r in self._runs.values() if r.status == "active"]

    def load(self) -> None:
        """Load SAM3.1 multiplex predictor lazily."""
        if self._predictor is not None:
            return

        sam3_src = str(Path("rose/vision/sam3").resolve())
        if sam3_src not in sys.path:
            sys.path.insert(0, sam3_src)

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3.1 multiplex requires CUDA.")
        torch.backends.cudnn.benchmark = True

        from sam3.model_builder import build_sam3_multiplex_video_predictor

        ckpt = self._resolve_checkpoint()
        # multiplex_count is BUCKET CAPACITY (must be 16 for the released checkpoint).
        # max_num_objects is the TOTAL OBJECT CAP — multiplex auto-allocates additional
        # buckets up to ceil(max_num_objects / multiplex_count).  E.g. 50 objects on a
        # single GPU = 4 buckets internally, propagation cost grows linearly with buckets.
        multiplex_count = 16  # locked by checkpoint training
        max_objects = max(int(getattr(self.config, "max_active_tracks", 50) or 50), 16)

        self._predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=ckpt,
            multiplex_count=multiplex_count,
            max_num_objects=max_objects,
            use_fa3=self.config.use_fa3,
            use_rope_real=True,
            compile=self.config.enable_compile,
            warm_up=self.config.enable_compile,
            async_loading_frames=False,
        )
        _patch_init_state_kwargs_filter(self._predictor)
        _patch_fast_video_loader()

        # Threshold override (multiplex exposes default_output_prob_thresh)
        try:
            self._predictor.default_output_prob_thresh = (
                self.config.score_threshold_detection
            )
        except Exception:
            pass

        # SPEED: skip video-resolution mask interpolation inside
        # propagate_in_video.  The multiplex caller
        # (`_propogate_tracker_one_frame_local_gpu`) receives `(low_res,
        # video_res)` but DISCARDS video_res via `_`, then runs its own
        # low→video conversion later via `_convert_low_res_mask_to_video_res`.
        # So the bilinear upsample inside the tracker's
        # `_get_orig_video_res_output` is pure waste:
        # ~N_obj × H_video × W_video ops per frame.
        #
        # NOTE: do NOT also disable ``non_overlap_masks_for_output`` —
        # turning it off reduces multiplex track quality (track count
        # drops, some obs go from 32 → 4 because overlapping mask
        # decisions diverge).  Keep that ON.
        try:
            tracker = self._predictor.model.tracker
            if hasattr(tracker, "_get_orig_video_res_output"):
                def _skip_video_res(self, inference_state, any_res_masks):
                    return any_res_masks, None
                tracker._get_orig_video_res_output = _skip_video_res.__get__(
                    tracker, type(tracker),
                )
                logger.info("Patched tracker._get_orig_video_res_output to skip video_res (speed)")
        except Exception:
            logger.debug("Could not apply tracker speed patches", exc_info=True)

        self._discovery_backward_window = int(self.config.discovery_backward_window)
        logger.info(
            "SAM3.1 multiplex predictor loaded from %s (multiplex=%d, max_obj=%d, fa3=%s)",
            ckpt, multiplex_count, max_objects, self.config.use_fa3,
        )

    def _resolve_checkpoint(self) -> Optional[str]:
        # Prefer explicit multiplex_model_path if set, else fall back to model_path.
        path_str = getattr(self.config, "multiplex_model_path", None) or self.config.model_path
        p = Path(path_str)
        if p.is_file():
            return str(p.resolve())
        if p.is_dir():
            # Look for the multiplex checkpoint specifically
            for name in ("sam3.1_multiplex.pt", "sam3.1_multiplex.safetensors"):
                cand = p / name
                if cand.is_file():
                    return str(cand.resolve())
            # Fall back to any .pt
            cands = sorted(p.glob("*.pt"))
            if cands:
                return str(cands[0].resolve())
        raise FileNotFoundError(
            f"SAM3.1 multiplex checkpoint not found at {p}. "
            "Set sam3.multiplex_model_path or sam3.model_path to the directory."
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def set_video_dir(self, video_dir: Union[str, Path]) -> None:
        """Equivalent to base wrapper's set_video_dir — opens session."""
        self._video_dir = Path(video_dir)
        self._open_session(str(self._video_dir))

    def set_video_frames(self, pil_images: list) -> None:
        """FAST PATH: build (N, 3, image_size, image_size) tensor directly on
        GPU via torch.nn.functional.interpolate, bypassing the CPU PIL+JPEG
        roundtrip in SAM3's load_resource_as_video_frames (~9s → ~0.2s).

        Approach:
          1. Stack PIL/np frames → uint8 numpy (N, H, W, 3)
          2. Move to GPU, permute to (N, 3, H, W), normalize to fp16 [0,1]
          3. F.interpolate to (image_size, image_size)
          4. Normalize by SAM3's mean/std
          5. Pass tensor directly to init_state (multiplex's loader detects
             pre-built tensor list and uses it, but here we monkey-patch
             load_resource_as_video_frames to detect our marker.)
        """
        import torch.nn.functional as F
        import numpy as np

        # Coerce to numpy uint8 RGB
        first = pil_images[0]
        if hasattr(first, "convert"):  # PIL.Image
            np_frames = [np.array(img.convert("RGB"), dtype=np.uint8) for img in pil_images]
        else:
            np_frames = [np.asarray(img, dtype=np.uint8) for img in pil_images]
        h, w = np_frames[0].shape[:2]
        n = len(np_frames)
        stacked = np.stack(np_frames)  # (N, H, W, 3) uint8

        # Move to GPU + convert
        image_size = self._predictor.model.image_size  # typically 1008
        t = torch.from_numpy(stacked).cuda(non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous().to(torch.float16) / 255.0  # (N, 3, H, W)
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        # SAM3 uses mean=std=(0.5, 0.5, 0.5), so normalize: (x - 0.5) / 0.5 = 2x - 1
        t = t * 2.0 - 1.0

        if self.config.offload_video_to_cpu:
            t = t.cpu()
        self._n_frames = n
        # Stash for the patched loader
        SAM3MultiplexSharedSessionManager._FAST_LOADED_FRAMES = (t, h, w)
        self._open_session("__FAST_LOADED_FRAMES__")
        SAM3MultiplexSharedSessionManager._FAST_LOADED_FRAMES = None

    # Class-level slot used to pass pre-built image tensor through to
    # init_state without changing SAM3 source.
    _FAST_LOADED_FRAMES = None

    def _open_session(self, resource_path: str) -> None:
        if self._session_id is not None:
            self._close_session_safe()
        resp = self._predictor.handle_request({
            "type": "start_session",
            "resource_path": resource_path,
            "offload_video_to_cpu": self.config.offload_video_to_cpu,
        })
        self._session_id = resp["session_id"]
        # Reset bookkeeping
        self._runs.clear()
        self._next_run_id = 0
        self._obj_id_to_run_id.clear()
        self._obj_id_to_discovery_frame.clear()
        self._next_obj_id = 0
        self._propagation_cache.clear()
        self._propagation_dirty = True

    def _close_session_safe(self) -> None:
        if self._session_id is not None and self._predictor is not None:
            try:
                self._predictor.handle_request({
                    "type": "close_session", "session_id": self._session_id,
                })
            except Exception:
                logger.warning("close_session failed", exc_info=True)
        self._session_id = None

    def end_all_runs(self) -> None:
        """Compat with base wrapper."""
        self._close_session_safe()
        for r in self._runs.values():
            r.status = "ended"
        self._cleanup_tmp_dir()

    def _cleanup_tmp_dir(self) -> None:
        if self._tmp_video_dir is not None:
            import shutil
            shutil.rmtree(str(self._tmp_video_dir), ignore_errors=True)
            self._tmp_video_dir = None

    # ------------------------------------------------------------------
    # Prompt API
    # ------------------------------------------------------------------

    def add_bboxes_batch_multi_frame(
        self,
        bboxes_by_frame: Dict[int, List[List[float]]],
    ) -> Dict[int, List[int]]:
        """Multi-anchor batched bbox add — single session for all anchors.

        Patches SAM 3.1 multiplex's reset-on-bbox-add behaviour by skipping
        reset_state on subsequent calls.  Result: ONE inference session
        holds prompts at multiple frames, enabling joint propagation and
        unified memory bank across all anchors (no Phase B-8 fresh session
        needed).

        Returns: dict mapping frame_idx → list of obj_ids assigned at that
        frame.  Total obj_ids assigned across all frames are unique.
        """
        result: Dict[int, List[int]] = {}
        if not bboxes_by_frame:
            return result
        # Process in ascending frame order — first call resets (init session),
        # subsequent calls skip_reset to preserve prior prompts.
        first = True
        for fidx in sorted(bboxes_by_frame.keys()):
            boxes = bboxes_by_frame[fidx]
            if not boxes:
                result[fidx] = []
                continue
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor([1] * len(boxes), dtype=torch.long)
            req = {
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": int(fidx),
                "bounding_boxes": boxes_t,
                "bounding_box_labels": labels_t,
            }
            if not first:
                req["_skip_reset"] = True
            resp = self._predictor.handle_request(req)
            outputs = resp.get("outputs", {}) or {}
            ids = outputs.get("out_obj_ids", None)
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().numpy().tolist()
            elif ids is None:
                ids = []
            ids = [int(o) for o in ids]
            result[fidx] = ids
            # Bookkeeping — one run per frame.
            run_id = self._next_run_id
            self._next_run_id += 1
            run = SAM3SharedTagRun(run_id=run_id, tag=f"batched_bbox_f{fidx}",
                                    start_frame=int(fidx), status="active")
            self._runs[run_id] = run
            for oid in ids:
                self._obj_id_to_run_id[oid] = run_id
                self._obj_id_to_discovery_frame[oid] = int(fidx)
                run.obj_ids.add(oid)
            if ids:
                self._next_obj_id = max(self._next_obj_id, max(ids) + 1)
            first = False
        self._propagation_dirty = True
        return result

    def add_bboxes_batch(
        self,
        boxes_xywh: List[List[float]],
        frame_idx: int,
    ) -> List[int]:
        """Multiplex FAST PATH: register all bboxes in ONE add_prompt call.

        Multiplex's bbox-grounding path (super().add_prompt with boxes_xywh)
        accepts a multi-bbox tensor and returns multiple obj_ids in a single
        joint pass.  This is ~100× faster than calling add_prompt per-bbox
        (~0.2s for 16 bboxes vs ~1s each = 16s).

        ⚠️ This RESETS the inference state — must only be called once per
        session.  Returns list of obj_ids assigned by multiplex grounding.
        """
        if not boxes_xywh:
            return []
        boxes_t = torch.tensor(boxes_xywh, dtype=torch.float32)
        labels_t = torch.tensor([1] * len(boxes_xywh), dtype=torch.long)
        resp = self._predictor.handle_request({
            "type": "add_prompt",
            "session_id": self._session_id,
            "frame_index": frame_idx,
            "bounding_boxes": boxes_t,
            "bounding_box_labels": labels_t,
        })
        # Multiplex returns out_obj_ids in response
        outputs = resp.get("outputs", {}) or {}
        ids = outputs.get("out_obj_ids", None)
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy().tolist()
        elif ids is None:
            ids = []
        # Bookkeeping: register a single run for all batched bboxes
        run_id = self._next_run_id
        self._next_run_id += 1
        run = SAM3SharedTagRun(run_id=run_id, tag="batched_bbox", start_frame=frame_idx, status="active")
        self._runs[run_id] = run
        for oid in ids:
            self._obj_id_to_run_id[int(oid)] = run_id
            self._obj_id_to_discovery_frame[int(oid)] = frame_idx
            run.obj_ids.add(int(oid))
        # Update _next_obj_id to avoid collisions if anyone else adds points later
        if len(ids) > 0:
            self._next_obj_id = max(self._next_obj_id, max(int(o) for o in ids) + 1)
        self._propagation_dirty = True
        return [int(o) for o in ids]

    def create_run_with_initial_bboxes(
        self,
        boxes_xywh: List[List[float]],
        box_labels: List[int],
        frame_idx: int,
        tag: str,
    ) -> Tuple[int, List[SAM3SharedMask]]:
        """Register bbox prompts as SAM2 tracker objects.

        Multiplex's add_prompt with bounding_boxes (no text) routes to SAM3
        grounding which expects text-conditioned detections — NOT what we
        want for FastSAM-supplied bboxes.  Instead, convert each bbox to its
        CENTER POINT and register via add_sam2_new_points (same path as
        ``add_object_point``).  This creates a proper tracker state that
        propagate_in_video will track across all frames.
        """
        run_id = self._next_run_id
        self._next_run_id += 1
        run = SAM3SharedTagRun(run_id=run_id, tag=tag, start_frame=frame_idx, status="active")
        self._runs[run_id] = run

        init_masks: List[SAM3SharedMask] = []
        for bbox, lab in zip(boxes_xywh, box_labels):
            # bbox = [x, y, w, h] normalized; center = (x + w/2, y + h/2)
            cx = float(bbox[0] + bbox[2] / 2.0)
            cy = float(bbox[1] + bbox[3] / 2.0)
            obj_id = self._next_obj_id
            self._next_obj_id += 1
            self._obj_id_to_run_id[obj_id] = run_id
            self._obj_id_to_discovery_frame[obj_id] = frame_idx
            run.obj_ids.add(obj_id)

            points_tensor = torch.tensor([[cx, cy]], dtype=torch.float32)
            labels_tensor = torch.tensor([1], dtype=torch.int32)
            resp = self._predictor.handle_request({
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": frame_idx,
                "points": points_tensor,
                "point_labels": labels_tensor,
                "obj_id": obj_id,
                "rel_coordinates": True,
            })
            outputs = resp.get("outputs", {}) or {}
            for sm in self._extract_frame_masks(outputs):
                if sm.obj_id_local == obj_id:
                    init_masks.append(SAM3SharedMask(
                        run_id=run_id, obj_id_local=obj_id,
                        mask=sm.mask, score=sm.score,
                    ))
                    break

        self._propagation_dirty = True
        return run_id, init_masks

    def refine_object_with_point(
        self, obj_id: int, frame_idx: int, point_xy: Tuple[float, float],
        label: int = 1,
    ) -> None:
        """Add a SAM2 refinement point to an EXISTING obj_id (no new id).

        Used to convert a multiplex grounding result (from add_bboxes_batch)
        into a SAM2-tracked object.  After at least one prior propagation has
        run, calling propagate_in_video again triggers ``propagation_partial``
        which uses SAM2 memory tracking — gives full per-frame masks across
        the entire video instead of the ~3 frames produced by VG grounding
        alone.
        """
        points_tensor = torch.tensor([list(point_xy)], dtype=torch.float32)
        labels_tensor = torch.tensor([label], dtype=torch.int32)
        self._predictor.handle_request({
            "type": "add_prompt",
            "session_id": self._session_id,
            "frame_index": frame_idx,
            "points": points_tensor,
            "point_labels": labels_tensor,
            "obj_id": int(obj_id),
            "rel_coordinates": True,
            "clear_old_points": False,  # keep prior history; add a new positive
        })
        self._propagation_dirty = True

    def add_object_point(
        self, frame_idx: int, point_xy: Tuple[float, float], label: int = 1,
    ) -> int:
        """Add a new tracked object via point prompt (mid-video discovery)."""
        if not self.active_runs:
            # Auto-create a run so we always have an owner
            run = SAM3SharedTagRun(
                run_id=self._next_run_id, tag="discovery",
                start_frame=frame_idx, status="active",
            )
            self._runs[self._next_run_id] = run
            self._next_run_id += 1
        else:
            run = self.active_runs[0]

        obj_id = self._next_obj_id
        self._next_obj_id += 1
        self._obj_id_to_run_id[obj_id] = run.run_id
        self._obj_id_to_discovery_frame[obj_id] = frame_idx
        run.obj_ids.add(obj_id)

        points_tensor = torch.tensor([list(point_xy)], dtype=torch.float32)
        labels_tensor = torch.tensor([label], dtype=torch.int32)
        self._predictor.handle_request({
            "type": "add_prompt",
            "session_id": self._session_id,
            "frame_index": frame_idx,
            "points": points_tensor,
            "point_labels": labels_tensor,
            "obj_id": obj_id,
            "rel_coordinates": True,
        })
        self._propagation_dirty = True
        return obj_id

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def precompute_backbone_features(self, vg_stride: int = 0) -> None:
        """No-op for multiplex (handled internally during propagate)."""
        return

    def _propagate(self, start_frame: int = 0) -> None:
        """Run propagate_in_video and populate _propagation_cache.

        ``start_frame`` > 0 propagates only from that frame onward, PRESERVING
        the already-cached masks for earlier frames.  Used for incremental
        late-discovery: a late object first appears at frame f, so there is no
        point re-propagating the whole bucket over frames [0, f).

        Multiplex output format per frame:
            outputs = {
                'out_obj_ids': np.ndarray of shape (N,),
                'out_binary_masks': np.ndarray of shape (N, H, W) bool,
                'output_probs': np.ndarray of shape (N,) or (N, H, W),
            }
        """
        if not self._propagation_dirty:
            return
        start_frame = max(0, int(start_frame))
        if start_frame <= 0:
            self._propagation_cache.clear()
        else:
            for _f in list(self._propagation_cache.keys()):
                if _f >= start_frame:
                    del self._propagation_cache[_f]
        # Multiplex's previous_stages_out flag is set only by VG (text/box-grounding)
        # add_prompt path; our point-prompt path (add_sam2_new_points) doesn't set it.
        # Pass start_frame_index=0 explicitly to skip the "no prompts" guard.
        callback = getattr(self, "_per_frame_callback", None)
        for response in self._predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": self._session_id,
            "start_frame_index": start_frame,
            # Forward only: prompts are at frame 0, no need for reverse pass.
            # Multiplex's default is "both" which doubles propagation cost.
            "propagation_direction": "forward",
        }):
            fid = int(response.get("frame_index", -1))
            frame_masks = self._extract_frame_masks(response.get("outputs", {}) or {})
            self._propagation_cache[fid] = frame_masks
            # Streaming hook: call user callback as soon as each frame is ready.
            # Used by warm_server to build 4DSG incrementally in parallel with
            # the GPU-bound propagation, hiding ~0.5s of CPU work.
            if callback is not None:
                try:
                    callback(fid, frame_masks)
                except Exception as e:
                    logger.warning("per-frame callback raised: %s", e, exc_info=True)
        self._propagation_dirty = False

    def _extract_frame_masks(self, outputs: dict) -> List[SAM3SharedMask]:
        """Convert multiplex outputs dict → list of SAM3SharedMask."""
        out: List[SAM3SharedMask] = []
        obj_ids = outputs.get("out_obj_ids", None)
        masks = outputs.get("out_binary_masks", None)
        probs = outputs.get("output_probs", None)
        if obj_ids is None or masks is None:
            return out
        # Coerce to lists
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.detach().cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        score_arr = None
        if probs is not None:
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            # If probs is per-pixel, take max over spatial dims as score
            if hasattr(probs, "ndim") and probs.ndim >= 3:
                score_arr = probs.reshape(probs.shape[0], -1).max(axis=1)
            else:
                score_arr = probs
        n_total = len(obj_ids)
        n_empty = 0
        for idx, oid in enumerate(obj_ids):
            mask = masks[idx]
            if not mask.any():
                n_empty += 1
                continue
            run_id = self._obj_id_to_run_id.get(int(oid), 0)
            score = float(score_arr[idx]) if score_arr is not None and idx < len(score_arr) else 1.0
            out.append(SAM3SharedMask(
                run_id=run_id, obj_id_local=int(oid),
                mask=mask.astype(bool, copy=False), score=score,
            ))
        if n_total > 0:
            logger.debug("_extract_frame_masks: %d/%d empty (kept %d)",
                         n_empty, n_total, len(out))
        return out

    def propagate_all(self, frame_idx: int) -> List[SAM3SharedMask]:
        """Return masks for *frame_idx* (re-propagate if dirty)."""
        if self._propagation_dirty:
            self._propagate()
        return self._propagation_cache.get(frame_idx, [])

    def propagate_new_objects(self, start_frame: int = 0) -> None:
        """Multiplex re-runs propagation as a single joint pass.

        ``start_frame`` > 0 → incremental: keep cached masks for earlier frames
        and only re-propagate from ``start_frame`` (where the new objects appear).
        """
        self._propagation_dirty = True
        self._propagate(start_frame=start_frame)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_mask(m) -> Optional[np.ndarray]:
        """Convert multiplex output mask to a HxW bool numpy array."""
        if m is None:
            return None
        if isinstance(m, dict):
            # Look for common keys
            for k in ("mask", "segmentation", "binary_mask"):
                if k in m:
                    m = m[k]
                    break
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu()
            if m.dtype != torch.bool:
                m = m > 0
            return m.squeeze().numpy().astype(bool)
        if isinstance(m, np.ndarray):
            if m.ndim > 2:
                m = m.squeeze()
            return m.astype(bool)
        return None
