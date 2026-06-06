"""SAM3 wrapper with a single shared predictor session.

Supports two prompt modes:
- **Bbox prompts** via ``create_run_with_initial_bboxes()`` for initial
  object discovery.  SAM3's ``add_prompt()`` with bboxes calls
  ``reset_state()``, so this must only be called once per video.
- **Point prompts** via ``add_object_point()`` for incrementally adding
  new objects mid-video without destroying existing tracker state.
  This routes to SAM3's ``add_tracker_new_points()``.

After adding new objects via point prompts, call ``propagate_new_objects()``
to trigger SAM3's ``propagation_partial`` mode which propagates only the
new objects and merges with cached results for existing objects.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import inspect

import numpy as np

from rose.engine.config.rose_config import SAM3Config

logger = logging.getLogger(__name__)


def _enable_fa3_on_sam3(model) -> int:
    """Flip use_fa3=True on every SAM3 attention module that supports it.

    SAM3's ``Sam3VideoPredictor`` (the path used here) builds the model via
    ``build_sam3_video_model`` which never propagates the ``use_fa3`` flag, and
    ``_create_tracker_transformer`` hardcodes ``use_fa3=False``.  So at module
    construction every attention has ``use_fa3=False``.

    The runtime FA3 dispatch in:
      - sam3/sam/transformer.py:248,343 (RoPEAttention/Attention)
      - sam3/model/vitdet.py:610 (Vanilla ViT attention)
    is gated only by ``self.use_fa3``.  Setting the flag post-load routes
    those forward paths through ``flash_attn_interface.flash_attn_func``
    (FP8, Hopper) instead of PyTorch SDPA — no weight changes required.
    """
    n = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "use_fa3") and not mod.use_fa3:
            mod.use_fa3 = True
            n += 1
    return n


def _start_session_compat(predictor, resource_path, config: SAM3Config) -> dict:
    """Call predictor.start_session() with offload kwargs only if supported."""
    sig = inspect.signature(predictor.start_session)
    kwargs: dict = {"resource_path": resource_path}
    if "offload_state_to_cpu" in sig.parameters:
        kwargs["offload_state_to_cpu"] = config.offload_state_to_cpu
    if "offload_video_to_cpu" in sig.parameters:
        kwargs["offload_video_to_cpu"] = config.offload_video_to_cpu
    return predictor.start_session(**kwargs)


@dataclass
class SAM3SharedTagRun:
    """One tag prompt inside the shared SAM3 session."""

    run_id: int
    tag: str
    start_frame: int
    status: str = "created"  # created | active | ended
    last_propagated_frame: int = -1
    obj_ids: Set[int] = field(default_factory=set)


@dataclass
class SAM3SharedMask:
    """A single mask output from SAM3 shared-session mode."""

    run_id: int
    obj_id_local: int
    mask: np.ndarray  # (H, W) bool
    score: float


class SAM3SharedSessionManager:
    """Manage SAM3 bbox prompts in a single shared predictor session."""

    def __init__(self, config: Optional[SAM3Config] = None):
        self.config = config or SAM3Config()
        self._predictor = None
        self._video_dir: Optional[Path] = None
        self._video_frames: Optional[list] = None
        self._session_id: Optional[str] = None
        self._session_last_propagated_frame: int = -1
        self._runs: Dict[int, SAM3SharedTagRun] = {}
        self._next_run_id: int = 0
        self._obj_id_to_run_id: Dict[int, int] = {}
        # Frame at which each new (point-prompted) obj_id was discovered.
        # Used by propagate_new_objects to bound the backward-propagation
        # window: an object discovered at frame F is unlikely to exist
        # before max(0, F - backward_window).
        self._obj_id_to_discovery_frame: Dict[int, int] = {}
        self._discovery_backward_window: int = 5  # frames; configurable via setter
        # Cache propagation results per-frame.  SAM3's action_history
        # mechanism switches to fetch-only mode after two consecutive
        # propagate_in_video calls, so we must propagate ALL remaining
        # frames in a single call and cache results here.
        self._propagation_cache: Dict[int, List[SAM3SharedMask]] = {}

    def load(self) -> None:
        """Load SAM3 predictor lazily."""
        if self._predictor is not None:
            return

        sam3_src = str(Path("rose/vision/sam3").resolve())
        if sam3_src not in sys.path:
            sys.path.insert(0, sam3_src)
        # Flash Attention 3 (Hopper) — pre-compiled .so for H200/sm90
        fa3_src = "/tmp/flash-attention/hopper"
        if fa3_src not in sys.path:
            sys.path.insert(0, fa3_src)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires CUDA. No CUDA device found.")
        # Auto-tune conv algorithms for fixed-size inputs (1008×1008).
        torch.backends.cudnn.benchmark = True

        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        checkpoint_path = self._resolve_checkpoint()

        self._predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            bpe_path=None,
            has_presence_token=True,
            geo_encoder_use_img_cross_attn=True,
            strict_state_dict_loading=True,
            async_loading_frames=False,
            apply_temporal_disambiguation=True,
            compile=self.config.enable_compile,
        )

        # Override threshold and memory controls.
        self._predictor.model.score_threshold_detection = (
            self.config.score_threshold_detection
        )
        self._predictor.model.trim_past_non_cond_mem_for_eval = (
            self.config.trim_past_non_cond_mem_for_eval
        )
        # Disable hotstart: the default hotstart_delay=15 buffers the
        # first 15 frames and retroactively removes objects that aren't
        # "confirmed" during that window.  With bbox prompts from YOLO
        # (already filtered for quality) this causes valid objects to be
        # dropped.  Setting hotstart_delay=0 disables the mechanism.
        self._predictor.model.hotstart_delay = 0

        if self.config.use_fa3:
            n_patched = _enable_fa3_on_sam3(self._predictor.model)
            logger.info("FA3 enabled on %d SAM3 attention modules", n_patched)

        # D4: configure bounded backward propagation window
        self._discovery_backward_window = int(self.config.discovery_backward_window)

        logger.info(
            "SAM3 shared-session predictor loaded from %s (score_thresh=%.2f)",
            checkpoint_path or "HuggingFace",
            self.config.score_threshold_detection,
        )

    def _resolve_checkpoint(self) -> Optional[str]:
        model_path = Path(self.config.model_path)
        if model_path.is_file():
            return str(model_path.resolve())
        if model_path.is_dir():
            candidates = sorted(model_path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .pt checkpoint files found in {model_path}. "
                    f"Place SAM3 weights (sam3.pt) in {model_path} or set "
                    f"sam3.model_path to the checkpoint file directly."
                )
            path = str(candidates[0].resolve())
            logger.info("Auto-selected SAM3 checkpoint: %s", path)
            return path
        raise FileNotFoundError(
            f"SAM3 model_path not found: {model_path}. "
            f"Download weights and place in {model_path}."
        )

    def set_video_dir(self, video_dir: Union[str, Path]) -> None:
        self._video_dir = Path(video_dir)
        self._video_frames = None  # clear PIL path
        # Restart the session on new video to avoid stale frame caches.
        self._close_session()
        self._session_last_propagated_frame = -1
        self._runs.clear()
        self._next_run_id = 0
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()

    def set_video_frames(self, pil_images: list) -> None:
        """Set video frames directly as a list of PIL.Image objects.

        This avoids the JPEG round-trip (write-to-disk + read-from-disk)
        that ``set_video_dir`` requires.  ``_ensure_session_started``
        will pass the list straight to SAM3's ``init_state`` →
        ``load_resource_as_video_frames``, which already supports PIL
        input.
        """
        self._video_frames = pil_images
        self._video_dir = None  # clear dir path
        # Restart the session on new video to avoid stale frame caches.
        self._close_session()
        self._session_last_propagated_frame = -1
        self._runs.clear()
        self._next_run_id = 0
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()

    def _ensure_session_started(self) -> None:
        if self._session_id is not None:
            return
        if self._video_dir is None and self._video_frames is None:
            raise RuntimeError("Call set_video_dir() or set_video_frames() before creating tags.")

        resource = self._video_frames if self._video_frames is not None else str(self._video_dir)
        result = _start_session_compat(
            self._predictor, resource, self.config,
        )
        self._session_id = result["session_id"]
        self._session_last_propagated_frame = -1
        logger.debug("Started SAM3 shared session %s", self._session_id)

    def create_run_with_initial_bboxes(
        self,
        boxes_xywh: List[List[float]],
        box_labels: Optional[List[int]],
        frame_idx: int,
        tag: str = "bbox",
    ) -> Tuple[SAM3SharedTagRun, List[SAM3SharedMask]]:
        """Initialise SAM3 with multiple bboxes and return initial masks.

        SAM3's ``add_prompt(boxes_xywh=...)`` is a *semantic-level* prompt:
        it treats the first box as a visual prompt for its detector and the
        rest as geometric refinement — it does **not** create one tracker
        per box.  To get per-object tracking we therefore:

        1. Send only the **first** bbox via ``add_prompt`` (visual prompt
           initialisation + reset_state).
        2. For every remaining bbox, compute its normalised centre and call
           ``add_object_point()`` which routes to
           ``add_tracker_new_points()`` — one dedicated tracker per object.

        Args:
            boxes_xywh: List of normalised [xmin, ymin, w, h] boxes in [0, 1].
            box_labels: Per-box labels (1 = foreground). Defaults to all-1 if None.
            frame_idx: Frame index where boxes are added.
            tag: Debug label attached to this run.
        """
        if not boxes_xywh:
            raise ValueError("boxes_xywh cannot be empty")
        self.load()
        self._ensure_session_started()

        if box_labels is None:
            box_labels = [1] * len(boxes_xywh)
        if len(boxes_xywh) != len(box_labels):
            raise ValueError("boxes_xywh and box_labels must have the same length")

        # Mirror SAM3's reset_state on our side.
        for run in self._runs.values():
            if run.status == "active":
                run.status = "ended"
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()
        self._purge_ended_runs()

        run_id = self._next_run_id
        self._next_run_id += 1
        run = SAM3SharedTagRun(run_id=run_id, tag=tag, start_frame=frame_idx)
        self._runs[run_id] = run

        # --- Send FIRST bbox only as visual prompt ---
        # SAM3's add_prompt with boxes treats the first box as a visual
        # prompt and remaining boxes as geometric refinement — not as
        # separate objects.  Remaining detections must be added via point
        # prompts AFTER the first propagate_in_video call (SAM3 requires
        # cached_frame_outputs before add_tracker_new_points can run).
        prompt_result = self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_idx,
            text="visual",
            bounding_boxes=[boxes_xywh[0]],
            bounding_box_labels=[box_labels[0]],
        )
        outputs = prompt_result.get("outputs", {})
        # Use score_threshold=0.0 for prompt-frame masks: accept all objects
        # that SAM3 detected from our bboxes, regardless of initial confidence.
        # The tracker will refine scores during propagation.
        initial_masks = self._outputs_to_masks(
            run_id=run_id,
            outputs=outputs,
            assign_new_obj_to_run=True,
            score_threshold=0.0,
        )

        # --- Prune VG init masks to max_init_masks (keep highest-score) ---
        max_init = self.config.max_init_masks
        if max_init > 0 and len(initial_masks) > max_init:
            # Sort by score descending, keep top N
            sorted_masks = sorted(initial_masks, key=lambda m: m.score, reverse=True)
            keep_masks = sorted_masks[:max_init]
            prune_masks = sorted_masks[max_init:]
            keep_ids = {m.obj_id_local for m in keep_masks}
            pruned_count = 0
            for m in prune_masks:
                if self.remove_object(m.obj_id_local):
                    pruned_count += 1
            logger.info(
                "VG init pruning: %d masks → kept top %d (pruned %d objects from tracker)",
                len(initial_masks), len(keep_masks), pruned_count,
            )
            initial_masks = keep_masks

        run.last_propagated_frame = frame_idx
        run.status = "active"
        # After add_prompt at frame_idx the session's internal cursor is at
        # frame_idx; next propagate_all must start from frame_idx+1.
        self._session_last_propagated_frame = frame_idx
        logger.debug(
            "Bbox prompt at frame %d: 1 visual bbox → %d initial masks (run %d).  "
            "Remaining %d bboxes should be added as point prompts after propagation.",
            frame_idx, len(initial_masks), run_id, max(0, len(boxes_xywh) - 1),
        )
        return run, initial_masks

    def _outputs_to_masks(
        self,
        run_id: int,
        outputs: Dict[str, np.ndarray],
        assign_new_obj_to_run: bool = False,
        require_owner_match: bool = True,
        default_run_id_when_unowned: int = -1,
        score_threshold: Optional[float] = None,
    ) -> List[SAM3SharedMask]:
        out: List[SAM3SharedMask] = []
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        probs = outputs.get("out_probs", np.array([]))
        masks = outputs.get("out_binary_masks", np.array([]))
        if score_threshold is None:
            score_threshold = self.config.score_threshold_detection

        for i, obj_id in enumerate(obj_ids):
            score = float(probs[i]) if i < len(probs) else 0.0
            if score < score_threshold:
                continue
            mask = masks[i] if i < len(masks) else None
            if mask is None:
                continue

            obj_id_int = int(obj_id)
            owner_run_id = self._obj_id_to_run_id.get(obj_id_int)
            if owner_run_id is None:
                if assign_new_obj_to_run:
                    self._obj_id_to_run_id[obj_id_int] = run_id
                    owner_run_id = run_id
                    if owner_run_id in self._runs:
                        self._runs[owner_run_id].obj_ids.add(obj_id_int)
                else:
                    owner_run_id = default_run_id_when_unowned

            if require_owner_match and run_id >= 0 and owner_run_id != run_id:
                continue

            out.append(
                SAM3SharedMask(
                    run_id=owner_run_id,
                    obj_id_local=obj_id_int,
                    mask=mask.astype(bool),
                    score=score,
                )
            )
        return out

    def _force_full_propagation(self) -> None:
        """Clear action_history so the next propagate_in_video uses full VG propagation.

        Normally, ``add_object_point`` records "add" actions which cause the
        next propagation to be ``propagation_partial`` (tracker-only).  Call
        this method after adding frame-0 point prompts but BEFORE calling
        ``propagate_all`` to force a single combined full propagation that
        tracks both bbox and point objects together — eliminating the need
        for a separate ``propagate_new_objects`` pass.
        """
        if self._predictor is None or self._session_id is None:
            return
        session = self._predictor._all_inference_states.get(self._session_id)
        if session is None:
            return
        inference_state = session["state"]
        inference_state["action_history"].clear()

        # Pre-populate hotstart metadata for point-prompt objects.
        # The VG full propagation pipeline's _process_hotstart expects
        # obj_first_frame_idx and trk_keep_alive entries for ALL tracked
        # objects.  Point-prompt objects aren't created by the VG detector
        # so they lack these entries — causing KeyError at runtime.
        tracker_metadata = inference_state.get("tracker_metadata", {})
        rank0_metadata = tracker_metadata.get("rank0_metadata")
        if rank0_metadata is not None:
            obj_first = rank0_metadata.get("obj_first_frame_idx", {})
            trk_alive = rank0_metadata.get("trk_keep_alive", {})
            all_obj_ids = tracker_metadata.get("obj_ids_all_gpu", [])
            # Point-prompt objects need high trk_keep_alive so the VG
            # pipeline's suppression logic doesn't remove them for being
            # "unmatched" with VG detections (they were never VG-detected).
            max_keep_alive = getattr(
                self._predictor.model, "max_trk_keep_alive", 8
            )
            for obj_id in all_obj_ids:
                obj_id = int(obj_id)
                if obj_id not in obj_first:
                    obj_first[obj_id] = 0  # treat as appearing on frame 0
                if obj_id not in trk_alive:
                    trk_alive[obj_id] = max_keep_alive

        logger.debug("Cleared action_history to force full propagation")

    def precompute_backbone_features(self, vg_stride: int = 0) -> None:
        """Pre-compute tracker backbone features for frames that don't need VG.

        Populates the feature cache with backbone-only features so that
        ``run_backbone_and_detection`` will hit its fast path and skip
        the expensive VG detector forward pass during full propagation.

        Args:
            vg_stride: Run full VG detection every N-th frame (for
                reconditioning, hotstart, and stale track cleanup).
                When > 0, frames where ``(frame_idx % vg_stride) == 0``
                are left WITHOUT pre-computed features so the full VG
                pipeline runs on them.  When 0 (default), ALL non-prompt
                frames get backbone-only features (no VG at all).

        Must be called AFTER ``create_run_with_initial_bboxes`` (which
        sets up the session) and BEFORE the first ``propagate_all``.
        """
        import torch

        if self._predictor is None or self._session_id is None:
            return
        session = self._predictor._all_inference_states.get(self._session_id)
        if session is None:
            return
        inference_state = session["state"]
        feature_cache = inference_state["feature_cache"]
        img_batch = inference_state["input_batch"].img_batch
        num_frames = len(img_batch)

        model = self._predictor.model
        detector = model.detector
        sam_mask_decoder = model.tracker.sam_mask_decoder
        device = detector.device

        n_precomputed = 0
        n_vg_frames = 0
        with torch.inference_mode():
            for frame_idx in range(num_frames):
                if frame_idx in feature_cache:
                    continue  # skip frames with existing features (e.g. prompt frame)
                # Leave periodic frames for full VG detection
                if vg_stride > 0 and frame_idx % vg_stride == 0:
                    n_vg_frames += 1
                    continue
                img = img_batch[frame_idx]
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                img = img.to(dtype=torch.float32, device=device).unsqueeze(0)
                bb_out = detector.backbone.forward_image(img)
                sam2_feats = bb_out["sam2_backbone_out"]
                # Apply the same conv projections as run_backbone_and_detection
                tracker_backbone_fpn = [
                    sam_mask_decoder.conv_s0(sam2_feats["backbone_fpn"][0]),
                    sam_mask_decoder.conv_s1(sam2_feats["backbone_fpn"][1]),
                    sam2_feats["backbone_fpn"][2],
                ]
                tracker_backbone_out = {
                    "vision_features": tracker_backbone_fpn[-1],
                    "vision_pos_enc": sam2_feats["vision_pos_enc"],
                    "backbone_fpn": tracker_backbone_fpn,
                }
                feature_cache[frame_idx] = (
                    img_batch[frame_idx],
                    {"tracker_backbone_out": tracker_backbone_out},
                )
                n_precomputed += 1

        logger.info(
            "Pre-computed backbone features for %d/%d frames "
            "(vg_stride=%d, %d frames reserved for full VG)",
            n_precomputed, num_frames, vg_stride, n_vg_frames,
        )

    def propagate_all(self, frame_idx: int) -> List[SAM3SharedMask]:
        """Propagate the shared session to a target frame and return frame masks.

        SAM3's ``action_history`` mechanism switches to fetch-only mode after
        two consecutive ``propagate_in_video`` calls (it assumes forward +
        backward passes are done).  To avoid this, we propagate ALL remaining
        frames in a **single** ``propagate_in_video`` call on the first
        invocation and cache every frame's results.  Subsequent calls just
        return from the cache.
        """
        # Fast path: return from cache if already propagated.
        if frame_idx in self._propagation_cache:
            return self._propagation_cache[frame_idx]

        empty: List[SAM3SharedMask] = []
        if self._predictor is None or self._session_id is None:
            return empty
        if not self._runs:
            return empty
        if frame_idx < 0:
            return empty

        start_idx = max(0, self._session_last_propagated_frame + 1)
        if start_idx > frame_idx:
            return empty

        active_run_ids = {
            run_id for run_id, run in self._runs.items() if run.status == "active"
        }
        if not active_run_ids:
            return empty

        try:
            # Propagate ALL remaining frames in one call so SAM3's
            # action_history only records a single propagation entry.
            for out in self._predictor.propagate_in_video(
                session_id=self._session_id,
                propagation_direction="forward",
                start_frame_idx=start_idx,
                max_frame_num_to_track=None,  # propagate through ALL frames
            ):
                fid = int(out.get("frame_index", -1))
                outputs = out.get("outputs", {})
                self._session_last_propagated_frame = max(
                    self._session_last_propagated_frame,
                    fid,
                )
                for run_id in active_run_ids:
                    run = self._runs.get(run_id)
                    if run is not None and run.status == "active":
                        run.last_propagated_frame = max(
                            run.last_propagated_frame, fid
                        )

                frame_masks: List[SAM3SharedMask] = []
                for m in self._outputs_to_masks(
                    run_id=-1,
                    outputs=outputs,
                    assign_new_obj_to_run=False,
                    require_owner_match=False,
                    default_run_id_when_unowned=-1,
                    score_threshold=self.config.score_threshold_detection,
                ):
                    # Only keep masks owned by active runs; reject unowned
                    # objects (run_id=-1) that SAM3 auto-discovered without
                    # a YOLO bbox prompt — these hurt tracking stability.
                    if m.run_id in active_run_ids:
                        frame_masks.append(m)
                self._propagation_cache[fid] = frame_masks
        except Exception as e:
            logger.warning("SAM3 shared propagation failed: %s", e, exc_info=True)
            self._close_session()
            for run in self._runs.values():
                run.status = "ended"
            self._session_last_propagated_frame = -1
            self._obj_id_to_run_id.clear()
            self._propagation_cache.clear()
            return empty

        # Post-propagation pruning: remove objects that are consistently low-score
        self._prune_low_score_objects()

        return self._propagation_cache.get(frame_idx, empty)

    def add_object_point(
        self,
        frame_idx: int,
        point_xy: tuple,
        label: int = 1,
    ) -> int:
        """Add a new tracked object via point prompt (non-destructive).

        Uses SAM3's ``add_tracker_new_points()`` path which appends a new
        tracker state without calling ``reset_state()``.  Existing objects
        continue to be tracked undisturbed.

        Args:
            frame_idx: Frame where the object is visible.
            point_xy: ``(x, y)`` coordinates of the object center,
                **normalised to [0, 1]** (i.e. ``x / image_width``,
                ``y / image_height``).  SAM3's ``add_tracker_new_points``
                defaults to ``rel_coordinates=True`` and the predictor
                layer does not expose this parameter.
            label: Point label (1 = foreground, 0 = background).

        Returns:
            The obj_id assigned to the new object.
        """
        active_runs = self.active_runs
        if not active_runs:
            raise RuntimeError(
                "No active run. Call create_run_with_initial_bboxes first."
            )
        run = active_runs[0]

        # Pick the next obj_id above all known IDs.
        existing_obj_ids = set(self._obj_id_to_run_id.keys())
        new_obj_id = max(existing_obj_ids, default=-1) + 1

        self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_idx,
            points=[list(point_xy)],
            point_labels=[label],
            obj_id=new_obj_id,
        )

        # Register ownership so propagate_new_objects can include it.
        self._obj_id_to_run_id[new_obj_id] = run.run_id
        # Track discovery frame for bounded backward-window propagation (D4).
        self._obj_id_to_discovery_frame[new_obj_id] = frame_idx
        run.obj_ids.add(new_obj_id)

        logger.debug(
            "Added point prompt at frame %d: obj_id=%d, point=%s (run %d)",
            frame_idx, new_obj_id, point_xy, run.run_id,
        )
        return new_obj_id

    def propagate_new_objects(self) -> None:
        """Run partial propagation for newly added objects, update cache.

        After calling ``add_object_point()`` one or more times, invoke
        this method to propagate the new objects through all frames.
        SAM3's action_history mechanism will select ``propagation_partial``
        mode, running the tracker only for new objects and merging their
        masks with the cached results from the initial full propagation.

        Bounded backward propagation (D4):
            Instead of always restarting from frame 0, start from the
            earliest discovery frame minus ``_discovery_backward_window``
            (default 5).  Frames before that are skipped: a discovery
            object is unlikely to be visible long before it was first
            detected by FastSAM.  Cuts Phase 2c cost ~3x with no recall
            loss for typical videos.
        """
        if self._predictor is None or self._session_id is None:
            return

        active_run_ids = {
            rid for rid, r in self._runs.items() if r.status == "active"
        }
        if not active_run_ids:
            return

        # Compute bounded start frame from discovery frames of pending
        # new objects (those with no propagation cache yet, i.e. frame_idx
        # not in self._propagation_cache for their discovery moment).
        if self._obj_id_to_discovery_frame:
            min_discovery = min(self._obj_id_to_discovery_frame.values())
            start_frame_idx = max(0, min_discovery - self._discovery_backward_window)
        else:
            start_frame_idx = 0

        # Always log the bounding decision (debug-friendly)
        logger.info(
            "propagate_new_objects: start_frame=%d (n_pending=%d, discovery_frames=%s, window=%d)",
            start_frame_idx,
            len(self._obj_id_to_discovery_frame),
            sorted(set(self._obj_id_to_discovery_frame.values()))[:8],
            self._discovery_backward_window,
        )

        # Snapshot which obj_ids are being processed in THIS propagation call.
        # After the call completes, clear them from _obj_id_to_discovery_frame
        # so the NEXT propagate_new_objects call only sees objects added since.
        # Without this, frame-0 supplement bboxes (added during Phase 2a) are
        # forever recorded with frame_idx=0, dragging min_discovery down to 0
        # and silently disabling D4 bounding for the Phase 2c call.
        objs_being_propagated = set(self._obj_id_to_discovery_frame.keys())

        try:
            for out in self._predictor.propagate_in_video(
                session_id=self._session_id,
                propagation_direction="forward",
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=None,
            ):
                fid = int(out.get("frame_index", -1))
                outputs = out.get("outputs", {})

                frame_masks: List[SAM3SharedMask] = []
                for m in self._outputs_to_masks(
                    run_id=-1,
                    outputs=outputs,
                    assign_new_obj_to_run=True,
                    require_owner_match=False,
                    default_run_id_when_unowned=-1,
                    score_threshold=self.config.score_threshold_detection,
                ):
                    if m.run_id in active_run_ids:
                        frame_masks.append(m)
                self._propagation_cache[fid] = frame_masks
        except Exception as e:
            logger.warning(
                "SAM3 partial propagation failed: %s", e, exc_info=True
            )
        finally:
            # Clear discovery records for the objects that have now been
            # propagated.  Future calls (e.g., a second propagate_new_objects
            # in Phase 2c) will only consider objects added since this point.
            for oid in objs_being_propagated:
                self._obj_id_to_discovery_frame.pop(oid, None)

    def remove_object(self, obj_id: int) -> bool:
        """Remove an object from SAM3 tracking to free compute resources.

        Calls SAM3's ``remove_object`` API to stop propagating this object
        and cleans up wrapper state (ownership map, cache, run obj_ids).

        Returns True if the object was successfully removed.
        """
        if self._predictor is None or self._session_id is None:
            return False
        if obj_id not in self._obj_id_to_run_id:
            return False

        try:
            self._predictor.remove_object(
                session_id=self._session_id,
                obj_id=obj_id,
                is_user_action=False,
            )
        except Exception as e:
            logger.debug("remove_object(%d) failed: %s", obj_id, e)
            return False

        # Clean up wrapper state
        run_id = self._obj_id_to_run_id.pop(obj_id, None)
        if run_id is not None and run_id in self._runs:
            self._runs[run_id].obj_ids.discard(obj_id)

        # Remove from propagation cache
        for fid in list(self._propagation_cache.keys()):
            self._propagation_cache[fid] = [
                m for m in self._propagation_cache[fid]
                if m.obj_id_local != obj_id
            ]

        return True

    def _prune_low_score_objects(self) -> int:
        """Remove objects that have low scores across ALL cached frames.

        After full propagation, scan the cache: if an object never reaches
        the config score threshold in any frame, remove it from SAM3's tracker
        to free compute for subsequent partial propagations.

        Returns the number of objects pruned.
        """
        if not self._propagation_cache:
            return 0

        threshold = self.config.score_threshold_detection
        # Collect max score per obj_id across all cached frames
        obj_max_score: Dict[int, float] = {}
        for fid, masks in self._propagation_cache.items():
            for m in masks:
                oid = m.obj_id_local
                obj_max_score[oid] = max(obj_max_score.get(oid, 0.0), m.score)

        # Prune objects whose max score is always below threshold
        pruned = 0
        for oid, max_score in obj_max_score.items():
            if max_score < threshold and oid in self._obj_id_to_run_id:
                if self.remove_object(oid):
                    pruned += 1
        if pruned > 0:
            logger.info(
                "Post-propagation pruning: removed %d objects (all-frame score < %.2f)",
                pruned, threshold,
            )
        return pruned

    def debug_runs(self) -> List[Dict[str, object]]:
        return [
            {
                "run_id": run.run_id,
                "tag": run.tag,
                "status": run.status,
                "start_frame": run.start_frame,
                "last_propagated_frame": run.last_propagated_frame,
                "obj_count": len(run.obj_ids),
                "has_session": self._session_id is not None,
            }
            for run in self._runs.values()
        ]

    @property
    def num_runs(self) -> int:
        return len(self._runs)

    @property
    def active_runs(self) -> List[SAM3SharedTagRun]:
        return [r for r in self._runs.values() if r.status == "active"]

    def _close_session(self) -> None:
        if self._predictor is None or self._session_id is None:
            return
        try:
            self._predictor.close_session(self._session_id)
        except Exception:
            pass
        self._session_id = None

    def _purge_ended_runs(self) -> None:
        """Remove ended runs from _runs to prevent unbounded accumulation."""
        ended_ids = [rid for rid, r in self._runs.items() if r.status == "ended"]
        for rid in ended_ids:
            del self._runs[rid]

    def end_all_runs(self) -> None:
        self._close_session()
        for run in self._runs.values():
            run.status = "ended"
        # Clear ownership map so stale bindings do not leak into future calls.
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()
