"""SAM3 wrapper for ROSE pipeline.

Implements Step 3: multi-run text-guided segmentation and tracking.

Design (from spec):
- Each new tag triggers a new independent run (independent session/memory bank).
- Each run propagates every frame, producing per-object masks with local obj_ids.
- Within a single run, SAM3 handles multi-instance detection automatically
  (score > score_threshold_detection, text prompts enable auto-detection).
- Cross-run deduplication is handled in Step 5 (GlobalIDFusion), not here.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import inspect

import numpy as np

from rose.engine.config.rose_config import SAM3Config

logger = logging.getLogger(__name__)


def _start_session_compat(predictor, resource_path: str, config: SAM3Config) -> dict:
    """Call predictor.start_session() with offload kwargs only if supported."""
    sig = inspect.signature(predictor.start_session)
    kwargs: dict = {"resource_path": resource_path}
    if "offload_state_to_cpu" in sig.parameters:
        kwargs["offload_state_to_cpu"] = config.offload_state_to_cpu
    if "offload_video_to_cpu" in sig.parameters:
        kwargs["offload_video_to_cpu"] = config.offload_video_to_cpu
    return predictor.start_session(**kwargs)


@dataclass
class SAM3Mask:
    """A single mask output from SAM3."""
    run_id: int
    obj_id_local: int
    mask: np.ndarray   # (H, W) bool
    score: float


@dataclass
class SAM3Run:
    """A single SAM3 run triggered by one tag.

    Lifecycle: created -> active -> ended (only at video end).
    """
    run_id: int
    tag: str
    start_frame: int
    session_id: Optional[str] = None
    status: str = "created"  # created | active | ended
    last_propagated_frame: int = -1


class SAM3RunManager:
    """Manages multiple independent SAM3 runs.

    Each tag gets its own run with an independent SAM3 session.
    Runs are never pruned mid-video (natural growth strategy).
    """

    def __init__(self, config: Optional[SAM3Config] = None):
        self.config = config or SAM3Config()
        self._predictor = None
        self._runs: Dict[int, SAM3Run] = {}
        self._next_run_id: int = 0
        self._video_dir: Optional[Path] = None

    def load(self) -> None:
        """Load the SAM3 video predictor. Called lazily."""
        if self._predictor is not None:
            return

        sam3_src = str(Path("rose/vision/sam3").resolve())
        if sam3_src not in sys.path:
            sys.path.insert(0, sam3_src)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3 requires CUDA. No CUDA device found."
            )

        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        # Resolve checkpoint from config.model_path
        checkpoint_path = self._resolve_checkpoint()

        self._predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            bpe_path=None,
            has_presence_token=True,
            geo_encoder_use_img_cross_attn=True,
            strict_state_dict_loading=True,
            async_loading_frames=False,
            apply_temporal_disambiguation=True,
        )

        # Override internal score threshold (SAM3 builder hardcodes 0.5)
        self._predictor.model.score_threshold_detection = (
            self.config.score_threshold_detection
        )
        self._predictor.model.trim_past_non_cond_mem_for_eval = (
            self.config.trim_past_non_cond_mem_for_eval
        )

        logger.info(
            "SAM3 predictor loaded from %s (score_thresh=%.2f)",
            checkpoint_path or "HuggingFace",
            self.config.score_threshold_detection,
        )

    def _resolve_checkpoint(self) -> Optional[str]:
        """Resolve SAM3 checkpoint path from config.model_path.

        Returns:
            Absolute path to .pt file, or None if model_path is not set
            / doesn't exist (falls back to HF download).
        """
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
        """Set the directory containing video frames (JPEG files).

        SAM3 requires a directory of frames or an MP4 path for
        start_session(). Call this before create_run / propagate.
        """
        self._video_dir = Path(video_dir)

    def create_run(self, tag: str, frame_idx: int) -> SAM3Run:
        """Create a new run for a newly discovered tag.

        This starts a new SAM3 session and adds a text prompt on the given frame.

        Args:
            tag: The RAM++ tag string (e.g. "car").
            frame_idx: The frame where this tag was first seen.

        Returns:
            The created SAM3Run.
        """
        self.load()

        run_id = self._next_run_id
        self._next_run_id += 1

        run = SAM3Run(
            run_id=run_id,
            tag=tag,
            start_frame=frame_idx,
        )

        # Start a new session for this run
        if self._video_dir is None:
            raise RuntimeError("Call set_video_dir() before create_run()")

        result = _start_session_compat(
            self._predictor, str(self._video_dir), self.config,
        )
        run.session_id = result["session_id"]

        # Add text prompt on the trigger frame
        self._predictor.add_prompt(
            session_id=run.session_id,
            frame_idx=frame_idx,
            text=tag,
        )
        run.last_propagated_frame = frame_idx
        run.status = "active"

        self._runs[run_id] = run
        logger.debug("Created SAM3 run %d for tag '%s' at frame %d", run_id, tag, frame_idx)
        return run

    def create_run_with_initial_masks(
        self,
        tag: str,
        frame_idx: int,
    ) -> Tuple[SAM3Run, List[SAM3Mask]]:
        """Create a run and return initial detections from add_prompt()."""
        self.load()

        run_id = self._next_run_id
        self._next_run_id += 1

        run = SAM3Run(
            run_id=run_id,
            tag=tag,
            start_frame=frame_idx,
        )

        if self._video_dir is None:
            raise RuntimeError("Call set_video_dir() before create_run_with_initial_masks()")

        session = _start_session_compat(
            self._predictor, str(self._video_dir), self.config,
        )
        run.session_id = session["session_id"]

        prompt_result = self._predictor.add_prompt(
            session_id=run.session_id,
            frame_idx=frame_idx,
            text=tag,
        )
        outputs = prompt_result.get("outputs", {})
        initial_masks = self._outputs_to_masks(run.run_id, outputs)

        run.last_propagated_frame = frame_idx
        run.status = "active"
        self._runs[run_id] = run
        logger.debug(
            "Created SAM3 run %d with %d initial masks for tag '%s' at frame %d",
            run_id,
            len(initial_masks),
            tag,
            frame_idx,
        )
        return run, initial_masks

    def create_run_with_initial_boxes(
        self,
        box_xywh: List[float],
        frame_idx: int,
        tag: str = "box",
        box_label: int = 1,
    ) -> Tuple[SAM3Run, List[SAM3Mask]]:
        """Create a run with a single box visual prompt and return initial masks."""
        self.load()

        run_id = self._next_run_id
        self._next_run_id += 1

        run = SAM3Run(
            run_id=run_id,
            tag=tag,
            start_frame=frame_idx,
        )

        if self._video_dir is None:
            raise RuntimeError("Call set_video_dir() before create_run_with_initial_boxes()")

        session = _start_session_compat(
            self._predictor, str(self._video_dir), self.config,
        )
        run.session_id = session["session_id"]

        prompt_result = self._predictor.add_prompt(
            session_id=run.session_id,
            frame_idx=frame_idx,
            text="visual",
            bounding_boxes=[box_xywh],
            bounding_box_labels=[box_label],
        )
        outputs = prompt_result.get("outputs", {})
        initial_masks = self._outputs_to_masks(run.run_id, outputs)

        run.last_propagated_frame = frame_idx
        run.status = "active"
        self._runs[run_id] = run
        logger.debug(
            "Created SAM3 run %d with %d initial masks for box '%s' at frame %d",
            run_id,
            len(initial_masks),
            tag,
            frame_idx,
        )
        return run, initial_masks

    def propagate_all(self, frame_idx: int) -> List[SAM3Mask]:
        """Propagate all active runs for the given frame.

        This calls propagate_in_video for each active run, collecting masks.
        Each run is propagated forward one frame at a time.

        Args:
            frame_idx: Current frame index.

        Returns:
            List of SAM3Mask from all runs (may have cross-run overlaps).
        """
        all_masks: List[SAM3Mask] = []

        for run in self._runs.values():
            if run.status != "active" or run.session_id is None:
                continue
            if frame_idx < run.start_frame:
                continue

            start_idx = max(run.start_frame, run.last_propagated_frame + 1)
            if start_idx > frame_idx:
                continue

            try:
                # Propagate only unseen frames for this run.
                for out in self._predictor.propagate_in_video(
                    session_id=run.session_id,
                    propagation_direction="forward",
                    start_frame_idx=start_idx,
                    max_frame_num_to_track=frame_idx - start_idx + 1,
                ):
                    fid = int(out.get("frame_index", -1))
                    outputs = out.get("outputs", {})
                    # only advance to the requested frame; SAM3 may return one
                    # extra frame due to tracking-order semantics.
                    if fid <= frame_idx:
                        run.last_propagated_frame = max(run.last_propagated_frame, fid)
                    if fid != frame_idx:
                        continue
                    all_masks.extend(self._outputs_to_masks(run.run_id, outputs))
            except Exception as e:
                logger.warning("SAM3 propagation failed for run %d: %s", run.run_id, e)
                try:
                    self._predictor.close_session(run.session_id)
                except Exception:
                    pass
                run.session_id = None
                run.status = "ended"

        return all_masks

    def end_all_runs(self) -> None:
        """Close all sessions. Called at video end."""
        if self._predictor is None:
            return
        for run in self._runs.values():
            if run.session_id is not None:
                try:
                    self._predictor.close_session(run.session_id)
                except Exception:
                    pass
                run.status = "ended"
                run.session_id = None

    @property
    def num_runs(self) -> int:
        return len(self._runs)

    @property
    def active_runs(self) -> List[SAM3Run]:
        return [r for r in self._runs.values() if r.status == "active"]

    def debug_runs(self) -> List[Dict[str, object]]:
        """Return lightweight run diagnostics for logging and troubleshooting."""
        return [
            {
                "run_id": run.run_id,
                "tag": run.tag,
                "status": run.status,
                "start_frame": run.start_frame,
                "last_propagated_frame": run.last_propagated_frame,
                "has_session": run.session_id is not None,
            }
            for run in self._runs.values()
        ]

    def _outputs_to_masks(self, run_id: int, outputs: Dict[str, np.ndarray]) -> List[SAM3Mask]:
        """Convert raw SAM3 outputs to filtered SAM3Mask list."""
        out: List[SAM3Mask] = []
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        probs = outputs.get("out_probs", np.array([]))
        masks = outputs.get("out_binary_masks", np.array([]))

        for i, obj_id in enumerate(obj_ids):
            score = float(probs[i]) if i < len(probs) else 0.0
            if score < self.config.score_threshold_detection:
                continue
            mask = masks[i] if i < len(masks) else None
            if mask is None:
                continue
            out.append(
                SAM3Mask(
                    run_id=run_id,
                    obj_id_local=int(obj_id),
                    mask=mask.astype(bool),
                    score=score,
                )
            )
        return out
