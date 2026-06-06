"""End-to-end ROSE orchestrator.

This module wraps the vision models (DA3, FastSAM, SAM3) and feeds
their per-frame outputs into the model-agnostic ROSEPipeline.

Architecture:
    Phase 1 (batch):  DA3.infer_batch(all_frames) → depth/K/T_wc
                      with inter-frame consistent poses.
    Phase 2 (two-pass GPU):
        2a: FastSAM frame 0 → SAM3 init + full propagation (cache all frames)
        2b: FastSAM per-frame → discover new objects via IoU comparison
        2c: SAM3 partial propagation for new objects (merge with cached)
        2d: Build FastFrameInput per frame → CPU worker thread

Usage:
    config = ROSEConfig()
    e2e = ROSEEndToEnd(config)
    result = e2e.process_video("video.mp4", "What is in front of the car?")
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.pipeline.rose_pipeline import (
    ROSEPipeline,
    FastFrameInput,
    FastLocalDetection,
)
from rose.vision.perception.da3_wrapper import DA3Wrapper, compute_chunks
from rose.vision.perception.fastsam_wrapper import FastSAMWrapper
from rose.vision.perception.sam3_shared_session_wrapper import (
    SAM3SharedMask,
    SAM3SharedSessionManager,
)

logger = logging.getLogger(__name__)

_SENTINEL = None  # poison pill for CPU worker queue


# ------------------------------------------------------------------
# Two-pass discovery helpers
# ------------------------------------------------------------------

def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Return (row, col) centroid of a boolean mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of the same shape."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _any_mask_iou_above(query_mask: np.ndarray, cached_masks, threshold: float) -> bool:
    """Check if query_mask overlaps with any cached SAM3 mask above threshold."""
    for m in cached_masks:
        if _mask_iou(query_mask, m.mask) >= threshold:
            return True
    return False


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (y_min, x_min, y_max, x_max) bounding box of a boolean mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two (y_min, x_min, y_max, x_max) bounding boxes."""
    y0 = max(a[0], b[0])
    x0 = max(a[1], b[1])
    y1 = min(a[2], b[2])
    x1 = min(a[3], b[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = (y1 - y0) * (x1 - x0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _mask_main_component_fraction(mask: np.ndarray) -> float:
    """Largest connected-component area divided by total mask area.

    1.0 = single coherent blob.  ~0.5 = two pieces of similar size
    (likely SAM3 fused two distinct objects into one mask).

    OPT: crop to bbox before connectedComponents — runs on the masked
    region only (often ~10% of full image) instead of the full frame.
    """
    if not mask.any():
        return 0.0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.flatnonzero(rows)
    c_idx = np.flatnonzero(cols)
    if r_idx.size == 0 or c_idx.size == 0:
        return 0.0
    r0, r1 = int(r_idx[0]), int(r_idx[-1]) + 1
    c0, c1 = int(c_idx[0]), int(c_idx[-1]) + 1
    sub = np.ascontiguousarray(mask[r0:r1, c0:c1]).astype(np.uint8)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(
        sub, connectivity=8,
    )
    if num <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = float(areas.sum())
    if total <= 0:
        return 0.0
    return float(areas.max()) / total


def _mask_is_fragmented(mask: np.ndarray, frac_thresh: float = 0.6) -> bool:
    """True when the main component covers < ``frac_thresh`` of the mask.

    Two FAST-PATHs that together catch ~95% of legitimate (single-blob)
    masks without ever calling cv2.connectedComponents:

      (a) Dense-in-bbox (mask area / bbox area > 0.30): a single-component
          mask with this density cannot be "fragmented into comparable
          pieces" because the largest CC must dominate.

      (b) Centroid-inside-mask: for a single connected blob the area-
          centroid pixel is inside the blob.  For two well-separated
          components the centroid falls in the gap (outside both).

    Only when BOTH fast-paths are inconclusive do we run the expensive
    cv2.connectedComponentsWithStats fallback.  On typical videos this
    reduces the fragment check from ~4s to <0.2s.
    """
    if not mask.any():
        return False
    # FAST-PATH 1: dense-in-bbox
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return False
    r_idx = np.flatnonzero(rows)
    c_idx = np.flatnonzero(cols)
    r0, r1 = int(r_idx[0]), int(r_idx[-1])
    c0, c1 = int(c_idx[0]), int(c_idx[-1])
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    mask_area = int(mask.sum())
    if bh * bw <= 0:
        return False
    if mask_area / (bh * bw) > 0.30:
        return False
    # FAST-PATH 2: centroid-inside test (cheap)
    sub = mask[r0:r1 + 1, c0:c1 + 1]
    ys, xs = np.where(sub)
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))
    if sub[cy, cx]:
        # Centroid pixel is inside mask → single coherent blob (very likely).
        return False
    # Inconclusive — fall through to the slow CC count.
    return _mask_main_component_fraction(mask) < frac_thresh


def _mask_is_low_texture(
    image: np.ndarray,
    mask: np.ndarray,
    std_thresh: float = 20.0,
    edge_density_thresh: float = 0.015,
) -> bool:
    """True when the masked region is uniform-color + edge-poor.

    Catches ground / sky / wall fragments that FastSAM segments as objects.
    Both interior color std AND edge density (Canny inside mask) must be low
    — a real textured object will fail one of the two.
    """
    if not mask.any():
        return False
    pixels = image[mask]
    if pixels.size == 0:
        return False
    if float(pixels.std()) >= std_thresh:
        return False
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin = int(np.argmax(rows))
    rmax = mask.shape[0] - 1 - int(np.argmax(rows[::-1]))
    cmin = int(np.argmax(cols))
    cmax = mask.shape[1] - 1 - int(np.argmax(cols[::-1]))
    region = image[rmin:rmax + 1, cmin:cmax + 1]
    region_mask = mask[rmin:rmax + 1, cmin:cmax + 1]
    if region.size == 0 or not region_mask.any():
        return False
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_pixels = int(((edges > 0) & region_mask).sum())
    mask_area = int(region_mask.sum())
    if mask_area == 0:
        return False
    edge_density = edge_pixels / mask_area
    return edge_density < edge_density_thresh


def _crop_is_uniform(crop: np.ndarray, std_thresh: float = 12.0) -> bool:
    """True when the crop is a near-uniform color block (empty / background)."""
    if crop.size == 0:
        return True
    return float(crop.std()) < std_thresh


def _crop_object_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: float = 0.2,
    size: int = 256,
) -> np.ndarray:
    """Crop an object region from the image using the mask's bounding box.

    Computes the bounding box of the mask, adds padding, and crops the
    full rectangular region (preserving background context for VLM
    recognition).  Resizes to (size, size).

    Args:
        image: (H, W, 3) uint8 RGB image.
        mask: (H, W) boolean mask.
        padding: Padding ratio relative to bbox dimensions.
        size: Target square size in pixels.

    Returns:
        (size, size, 3) uint8 RGB bbox crop with background context.
    """
    h, w = mask.shape[:2]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.zeros((size, size, 3), dtype=np.uint8)

    y_min = int(np.argmax(rows))
    y_max = h - 1 - int(np.argmax(rows[::-1]))
    x_min = int(np.argmax(cols))
    x_max = w - 1 - int(np.argmax(cols[::-1]))
    bh = y_max - y_min + 1
    bw = x_max - x_min + 1
    pad_y = int(bh * padding)
    pad_x = int(bw * padding)
    y0 = max(0, y_min - pad_y)
    y1 = min(h, y_max + pad_y + 1)
    x0 = max(0, x_min - pad_x)
    x1 = min(w, x_max + pad_x + 1)

    crop = image[y0:y1, x0:x1].copy()

    if crop.shape[0] > 0 and crop.shape[1] > 0:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    else:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    return crop


def _make_chunk_frame_dir(
    frame_dir: Path,
    chunk_start: int,
    chunk_end: int,
) -> Path:
    """Create a temp directory with symlinks to only the chunk's frames.

    SAM3's ``load_video_frames_from_image_folder`` reads ALL images in a
    directory.  We create a subdirectory with symlinks renumbered as
    ``000000.jpg``, ``000001.jpg``, ... so SAM3 sees them as a contiguous
    short video.
    """
    chunk_dir = Path(
        tempfile.mkdtemp(prefix=f"sam3_chunk_{chunk_start}_{chunk_end}_")
    )
    for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
        src = frame_dir / f"{global_idx:06d}.jpg"
        dst = chunk_dir / f"{local_idx:06d}.jpg"
        dst.symlink_to(src)
    return chunk_dir


def _stitch_chunk_ids(
    chunk_caches: List[Dict[int, List[SAM3SharedMask]]],
    chunks: List[Tuple[int, int]],
    iou_threshold: float = 0.3,
) -> None:
    """Remap object identities across chunks using mask IoU in overlap frames.

    For each pair of adjacent chunks, compute average mask IoU in the overlap
    region and greedily match objects.  Then apply a Union-Find transitive
    closure so all chunks reference a single canonical identity per object.

    Modifies ``chunk_caches`` in-place (rewrites ``run_id`` and
    ``obj_id_local`` on matched masks).
    """
    if len(chunk_caches) < 2:
        return

    # Union-Find on (run_id, obj_id_local) keys
    parent: Dict[Tuple, Tuple] = {}

    def _find(x: Tuple) -> Tuple:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(keep: Tuple, drop: Tuple) -> None:
        rk, rd = _find(keep), _find(drop)
        if rk != rd:
            parent[rd] = rk

    for ci in range(len(chunks) - 1):
        overlap_start = chunks[ci + 1][0]
        overlap_end = chunks[ci][1]
        if overlap_start >= overlap_end:
            continue
        overlap_frames = list(range(overlap_start, overlap_end))

        # Accumulate per-pair mean similarity across overlap frames.
        # Use max(mask_IoU, bbox_IoU) to catch objects with matching
        # bounding boxes but different mask boundaries.
        pair_sim_sums: Dict[Tuple[Tuple, Tuple], float] = {}
        pair_sim_counts: Dict[Tuple[Tuple, Tuple], int] = {}

        for fidx in overlap_frames:
            masks_a = chunk_caches[ci].get(fidx, [])
            masks_b = chunk_caches[ci + 1].get(fidx, [])
            for ma in masks_a:
                key_a = (ma.run_id, ma.obj_id_local)
                bbox_a = _mask_bbox(ma.mask)
                for mb in masks_b:
                    key_b = (mb.run_id, mb.obj_id_local)
                    m_iou = _mask_iou(ma.mask, mb.mask)
                    b_iou = _bbox_iou(bbox_a, _mask_bbox(mb.mask))
                    sim = max(m_iou, b_iou)
                    pair_key = (key_a, key_b)
                    pair_sim_sums[pair_key] = pair_sim_sums.get(pair_key, 0.0) + sim
                    pair_sim_counts[pair_key] = pair_sim_counts.get(pair_key, 0) + 1

        # Greedy 1:1 matching by descending mean similarity.
        mean_ious = [
            (pair_sim_sums[k] / pair_sim_counts[k], k)
            for k in pair_sim_sums
        ]
        mean_ious.sort(key=lambda x: x[0], reverse=True)

        matched_a: set = set()
        matched_b: set = set()
        for mean_iou, (key_a, key_b) in mean_ious:
            if mean_iou < iou_threshold:
                break
            if key_a in matched_a or key_b in matched_b:
                continue
            _union(key_a, key_b)
            matched_a.add(key_a)
            matched_b.add(key_b)
            logger.debug(
                "Stitch chunk %d↔%d: %s → %s (IoU=%.3f)",
                ci, ci + 1, key_b, key_a, mean_iou,
            )

    # Apply canonical remapping to all chunk caches.
    for cc in chunk_caches:
        for fidx, masks in cc.items():
            for m in masks:
                key = (m.run_id, m.obj_id_local)
                canonical = _find(key)
                if canonical != key:
                    m.run_id = canonical[0]
                    m.obj_id_local = canonical[1]


def _merge_chunk_caches(
    chunk_caches: List[Dict[int, List[SAM3SharedMask]]],
) -> Dict[int, List[SAM3SharedMask]]:
    """Merge per-chunk mask caches into a single cache.

    For overlap frames (present in multiple chunks), keep the highest-score
    mask per unique ``(run_id, obj_id_local)`` key.
    """
    merged: Dict[int, List[SAM3SharedMask]] = {}
    for cc in chunk_caches:
        for fidx, masks in cc.items():
            if fidx not in merged:
                merged[fidx] = list(masks)
            else:
                existing_keys = {}
                for i, m in enumerate(merged[fidx]):
                    k = (m.run_id, m.obj_id_local)
                    prev = existing_keys.get(k)
                    if prev is None or m.score > merged[fidx][prev].score:
                        existing_keys[k] = i
                for m in masks:
                    k = (m.run_id, m.obj_id_local)
                    if k in existing_keys:
                        idx = existing_keys[k]
                        if m.score > merged[fidx][idx].score:
                            merged[fidx][idx] = m
                    else:
                        existing_keys[k] = len(merged[fidx])
                        merged[fidx].append(m)
    return merged


@dataclass
class _KeyframeDirMixin:
    """Shared cleanup logic for result objects that own a keyframe directory.

    The caller **must** call :meth:`cleanup` when the keyframe images
    referenced by ``four_dsg_dict["metadata"]["visual_anchor"]`` are no
    longer needed.  Preferred usage::

        result = e2e.process_video(...)
        try:
            ...  # use result.four_dsg_dict, result.scene_json, etc.
        finally:
            result.cleanup()
    """
    keyframe_dir: Optional[Path] = None

    def cleanup(self) -> None:
        """Remove the temporary keyframe directory."""
        if self.keyframe_dir is not None:
            shutil.rmtree(self.keyframe_dir, ignore_errors=True)
            self.keyframe_dir = None


@dataclass
class ROSEE2EResult(_KeyframeDirMixin):
    """End-to-end pipeline result (process_video)."""
    answer: str = ""
    scene_json: str = ""
    four_dsg_dict: Dict = field(default_factory=dict)
    step01_trace: List[Dict] = field(default_factory=list)
    keyframe_dir: Optional[Path] = None


@dataclass
class ROSE4DSGResult(_KeyframeDirMixin):
    """4DSG-only pipeline result (build_4dsg_from_video)."""
    four_dsg_dict: Dict = field(default_factory=dict)
    scene_json: str = ""
    keyframe_dir: Optional[Path] = None


class ROSEEndToEnd:
    """End-to-end orchestrator: video -> DA3/FastSAM/SAM3 -> ROSEPipeline -> VLM."""

    def __init__(self, config: Optional[ROSEConfig] = None):
        self.config = config or ROSEConfig()
        self._da3 = DA3Wrapper(self.config.da3)
        self._fastsam = FastSAMWrapper(self.config.fastsam)
        self._sam3 = SAM3SharedSessionManager(self.config.sam3)
        self._vlm_client = None
        # SAM3 init state (reset per video)
        self._sam3_initialized: bool = False
        # 2026-05-31: perception now delegates to WarmModelPool (warm_server) — the
        # SAME path the production warm_client serves — so ROSEEndToEnd honours ALL
        # the speed levers (stride / incremental_late_prop / flow_warp / early_dedup /
        # multiplex / FA3) instead of the divergent legacy _process_sam3_chunk path,
        # which honoured none of them. The legacy path is kept below (unused) for
        # offline/debug reference. Pool is built lazily on first call (one model load).
        self._pool = None

    def _get_pool(self):
        """Lazily build + warm the WarmModelPool (production perception path).

        Imported here (not at module top) to avoid the warm_server↔rose_e2e
        circular import (warm_server imports this module).
        """
        if self._pool is None:
            from rose.engine.server.warm_server import WarmModelPool
            pool = WarmModelPool(self.config)
            pool.load_all()
            try:
                pool.warmup_cuda()
            except Exception as e:  # warmup is best-effort
                logger.warning("WarmModelPool warmup_cuda failed (non-fatal): %s", e)
            pool._status = "ready"
            self._pool = pool
        return self._pool

    @staticmethod
    def _gc_gpu(tag: str = "") -> None:
        """Force-release cached CUDA memory between pipeline phases."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared (%s)", tag)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # SAM3 chunked inference
    # ------------------------------------------------------------------

    def _estimate_sam3_chunk_size(self, n_frames: int) -> int:
        """Estimate max frames per SAM3 chunk from available CPU memory.

        Memory model (empirical, with offload_state_to_cpu=True):
          base ≈ 4 GB (model + CUDA context)
          per_frame ≈ 0.55 GB (image tensor + cached state)
        Targets 60 % of available CPU memory.
        """
        base_gb = 4.0
        per_frame_gb = 0.55
        fallback = 50

        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                        avail_gb = avail_kb / (1024 * 1024)
                        break
                else:
                    avail_gb = -1.0
        except (OSError, ValueError):
            avail_gb = -1.0

        if avail_gb <= 0:
            logger.info("Cannot query CPU memory; SAM3 chunk_size fallback=%d", fallback)
            return min(fallback, n_frames)

        safe_n = int((avail_gb * 0.6 - base_gb) / per_frame_gb)
        safe_n = max(self.config.sam3.chunk_overlap + 2, safe_n)
        safe_n = min(safe_n, n_frames)
        logger.info(
            "SAM3 auto chunk_size=%d (avail_cpu=%.1fGB, per_frame=%.2fGB, n=%d)",
            safe_n, avail_gb, per_frame_gb, n_frames,
        )
        return safe_n

    def _process_sam3_chunk(
        self,
        frames: List[np.ndarray],
        frame_dir: Path,
        chunk_start: int,
        chunk_end: int,
        chunk_idx: int,
        prev_overlap_masks: Optional[Dict[int, List[SAM3SharedMask]]] = None,
        discovery_budget_remaining: Optional[List[int]] = None,
    ) -> Dict[int, List[SAM3SharedMask]]:
        """Run Phase 2a/2b/2c on a single chunk of frames.

        Creates a temporary frame directory with symlinks, initialises a
        fresh SAM3 session, runs FastSAM+SAM3 two-pass detection and
        propagation, and returns a mask cache indexed by **global** frame
        indices with run_ids namespaced by *chunk_idx*.

        Args:
            prev_overlap_masks: Masks from the previous chunk in the overlap
                region, indexed by **global** frame index.  Used during
                discovery to suppress re-detection of already-tracked objects.
            discovery_budget_remaining: Single-element list ``[N]`` shared
                across chunks.  Decremented in-place as new objects are
                discovered.  ``None`` means use per-chunk budget from config.
        """
        n_local = chunk_end - chunk_start
        logger.info(
            "SAM3 chunk %d: frames [%d, %d) (%d frames)",
            chunk_idx, chunk_start, chunk_end, n_local,
        )

        # Create chunk directory with symlinks
        chunk_dir = _make_chunk_frame_dir(frame_dir, chunk_start, chunk_end)
        try:
            self._sam3.set_video_dir(chunk_dir)

            # -- Phase 2a: Init + Full Propagation --
            # Try chunk_start first; if FastSAM finds nothing, scan
            # subsequent frames and pick the one with the most detections.
            MAX_INIT_FRAME_ATTEMPTS = min(n_local, 10)
            init_local_frame = 0  # local index within chunk
            _raw0 = self._fastsam.detect(frames[chunk_start])
            # P1-1 + P1-2: drop fragmented & low-texture init detections.
            fastsam_dets_0 = [
                d for d in _raw0
                if not _mask_is_fragmented(d.mask)
                and not _mask_is_low_texture(frames[chunk_start], d.mask)
            ]

            if not fastsam_dets_0:
                best_offset = -1
                best_dets = None
                best_count = 0
                for try_offset in range(1, MAX_INIT_FRAME_ATTEMPTS):
                    raw = self._fastsam.detect(frames[chunk_start + try_offset])
                    dets = [
                        d for d in raw
                        if not _mask_is_fragmented(d.mask)
                        and not _mask_is_low_texture(frames[chunk_start + try_offset], d.mask)
                    ]
                    if len(dets) > best_count:
                        best_offset = try_offset
                        best_dets = dets
                        best_count = len(dets)
                if best_dets is not None:
                    init_local_frame = best_offset
                    fastsam_dets_0 = best_dets
            cur_bboxes = (
                [list(d.bbox_xywh_norm) for d in fastsam_dets_0]
                if fastsam_dets_0
                else []
            )

            chunk_initialised = False
            if cur_bboxes:
                _, init_masks = self._sam3.create_run_with_initial_bboxes(
                    boxes_xywh=cur_bboxes,
                    box_labels=[1] * len(cur_bboxes),
                    frame_idx=init_local_frame,
                    tag=f"chunk{chunk_idx}_bbox",
                )
                chunk_initialised = True
                if init_local_frame > 0:
                    logger.info(
                        "  chunk %d: FastSAM fallback — using frame %d (local %d) as init",
                        chunk_idx, chunk_start + init_local_frame, init_local_frame,
                    )
                logger.info(
                    "  chunk %d: SAM3 init with %d bboxes → %d masks",
                    chunk_idx, len(cur_bboxes), len(init_masks),
                )

                # Apply inference acceleration settings from config
                sam3_cfg = self.config.sam3
                if sam3_cfg.retain_backbone_cache:
                    self._sam3._predictor.model.retain_feature_cache = True
                if sam3_cfg.num_maskmem != 7:
                    self._sam3._predictor.model.tracker.num_maskmem = sam3_cfg.num_maskmem
                if sam3_cfg.memory_temporal_stride != 1:
                    self._sam3._predictor.model.tracker.memory_temporal_stride_for_eval = sam3_cfg.memory_temporal_stride

                # Free FastSAM GPU before heavy SAM3 propagation
                self._fastsam.unload()
                self._gc_gpu(f"chunk{chunk_idx}-pre-propagate")

                # Pre-compute backbone features (backbone-only, no VG detection head)
                # so that full propagation skips the expensive VG forward pass.
                # vg_stride>0 leaves every N-th frame for full VG (reconditioning).
                if sam3_cfg.retain_backbone_cache:
                    self._sam3.precompute_backbone_features(
                        vg_stride=sam3_cfg.vg_stride,
                    )

                # Full propagation — caches all chunk frames
                for fidx in range(n_local):
                    self._sam3.propagate_all(fidx)

                # Add remaining init-frame bboxes as point prompts, filtering
                # out those that overlap with already-tracked SAM3 masks.
                MAX_FRAME0_POINTS = 8
                frame0_point_count = 0
                if len(cur_bboxes) > 1:
                    remaining = list(range(1, len(cur_bboxes)))
                    remaining.sort(
                        key=lambda idx: fastsam_dets_0[idx].mask.sum(),
                        reverse=True,
                    )
                    for i in remaining:
                        if frame0_point_count >= MAX_FRAME0_POINTS:
                            break
                        if init_masks and _any_mask_iou_above(
                            fastsam_dets_0[i].mask, init_masks, 0.5,
                        ):
                            continue
                        bx, by, bw, bh = cur_bboxes[i]
                        self._sam3.add_object_point(
                            init_local_frame, (bx + bw / 2.0, by + bh / 2.0)
                        )
                        frame0_point_count += 1

                # Propagate init-frame point prompts NOW so discovery (Phase 2b)
                # can see all tracked objects in the cache.
                if frame0_point_count > 0:
                    logger.info(
                        "  chunk %d: propagating %d init-frame point prompts",
                        chunk_idx, frame0_point_count,
                    )
                    self._sam3.propagate_new_objects()
            else:
                logger.warning(
                    "  chunk %d: FastSAM detected 0 objects on first %d frames (from frame %d)",
                    chunk_idx, MAX_INIT_FRAME_ATTEMPTS, chunk_start,
                )

            # -- Phase 2b: Discovery --
            new_obj_count = 0
            discovery_thresh = self.config.fastsam.discovery_iou_thresh
            max_disc = self.config.sam3.max_discovery_per_frame
            min_mask_frac = self.config.sam3.discovery_min_mask_frac
            # Use shared budget if provided, else per-chunk budget from config
            if discovery_budget_remaining is not None:
                _budget_is_shared = True
            else:
                _budget_is_shared = False
                max_total = self.config.sam3.max_discovery_total
            # Cap discovery budget based on max_active_tracks
            max_active = self.config.sam3.max_active_tracks
            if max_active > 0:
                current_active = len(self._sam3._obj_id_to_run_id)
                active_headroom = max(0, max_active - current_active)
                if _budget_is_shared:
                    discovery_budget_remaining[0] = min(
                        discovery_budget_remaining[0], active_headroom
                    )
                else:
                    max_total = min(max_total, active_headroom) if max_total > 0 else active_headroom
                logger.info(
                    "  chunk %d: discovery cap — %d active objects, headroom %d",
                    chunk_idx, current_active, active_headroom,
                )
            discovery_stride = self.config.sam3.full_propagation_stride
            if chunk_initialised and self._sam3.active_runs:
                for local_fidx in range(init_local_frame + 1, n_local):
                    # Stride discovery to match full propagation stride —
                    # skipped frames have reused masks that are valid reference
                    # but running FastSAM on every frame is wasteful.
                    if discovery_stride > 1 and local_fidx % discovery_stride != 0:
                        continue
                    global_fidx = chunk_start + local_fidx
                    # Check budget
                    if _budget_is_shared:
                        if discovery_budget_remaining[0] <= 0:
                            break
                    else:
                        if max_total > 0 and new_obj_count >= max_total:
                            break
                    fastsam_dets = self._fastsam.detect(frames[global_fidx])
                    # FastSAM returns detections sorted by mask area descending.
                    # After min/max_mask_frac filtering, larger remaining objects
                    # are more likely to be foreground subjects (person, animal,
                    # vehicle) than small fragments.  Keep area ordering.
                    cached_masks = self._sam3.propagate_all(local_fidx)
                    # Combine SAM3 cache with previous chunk's overlap masks
                    ref_masks = list(cached_masks)
                    if prev_overlap_masks is not None and global_fidx in prev_overlap_masks:
                        ref_masks.extend(prev_overlap_masks[global_fidx])
                    frame_disc = 0
                    for det in fastsam_dets:
                        if max_disc > 0 and frame_disc >= max_disc:
                            break
                        if _budget_is_shared:
                            if discovery_budget_remaining[0] <= 0:
                                break
                        else:
                            if max_total > 0 and new_obj_count >= max_total:
                                break
                        # Skip tiny detections (noise / spurious segments)
                        if min_mask_frac > 0:
                            h, w = det.mask.shape[:2]
                            if det.mask.sum() < min_mask_frac * h * w:
                                continue
                        # P1-1: skip fragmented FastSAM masks (multiple comparable blobs)
                        if _mask_is_fragmented(det.mask):
                            continue
                        # P1-2: skip low-texture ground / sky / wall segments
                        if _mask_is_low_texture(frames[global_fidx], det.mask):
                            continue
                        if not _any_mask_iou_above(
                            det.mask, ref_masks, discovery_thresh
                        ):
                            cy_px, cx_px = _mask_centroid(det.mask)
                            h, w = det.mask.shape[:2]
                            self._sam3.add_object_point(
                                local_fidx, (cx_px / w, cy_px / h)
                            )
                            new_obj_count += 1
                            frame_disc += 1
                            if _budget_is_shared:
                                discovery_budget_remaining[0] -= 1

            self._fastsam.unload()
            self._gc_gpu(f"chunk{chunk_idx}-post-discovery")

            # -- Phase 2c: Partial Propagation for discovery objects --
            # Frame-0 point prompts were already propagated before discovery,
            # so only discovery-added objects need propagation here.
            if new_obj_count > 0:
                logger.info(
                    "  chunk %d: partial propagation for %d discovery objects",
                    chunk_idx, new_obj_count,
                )
                self._sam3.propagate_new_objects()

            # -- Collect masks and re-index to global frame indices --
            mask_cache: Dict[int, List[SAM3SharedMask]] = {}
            for local_fidx in range(n_local):
                global_fidx = chunk_start + local_fidx
                raw_masks = list(self._sam3.propagate_all(local_fidx))
                # Namespace run_ids with chunk_idx to ensure cross-chunk uniqueness
                for m in raw_masks:
                    m.run_id = (chunk_idx, m.run_id)
                mask_cache[global_fidx] = raw_masks

            if _budget_is_shared:
                logger.info(
                    "  chunk %d: discovery used %d (budget remaining: %d)",
                    chunk_idx, new_obj_count, discovery_budget_remaining[0],
                )
            logger.info(
                "  chunk %d: done — %d frames, %d discovery objects",
                chunk_idx, len(mask_cache), new_obj_count,
            )
            return mask_cache

        finally:
            self._sam3.end_all_runs()
            shutil.rmtree(chunk_dir, ignore_errors=True)
            self._gc_gpu(f"chunk{chunk_idx}-cleanup")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: Union[str, Path],
        question: str,
    ) -> ROSEE2EResult:
        """Process video through the full ROSE pipeline.

        Phase 1: DA3 batch inference on all frames (consistent poses).
        Phase 2: Two-pass FastSAM + SAM3 (GPU) with CPU-overlap pipeline.

        Args:
            video_path: Path to MP4 video.
            question: Question to answer about the scene.

        Returns:
            ROSEE2EResult with answer, JSON, and 4DSG dict.
        """
        # Delegate to WarmModelPool (production perception path + VLM). run_inference
        # builds the 4DSG with all speed levers and answers the question in one call.
        from rose.engine.server.warm_server import InferenceRequest
        resp = self._get_pool().run_inference(
            InferenceRequest(video_path=str(video_path), question=question)
        )
        if resp.status != "ok":
            raise RuntimeError(f"ROSE pipeline failed: {getattr(resp, 'error', resp.status)}")
        return ROSEE2EResult(
            answer=resp.answer or "",
            scene_json=resp.scene_json or "",
            four_dsg_dict=resp.four_dsg_dict or {},
            step01_trace=[],
            keyframe_dir=Path(resp.keyframe_dir) if resp.keyframe_dir else None,
        )

    def build_4dsg_from_video(
        self,
        video_path: Union[str, Path],
    ) -> ROSE4DSGResult:
        """Process video to 4DSG without VLM inference.

        Returns:
            ROSE4DSGResult with four_dsg_dict, scene_json, and
            keyframe_dir.  The caller **must** call ``result.cleanup()``
            when the keyframe images are no longer needed.
        """
        # Delegate to WarmModelPool (production perception path), no VLM (question=None).
        from rose.engine.server.warm_server import InferenceRequest
        resp = self._get_pool().run_inference(
            InferenceRequest(video_path=str(video_path), question=None)
        )
        if resp.status != "ok":
            raise RuntimeError(f"ROSE 4DSG build failed: {getattr(resp, 'error', resp.status)}")
        return ROSE4DSGResult(
            four_dsg_dict=resp.four_dsg_dict or {},
            scene_json=resp.scene_json or "",
            keyframe_dir=Path(resp.keyframe_dir) if resp.keyframe_dir else None,
        )

    # ------------------------------------------------------------------
    # Internal: core pipeline (shared by process_video / build_4dsg)
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        video_path: Union[str, Path],
    ) -> Tuple[Dict, str, List[Dict], Path]:
        """Run the full 4DSG pipeline.

        Phase 1: DA3 batch on all frames → consistent depth + pose.
        Phase 2: Two-pass FastSAM + SAM3 → masks → pipeline (CPU worker).

        Returns:
            (four_dsg_dict, scene_json, step01_trace, frame_dir)
        """
        # Step 0: Extract frames
        frames, frame_dir, source_indices, keyframe_paths, timestamps_s = self._extract_frames(video_path)
        try:
            # Phase 1: DA3 batch inference — run in background thread so
            # SAM3 model loading can overlap with DA3 GPU inference.
            logger.info("Phase 1: DA3 batch inference on %d frames (async)...", len(frames))
            da3_holder: Dict[str, Any] = {}

            def _da3_worker() -> None:
                try:
                    da3_holder["results"] = self._da3.infer_batch(frames)
                    self._da3.unload()
                    self._gc_gpu("post-DA3")
                except Exception as exc:
                    da3_holder["error"] = exc

            da3_thread = threading.Thread(target=_da3_worker, daemon=True)
            da3_thread.start()

            # Phase 2: Two-pass FastSAM + SAM3 (with optional chunking)
            logger.info("Phase 2: FastSAM + SAM3 two-pass on %d frames...", len(frames))

            step01_trace: List[Dict] = []

            # Determine SAM3 chunk size
            sam3_cs = self.config.sam3.chunk_size
            if sam3_cs == 0:
                sam3_cs = self._estimate_sam3_chunk_size(len(frames))
            elif sam3_cs < 0:
                sam3_cs = len(frames)  # disable chunking

            # Load SAM3 model once (lazy; subsequent set_video_dir reuses it)
            self._sam3.load()

            if len(frames) <= sam3_cs:
                # ---- Single chunk: use _process_sam3_chunk directly ----
                mask_cache = self._process_sam3_chunk(
                    frames, frame_dir, 0, len(frames), chunk_idx=0,
                )
            else:
                # ---- Multiple chunks ----
                sam3_overlap = self.config.sam3.chunk_overlap
                chunks = compute_chunks(len(frames), sam3_cs, sam3_overlap)
                logger.info(
                    "SAM3 chunked: %d frames → %d chunks (size=%d, overlap=%d)",
                    len(frames), len(chunks), sam3_cs, sam3_overlap,
                )
                chunk_caches: List[Dict[int, List[SAM3SharedMask]]] = []
                # Shared discovery budget across all chunks
                shared_budget = [self.config.sam3.max_discovery_total]
                prev_overlap: Optional[Dict[int, List[SAM3SharedMask]]] = None
                for ci, (start, end) in enumerate(chunks):
                    cc = self._process_sam3_chunk(
                        frames, frame_dir, start, end, chunk_idx=ci,
                        prev_overlap_masks=prev_overlap,
                        discovery_budget_remaining=shared_budget,
                    )
                    chunk_caches.append(cc)
                    # Extract overlap masks for the next chunk
                    if ci + 1 < len(chunks):
                        next_start = chunks[ci + 1][0]
                        prev_overlap = {
                            fidx: masks
                            for fidx, masks in cc.items()
                            if fidx >= next_start
                        }

                # Stitch IDs across chunk boundaries via mask IoU
                _stitch_chunk_ids(
                    chunk_caches, chunks,
                    iou_threshold=self.config.fastsam.discovery_iou_thresh,
                )
                mask_cache = _merge_chunk_caches(chunk_caches)

            # ---- Wait for DA3 to complete before Phase 2d ----
            da3_thread.join()
            if "error" in da3_holder:
                raise da3_holder["error"]
            da3_results = da3_holder["results"]
            logger.info("DA3 ready: %d results", len(da3_results))

            # ---- Phase 2d: Build FastFrameInputs → CPU pipeline ----
            # Also collect per-object best crops during this loop.
            pipeline = ROSEPipeline(self.config)
            crop_pad = self.config.vlm.object_crop_padding
            crop_sz = self.config.vlm.object_crop_size
            best_crops: Dict[tuple, Tuple[np.ndarray, float, int, float]] = {}  # (crop, score, src_idx, brightness)

            cpu_queue: queue.Queue[Optional[FastFrameInput]] = queue.Queue()
            cpu_error: List[Optional[BaseException]] = [None]

            def _cpu_worker() -> None:
                try:
                    while True:
                        item = cpu_queue.get()
                        if item is _SENTINEL:
                            break
                        pipeline.process_frame(item)
                except Exception as exc:
                    cpu_error[0] = exc

            worker = threading.Thread(target=_cpu_worker, daemon=True)
            worker.start()

            try:
                for sam3_idx, image in enumerate(frames):
                    frame_masks = mask_cache.get(sam3_idx, [])

                    # Collect best crop per (run_id, obj_id_local).
                    # Prefer highest score; skip expensive crop computation
                    # when score is strictly worse than the current best.
                    for m in frame_masks:
                        key = (m.run_id, m.obj_id_local)
                        prev = best_crops.get(key)
                        if prev is not None and m.score < prev[1]:
                            continue
                        crop = _crop_object_from_mask(
                            image, m.mask, padding=crop_pad, size=crop_sz,
                        )
                        brightness = float(crop.mean())
                        if prev is None or (m.score, brightness) > (prev[1], prev[3]):
                            best_crops[key] = (crop, m.score, source_indices[sam3_idx], brightness)

                    fi, trace = self._build_frame_input(
                        image=image,
                        da3_result=da3_results[sam3_idx],
                        sam3_frame_idx=sam3_idx,
                        source_frame_idx=source_indices[sam3_idx],
                        timestamp_s=timestamps_s[sam3_idx],
                        frame_masks=frame_masks,
                    )
                    step01_trace.append(trace)
                    cpu_queue.put(fi)
            finally:
                cpu_queue.put(_SENTINEL)
                worker.join()

            if cpu_error[0] is not None:
                raise cpu_error[0]  # type: ignore[misc]

            # Save per-object crops and resolve global IDs
            crops_dir = frame_dir / "crops"
            crops_dir.mkdir(exist_ok=True)
            object_crops: Dict[int, Dict[str, object]] = {}
            for key, gid in pipeline._local_to_global.items():
                if key in best_crops and gid not in object_crops:
                    crop_rgb, _score, src_idx, _bright = best_crops[key]
                    if _crop_is_uniform(crop_rgb):
                        continue  # skip uniform-color / empty crops
                    crop_path = crops_dir / f"obj_{gid:04d}.jpg"
                    cv2.imwrite(
                        str(crop_path),
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                    )
                    object_crops[gid] = {
                        "path": str(crop_path),
                        "source_frame_idx": src_idx,
                    }
            logger.info("Saved %d object crops to %s", len(object_crops), crops_dir)

            # Post-hoc dedup: merge re-tracked objects with similar crops & centroids.
            n_before = len(object_crops)
            object_crops = pipeline.merge_duplicate_tracks(object_crops)
            n_after = len(object_crops)
            if n_before != n_after:
                logger.info("Dedup: %d tracks → %d tracks (merged %d duplicates)", n_before, n_after, n_before - n_after)

            four_dsg_dict = pipeline.build_4dsg_dict(
                object_crops=object_crops,
            )
            scene_json = json.dumps(four_dsg_dict, separators=(",", ":"), sort_keys=False)
            return four_dsg_dict, scene_json, step01_trace, frame_dir
        except BaseException:
            shutil.rmtree(frame_dir, ignore_errors=True)
            raise
        finally:
            try:
                self._sam3.end_all_runs()
            except Exception:
                logger.warning("Failed to close SAM3 sessions", exc_info=True)

    # ------------------------------------------------------------------
    # Internal: frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: Union[str, Path],
    ) -> Tuple[List[np.ndarray], Path, List[int], List[Tuple[int, Path]], List[float]]:
        """Extract sampled frames from video, save as JPEGs for SAM3.

        Sampling follows Step 0 config:
        - default target_fps=10.0 (10 Hz)
        - optional clip by max_frames

        The caller is responsible for cleaning up the returned frame_dir
        (via shutil.rmtree) after SAM3 sessions are closed.

        Returns:
            Tuple of (frames_rgb, frame_dir, source_indices, keyframe_paths, timestamps_s)
            where keyframe_paths is ``[(source_frame_idx, jpeg_path), ...]``
            for visual anchor / VLM multimodal input (spec §4.3, line 445),
            and timestamps_s is the physical timestamp in seconds for each frame.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_dir = Path(tempfile.mkdtemp(prefix="rose_frames_"))
        frames: List[np.ndarray] = []
        source_indices: List[int] = []
        timestamps_s: List[float] = []
        keyframe_paths: List[Tuple[int, Path]] = []
        save_idx = 0
        src_idx = 0
        target_fps = float(self.config.sampling.target_fps)
        if target_fps <= 0:
            cap.release()
            raise ValueError(f"sampling.target_fps must be > 0, got {target_fps}")
        max_frames = self.config.sampling.max_frames
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            logger.warning(
                "Video metadata reports invalid FPS (%.3f); fallback to sampling all frames.",
                source_fps,
            )
            sample_interval_s = 0.0
            source_fps = target_fps
        else:
            sample_interval_s = 1.0 / target_fps
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
            jpeg_path = frame_dir / f"{save_idx:06d}.jpg"
            cv2.imwrite(str(jpeg_path), frame_bgr)
            keyframe_paths.append((src_idx - 1, jpeg_path))
            save_idx += 1
            if max_frames is not None and save_idx >= max_frames:
                break

        cap.release()
        logger.info(
            "Extracted %d sampled frames (target_fps=%.3f, source_fps=%.3f) to %s",
            len(frames),
            target_fps,
            source_fps,
            frame_dir,
        )
        return frames, frame_dir, source_indices, keyframe_paths, timestamps_s

    # ------------------------------------------------------------------
    # Internal: per-frame vision model inference (Steps 1-3)
    # ------------------------------------------------------------------

    def _build_frame_input(
        self,
        image: np.ndarray,
        da3_result,
        sam3_frame_idx: int,
        source_frame_idx: int,
        timestamp_s: float = 0.0,
        frame_masks=None,
    ) -> Tuple[FastFrameInput, Dict]:
        """Build FastFrameInput from pre-computed SAM3 masks and DA3 results.

        In the two-pass architecture, SAM3 masks are pre-computed during
        Phases 2a-2c.  This method only converts them to FastLocalDetection
        and packages them with DA3 depth/pose into a FastFrameInput.

        Args:
            image: RGB image (H, W, 3).
            da3_result: Pre-computed DA3Result for this frame.
            sam3_frame_idx: Sequential index matching saved JPEG filenames.
            source_frame_idx: Original video frame number (used as frame_idx).
            timestamp_s: Physical timestamp in seconds from video start.
            frame_masks: Pre-computed SAM3SharedMask list for this frame.
        """
        if frame_masks is None:
            frame_masks = []

        # Deduplicate by (run_id, obj_id_local), keep highest score.
        best_by_key = {}
        for m in frame_masks:
            key = (m.run_id, m.obj_id_local)
            prev = best_by_key.get(key)
            if prev is None or m.score > prev.score:
                best_by_key[key] = m
        frame_masks = list(best_by_key.values())

        # P1-1: drop SAM3 masks fragmented into multiple comparable pieces
        # (typically two distinct objects mistakenly grouped into one mask).
        frame_masks = [m for m in frame_masks if not _mask_is_fragmented(m.mask)]

        # D2: cross-run frame-level mask IoU dedup.
        if len(frame_masks) > 1:
            from rose.engine.server.warm_server import _dedup_masks_by_iou
            frame_masks = _dedup_masks_by_iou(frame_masks, iou_threshold=0.95)

        # Convert SAM3 masks to FastLocalDetection
        detections = [
            FastLocalDetection(
                run_id=m.run_id,
                local_obj_id=m.obj_id_local,
                mask=m.mask,
                score=m.score,
            )
            for m in frame_masks
        ]

        frame_input = FastFrameInput(
            frame_idx=source_frame_idx,
            depth_t=da3_result.depth,
            K_t=da3_result.K,
            T_wc_t=da3_result.T_wc,
            detections=detections,
            depth_conf_t=da3_result.depth_conf,
            depth_is_metric=da3_result.is_metric,
            timestamp_s=timestamp_s,
        )
        trace = {
            "sam3_frame_idx": sam3_frame_idx,
            "source_frame_idx": source_frame_idx,
            "frame_idx": source_frame_idx,
            "mask_count": len(frame_masks),
            "active_runs": self._sam3.num_runs,
        }
        return frame_input, trace

    # ------------------------------------------------------------------
    # Internal: VLM
    # ------------------------------------------------------------------

    def _query_vlm(
        self,
        four_dsg_dict: Dict,
        question: str,
        video_frames: Optional[List[Path]] = None,
    ) -> str:
        """Query VLM with interleaved object crops + STEP tokens.

        Prompt layout (same logical structure for both APIs):
            [Preamble]           — instructions, coordinate system
            [VIDEO FRAMES]       — sampled video frames (passed by caller)
            [TRACKED OBJECTS]    — per-object: crop image + STEP token text
            [QUERY]              — user question

        Supports two providers via config.vlm.provider:
            - "openai": OpenAI API (GPT-5.2, etc.)
            - "google": Google genai API (Gemini, Gemma, etc.)
        """
        provider = self.config.vlm.provider

        if provider == "openai":
            return self._query_vlm_openai(four_dsg_dict, question, video_frames)
        elif provider == "google":
            return self._query_vlm_google(four_dsg_dict, question, video_frames)
        else:
            raise ValueError(f"Unsupported VLM provider: {provider!r}. Use 'openai' or 'google'.")

    # ------------------------------------------------------------------
    # 4DSG text builders
    # ------------------------------------------------------------------

    def _build_preamble_text(self, four_dsg_dict: Dict) -> str:
        """Build the preamble / instructions section.

        Reads the ``reasoning_guide`` from 4DSG metadata (self-documenting SG),
        then appends video-level stats (num_frames, num_tracks).
        """
        meta = four_dsg_dict.get("metadata", {})
        guide = meta.get("reasoning_guide", "")
        return (
            f"{guide}\n\n"
            f"Frames: {meta.get('num_frames', '?')}, "
            f"Tracked objects: {meta.get('num_tracks', '?')}\n"
            f"Coordinate system: {meta.get('coordinate_system', 'unknown')}\n"
        )

    def _build_object_text(self, track: Dict) -> str:
        """Build compact text for a single object track."""
        oid = track.get("object_id", "?")
        fk = track.get("F_k", [])
        if not fk:
            return f"=== Object {oid} ===\nNo observations.\n"

        theta = track.get("theta", [0, 0])
        extent = track.get("extent", [0, 0, 0])
        extent_str = f"{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}m"

        img_pos = track.get("image_position", "center")

        header = f"Time: {theta[0]}s-{theta[1]}s ({len(fk)} obs). Size: {extent_str}. Image: {img_pos}."
        motion = track.get("motion")
        if motion:
            header = header[:-1] + f" Motion: {motion}."

        lines = [
            f"=== Object {oid} ===",
            header,
        ]
        for obs in fk:
            t = obs.get("t", "?")
            c = obs.get("c", [])
            pos_str = f"[{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]" if len(c) == 3 else str(c)
            lines.append(f"  t={t}s: pos={pos_str}")
        return "\n".join(lines) + "\n"


    def _get_object_crop_path(self, track: Dict) -> Optional[Path]:
        """Get the crop image path for a track, if it exists."""
        va = track.get("visual_anchor")
        if va is None:
            return None
        p = Path(va["path"])
        return p if p.exists() else None

    _QUERY_SUFFIX = (
        "You are given a 4D scene graph (4DSG) with per-object crop images, "
        "sampled video frames, and 3D tracking data. "
        "Use BOTH the visual information (video frames, object crops) and the 3D data "
        "(3D coordinates and shape stats) to reason about the answer.\n\n"
        "Answer the given multiple-choice question step by step. "
        "First, identify the relevant objects from their crop images. "
        "Then, analyze spatial/temporal relations using the 3D trajectory data. "
        "In the last sentence of your response, you must conclude by stating "
        "the final answer using the following format: "
        "'Therefore, the final answer is: $LETTER' (without quotes), "
        "where $LETTER must be only one of the options (A or B or C or D)."
    )

    def _query_vlm_openai(self, four_dsg_dict: Dict, question: str, video_frames: Optional[List[Path]] = None) -> str:
        """Query VLM via OpenAI API (GPT-5.2, etc.)."""
        import base64
        import os

        if self._vlm_client is None:
            from openai import OpenAI
            api_key = os.environ.get(self.config.vlm.api_key_env)
            if not api_key:
                raise ValueError(f"{self.config.vlm.api_key_env} env var required")
            kwargs = {"api_key": api_key}
            if self.config.vlm.base_url:
                kwargs["base_url"] = self.config.vlm.base_url
            self._vlm_client = OpenAI(**kwargs)

        content: List[Dict] = []

        def _add_text(text: str) -> None:
            content.append({"type": "text", "text": text})

        def _add_image(path: Path) -> None:
            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        # 1) Preamble
        _add_text(self._build_preamble_text(four_dsg_dict))

        # 2) Video frames (passed by caller, not from 4DSG)
        if video_frames:
            _add_text("[VIDEO FRAMES]")
            for vf_path in video_frames:
                _add_image(vf_path)

        # 3) Tracked objects — interleaved crop + text
        tracks = four_dsg_dict.get("tracks", [])
        if tracks:
            _add_text("\n[TRACKED OBJECTS]")
            for track in tracks:
                oid = track.get("object_id", "?")
                crop_path = self._get_object_crop_path(track)
                if crop_path is not None:
                    _add_text(f"Object {oid}:")
                    _add_image(crop_path)
                _add_text(self._build_object_text(track))

        # 4) Question
        _add_text(f"[QUERY]\n{question}\n\n{self._QUERY_SUFFIX}")

        # GPT-5.x, o1, o3, o4 use max_completion_tokens; older models use max_tokens
        model_lower = self.config.vlm.model.lower()
        use_new_param = any(p in model_lower for p in ['o1', 'o3', 'o4', 'gpt-5'])
        token_param = "max_completion_tokens" if use_new_param else "max_tokens"

        response = self._vlm_client.chat.completions.create(
            model=self.config.vlm.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.config.vlm.temperature,
            **{token_param: self.config.vlm.max_output_tokens},
        )
        return response.choices[0].message.content

    def _query_vlm_google(self, four_dsg_dict: Dict, question: str, video_frames: Optional[List[Path]] = None) -> str:
        """Query VLM via Google genai API (Gemini, Gemma, etc.)."""
        import os

        if self._vlm_client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai required for provider='google'")
            api_key = os.environ.get(self.config.vlm.api_key_env)
            if not api_key:
                raise ValueError(f"{self.config.vlm.api_key_env} env var required")
            self._vlm_client = genai.Client(api_key=api_key)

        from google.genai import types

        contents: list = []

        def _add_text(text: str) -> None:
            contents.append(types.Part.from_text(text=text))

        def _add_image(path: Path) -> None:
            contents.append(types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type="image/jpeg",
            ))

        # 1) Preamble
        _add_text(self._build_preamble_text(four_dsg_dict))

        # 2) Video frames (passed by caller, not from 4DSG)
        if video_frames:
            _add_text("[VIDEO FRAMES]")
            for vf_path in video_frames:
                _add_image(vf_path)

        # 3) Tracked objects — interleaved crop + text
        tracks = four_dsg_dict.get("tracks", [])
        if tracks:
            _add_text("\n[TRACKED OBJECTS]")
            for track in tracks:
                oid = track.get("object_id", "?")
                crop_path = self._get_object_crop_path(track)
                if crop_path is not None:
                    _add_text(f"Object {oid}:")
                    _add_image(crop_path)
                _add_text(self._build_object_text(track))

        # 4) Question
        _add_text(f"[QUERY]\n{question}\n\n{self._QUERY_SUFFIX}")

        response = self._vlm_client.models.generate_content(
            model=self.config.vlm.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.vlm.max_output_tokens,
                temperature=self.config.vlm.temperature,
            ),
        )
        return response.text
