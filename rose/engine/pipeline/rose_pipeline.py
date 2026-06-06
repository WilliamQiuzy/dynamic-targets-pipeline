"""ROSE runtime pipeline.

Implements the ROSE route in docs/roadmap/ROSE_IMPLEMENTATION.md:
- Step 4: mask + depth backprojection and geometric filtering
- Step 5: cross-run global ID fusion
- Step 6: STEP token construction
- Step 7: temporal tracks (F_k)
- Step 8: strict 1:1 JSON serialization

This module is model-agnostic: callers provide per-frame SAM3 detections and DA3 outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rose.engine.config.rose_config import ROSEConfig
from rose.reasoning.tokens.geometry_tokens import build_centroid_token, build_shape_token
from rose.reasoning.tokens.patch_tokenizer import PatchToken, mask_to_patch_tokens
from rose.reasoning.tokens.step_encoding import STEPToken
from rose.reasoning.tokens.temporal_tokens import TemporalToken


RunKey = Tuple[Hashable, Hashable]


# ── Deep re-ID embedder (DINOv2) ─────────────────────────────────────────────
# A single-best-crop DINOv2 embedding is a far more discriminative object
# appearance descriptor than the raw-pixel crop cosine the pipeline used (which
# false-merged on color alone in blurry scenes).  Loaded lazily + cached so it
# costs one ~3 s load per process and ~0.3 s / video for the embeddings.
_REID_STATE: Dict[str, object] = {}


def _get_reid_embedder(model_name: str):
    if _REID_STATE.get("name") == model_name and "model" in _REID_STATE:
        return _REID_STATE["model"], _REID_STATE["mean"], _REID_STATE["std"]
    import timm  # local import: only loaded when re-ID merge is enabled
    import torch
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    cfg = timm.data.resolve_model_data_config(model)
    dev = next(model.parameters()).device
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1).to(dev)
    std = torch.tensor(cfg["std"]).view(3, 1, 1).to(dev)
    _REID_STATE.update(name=model_name, model=model, mean=mean, std=std, size=cfg["input_size"][-1])
    return model, mean, std


def _embed_crop_paths(paths: List[str], model_name: str) -> Dict[str, np.ndarray]:
    """Return {path: L2-normalised DINOv2 embedding} for crop image files."""
    import cv2 as _cv2
    import torch
    model, mean, std = _get_reid_embedder(model_name)
    size = int(_REID_STATE.get("size", 518))
    dev = next(model.parameters()).device
    valid, batch = [], []
    for p in paths:
        img = _cv2.imread(p)
        if img is None:
            continue
        img = _cv2.cvtColor(_cv2.resize(img, (size, size)), _cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0
        batch.append((t - mean) / std)
        valid.append(p)
    if not batch:
        return {}
    with torch.no_grad():
        feats = torch.nn.functional.normalize(model(torch.stack(batch)), dim=1).cpu().numpy()
    return {p: feats[i] for i, p in enumerate(valid)}


@dataclass(frozen=True)
class FastLocalDetection:
    """One SAM3 detection from a single run in one frame."""

    run_id: Hashable
    local_obj_id: Hashable
    mask: np.ndarray  # (H, W) bool
    score: float = 1.0


@dataclass(frozen=True)
class FastFrameInput:
    """Per-frame inputs required by ROSE Step 4+.

    Attributes:
        frame_idx: Frame index.
        depth_t: DA3 depth map (H, W), meters.
        K_t: Camera intrinsics (3, 3).
        T_wc_t: DA3 world->camera transform (4, 4).
        detections: SAM3 detections from all active runs for this frame.
        depth_conf_t: DA3 depth confidence map (H, W), values in [0, 1].
            When None, all pixels are trusted (equivalent to ones_like(depth_t)).
        depth_is_metric: Whether depth_t is in absolute metres (True) or
            relative/arbitrary scale (False).  When False, max_extent
            filtering is skipped since the threshold is meaningless.
    """

    frame_idx: int
    depth_t: np.ndarray
    K_t: np.ndarray
    T_wc_t: np.ndarray
    detections: Sequence[FastLocalDetection]
    depth_conf_t: Optional[np.ndarray] = None
    depth_is_metric: bool = True
    timestamp_s: float = 0.0  # Physical timestamp in seconds from video start


@dataclass(frozen=True)
class _FrameObservation:
    frame_idx: int
    timestamp_s: float
    step: STEPToken
    mask_center_2d: Tuple[float, float] = (0.5, 0.5)  # normalized (cx, cy) in [0,1]
    # Dynamic-targets export extras (populated only when dynamic_targets.enabled).
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (rmin, cmin, rmax, cmax) pixels
    obb: Optional[dict] = None  # per-frame 9DOF gravity-aligned box (center/size/yaw/corners)


@dataclass
class _TrackState:
    track_id: int
    observations: List[_FrameObservation] = field(default_factory=list)
    status: str = "active"  # active|lost|archived
    missing_streak: int = 0
    last_seen_t: Optional[int] = None
    last_centroid: Optional[np.ndarray] = None

    def observe(self, frame_idx: int, timestamp_s: float, step: STEPToken, centroid: np.ndarray,
                mask_center_2d: Tuple[float, float] = (0.5, 0.5),
                bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
                obb: Optional[dict] = None) -> None:
        if self.status == "archived":
            raise RuntimeError(
                f"Bug: observe() called on archived track {self.track_id}. "
                f"Archived tracks must not be re-identified."
            )
        self.observations.append(_FrameObservation(
            frame_idx=frame_idx, timestamp_s=timestamp_s, step=step,
            mask_center_2d=mask_center_2d, bbox=bbox, obb=obb,
        ))
        self.last_seen_t = frame_idx
        self.last_centroid = centroid
        self.status = "active"
        self.missing_streak = 0

    def miss(self, cfg: ROSEConfig) -> None:
        if self.status == "archived":
            return
        self.missing_streak += 1
        if self.missing_streak >= cfg.fusion.lost_patience + cfg.fusion.archive_patience:
            self.status = "archived"
        elif self.missing_streak >= cfg.fusion.lost_patience:
            self.status = "lost"


@dataclass(frozen=True)
class _Candidate:
    run_id: Hashable
    local_obj_id: Hashable
    mask: np.ndarray
    score: float
    centroid_xyz: np.ndarray
    step: STEPToken
    mask_center_2d: Tuple[float, float] = (0.5, 0.5)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (rmin, cmin, rmax, cmax) for fast overlap check
    obb: Optional[dict] = None  # per-frame 9DOF gravity-aligned box (dynamic_targets only)


class ROSEPipeline:
    """ROSE pipeline implementation for precomputed model outputs."""

    def __init__(self, config: Optional[ROSEConfig] = None):
        self.config = config or ROSEConfig()
        self.reset()

    def reset(self) -> None:
        self._next_global_id = 0
        self._local_to_global: Dict[RunKey, int] = {}
        self._tracks: Dict[int, _TrackState] = {}
        self._ego_poses_cw: Dict[int, np.ndarray] = {}
        self._frame_timestamps: Dict[int, float] = {}  # frame_idx → timestamp_s
        self._latest_frame_idx: Optional[int] = None

    def process_frames(self, frames: Iterable[FastFrameInput], reset: bool = True) -> None:
        """Process a sequence of frame inputs."""
        if reset:
            self.reset()

        for frame in sorted(frames, key=lambda x: x.frame_idx):
            self.process_frame(frame)

    def compute_candidates(self, frame: FastFrameInput):
        """PURE per-frame work (Step 4): backproject each mask → 3D + STEP tokens.

        Does NOT mutate any shared pipeline state (_tracks / _local_to_global /
        global-id counter) — all of that happens in the SEQUENTIAL fuse+observe
        pass in process_frame. Therefore this is safe to run across frames IN
        PARALLEL (a thread pool). The heavy numpy here (np.nonzero / gather /
        K_inv@uv1 matmul / np.percentile / mask_to_patch_tokens) releases the
        GIL, so threading gives real multi-core speedup. Returns (T_cw_t, candidates).
        """
        T_cw_t = np.linalg.inv(frame.T_wc_t)
        K_inv = np.linalg.inv(frame.K_t.astype(np.float64))
        depth_t = frame.depth_t
        depth_conf_t = frame.depth_conf_t
        depth_ok = np.isfinite(depth_t) & (depth_t > 0.0)
        if depth_conf_t is not None:
            depth_ok &= depth_conf_t >= self.config.depth_filter.conf_thresh
        candidates = self._build_candidates(
            frame, T_cw_t, K_inv, frame_ctx={"depth_ok": depth_ok}
        )
        return T_cw_t, candidates

    def process_frame(self, frame: FastFrameInput, precomputed=None) -> None:
        """Process one frame (Step 4-7).

        If *precomputed* = (T_cw_t, candidates) is supplied (from a PARALLEL
        compute_candidates pass), the per-frame backprojection is skipped here and
        only the SEQUENTIAL, order-dependent fuse + observe + miss runs — keeping
        track-state updates deterministic while the heavy lift was parallelised.
        """
        self._latest_frame_idx = frame.frame_idx
        if precomputed is not None:
            T_cw_t, candidates = precomputed
        else:
            T_cw_t, candidates = self.compute_candidates(frame)
        self._ego_poses_cw[frame.frame_idx] = T_cw_t
        self._frame_timestamps[frame.frame_idx] = frame.timestamp_s
        winners = self._fuse_candidates(frame.frame_idx, candidates, frame.depth_is_metric)

        touched: set[int] = set()

        for gid, cand in winners.items():
            state = self._tracks.setdefault(gid, _TrackState(track_id=gid))
            state.observe(frame.frame_idx, frame.timestamp_s, cand.step, cand.centroid_xyz,
                         mask_center_2d=cand.mask_center_2d, bbox=cand.bbox, obb=cand.obb)
            touched.add(gid)

        for gid, state in self._tracks.items():
            if gid not in touched:
                state.miss(self.config)

    def build_4dsg_dict(
        self,
        object_crops: Optional[Dict[int, Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Build compact ROSE 4DSG JSON object (Step 8).

        All ``t`` fields use real seconds (from ``FastFrameInput.timestamp_s``).
        Tracks without a crop image are dropped (VLM cannot identify them).

        Args:
            object_crops: Per-track masked crop references, mapping
                ``global_track_id → {"path": str, "source_frame_idx": int}``.
                Each crop is a bbox-padded masked image of the object.
        """
        _R = 2  # decimal places

        tracks_entries: List[Dict[str, object]] = []
        # Quality-filter thresholds (FusionConfig-driven).
        min_obs = int(getattr(self.config.fusion, "min_track_observations", 3))
        max_ext = float(getattr(self.config.fusion, "max_track_extent", 0.7))

        for gid in sorted(self._tracks.keys()):
            # Skip tracks without a crop — VLM cannot identify them.
            if object_crops is None or gid not in object_crops:
                continue

            state = self._tracks[gid]
            if not state.observations:
                continue

            obs_sorted = sorted(state.observations, key=lambda x: x.frame_idx)

            # Quality filter 1: drop ghost tracks (too few observations).
            if len(obs_sorted) < min_obs:
                continue

            # Track-level 3D extent: median (max-min) per axis across observations.
            extents_x, extents_y, extents_z = [], [], []
            for obs in obs_sorted:
                s = obs.step.shape
                extents_x.append(s.x_max - s.x_min)
                extents_y.append(s.y_max - s.y_min)
                extents_z.append(s.z_max - s.z_min)
            extent = [
                round(float(np.median(extents_x)), _R),
                round(float(np.median(extents_y)), _R),
                round(float(np.median(extents_z)), _R),
            ]

            # Quality filter 2: drop depth-broken tracks.
            # The original heuristic "≥2 axes > 0.7" caught water/sky blobs
            # (e.g. ext=[1.17, 0.09, 1.17]) but ALSO over-killed legitimate
            # moving cars (e.g. ext=[1.25, 0.26, 1.52] from a car driving
            # laterally + receding) because the geometry is identical.
            #
            # Fix: distinguish stationary blobs from moving objects via
            # centroid displacement.  Water/sky blobs have NO trajectory
            # motion; cars sweep ≥0.3 units across the scene.  Drop only
            # when the track has BOTH a "blob" extent signature AND a
            # near-stationary centroid.
            n_huge = sum(1 for e in extent if e > 0.7)
            first_c = obs_sorted[0].step.centroid
            last_c  = obs_sorted[-1].step.centroid
            disp_max = max(abs(last_c.x - first_c.x),
                           abs(last_c.y - first_c.y),
                           abs(last_c.z - first_c.z))
            # Metric depth: an axis extent > max_track_extent metres means the
            # "object" is a background / scene region (sky, far trees, a whole
            # wall), not a discrete object → drop.  Discrete foreground subjects,
            # including a close-up animal (~1 m) or a slalom banner (~5 m), are
            # well under the threshold and kept (this is what un-broke dog/close-
            # up videos: size, not motion, is the right metric-depth signal).
            if max(extent) > max_ext:
                continue
            # Motion descriptor from FULL trajectory (before any truncation).
            motion = self._compute_motion_descriptor(obs_sorted, _R)

            # Optional sliding window: keep only the most recent T observations
            # for VLM serialization.  T=0 (default) means keep all.
            T = self.config.step.temporal_window
            if T > 0 and len(obs_sorted) > T:
                obs_sorted = obs_sorted[-T:]

            # Per-observation: only t and c (centroid).
            fk: List[Dict[str, object]] = []
            for obs in obs_sorted:
                c = obs.step.centroid
                fk.append({
                    "t": round(obs.timestamp_s, _R),
                    "c": [round(float(c.x), _R), round(float(c.y), _R), round(float(c.z), _R)],
                })

            # Image position from first observation's 2D mask center.
            first_obs = sorted(state.observations, key=lambda x: x.frame_idx)[0]
            image_position = self._center_to_position_9(first_obs.mask_center_2d)

            # Only keep the crop path for VLM; drop internal fields like source_frame_idx.
            va = {"path": object_crops[gid]["path"]}

            track_entry: Dict[str, object] = {
                "object_id": gid,
                "visual_anchor": va,
                "extent": extent,
                # "motion": motion,  # TODO: re-enable after phase-aware descriptor is validated
                "image_position": image_position,
                "F_k": fk,
            }
            tracks_entries.append(track_entry)

        metadata: Dict[str, object] = {
            "num_frames": len(self._ego_poses_cw),
            "num_tracks": len(tracks_entries),
            "coordinate_system": "World frame = first frame camera. X=right, Y=down, Z=forward. Units: relative scale (not calibrated).",
            "reasoning_guide": (
                "This is a 4D scene graph (4DSG): 3D tracking data + crop images from a video.\n"
                "All coordinates are in WORLD frame (camera motion already compensated).\n\n"
                "COORDINATE SYSTEM (world frame = camera at frame 0):\n"
                "  X axis → viewer's right    (positive = right, negative = left)\n"
                "  Y axis → downward          (positive = down, negative = up)\n"
                "  Z axis → into the scene    (positive = farther, negative = closer)\n\n"
                "HOW TO DETERMINE MOTION DIRECTION:\n"
                "  Compare an object's 3D position [x,y,z] across its trajectory.\n"
                "  - x increases → object moved RIGHT;  x decreases → moved LEFT\n"
                "  - y increases → object moved DOWN;    y decreases → moved UP\n"
                "  - z increases → object moved AWAY;    z decreases → moved CLOSER\n\n"
                "HOW TO DETERMINE SPATIAL RELATIONS:\n"
                "  Compare [x,y,z] of two objects AT THE SAME TIMESTAMP.\n"
                "  - Object A has smaller x than B → A is to the LEFT of B\n"
                "  - Object A has smaller y than B → A is ABOVE B\n"
                "  - Object A has smaller z than B → A is CLOSER to the camera\n\n"
                "PER-OBJECT FIELDS:\n"
                "  visual_anchor — crop image showing what the object looks like.\n"
                "  extent [dx, dy, dz] — 3D bounding-box size (width, height, depth).\n"
                "  image_position — where the object appeared in the video frame (e.g. top-left).\n"
                "  F_k — trajectory: list of {t: timestamp, c: [x, y, z] position}.\n\n"
                "IMPORTANT:\n"
                "  - F_k positions are world-frame — they reflect true object motion, not camera motion.\n"
                "  - 'left/right' in questions usually means the viewer's left/right (X axis)."
            ),
        }

        return {
            "metadata": metadata,
            "tracks": tracks_entries,
        }

    # ------------------------------------------------------------------
    # Dynamic-targets export (动态目标管线 ALL_FRAMES schema)
    # ------------------------------------------------------------------
    def _surviving_track_gids(self, object_crops: Dict[int, Dict[str, object]]) -> List[int]:
        """Tracks that pass the SAME quality gate as build_4dsg_dict (crop present,
        enough observations, not a depth-broken background blob). Keeps the
        dynamic-targets object set identical to the 4DSG tracks."""
        min_obs = int(getattr(self.config.fusion, "min_track_observations", 3))
        max_ext = float(getattr(self.config.fusion, "max_track_extent", 0.7))
        out: List[int] = []
        for gid in sorted(self._tracks.keys()):
            if object_crops is None or gid not in object_crops:
                continue
            state = self._tracks[gid]
            if not state.observations or len(state.observations) < min_obs:
                continue
            obs_sorted = sorted(state.observations, key=lambda x: x.frame_idx)
            ex = [np.median([o.step.shape.x_max - o.step.shape.x_min for o in obs_sorted]),
                  np.median([o.step.shape.y_max - o.step.shape.y_min for o in obs_sorted]),
                  np.median([o.step.shape.z_max - o.step.shape.z_min for o in obs_sorted])]
            if max(ex) > max_ext:
                continue
            out.append(gid)
        return out

    def _smooth_centroids(self, obs_sorted) -> np.ndarray:
        """Moving-average smooth the world-centroid trajectory (N,3).

        Monocular per-frame depth jitters the mask centroid by centimetres; over a
        0.1 s frame step that becomes metres-per-second of spurious velocity on a
        STATIC object. A short centred moving average removes that jitter while
        preserving real translation."""
        cs = np.array([[o.step.centroid.x, o.step.centroid.y, o.step.centroid.z]
                       for o in obs_sorted], dtype=np.float64)
        w = int(getattr(self.config.dynamic_targets, "smooth_window", 1))
        n = cs.shape[0]
        if w <= 1 or n < 3:
            return cs
        half = w // 2
        out = np.empty_like(cs)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            out[i] = cs[lo:hi].mean(axis=0)
        return out

    def _kalman_smooth(self, obs_sorted):
        """Constant-velocity RTS Kalman smoother with innovation gating + depth-adaptive
        measurement noise. Returns (positions Nx3, velocities Nx3) in world frame.

        Optimally fuses the whole 3D-center trajectory under a smooth-motion prior:
        the gate soft-rejects bad-depth outlier frames (which otherwise spike
        velocity) and R∝depth^2 down-weights noisy far measurements. Benchmarked at
        ~3.7x lower velocity RMSE than moving-average + windowed least squares."""
        dtc = self.config.dynamic_targets
        P = np.array([[o.step.centroid.x, o.step.centroid.y, o.step.centroid.z]
                      for o in obs_sorted], dtype=np.float64)
        T = np.array([o.timestamp_s for o in obs_sorted], dtype=np.float64)
        n = len(obs_sorted)
        if n == 1:
            return P, np.zeros((1, 3))
        sa = float(dtc.kalman_sigma_a); br = float(dtc.kalman_base_r)
        zref = float(dtc.kalman_z_ref); gate = float(dtc.kalman_gate)
        Z = P[:, 2]
        xs = np.zeros((n, 3, 2)); Ps_ = np.zeros((n, 3, 2, 2))
        xp = np.zeros((n, 3, 2)); Pp = np.zeros((n, 3, 2, 2))
        x = np.stack([P[0], np.zeros(3)], 1)
        Pc = np.tile(np.diag([br ** 2, 1.0]), (3, 1, 1)).astype(np.float64)
        H = np.array([1.0, 0.0])
        I2 = np.eye(2)
        for i in range(n):
            dt = (T[i] - T[i - 1]) if i > 0 else 0.0
            F = np.array([[1.0, dt], [0.0, 1.0]])
            G = np.array([0.5 * dt * dt, dt]); Q = np.outer(G, G) * sa ** 2
            if i > 0:
                for ax in range(3):
                    x[ax] = F @ x[ax]; Pc[ax] = F @ Pc[ax] @ F.T + Q
            xp[i] = x; Pp[i] = Pc
            r0 = (br * max(Z[i] / zref, 1.0) ** 2) ** 2
            innov = np.array([P[i, ax] - H @ x[ax] for ax in range(3)])
            svar = np.array([H @ Pc[ax] @ H + r0 for ax in range(3)])
            d2 = float((innov ** 2 / np.maximum(svar, 1e-9)).sum())
            r = r0 * (1.0 + (d2 / (gate ** 2 * 3)) ** 2 * 50.0) if (gate and d2 > gate ** 2 * 3) else r0
            for ax in range(3):
                S = H @ Pc[ax] @ H + r; K = (Pc[ax] @ H) / S
                x[ax] = x[ax] + K * (P[i, ax] - H @ x[ax]); Pc[ax] = (I2 - np.outer(K, H)) @ Pc[ax]
            xs[i] = x; Ps_[i] = Pc
        pos = np.zeros((n, 3)); vel = np.zeros((n, 3))
        for ax in range(3):
            xsm = xs[:, ax].copy()
            for i in range(n - 2, -1, -1):
                dt = (T[i + 1] - T[i]); F = np.array([[1.0, dt], [0.0, 1.0]])
                C = Ps_[i, ax] @ F.T @ np.linalg.inv(Pp[i + 1, ax])
                xsm[i] = xs[i, ax] + C @ (xsm[i + 1] - xp[i + 1, ax])
            pos[:, ax] = xsm[:, 0]; vel[:, ax] = xsm[:, 1]
        return pos, vel

    def _smooth_yaw(self, obs_sorted) -> Dict[int, float]:
        """Circular moving-average of per-frame box yaw → stable 9DOF corners.
        frame_idx → smoothed yaw (radians)."""
        w = int(getattr(self.config.dynamic_targets, "yaw_smooth_window", 1))
        yaws = [(o.frame_idx, (o.obb["yaw"] if o.obb is not None else None)) for o in obs_sorted]
        n = len(yaws)
        out: Dict[int, float] = {}
        for i, (fidx, y) in enumerate(yaws):
            if y is None:
                out[fidx] = 0.0; continue
            if w <= 1:
                out[fidx] = y; continue
            half = w // 2
            cs = ss = 0.0
            for j in range(max(0, i - half), min(n, i + half + 1)):
                yj = yaws[j][1]
                if yj is None:
                    continue
                # yaw is a 180-deg-ambiguous axis direction → average on the 2*theta circle.
                cs += math.cos(2 * yj); ss += math.sin(2 * yj)
            out[fidx] = 0.5 * math.atan2(ss, cs) if (cs or ss) else y
        return out

    def _track_velocities(self, obs_sorted, centroids: Optional[np.ndarray] = None) -> Dict[int, List[float]]:
        """World-frame absolute velocity (m/s) per observation. frame_idx → [vx,vy,vz].

        Uses a windowed least-squares slope (regress position vs. time over a
        ±window/2 neighbourhood) rather than a raw two-frame difference: the
        regression is robust to per-frame depth outliers (a single noisy far-object
        depth no longer spikes the velocity) and naturally smooths. Falls back to
        a finite difference for very short tracks."""
        cs = centroids if centroids is not None else np.array(
            [[o.step.centroid.x, o.step.centroid.y, o.step.centroid.z] for o in obs_sorted],
            dtype=np.float64)
        ts = np.array([o.timestamp_s for o in obs_sorted], dtype=np.float64)
        n = len(obs_sorted)
        win = max(3, int(getattr(self.config.dynamic_targets, "smooth_window", 5)))
        half = win // 2
        central = bool(self.config.dynamic_targets.velocity_central_diff)
        vel: Dict[int, List[float]] = {}
        for i, o in enumerate(obs_sorted):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            tw = ts[lo:hi]
            pw = cs[lo:hi]
            if tw.shape[0] >= 3 and (tw.max() - tw.min()) > 1e-6:
                # least-squares slope per axis: v = cov(t, p) / var(t)
                tc = tw - tw.mean()
                denom = float((tc * tc).sum())
                v = (tc[:, None] * (pw - pw.mean(axis=0))).sum(axis=0) / max(denom, 1e-9)
            elif central and 0 < i < n - 1 and (ts[i + 1] - ts[i - 1]) > 1e-6:
                v = (cs[i + 1] - cs[i - 1]) / (ts[i + 1] - ts[i - 1])
            elif i < n - 1 and (ts[i + 1] - ts[i]) > 1e-6:
                v = (cs[i + 1] - cs[i]) / (ts[i + 1] - ts[i])
            elif i > 0 and (ts[i] - ts[i - 1]) > 1e-6:
                v = (cs[i] - cs[i - 1]) / (ts[i] - ts[i - 1])
            else:
                v = np.zeros(3)
            vel[o.frame_idx] = [float(v[0]), float(v[1]), float(v[2])]
        return vel

    def build_dynamic_targets_dict(
        self,
        object_crops: Dict[int, Dict[str, object]],
        instance_names: Optional[Dict[int, str]] = None,
        image_hw: Optional[Tuple[int, int]] = None,
        source_fps: Optional[float] = None,
    ) -> Dict[str, object]:
        """Build the client ALL_FRAMES JSON (per-frame visible objects with 9DOF
        oriented boxes, 2D pixel boxes, instance names, absolute velocity).

        Args:
            object_crops: surviving-track gid → crop ref (same dict the 4DSG uses,
                AFTER reid/duplicate merges). Defines the object set.
            instance_names: gid → semantic class name (from the VLM namer). Missing
                gids default to "object".
            image_hw: (H, W) pixel resolution the 2D bboxes are in (for metadata).
            source_fps: original video fps (metadata only).
        """
        from rose.engine.export.obb import corners_from_center_size_yaw

        cfg = self.config.dynamic_targets
        R = int(cfg.round_decimals)
        instance_names = instance_names or {}

        def _rnd(x):
            return round(float(x), R)

        def _rnd_list(xs):
            return [round(float(v), R) for v in xs]

        gids = self._surviving_track_gids(object_crops)

        # Per-object precompute: sorted obs, velocity, stabilized box size, dynamic flag.
        obj_obs: Dict[int, list] = {}
        obj_vel: Dict[int, Dict[int, List[float]]] = {}
        obj_size: Dict[int, List[float]] = {}
        obj_center: Dict[int, Dict[int, List[float]]] = {}  # frame_idx → smoothed center
        obj_yaw: Dict[int, Dict[int, float]] = {}           # frame_idx → smoothed yaw
        keep: List[int] = []
        use_kalman = bool(getattr(cfg, "velocity_kalman", True))
        use_sm_center = bool(cfg.use_smoothed_center) or use_kalman
        for gid in gids:
            obs_sorted = sorted(self._tracks[gid].observations, key=lambda x: x.frame_idx)
            if use_kalman:
                # RTS Kalman: smoothed center (position state) + velocity (velocity state).
                pos, kv = self._kalman_smooth(obs_sorted)
                sm = pos
                vel = {o.frame_idx: [float(kv[i][0]), float(kv[i][1]), float(kv[i][2])]
                       for i, o in enumerate(obs_sorted)}
            else:
                sm = self._smooth_centroids(obs_sorted)
                vel = self._track_velocities(obs_sorted, centroids=sm)
            obj_center[gid] = {o.frame_idx: [float(sm[i][0]), float(sm[i][1]), float(sm[i][2])]
                               for i, o in enumerate(obs_sorted)}
            obj_yaw[gid] = self._smooth_yaw(obs_sorted)
            # Stabilized (rigid-object) box size = median of per-frame OBB sizes.
            obb_sizes = [o.obb["size"] for o in obs_sorted if o.obb is not None]
            if obb_sizes:
                med = np.median(np.array(obb_sizes, dtype=np.float64), axis=0)
                size_stab = [float(med[0]), float(med[1]), float(med[2])]
            else:
                # fall back to axis-aligned extent (world axes) median
                size_stab = [
                    float(np.median([o.step.shape.x_max - o.step.shape.x_min for o in obs_sorted])),
                    float(np.median([o.step.shape.y_max - o.step.shape.y_min for o in obs_sorted])),
                    float(np.median([o.step.shape.z_max - o.step.shape.z_min for o in obs_sorted])),
                ]
            # only_dynamic filter (mean speed over track).
            speeds = [float(np.linalg.norm(v)) for v in vel.values()]
            mean_speed = float(np.mean(speeds)) if speeds else 0.0
            if cfg.only_dynamic and mean_speed < cfg.min_speed_dynamic:
                continue
            # Planar-background filter: drop tracks that are BIG in the image AND
            # geometrically planar (wall/floor/surface) — keeps real large close-up
            # objects (which have 3D thickness).
            if getattr(cfg, "drop_planar_background", False) and image_hw is not None:
                _H, _W = image_hw
                if _H and _W:
                    fracs = []
                    for o in obs_sorted:
                        rmin, cmin, rmax, cmax = o.bbox
                        fracs.append((rmax - rmin) * (cmax - cmin) / float(_H * _W))
                    if (np.median(fracs) > cfg.bg_bbox_frac
                            and min(size_stab) < cfg.bg_min_thickness):
                        continue
            obj_obs[gid] = obs_sorted
            obj_vel[gid] = vel
            obj_size[gid] = size_stab
            keep.append(gid)

        # Index observations by frame for the per-frame inversion.
        # frame_idx → list of (gid, obs)
        by_frame: Dict[int, List[Tuple[int, object]]] = {}
        for gid in keep:
            for o in obj_obs[gid]:
                by_frame.setdefault(o.frame_idx, []).append((gid, o))

        frames_sorted = sorted(self._ego_poses_cw.keys())
        all_frames: Dict[str, object] = {}
        for seq, fidx in enumerate(frames_sorted):
            ts = self._frame_timestamps.get(fidx, 0.0)
            visible = []
            for gid, o in sorted(by_frame.get(fidx, []), key=lambda x: x[0]):
                # 9DOF box: stabilized size + smoothed per-frame center & yaw.
                size = obj_size[gid]
                if o.obb is not None:
                    center = o.obb["center"]
                else:
                    c = o.step.centroid
                    center = [c.x, c.y, c.z]
                yaw = obj_yaw[gid].get(o.frame_idx, o.obb["yaw"] if o.obb is not None else 0.0)
                if use_sm_center:
                    center = obj_center[gid].get(o.frame_idx, center)
                box_size = size if cfg.stabilize_size else (
                    o.obb["size"] if o.obb is not None else size)
                # Always recompute corners from the final center+size+yaw so the 8
                # corners are exactly consistent with the reported center & size.
                corners = corners_from_center_size_yaw(
                    np.array(center), np.array(box_size), yaw).tolist()
                # 2D pixel bbox (rmin,cmin,rmax,cmax) → xyxy [x_left, y_top, x_right, y_bottom].
                rmin, cmin, rmax, cmax = o.bbox
                pv_bbox = [int(cmin), int(rmin), int(cmax), int(rmax)]
                vel = obj_vel[gid].get(o.frame_idx, [0.0, 0.0, 0.0])
                visible.append({
                    "instance_id": int(gid),
                    "instance_name": instance_names.get(gid, "object"),
                    "Ori_9DOF_corners": [_rnd_list(c) for c in corners],
                    "Ori_9DOF_center": _rnd_list(center),
                    "Ori_9DOF_size": _rnd_list(box_size),
                    "instance_pv_bbox": pv_bbox,
                    "absolute_velocity": _rnd_list(vel),
                })
            all_frames[f"frame_{seq}"] = {
                "timestamp": _rnd(ts),
                "visible_objects": visible,
            }

        H, W = (image_hw if image_hw is not None else (None, None))
        metadata = {
            "num_frames": len(frames_sorted),
            "num_instances": len(keep),
            "image_height": H,
            "image_width": W,
            "source_fps": (round(float(source_fps), 2) if source_fps else None),
            "coordinate_system": (
                "World frame = camera at frame 0. X=right, Y=down (gravity +Y), "
                "Z=forward. Units: METRES (DA3 metric depth)."
            ),
            "field_guide": {
                "instance_id": "Unique tracking id, persistent across frames.",
                "instance_name": "Open-vocabulary class (Gemma-3-4b VLM on object crops).",
                "Ori_9DOF_corners": "8 world-frame [x,y,z] corners of the gravity-aligned 3D box.",
                "Ori_9DOF_center": "World-frame box center [x,y,z] (metres).",
                "Ori_9DOF_size": "Box size [L, H, W] (metres): L,W horizontal (object x',z'), H vertical.",
                "instance_pv_bbox": "2D pixel box [x_left, y_top, x_right, y_bottom] in image (W,H).",
                "absolute_velocity": "World-frame velocity [vx,vy,vz] m/s (camera-motion-compensated).",
            },
        }
        return {"metadata": metadata, "ALL_FRAMES": all_frames}

    # Minimum speed (m/s) to be considered "moving".
    _SPEED_THRESH = 0.05
    # Minimum amplitude (m) for oscillation to be reported (else stationary).
    _OSC_AMP_THRESH = 0.05
    # Minimum dominant-axis range (m) to trigger phase analysis.
    # Below this, use simple linear descriptor (noise-level movement).
    _PHASE_MIN_RANGE = 0.10

    def _compute_motion_descriptor(
        self, obs_sorted: List[_FrameObservation], decimals: int = 3
    ) -> str:
        """Compute a phase-aware motion descriptor from centroid trajectory.

        Detects direction changes and returns multi-phase descriptions:
        - "stationary" — negligible movement
        - "moving rightward at ~0.5m/s" — simple linear motion
        - "rightward (0-2s), then leftward (2-4s)" — direction reversal
        - "oscillating left-right, ~0.3m amplitude" — repeated oscillation
        """
        if len(obs_sorted) < 2:
            return "stationary"

        # Extract positions and timestamps.
        positions = np.array([
            [o.step.centroid.x, o.step.centroid.y, o.step.centroid.z]
            for o in obs_sorted
        ], dtype=float)
        times = np.array([o.timestamp_s for o in obs_sorted], dtype=float)

        total_dt = times[-1] - times[0]
        if total_dt < 1e-6:
            return "stationary"

        # Total path length (sum of consecutive distances).
        diffs = np.diff(positions, axis=0)
        seg_dists = np.linalg.norm(diffs, axis=1)
        path_length = float(seg_dists.sum())
        net_disp = positions[-1] - positions[0]
        net_dist = float(np.linalg.norm(net_disp))
        avg_speed = path_length / total_dt

        net_speed = net_dist / total_dt
        if avg_speed < self._SPEED_THRESH:
            return "stationary"

        # --- Classify each time-step as stationary / moving-positive / moving-negative ---
        # on the dominant motion axis.
        abs_net = np.abs(net_disp)
        if net_dist > 0.02:
            dom_ax = int(np.argmax(abs_net))
        else:
            # Net displacement ~0 but path_length > 0 → oscillation likely.
            abs_cumulative = np.abs(diffs).sum(axis=0)
            dom_ax = int(np.argmax(abs_cumulative))

        # If the dominant-axis range is below threshold, movement is too
        # small for meaningful phase analysis — use simple linear descriptor.
        dom_range = float(positions[:, dom_ax].max() - positions[:, dom_ax].min())
        if dom_range < self._PHASE_MIN_RANGE:
            # Tiny range: if net displacement is also negligible, it's just noise.
            if net_speed < self._SPEED_THRESH:
                return "stationary"
            dir_words = self._direction_to_words(net_disp)
            return f"moving {dir_words} at ~{net_speed:.1f}m/s"

        dt_segs = np.diff(times)
        dt_segs = np.where(dt_segs < 1e-6, 1e-6, dt_segs)
        # Per-segment 3D speed and dominant-axis velocity.
        speeds = seg_dists / dt_segs
        vel_dom = diffs[:, dom_ax] / dt_segs

        # Smooth to reduce noise.
        if len(vel_dom) >= 5:
            kernel = np.ones(3) / 3
            vel_dom = np.convolve(vel_dom, kernel, mode="same")
            speeds = np.convolve(speeds, kernel, mode="same")

        # Per-segment state: 0=stationary, +1=positive, -1=negative.
        states = np.zeros(len(vel_dom), dtype=int)
        for i in range(len(vel_dom)):
            if speeds[i] < self._SPEED_THRESH:
                states[i] = 0
            else:
                states[i] = 1 if vel_dom[i] >= 0 else -1

        # --- Build run-length-encoded phases ---
        # Each phase: (start_obs_idx, end_obs_idx, state).
        phases_raw: list = []  # [(start_idx, end_idx, state), ...]
        cur_state = states[0]
        cur_start = 0
        for i in range(1, len(states)):
            if states[i] != cur_state:
                phases_raw.append((cur_start, i, cur_state))
                cur_state = states[i]
                cur_start = i
        phases_raw.append((cur_start, len(states), cur_state))

        # Count direction reversals (ignoring stationary gaps).
        moving_dirs = [s for _, _, s in phases_raw if s != 0]
        n_reversals = 0
        for i in range(1, len(moving_dirs)):
            if moving_dirs[i] != moving_dirs[i - 1]:
                n_reversals += 1

        # --- Oscillation check ---
        if n_reversals >= 3:
            amp = float(positions[:, dom_ax].max() - positions[:, dom_ax].min())
            if amp < self._OSC_AMP_THRESH:
                return "stationary"
            _AX_LABELS = {0: "left-right", 1: "up-down", 2: "forward-backward"}
            ax_label = _AX_LABELS.get(dom_ax, "")
            return f"oscillating {ax_label}, ~{amp:.2f}m amplitude"

        # --- No reversals: simple linear ---
        if n_reversals == 0 and all(s != 0 for _, _, s in phases_raw):
            dir_words = self._direction_to_words(net_disp)
            return f"moving {dir_words} at ~{avg_speed:.1f}m/s"

        # --- Build human-readable phases ---
        # Convert raw phases to (t_start, t_end, description).
        phases_desc: list = []
        for start_idx, end_idx, state in phases_raw:
            t0 = times[start_idx]
            t1 = times[min(end_idx, len(times) - 1)]
            if state == 0:
                phases_desc.append((t0, t1, "stationary"))
            else:
                phase_disp = positions[min(end_idx, len(positions) - 1)] - positions[start_idx]
                dir_w = self._direction_to_words(phase_disp)
                phases_desc.append((t0, t1, dir_w if dir_w else "stationary"))

        # Merge consecutive phases with the same description.
        merged: list = [phases_desc[0]]
        for ph in phases_desc[1:]:
            if ph[2] == merged[-1][2]:
                merged[-1] = (merged[-1][0], ph[1], ph[2])
            else:
                merged.append(ph)

        if len(merged) == 1:
            t0, t1, dw = merged[0]
            if dw == "stationary":
                return "stationary"
            sp = net_dist / total_dt
            return f"moving {dw} at ~{sp:.1f}m/s"

        # Multi-phase: cap at 3 phases.
        parts = []
        for t0, t1, dw in merged[:3]:
            parts.append(f"{dw} ({t0:.1f}-{t1:.1f}s)")
        if len(merged) > 3:
            parts.append("...")
        return ", then ".join(parts)

    @staticmethod
    def _center_to_position_9(center: Tuple[float, float]) -> str:
        """Map normalized (cx, cy) to a 9-way image position label."""
        cx, cy = center
        if cx < 1 / 3:
            col = "left"
        elif cx > 2 / 3:
            col = "right"
        else:
            col = "center"

        if cy < 1 / 3:
            row = "top"
        elif cy > 2 / 3:
            row = "bottom"
        else:
            row = "center"

        if row == "center" and col == "center":
            return "center"
        if row == "center":
            return col
        if col == "center":
            return row
        return f"{row}-{col}"

    @staticmethod
    def _direction_to_words(disp: np.ndarray, thresh: float = 0.25) -> str:
        """Convert a 3D displacement vector to natural language direction.

        Maps axes to words (X=right/left, Y=down/up, Z=forward/backward),
        keeps the top-2 significant components (|value| > *thresh*),
        ordered by magnitude.  Returns '' for near-zero vectors.
        """
        _POS = {0: "rightward", 1: "downward", 2: "forward"}
        _NEG = {0: "leftward", 1: "upward", 2: "backward"}

        dist = float(np.linalg.norm(disp))
        if dist < 1e-6:
            return ""

        normed = disp / dist
        parts: list[tuple[float, str]] = []
        for ax in (2, 0, 1):  # priority: Z > X > Y
            v = normed[ax]
            if abs(v) < thresh:
                continue
            parts.append((abs(v), _POS[ax] if v > 0 else _NEG[ax]))

        if not parts:
            ax = int(np.argmax(np.abs(normed)))
            parts.append((abs(normed[ax]), _POS[ax] if normed[ax] > 0 else _NEG[ax]))

        parts.sort(key=lambda x: -x[0])
        return " and ".join(w for _, w in parts[:2])

    def _compute_camera_motion(self) -> str:
        """Compute a human-readable camera motion summary from ego poses.

        Uses the stored ``_ego_poses_cw`` (camera-to-world transforms) and
        ``_frame_timestamps`` to compute total displacement, average speed,
        and dominant direction of the camera over the video.
        """
        if len(self._ego_poses_cw) < 2:
            return "stationary"

        fidxs = sorted(self._ego_poses_cw.keys())
        positions = np.array([self._ego_poses_cw[f][:3, 3] for f in fidxs])
        t_start = self._frame_timestamps.get(fidxs[0], 0.0)
        t_end = self._frame_timestamps.get(fidxs[-1], 0.0)
        dt = t_end - t_start

        disp = positions[-1] - positions[0]
        total_disp = float(np.linalg.norm(disp))

        if dt < 1e-6 or total_disp < 0.01:
            return "stationary"

        dir_words = self._direction_to_words(disp)
        if total_disp < 0.3:
            return f"slight drift {dir_words}"
        elif total_disp < 1.0:
            return f"moving {dir_words}"
        else:
            return f"significant motion {dir_words}"

    def reid_merge_tracks(
        self,
        object_crops: Dict[int, Dict[str, object]],
    ) -> Dict[int, Dict[str, object]]:
        """Deep-appearance re-ID merge (runs BEFORE merge_duplicate_tracks).

        Merges tracks that are the same physical object using a DINOv2 embedding
        of each track's best crop — the descriptor mask-IoU / 2D-trajectory dedup
        cannot catch when the duplicate obj_ids mask the object on ALTERNATING
        frames (≈0 mutual mask-IoU) or when an object disappears and is
        re-tracked under a new id (no shared frames).  Two tracks merge when:
          cosine(emb_i, emb_j) ≥ reid_sim_thresh  AND  a spatio-temporal gate
          holds (concurrent → mean 2D-center distance ≤ reid_max_2d_jump;
          disjoint → time gap ≤ reid_max_gap_s).
        Validated on clip_001: thresh 0.68-0.78 merged 2/2 confirmed duplicates
        with 0 false merges, vs raw-pixel cosine's 2-9 false merges.
        """
        cfg = self.config.fusion
        if not getattr(cfg, "reid_merge", False):
            return object_crops
        gids = sorted(
            gid for gid in object_crops
            if gid in self._tracks and self._tracks[gid].observations
        )
        if len(gids) < 2:
            return object_crops

        # Embed each track's best crop with DINOv2.
        path_of = {gid: object_crops[gid].get("path") for gid in gids}
        try:
            emb_by_path = _embed_crop_paths(
                [p for p in path_of.values() if p], cfg.reid_model,
            )
        except Exception as e:  # pragma: no cover - never block the pipeline on re-ID
            import logging
            logging.getLogger(__name__).warning("re-ID embed failed (skipping): %s", e)
            return object_crops
        emb = {gid: emb_by_path[p] for gid, p in path_of.items()
               if p in emb_by_path}
        gids = [g for g in gids if g in emb]
        if len(gids) < 2:
            return object_crops

        # Per-track temporal span + 2D trajectory (normalised image coords).
        span, traj = {}, {}
        for gid in gids:
            obs = self._tracks[gid].observations
            traj[gid] = {o.frame_idx: np.array(o.mask_center_2d, float) for o in obs}
            ts = [o.timestamp_s for o in obs]
            span[gid] = (min(ts), max(ts))

        thr = float(getattr(cfg, "reid_sim_thresh", 0.72))
        max_2d = float(getattr(cfg, "reid_max_2d_jump", 0.25))
        max_gap = float(getattr(cfg, "reid_max_gap_s", 0.6))

        parent = {g: g for g in gids}
        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for a_i in range(len(gids)):
            for b_i in range(a_i + 1, len(gids)):
                a, b = gids[a_i], gids[b_i]
                if _find(a) == _find(b):
                    continue
                if float(np.dot(emb[a], emb[b])) < thr:
                    continue
                # spatio-temporal gate
                ov = min(span[a][1], span[b][1]) - max(span[a][0], span[b][0])
                if ov > 0:  # concurrent → must occupy a consistent image region
                    shared = set(traj[a]) & set(traj[b])
                    if shared:
                        md = float(np.mean([np.linalg.norm(traj[a][f] - traj[b][f]) for f in shared]))
                        if md > max_2d:
                            continue
                else:       # disjoint (disappear→reappear) → small time gap only
                    if abs(ov) > max_gap:
                        continue
                ra, rb = _find(a), _find(b)
                na = len(self._tracks[ra].observations); nb = len(self._tracks[rb].observations)
                if nb > na:
                    ra, rb = rb, ra
                parent[rb] = ra

        # Apply merges: collapse track states + rebuild object_crops.
        groups: Dict[int, List[int]] = {}
        for g in gids:
            groups.setdefault(_find(g), []).append(g)
        new_crops = dict(object_crops)
        for root, members in groups.items():
            if len(members) <= 1:
                continue
            best = max(members, key=lambda g: len(self._tracks[g].observations))
            for g in members:
                if g != best:
                    self._merge_track_states(best, g)
                    new_crops.pop(g, None)
            new_crops[best] = object_crops[best]
        return new_crops

    def merge_duplicate_tracks(
        self,
        object_crops: Dict[int, Dict[str, object]],
    ) -> Dict[int, Dict[str, object]]:
        """Post-hoc deduplication: merge tracks that are the same re-tracked object.

        SAM3 sometimes loses an object and re-tracks it under a new ID.
        This method detects such duplicates by comparing crop image
        cosine similarity (must be > dedup_crop_sim_thresh).

        Modifies ``self._tracks`` in-place (via ``_merge_track_states``)
        and returns an updated ``object_crops`` dict with merged entries.

        Args:
            object_crops: Per-track crop references, mapping
                ``global_track_id → {"path": str, "source_frame_idx": int}``.

        Returns:
            Updated object_crops with duplicate entries removed.
        """
        import cv2 as _cv2

        cfg = self.config.fusion
        if not getattr(cfg, "enable_post_dedup", True):
            # Master kill-switch: keep every fused track as-is.
            return object_crops
        gids = sorted(
            gid for gid in object_crops
            if gid in self._tracks and self._tracks[gid].observations
        )
        if len(gids) < 2:
            return object_crops

        # 1. Load crop images → L2-normalised feature vectors.
        FEAT_SZ = 64
        features: Dict[int, np.ndarray] = {}
        brightnesses: Dict[int, float] = {}
        for gid in gids:
            img = _cv2.imread(object_crops[gid]["path"])
            if img is None:
                continue
            brightnesses[gid] = float(img.mean())
            img = _cv2.resize(img, (FEAT_SZ, FEAT_SZ)).flatten().astype(np.float32)
            norm = np.linalg.norm(img)
            if norm > 0:
                img /= norm
            features[gid] = img

        # 2. Union-find (keep track with more observations as root).
        parent: Dict[int, int] = {gid: gid for gid in gids}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            na = len(self._tracks[ra].observations) if ra in self._tracks else 0
            nb = len(self._tracks[rb].observations) if rb in self._tracks else 0
            if nb > na:
                ra, rb = rb, ra
            parent[rb] = ra

        # 3. Pairwise comparison: crop cosine similarity AND image-space
        # proximity.  Crop-similarity alone over-merges on uniform-texture
        # scenes (e.g. running-track POV where every patch of red rubber
        # has cosine sim ≈ 0.98).  Require centroids to be close too.
        #
        # Build per-gid 2D trajectory once for reuse in pass 1 + pass 2 + pass 3.
        pass1_traj: Dict[int, Dict[int, np.ndarray]] = {}
        for gid in gids:
            if gid not in self._tracks:
                continue
            pass1_traj[gid] = {
                obs.frame_idx: np.array(obs.mask_center_2d, dtype=float)
                for obs in self._tracks[gid].observations
            }
        PASS1_PROX_MAX = 0.25  # mean centroid distance ≤ 25% image diagonal

        gid_list = sorted(features.keys())
        for i, gi in enumerate(gid_list):
            for gj in gid_list[i + 1:]:
                if _find(gi) == _find(gj):
                    continue
                sim = float(np.dot(features[gi], features[gj]))
                if sim < cfg.dedup_crop_sim_thresh:
                    continue
                # Proximity gate: tracks must occupy similar image region.
                ti = pass1_traj.get(gi); tj = pass1_traj.get(gj)
                if ti is None or tj is None:
                    continue
                shared = set(ti.keys()) & set(tj.keys())
                if len(shared) < 2:
                    # No temporal overlap → can't confirm same physical object.
                    # Skip; pass 3 may catch via co-motion if applicable.
                    continue
                dists = [float(np.linalg.norm(ti[f] - tj[f])) for f in shared]
                if (sum(dists) / len(dists)) > PASS1_PROX_MAX:
                    continue
                _union(gi, gj)

        # 5. Merge track states.
        groups: Dict[int, List[int]] = {}
        for gid in gids:
            groups.setdefault(_find(gid), []).append(gid)

        merged_count = 0
        for root, members in groups.items():
            if len(members) <= 1:
                continue
            for gid in members:
                if gid != root:
                    self._merge_track_states(root, gid)
                    merged_count += 1

        # 6. Rebuild object_crops: one entry per canonical track.
        # Prefer the crop from the track with the MOST observations (i.e. the
        # main track of the merged group, not a shorter sub-track), tiebreak
        # by brightness.  This avoids the failure mode where a small bright
        # sub-region (e.g. a boat's whitewater bow) wins over the full-object
        # crop just by virtue of higher pixel intensity.
        new_crops: Dict[int, Dict[str, object]] = {}
        for root, members in groups.items():
            def _score(gid: int) -> Tuple[int, float]:
                obs = (len(self._tracks[gid].observations)
                       if gid in self._tracks else 0)
                return (obs, brightnesses.get(gid, -1.0))
            best_gid = max(members, key=_score)
            new_crops[root] = object_crops[best_gid]

        # 7. SECOND-PASS trajectory-aware dedup using 2D IMAGE-FRAME centroids.
        # Background: DA3 reports RELATIVE depth, so the same physical object
        # tracked from two different anchor frames can have wildly different
        # world-frame Z coordinates (e.g. boat reports z=1.03 from frame-0
        # anchor and z=1.75 from frame-12 anchor).  3D centroid distance is
        # therefore unreliable for "same physical object" judgement.
        # Per-observation `mask_center_2d` is in normalized image coords [0,1]
        # which IS stable across tracks: if two tracks both follow the boat
        # at frame 12 they both report the boat's 2D image position there.
        traj_thresh = float(getattr(cfg, "traj_merge_dist", 0.05))
        gids2 = sorted(new_crops.keys())
        if len(gids2) >= 2:
            traj: Dict[int, Dict[int, np.ndarray]] = {}
            for gid in gids2:
                if gid not in self._tracks:
                    continue
                traj[gid] = {
                    obs.frame_idx: np.array(obs.mask_center_2d, dtype=float)
                    for obs in self._tracks[gid].observations
                }

            parent2 = {gid: gid for gid in gids2}

            def _find2(x: int) -> int:
                while parent2[x] != x:
                    parent2[x] = parent2[parent2[x]]
                    x = parent2[x]
                return x

            # AND gate: trajectory proximity must be confirmed by weak crop
            # similarity, so two unrelated objects whose tracks happen to
            # share image-space (e.g. batter + ground patch both centered)
            # are not merged.
            traj_sim_min = 0.65   # crop cosine sim ≥ this when traj is close

            for i, gi in enumerate(gids2):
                ti = traj.get(gi)
                if ti is None:
                    continue
                for gj in gids2[i + 1:]:
                    tj = traj.get(gj)
                    if tj is None:
                        continue
                    if _find2(gi) == _find2(gj):
                        continue
                    shared = set(ti.keys()) & set(tj.keys())
                    if len(shared) < 4:  # require ≥4 shared frames (in_session has many)
                        continue
                    dists = [float(np.linalg.norm(ti[f] - tj[f])) for f in shared]
                    mean_d = sum(dists) / len(dists)
                    if mean_d >= traj_thresh:
                        continue
                    # Confirm with crop visual similarity at a weaker threshold
                    # than pass 1.  Both signals must hold.
                    feat_i = features.get(gi)
                    feat_j = features.get(gj)
                    if feat_i is None or feat_j is None:
                        continue
                    sim = float(np.dot(feat_i, feat_j))
                    if sim < traj_sim_min:
                        continue
                    # union; keep the longer track as root
                    ri, rj = _find2(gi), _find2(gj)
                    ni = len(self._tracks[ri].observations) if ri in self._tracks else 0
                    nj = len(self._tracks[rj].observations) if rj in self._tracks else 0
                    if nj > ni:
                        ri, rj = rj, ri
                    parent2[rj] = ri

            # ---- Pass 3: RIGID CO-MOTION -----------------------------
            # The strongest "same physical object" signal: two tracks whose
            # 2D image positions move with CONSTANT relative offset over many
            # shared frames.  E.g. SAM 3.1 often splits one person into a
            # "head" track and a "body" track — their relative offset is the
            # head→torso distance, near-constant as the person moves.
            #
            # Test: std(delta_x), std(delta_y) of (pos_A - pos_B) across
            # shared frames < tight threshold (rigid) AND both tracks
            # actually moved a meaningful amount (excludes coincidental
            # co-location of two static background blobs).
            #
            # This pass does NOT require crop similarity — head and torso of
            # the same person look totally different but co-move rigidly.
            MOTION_CONSTANT_STD_MAX  = 0.010  # rigid: delta varies < 1.0% image
            MIN_TRACK_MOTION_RANGE   = 0.08   # both tracks must have moved > 8%
            MIN_PASS3_SHARED_FRAMES  = 12     # need many samples for confidence
            MIN_PASS3_TRACK_OBS      = 5      # protect short tracks from absorption
            PASS3_CROP_SIM_MIN       = 0.50   # weak visual confirmation
            PASS3_REL_STD_RATIO      = 0.20   # std(delta) / motion_range < 0.2:
                                              # delta jitter must be << track motion
            PASS3_MEAN_DIST_MAX      = 0.20   # parts of one physical object must be within 20%
                                              # of image diagonal; PREVENTS merging distant
                                              # scene parts that happen to co-move because of
                                              # camera pan (e.g. running POV — track edge top-
                                              # left and bottom-right both "move" with camera
                                              # but are NOT one object).

            for i, gi in enumerate(gids2):
                ti = traj.get(gi)
                if ti is None or len(ti) < MIN_PASS3_TRACK_OBS:
                    continue
                for gj in gids2[i + 1:]:
                    tj = traj.get(gj)
                    if tj is None or len(tj) < MIN_PASS3_TRACK_OBS:
                        continue
                    if _find2(gi) == _find2(gj):
                        continue
                    shared = sorted(set(ti.keys()) & set(tj.keys()))
                    if len(shared) < MIN_PASS3_SHARED_FRAMES:
                        continue

                    a = np.array([ti[f] for f in shared], dtype=float)  # (N,2)
                    b = np.array([tj[f] for f in shared], dtype=float)  # (N,2)
                    delta = a - b                                        # (N,2)

                    std_dx = float(delta[:, 0].std())
                    std_dy = float(delta[:, 1].std())
                    if max(std_dx, std_dy) >= MOTION_CONSTANT_STD_MAX:
                        continue

                    # Proximity gate: parts of one physical object are CLOSE
                    # in image space.  Two far-apart scene elements that both
                    # "move with the camera" (POV / pan video) would otherwise
                    # be falsely merged.
                    mean_dist = float(np.linalg.norm(delta, axis=1).mean())
                    if mean_dist > PASS3_MEAN_DIST_MAX:
                        continue

                    # Both tracks must have shown real motion (otherwise we
                    # can't tell co-motion from coincidental co-location).
                    a_range = float(max(a[:, 0].max() - a[:, 0].min(),
                                        a[:, 1].max() - a[:, 1].min()))
                    b_range = float(max(b[:, 0].max() - b[:, 0].min(),
                                        b[:, 1].max() - b[:, 1].min()))
                    if min(a_range, b_range) < MIN_TRACK_MOTION_RANGE:
                        continue

                    # The delta-jitter must be << each track's motion range.
                    # E.g. if a track moves 0.20 image-units and delta jitters
                    # 0.01 (5%), they're truly rigid.  If a track moves 0.02
                    # but delta jitters 0.005 (25%), it's marginal — skip.
                    if max(std_dx, std_dy) >= PASS3_REL_STD_RATIO * min(a_range, b_range):
                        continue

                    # Weak crop-visual confirmation — same physical object's
                    # parts share rough color palette even when content differs.
                    fi = features.get(gi); fj = features.get(gj)
                    if fi is not None and fj is not None:
                        sim = float(np.dot(fi, fj))
                        if sim < PASS3_CROP_SIM_MIN:
                            continue

                    # union; keep longer as root
                    ri, rj = _find2(gi), _find2(gj)
                    ni = len(self._tracks[ri].observations) if ri in self._tracks else 0
                    nj = len(self._tracks[rj].observations) if rj in self._tracks else 0
                    if nj > ni:
                        ri, rj = rj, ri
                    parent2[rj] = ri

            # ---- Pass 4: SEQUENTIAL RE-LINK (disjoint "断帧") ----------------
            # Re-link a track that STARTS right after another ENDS when the
            # object is continuous across the gap.  This is the ONLY pass that
            # merges tracks with no shared frames (mask-IoU / overlap passes
            # cannot).  Every gate is required (small temporal gap AND small 3D
            # centroid jump AND small 2D image jump AND appearance match) so the
            # crop-similarity false positives that plague blurry / low-texture
            # scenes (uniform outdoor palette → sim≈0.9 between unrelated tracks)
            # cannot trigger a merge.  The 3D-continuity gate is the
            # discriminative one: over a <0.6 s gap a real object barely moves,
            # while two distinct objects are >>0.2 world-units apart.
            cfgf = self.config.fusion
            if getattr(cfgf, "enable_seq_relink", True):
                max_gap = float(getattr(cfgf, "seq_relink_max_gap_s", 0.6))
                max_3d = float(getattr(cfgf, "seq_relink_max_3d_jump", 0.2))
                max_2d = float(getattr(cfgf, "seq_relink_max_2d_jump", 0.2))
                sim_min = float(getattr(cfgf, "seq_relink_crop_sim", 0.88))

                def _c3(o):
                    c = o.step.centroid
                    return np.array([c.x, c.y, c.z], dtype=float)

                ends: Dict[int, dict] = {}
                for gid in gids2:
                    st = self._tracks.get(gid)
                    if not st or not st.observations:
                        continue
                    ob = sorted(st.observations, key=lambda o: o.frame_idx)
                    ends[gid] = dict(
                        t0=ob[0].timestamp_s, t1=ob[-1].timestamp_s,
                        c0=_c3(ob[0]), c1=_c3(ob[-1]),
                        p0=np.array(ob[0].mask_center_2d, dtype=float),
                        p1=np.array(ob[-1].mask_center_2d, dtype=float),
                    )

                eids = sorted(ends.keys())
                for a in eids:
                    for b in eids:
                        if a == b or _find2(a) == _find2(b):
                            continue
                        ea, eb = ends[a], ends[b]
                        gap = eb["t0"] - ea["t1"]          # require A ends before B starts
                        if gap < 0 or gap > max_gap:
                            continue
                        if float(np.linalg.norm(ea["c1"] - eb["c0"])) > max_3d:
                            continue
                        if float(np.linalg.norm(ea["p1"] - eb["p0"])) > max_2d:
                            continue
                        fa, fb = features.get(a), features.get(b)
                        if fa is None or fb is None or float(np.dot(fa, fb)) < sim_min:
                            continue
                        ra, rb = _find2(a), _find2(b)
                        na = len(self._tracks[ra].observations) if ra in self._tracks else 0
                        nb = len(self._tracks[rb].observations) if rb in self._tracks else 0
                        if nb > na:
                            ra, rb = rb, ra
                        parent2[rb] = ra

            groups2: Dict[int, List[int]] = {}
            for gid in gids2:
                groups2.setdefault(_find2(gid), []).append(gid)

            new_crops2: Dict[int, Dict[str, object]] = {}
            for root, members in groups2.items():
                if len(members) > 1:
                    for gid in members:
                        if gid != root:
                            self._merge_track_states(root, gid)
                # Use root's existing crop (already chosen as best in step 6)
                new_crops2[root] = new_crops[root]
            return new_crops2

        return new_crops

    def serialize_4dsg(
        self,
        object_crops: Optional[Dict[int, Dict[str, object]]] = None,
    ) -> str:
        """Serialize Step 8 JSON (compact, no indentation)."""
        return json.dumps(
            self.build_4dsg_dict(object_crops=object_crops),
            separators=(",", ":"),
            sort_keys=False,
        )

    def _build_candidates(self, frame: FastFrameInput, T_cw_t: np.ndarray,
                          K_inv: np.ndarray, frame_ctx: Optional[dict] = None) -> List[_Candidate]:
        out: List[_Candidate] = []
        for det in frame.detections:
            # Score filtering is done in Step 3 (SAM3 wrapper);
            # Step 4 only applies geometric filters (conf/min_points/max_extent).
            mask = det.mask.astype(bool, copy=False)
            points_world = self._backproject_mask_points(
                mask=mask,
                depth_t=frame.depth_t,
                K_inv=K_inv,
                T_cw_t=T_cw_t,
                depth_conf_t=frame.depth_conf_t,
                frame_depth_ok=(frame_ctx.get("depth_ok") if frame_ctx else None),
            )
            if points_world.shape[0] < self.config.depth_filter.min_points:
                continue

            shape = build_shape_token(points_world)
            # max_extent filter only meaningful with metric depth;
            # relative depth has arbitrary scale → skip to avoid false rejection.
            if frame.depth_is_metric:
                extents = np.array(
                    [
                        shape.x_max - shape.x_min,
                        shape.y_max - shape.y_min,
                        shape.z_max - shape.z_min,
                    ],
                    dtype=float,
                )
                if np.any(extents > self.config.depth_filter.max_extent):
                    continue

            # 2D mask bbox (normalized) for image-position descriptor.
            # Use np.any along axes to find bbox — O(H+W) instead of O(H*W).
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.argmax(rows), mask.shape[0] - 1 - np.argmax(rows[::-1])
            cmin, cmax = np.argmax(cols), mask.shape[1] - 1 - np.argmax(cols[::-1])
            _h, _w = mask.shape[:2]
            _cx = float((cmin + cmax) / 2) / _w
            _cy = float((rmin + rmax) / 2) / _h

            centroid = build_centroid_token(points_world)

            # Dynamic-targets export: fit a per-frame 9DOF gravity-aligned box from
            # this object's world points (gated; the default 5.56 path skips it).
            obb = None
            if getattr(self.config, "dynamic_targets", None) is not None and \
                    self.config.dynamic_targets.enabled:
                from rose.engine.export.obb import gravity_aligned_obb
                obb = gravity_aligned_obb(
                    points_world,
                    extent_pct=self.config.dynamic_targets.obb_extent_pct,
                    min_points=self.config.dynamic_targets.obb_min_points,
                )

            patch_tokens = mask_to_patch_tokens(
                mask,
                grid_size=self.config.step.grid_size,
                iou_threshold=self.config.step.iou_threshold,
            )
            step = STEPToken(
                patch_tokens=patch_tokens,
                centroid=centroid,
                shape=shape,
                temporal=TemporalToken(t_start=frame.timestamp_s, t_end=frame.timestamp_s),
            )

            out.append(
                _Candidate(
                    run_id=det.run_id,
                    local_obj_id=det.local_obj_id,
                    mask=mask,
                    score=float(det.score),
                    centroid_xyz=np.array([centroid.x, centroid.y, centroid.z], dtype=float),
                    step=step,
                    mask_center_2d=(_cx, _cy),
                    bbox=(int(rmin), int(cmin), int(rmax), int(cmax)),
                    obb=obb,
                )
            )
        return out

    def _fuse_candidates(self, frame_idx: int, candidates: Sequence[_Candidate], depth_is_metric: bool = True) -> Dict[int, _Candidate]:
        if not candidates:
            return {}

        # OPT (H200): precompute pairwise mask IoU on GPU once, reuse below.
        # Avoids ~2s/video of per-pair CPU mask_iou calls (60-80 pairs/frame
        # × 32 frames × ~1ms each = ~2s on (480x910) bool masks).
        iou_matrix = None
        try:
            import torch
            if torch.cuda.is_available() and len(candidates) >= 2:
                # Cross-run pairs only need IoU; same-run pairs are skipped
                # later anyway, but precomputing all is cheaper than branching.
                masks_np = np.stack([c.mask for c in candidates]).astype(np.float32, copy=False)
                K, H, W = masks_np.shape
                with torch.autocast(device_type="cuda", enabled=False):
                    t = torch.from_numpy(masks_np).cuda(non_blocking=True).view(K, H * W)
                    area = t.sum(dim=1)
                    inter = t @ t.t()
                    union = area[:, None] + area[None, :] - inter
                    iou_matrix = (inter / union.clamp(min=1.0)).cpu().numpy()
        except Exception:
            iou_matrix = None

        # 1) Assign provisional global IDs by local (run_id, obj_id)
        #    Archived tracks do NOT participate in matching (spec §5):
        #    if the old gid maps to an archived track, allocate a fresh gid.
        candidate_gids: List[int] = []
        keys: List[RunKey] = []
        for cand in candidates:
            key = (cand.run_id, cand.local_obj_id)
            keys.append(key)
            gid = self._local_to_global.get(key)
            if gid is not None:
                track = self._tracks.get(gid)
                if track is not None and track.status == "archived":
                    gid = None  # force new allocation
            if gid is None:
                gid = self._allocate_global_id()
                self._local_to_global[key] = gid
            candidate_gids.append(gid)

        # 2) Cross-run fusion with score-desc greedy order
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(keep: int, drop: int) -> int:
            rk = find(keep)
            rd = find(drop)
            if rk != rd:
                parent[rd] = rk
            return rk

        for gid in candidate_gids:
            parent.setdefault(gid, gid)

        order = sorted(range(len(candidates)), key=lambda i: candidates[i].score, reverse=True)
        for i_pos, i in enumerate(order):
            ci = candidates[i]
            gi = find(candidate_gids[i])
            # Skip if this track is archived (should not participate in merging)
            ti = self._tracks.get(gi)
            if ti is not None and ti.status == "archived":
                continue
            for j in order[i_pos + 1 :]:
                cj = candidates[j]
                if ci.run_id == cj.run_id:
                    continue
                gj = find(candidate_gids[j])
                if gi == gj:
                    continue
                # Skip if target track is archived
                tj = self._tracks.get(gj)
                if tj is not None and tj.status == "archived":
                    continue

                # Fast bbox overlap check — skip expensive mask IoU if bboxes
                # don't overlap (IoU must be 0).
                ri_min, ci_min, ri_max, ci_max = ci.bbox
                rj_min, cj_min, rj_max, cj_max = cj.bbox
                if ri_max < rj_min or rj_max < ri_min or ci_max < cj_min or cj_max < ci_min:
                    continue

                if iou_matrix is not None:
                    iou = float(iou_matrix[i, j])
                else:
                    iou = self._mask_iou(ci.mask, cj.mask)
                if iou <= self.config.fusion.cross_run_iou_thresh:
                    continue

                cdist = float(np.linalg.norm(ci.centroid_xyz - cj.centroid_xyz))
                if depth_is_metric:
                    if cdist >= self.config.fusion.merge_centroid_dist_m:
                        continue
                else:
                    # Relative depth: normalise distance by mean Z (depth) of the
                    # two centroids so the gate is scale-invariant.
                    mean_z = (abs(float(ci.centroid_xyz[2])) + abs(float(cj.centroid_xyz[2]))) / 2
                    if mean_z < 1e-6 or (cdist / mean_z) >= self.config.fusion.merge_centroid_dist_rel:
                        continue

                li = self._tracks[gi].last_seen_t if gi in self._tracks and self._tracks[gi].last_seen_t is not None else frame_idx
                lj = self._tracks[gj].last_seen_t if gj in self._tracks and self._tracks[gj].last_seen_t is not None else frame_idx
                if abs(li - lj) > self.config.fusion.merge_temporal_gap:
                    continue

                gi = union(gi, gj)

        # 3) Merge historical track states for merged global IDs
        for gid in sorted(set(candidate_gids)):
            root = find(gid)
            if gid != root:
                self._merge_track_states(root, gid)

        # Keep mapping consistent globally
        for key, gid in list(self._local_to_global.items()):
            self._local_to_global[key] = find(gid)

        # 4) Keep only highest-score candidate per fused global ID for this frame
        best_idx_for_gid: Dict[int, int] = {}
        for idx, cand in enumerate(candidates):
            gid = find(candidate_gids[idx])
            best_idx = best_idx_for_gid.get(gid)
            if best_idx is None or cand.score > candidates[best_idx].score:
                best_idx_for_gid[gid] = idx

        winners: Dict[int, _Candidate] = {}
        for gid, idx in best_idx_for_gid.items():
            winners[gid] = candidates[idx]

        # Update current local ID mapping to winning global IDs.
        for idx, key in enumerate(keys):
            self._local_to_global[key] = find(candidate_gids[idx])

        return winners

    def _merge_track_states(self, keep_gid: int, drop_gid: int) -> None:
        if keep_gid == drop_gid:
            return

        keep = self._tracks.get(keep_gid)
        drop = self._tracks.get(drop_gid)
        if drop is None:
            return

        if keep is None:
            keep = _TrackState(track_id=keep_gid)
            self._tracks[keep_gid] = keep

        # Merge and deduplicate observations: one per frame_idx.
        # Prefer keep's observation (higher-score winner) over drop's.
        obs_by_frame: Dict[int, _FrameObservation] = {}
        for obs in drop.observations:
            obs_by_frame[obs.frame_idx] = obs
        for obs in keep.observations:
            obs_by_frame[obs.frame_idx] = obs  # keep overwrites drop
        keep.observations = sorted(obs_by_frame.values(), key=lambda x: x.frame_idx)

        if keep.last_seen_t is None or (drop.last_seen_t is not None and drop.last_seen_t > keep.last_seen_t):
            keep.last_seen_t = drop.last_seen_t
            keep.last_centroid = drop.last_centroid

        status_rank = {"active": 0, "lost": 1, "archived": 2}
        keep.status = min((keep.status, drop.status), key=lambda s: status_rank.get(s, 99))
        keep.missing_streak = min(keep.missing_streak, drop.missing_streak)

        del self._tracks[drop_gid]

    def _backproject_mask_points(
        self,
        mask: np.ndarray,
        depth_t: np.ndarray,
        K_inv: np.ndarray,
        T_cw_t: np.ndarray,
        depth_conf_t: Optional[np.ndarray],
        frame_depth_ok: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Backproject masked depth pixels to 3D world coordinates.

        Args:
            K_inv: Pre-computed inverse intrinsics (3,3) float64.
                   Computed once per frame by the caller.
            frame_depth_ok: D7 optimization — pre-computed per-frame validity
                mask ``(depth_finite & depth_positive & conf >= thresh)``.
                When provided, we skip the redundant 3-way numpy reduction
                across the full HxW image for each mask.  ``None`` falls back
                to per-mask computation (backward compatible).
        """
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")
        if depth_t.ndim != 2:
            raise ValueError(f"depth_t must be 2D, got shape {depth_t.shape}")
        if mask.shape != depth_t.shape:
            raise ValueError(
                f"mask/depth shape mismatch: mask={mask.shape}, depth={depth_t.shape}"
            )

        if frame_depth_ok is not None:
            # D7 v2 fast path: scan ONLY mask pixels (size N), not full HxW grid (size 720*1280).
            # Most masks cover < 1% of image; scanning HxW per mask is the real cost.
            v, u = np.nonzero(mask)
            if v.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            ok = frame_depth_ok[v, u]
            if not ok.all():
                v = v[ok]; u = u[ok]
                if v.size == 0:
                    return np.zeros((0, 3), dtype=np.float32)
            d = depth_t[v, u].astype(np.float64)
        else:
            # Slow path: original logic (HxW scan per mask)
            if depth_conf_t is None:
                depth_conf_t = np.ones_like(depth_t, dtype=np.float32)
            elif depth_conf_t.shape != depth_t.shape:
                raise ValueError(
                    f"depth_conf_t shape mismatch: conf={depth_conf_t.shape}, depth={depth_t.shape}"
                )
            valid = mask & np.isfinite(depth_t) & (depth_t > 0.0)
            valid &= depth_conf_t >= self.config.depth_filter.conf_thresh
            v, u = np.nonzero(valid)
            if u.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            d = depth_t[v, u].astype(np.float64)
        uv1 = np.stack([u.astype(np.float64), v.astype(np.float64), np.ones_like(d)], axis=0)

        rays = K_inv @ uv1
        p_cam = rays * d  # (3, N)

        p_cam_h = np.vstack([p_cam, np.ones((1, p_cam.shape[1]), dtype=np.float64)])
        p_world_h = T_cw_t.astype(np.float64) @ p_cam_h
        # Cast to float32 for storage/serialization (spec §5.5)
        return p_world_h[:3, :].T.astype(np.float32)

    def _allocate_global_id(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

__all__ = [
    "ROSEPipeline",
    "FastFrameInput",
    "FastLocalDetection",
]
