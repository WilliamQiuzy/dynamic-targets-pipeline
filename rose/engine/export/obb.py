"""9DOF oriented 3D bounding box from a backprojected world-frame point cloud.

ROSE's world frame (set by DA3, world = camera at frame 0):
    X = right, Y = down (gravity points +Y), Z = forward (into scene).

For indoor / driving objects the physically-correct box is *gravity-aligned*:
its vertical axis is locked to world-up (-Y) and the only free rotation is the
yaw about the vertical axis. Estimating a full free SO(3) orientation from a
single-view, monocular-depth point cloud is unstable (the cloud is a thin shell
on the visible surface, so PCA tilts the box toward the viewing direction); the
gravity-aligned yaw-only box is far more robust and is the standard convention
for indoor 3D detection (SUN RGB-D, ScanNet) and autonomous driving (KITTI,
nuScenes). We therefore return a yaw-only 9DOF box: center (3) + size (3) +
rotation (yaw about Y; roll = pitch = 0).

All functions are pure NumPy and have no ROSE dependencies, so they unit-test in
isolation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def _yaw_rotation(yaw: float) -> np.ndarray:
    """Right-handed rotation about the world Y (down) axis by ``yaw`` radians.

    Maps object-frame coords to world-frame coords: p_world = R @ p_obj.
        [ cos  0  sin]
        [  0   1   0 ]
        [-sin  0  cos]
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
    )


def corners_from_center_size_yaw(
    center: np.ndarray, size: np.ndarray, yaw: float
) -> np.ndarray:
    """8 world-frame corners of a gravity-aligned box.

    Args:
        center: (3,) box center in world frame [x, y, z].
        size:   (3,) box extent [L, H, W] along object (x', y=world-down, z') axes.
        yaw:    rotation about world Y, radians.

    Returns:
        (8, 3) corner coordinates in world frame. Ordering (object-frame signs of
        [x', y, z']), deterministic so downstream consumers can index faces:
            0:(-,-,-) 1:(+,-,-) 2:(+,-,+) 3:(-,-,+)   (top face, y = -H/2 = up)
            4:(-,+,-) 5:(+,+,-) 6:(+,+,+) 7:(-,+,+)   (bottom face, y = +H/2 = down)
        (world Y points down, so y = -H/2 is the physically-upper face.)
    """
    L, H, W = float(size[0]), float(size[1]), float(size[2])
    hx, hy, hz = L / 2.0, H / 2.0, W / 2.0
    signs = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, -1, +1],
            [-1, -1, +1],
            [-1, +1, -1],
            [+1, +1, -1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    local = signs * np.array([hx, hy, hz], dtype=np.float64)  # (8,3) object frame
    R = _yaw_rotation(yaw)
    world = local @ R.T + np.asarray(center, dtype=np.float64)  # (8,3)
    return world


def gravity_aligned_obb(
    points_world: np.ndarray,
    extent_pct: float = 2.0,
    min_points: int = 12,
) -> Optional[dict]:
    """Fit a gravity-aligned (yaw-only) 9DOF box to a world-frame point cloud.

    Args:
        points_world: (N, 3) float, world frame (X=right, Y=down, Z=forward).
        extent_pct:   robust trim percentile per axis (2.0 → use 2nd..98th pct
                      bounds, matching ROSE's ShapeToken) to reject depth-edge
                      outliers. Falls back to raw min/max when N is small.
        min_points:   below this, return None (caller falls back / skips).

    Returns:
        dict with:
            center: [cx, cy, cz]  (world)
            size:   [L, H, W]     (object x', world-down y, object z')
            yaw:    float radians (rotation about world Y)
            corners: (8, 3) list   (world; see corners_from_center_size_yaw)
        or None if too few points.
    """
    if points_world is None:
        return None
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < min_points:
        return None

    # --- 0) Statistical outlier removal -------------------------------------
    # Monocular depth bleeds at object/background edges → a few points land far
    # behind/around the object and inflate the box (oversize) and tilt the yaw.
    # Drop points whose distance from the robust center exceeds median + 2.5*MAD.
    if pts.shape[0] >= 30:
        med = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - med, axis=1)
        dmed = np.median(dist)
        mad = np.median(np.abs(dist - dmed)) + 1e-6
        inliers = dist <= dmed + 2.5 * 1.4826 * mad
        if int(inliers.sum()) >= min_points:
            pts = pts[inliers]

    # --- 1) Yaw from PCA on the horizontal (X, Z) projection -----------------
    # The vertical axis is locked to world Y, so orientation is a single angle in
    # the ground plane. PCA on the horizontal footprint gives the dominant
    # ground-plane direction; that direction defines the object's x' axis.
    horiz = pts[:, [0, 2]]  # (N, 2): columns are world X, world Z
    horiz_mean = horiz.mean(axis=0)
    hc = horiz - horiz_mean
    cov = (hc.T @ hc) / max(hc.shape[0] - 1, 1)
    # Symmetric 2x2 → eigh gives ascending eigenvalues; take the largest.
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, int(np.argmax(evals))]  # (2,) = [dir_x, dir_z] in world (X,Z)
    # The object x' axis equals R(yaw) @ [1,0,0] = (cos yaw, -sin yaw) in world (X,Z),
    # so to align x' with the PCA principal direction: yaw = atan2(-dir_z, dir_x).
    # (Using atan2(+dir_z, dir_x) yields -yaw and de-aligns the size axes.)
    yaw = math.atan2(-float(principal[1]), float(principal[0]))

    # --- 2) Rotate points into the object frame (undo yaw) -------------------
    R = _yaw_rotation(yaw)
    pts_obj = (pts - 0.0) @ R  # p_obj = R^T @ p_world  (R is orthonormal → R^-1 = R^T)
    # pts_obj columns: x' (along principal), y (= world Y, unchanged), z'

    # --- 3) Robust axis-aligned bounds in the object frame -------------------
    n = pts_obj.shape[0]
    if extent_pct > 0.0 and n >= 20:
        mins = np.percentile(pts_obj, extent_pct, axis=0)
        maxs = np.percentile(pts_obj, 100.0 - extent_pct, axis=0)
    else:
        mins = pts_obj.min(axis=0)
        maxs = pts_obj.max(axis=0)
    size_obj = np.maximum(maxs - mins, 1e-4)  # [L, H, W], guard against zero
    center_obj = (mins + maxs) / 2.0
    # Object-frame center back to world frame.
    center_world = R @ center_obj

    corners = corners_from_center_size_yaw(center_world, size_obj, yaw)

    return {
        "center": [float(center_world[0]), float(center_world[1]), float(center_world[2])],
        "size": [float(size_obj[0]), float(size_obj[1]), float(size_obj[2])],
        "yaw": float(yaw),
        "corners": corners.tolist(),
    }
