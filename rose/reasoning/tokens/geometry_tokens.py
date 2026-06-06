"""Geometry tokens for STEP encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CentroidToken:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ShapeToken:
    x_mu: float
    x_sigma: float
    x_min: float
    x_max: float
    y_mu: float
    y_sigma: float
    y_min: float
    y_max: float
    z_mu: float
    z_sigma: float
    z_min: float
    z_max: float


def build_centroid_token(points_xyz: np.ndarray) -> CentroidToken:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    mean = points_xyz.mean(axis=0)
    return CentroidToken(x=float(mean[0]), y=float(mean[1]), z=float(mean[2]))


def build_shape_token(
    points_xyz: np.ndarray, extent_pct: float = 2.0
) -> ShapeToken:
    """Build a 3D shape token (per-axis mean/std/min/max) for an object.

    ``x_min/x_max`` etc. are ROBUST percentile bounds, not raw min/max.  A few
    backprojected mask pixels land at wildly wrong depths (mask-edge bleed onto
    background, depth-map noise on far objects).  With metric DA3 depth this
    inflated the 3D extent badly (a ~0.5 m gate reading 5.5 m deep).  Using the
    [extent_pct, 100-extent_pct] percentiles trims those outliers so the extent
    reflects the object's real size.  extent_pct=2.0 → 2nd/98th percentile.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    mu = points_xyz.mean(axis=0)
    sigma = points_xyz.std(axis=0)
    if extent_pct > 0.0 and points_xyz.shape[0] >= 20:
        mins = np.percentile(points_xyz, extent_pct, axis=0)
        maxs = np.percentile(points_xyz, 100.0 - extent_pct, axis=0)
    else:
        # Too few points for a stable percentile → raw min/max.
        mins = points_xyz.min(axis=0)
        maxs = points_xyz.max(axis=0)
    return ShapeToken(
        x_mu=float(mu[0]),
        x_sigma=float(sigma[0]),
        x_min=float(mins[0]),
        x_max=float(maxs[0]),
        y_mu=float(mu[1]),
        y_sigma=float(sigma[1]),
        y_min=float(mins[1]),
        y_max=float(maxs[1]),
        z_mu=float(mu[2]),
        z_sigma=float(sigma[2]),
        z_min=float(mins[2]),
        z_max=float(maxs[2]),
    )
