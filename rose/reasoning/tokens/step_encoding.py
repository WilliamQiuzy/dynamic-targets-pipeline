"""STEP token encoding.

Paper reference (Section 3.2, Eq. 4):
    S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t, c_t^k, s_t^k, θ_t^k}

    Where:
    - τ: Image patch tokens (16×16 grid, IoU > 0.5) — actual masked image crops
    - c: Centroid token (3D center)
    - s: Shape token (Gaussian statistics + extents)
    - θ: Temporal token (t_start, t_end)

The patch tokens τ are **visual tokens**: actual image regions cropped from the
masked image, not mere (row, col, iou) metadata.  They are fed to the VLM's
vision encoder so it can infer the object's semantic category.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from rose.reasoning.tokens.geometry_tokens import CentroidToken, ShapeToken, build_centroid_token, build_shape_token
from rose.reasoning.tokens.patch_tokenizer import PatchToken, mask_to_patch_tokens
from rose.reasoning.tokens.temporal_tokens import TemporalToken


@dataclass
class STEPToken:
    """Spatio-Temporal Tokenized Patch Encoding for a single object instance.

    This compact multimodal representation captures:
    - Semantic: image patch tokens (masked image crops → VLM vision input)
    - Geometric: centroid + shape tokens (3D structure → VLM text input)
    - Temporal: appearance/disappearance timestamps (track-level → VLM text input)

    Attributes:
        patch_tokens: List of PatchToken, each with (row, col, iou, image_crop).
        centroid: 3D center (x, y, z) from backprojected mask points.
        shape: Gaussian statistics (mu, sigma, min, max) x 3 axes.
        temporal: (t_start, t_end) track-level timestamps.
    """
    patch_tokens: List[PatchToken]
    centroid: CentroidToken
    shape: ShapeToken
    temporal: TemporalToken


def build_step_token(
    mask: np.ndarray,
    points_xyz: np.ndarray,
    t_start: int,
    t_end: int,
    grid_size: int = 16,
    iou_threshold: float = 0.5,
    image: Optional[np.ndarray] = None,
    mask_outside: bool = True,
    crop_size: Optional[int] = None,
) -> STEPToken:
    """Build STEP tokens for a single object instance.

    Paper workflow (Section 3.2):
    1. Isolate masked image (zero out non-mask pixels)
    2. Partition into 16x16 grid
    3. Retain cells with IoU > 0.5 as image patch tokens
    4. Compute centroid from 3D points
    5. Compute shape token (Gaussian + extents)
    6. Attach temporal token (t_start, t_end)

    Args:
        mask: (H, W) boolean mask from SAM segmentation.
        points_xyz: (N, 3) 3D points from backprojection.
        t_start: First appearance frame.
        t_end: Last appearance frame.
        grid_size: Grid dimension (default 16 for 256 patches).
        iou_threshold: Minimum IoU to retain patch (default 0.5).
        image: (H, W, 3) uint8 RGB image.  When provided, patch tokens will
            include actual image crops (the paper's "image patch tokens").
        mask_outside: Zero out non-mask pixels within each cell crop.
        crop_size: Resize each crop to (crop_size, crop_size); None = native.

    Returns:
        STEPToken with all components initialized.
    """
    patch_tokens = mask_to_patch_tokens(
        mask,
        grid_size=grid_size,
        iou_threshold=iou_threshold,
        image=image,
        mask_outside=mask_outside,
        crop_size=crop_size,
    )
    centroid = build_centroid_token(points_xyz)
    shape = build_shape_token(points_xyz)
    temporal = TemporalToken(t_start=t_start, t_end=t_end)
    return STEPToken(
        patch_tokens=patch_tokens,
        centroid=centroid,
        shape=shape,
        temporal=temporal,
    )


def update_temporal_token(
    step: STEPToken,
    t_start: int,
    t_end: int,
) -> STEPToken:
    """Update temporal token with track-level timestamps.

    Called after cross-frame association to replace single-frame placeholder
    temporal tokens with the true track-level span.
    """
    return STEPToken(
        patch_tokens=step.patch_tokens,
        centroid=step.centroid,
        shape=step.shape,
        temporal=TemporalToken(t_start=t_start, t_end=t_end),
    )
