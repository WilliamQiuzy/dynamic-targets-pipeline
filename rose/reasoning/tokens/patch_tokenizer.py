"""Patch tokenizer for STEP encoding.

Paper reference (Section 3.2, Page 3):
    "The object mask is isolated by coloring all in-mask pixels. The masked
    image is partitioned into a fixed 16x16 grid, yielding 256 patches. Each
    grid cell is evaluated by its IoU with the mask. Cells with IoU>0.5 are
    retained as image patch tokens."

In the paper, tau (patch tokens) are actual image regions -- visual tokens fed
to the VLM's vision encoder.  Each PatchToken therefore carries both the grid
coordinates (row, col, iou) **and** the pixel crop of that cell region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class PatchToken:
    """A single image patch token from the 16x16 grid.

    Attributes:
        row: Grid row index (0-based).
        col: Grid column index (0-based).
        iou: Mask coverage ratio within this cell (intersection / cell_area).
        image_crop: (cell_h, cell_w, 3) uint8 RGB crop of the masked image
            region.  ``None`` when the tokenizer is called without an image
            (backward-compatible fallback).
    """
    row: int
    col: int
    iou: float
    image_crop: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


def mask_to_patch_tokens(
    mask: np.ndarray,
    grid_size: int = 16,
    iou_threshold: float = 0.5,
    image: Optional[np.ndarray] = None,
    mask_outside: bool = True,
    crop_size: Optional[int] = None,
) -> List[PatchToken]:
    """Convert a binary mask into patch tokens based on IoU with grid cells.

    When *image* is provided, each retained cell also carries the actual pixel
    crop (the "image patch token" in the paper).  Non-mask pixels within the
    cell are zeroed out when *mask_outside* is True (paper default).

    Args:
        mask: (H, W) boolean mask.
        grid_size: Grid dimension (default 16 for 16x16 = 256 patches).
        iou_threshold: Minimum coverage ratio to retain a patch (default 0.5).
        image: (H, W, 3) uint8 RGB image.  When provided, each PatchToken
            will include the cropped image region.  When ``None``, tokens
            contain only grid coordinates and iou (backward-compatible).
        mask_outside: If True and *image* is given, zero out pixels outside
            the mask within each cell crop.  This is the paper's behaviour
            ("The object mask is isolated by coloring all in-mask pixels").
        crop_size: If not None, resize each cell crop to (crop_size, crop_size)
            pixels.  ``None`` keeps the native cell resolution.

    Returns:
        List of PatchToken for cells with IoU > threshold.

    Raises:
        ValueError: If mask is not 2D, smaller than grid_size, or if image
            dimensions don't match mask.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    h, w = mask.shape
    if h < grid_size or w < grid_size:
        raise ValueError(
            f"mask dimensions ({h}x{w}) must be >= grid_size ({grid_size}x{grid_size})"
        )

    if image is not None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")
        if image.shape[0] != h or image.shape[1] != w:
            raise ValueError(
                f"image/mask shape mismatch: image={image.shape[:2]}, mask={mask.shape}"
            )

    # Precompute masked image once (avoid per-cell masking)
    masked_image: Optional[np.ndarray] = None
    if image is not None and mask_outside:
        masked_image = image.copy()
        masked_image[~mask] = 0
    elif image is not None:
        masked_image = image  # no masking, use original

    cell_h = h // grid_size
    cell_w = w // grid_size

    # D7 vectorization: compute all 256 cell-IoUs in one numpy reduction
    # instead of 256 Python loops with .sum() each.
    #
    # Strategy:
    #   1. Crop mask to the largest sub-array divisible by (grid_size * cell_h, grid_size * cell_w).
    #      This handles the case where h or w is NOT divisible by grid_size — the original
    #      code special-cased the last row/col by extending y1=h, x1=w.  We replicate that
    #      by computing IoU for the "regular" interior cells via vectorization, then patch
    #      the edge row/column afterwards if h or w is not exactly grid_size * cell_h/w.
    interior_h = cell_h * grid_size
    interior_w = cell_w * grid_size

    # Reshape interior mask to (grid_size, cell_h, grid_size, cell_w),
    # then sum over (cell_h, cell_w) axes → (grid_size, grid_size) intersections.
    interior_mask = mask[:interior_h, :interior_w]
    intersections = interior_mask.reshape(
        grid_size, cell_h, grid_size, cell_w
    ).sum(axis=(1, 3))  # shape (grid_size, grid_size), int

    # Per-cell area (constant for interior cells; edge cells handled below)
    cell_area = cell_h * cell_w
    ious = intersections.astype(np.float32) / float(cell_area)

    # If h or w is not a multiple of grid_size, the last row/col cells
    # extend further (y1=h, x1=w in the original loop).  Recompute those.
    has_edge_row = interior_h < h
    has_edge_col = interior_w < w
    if has_edge_row:
        # Last row (r = grid_size - 1) covers y0 = (grid_size-1)*cell_h .. h
        last_row_mask = mask[(grid_size - 1) * cell_h:h, :]  # (extra_h, w)
        extra_h = last_row_mask.shape[0]
        last_row_inters = last_row_mask[:, :interior_w].reshape(
            extra_h, grid_size, cell_w
        ).sum(axis=(0, 2))
        ious[grid_size - 1, :] = last_row_inters.astype(np.float32) / float(extra_h * cell_w)
    if has_edge_col:
        last_col_mask = mask[:, (grid_size - 1) * cell_w:w]  # (h, extra_w)
        extra_w = last_col_mask.shape[1]
        last_col_inters = last_col_mask[:interior_h, :].reshape(
            grid_size, cell_h, extra_w
        ).sum(axis=(1, 2))
        ious[:, grid_size - 1] = last_col_inters.astype(np.float32) / float(cell_h * extra_w)
    if has_edge_row and has_edge_col:
        # Bottom-right corner cell — full extra block
        corner = mask[(grid_size - 1) * cell_h:h, (grid_size - 1) * cell_w:w]
        ious[grid_size - 1, grid_size - 1] = float(corner.sum()) / float(corner.size)

    # Find cells passing threshold (small list, ~10-30 entries typical)
    keep_mask = ious > iou_threshold
    rows, cols = np.nonzero(keep_mask)

    tokens: List[PatchToken] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        iou = float(ious[r, c])
        crop: Optional[np.ndarray] = None
        if masked_image is not None:
            y0, x0 = r * cell_h, c * cell_w
            y1 = h if r == grid_size - 1 else (r + 1) * cell_h
            x1 = w if c == grid_size - 1 else (c + 1) * cell_w
            crop = masked_image[y0:y1, x0:x1].copy()
            if crop_size is not None:
                crop = cv2.resize(
                    crop, (crop_size, crop_size),
                    interpolation=cv2.INTER_AREA,
                )
        tokens.append(PatchToken(row=int(r), col=int(c), iou=iou, image_crop=crop))

    return tokens
