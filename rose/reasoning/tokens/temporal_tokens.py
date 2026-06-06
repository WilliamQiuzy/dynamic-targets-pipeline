"""Temporal tokens for STEP encoding.

Paper reference (Section 3.2, Page 3, Lines 291-294):
    "a pair of temporal tokens θ_t^k = (t_start, t_end) encoding the time of
    first appearance and disappearance of the object."

Implementation Strategy:
    Temporal tokens are TRACK-LEVEL, not frame-level:
    - t_start: The FIRST frame where this object appeared
    - t_end: The LAST frame where this object was observed

    Two-Phase Construction:
    1. Phase 4 (STEP Encoding): Initialize with single-frame placeholder
       - During initial token creation: t_start = t_end = current_frame_idx
       - This allows STEP tokens to be created immediately without waiting for tracking

    2. Phase 6 (Object Tracking): Update with true track-level timestamps
       - After cross-frame association via ObjectTracker
       - t_start = min(track.frame_indices)
       - t_end = max(track.frame_indices)
       - The tracker maintains the full temporal history of each object

    Workflow:
        frame_t: Create STEPToken with TemporalToken(t_start=t, t_end=t)
        frame_t+1: Tracker associates object → update temporal token
        frame_t+2: Continue tracking → temporal token reflects [t_start, t_end]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalToken:
    """Temporal information for a tracked object.

    Attributes:
        t_start: Timestamp (seconds) of first appearance.
        t_end: Timestamp (seconds) of last observed appearance.

    Note:
        In Phase 4, both are set to current frame timestamp (placeholder).
        In Phase 6, tracker updates these to reflect the true track span.
    """
    t_start: float
    t_end: float
