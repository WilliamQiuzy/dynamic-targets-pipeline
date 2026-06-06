"""ROSE dynamic-targets export module.

Converts ROSE's internal per-object per-frame tracking + geometry into the
client-requested ``ALL_FRAMES`` JSON (per-frame visible objects with 9DOF
oriented 3D boxes, 2D pixel boxes, instance names, and absolute velocity).
See 动态目标管线关键字段.md for the target schema.
"""

from .obb import gravity_aligned_obb, corners_from_center_size_yaw

__all__ = ["gravity_aligned_obb", "corners_from_center_size_yaw"]
