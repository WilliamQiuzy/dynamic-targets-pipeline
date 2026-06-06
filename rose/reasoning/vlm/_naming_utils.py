"""Shared helpers for VLM instance namers (Gemma / Qwen)."""

from __future__ import annotations

import re
import sys
import types
from typing import List

import numpy as np


def install_device_mesh_shim() -> None:
    """Make ``torch.distributed.tensor.device_mesh.DeviceMesh`` importable on torch 2.4.

    transformers 5.9 expects this path which does not exist in torch 2.4; the
    shim is isolated to the namer modules (ROSE itself never imports transformers).
    """
    if "torch.distributed.tensor.device_mesh" in sys.modules:
        return
    try:
        import torch.distributed.tensor as _t  # exists on 2.4
    except ModuleNotFoundError:
        _t = types.ModuleType("torch.distributed.tensor")
        sys.modules["torch.distributed.tensor"] = _t
    if not hasattr(_t, "device_mesh"):
        from torch.distributed.device_mesh import DeviceMesh

        shim = types.ModuleType("torch.distributed.tensor.device_mesh")
        shim.DeviceMesh = DeviceMesh
        sys.modules["torch.distributed.tensor.device_mesh"] = shim
        _t.device_mesh = shim


# A single object may fill / overflow a close-up crop, so we tell the model the
# crop IS the object and to name the MAIN/foreground object (not the scene).
PROMPT = (
    "The image(s) are tight crops of ONE single object taken from different frames "
    "of an indoor video. The object may fill or be partially cut off by the crop.\n"
    "Name the MAIN foreground object — a concrete physical thing, NOT the scene, "
    "room, or activity.\n"
    "Reply with ONLY a short lowercase common-noun category (1-3 words). Examples: "
    "person, hand, office chair, cardboard box, water bottle, coffee mug, laptop, "
    "potted plant, book, remote control, knife, cutting board.\n"
    "Do NOT answer with a scene/room/activity word (e.g. 'kitchen', 'art studio', "
    "'office', 'rug pattern'). No description, no color, no punctuation."
)

# Reject degenerate / scene-like / non-answer outputs.
_BAD = {"", "object", "unknown", "thing", "item", "n/a", "none", "image", "crop",
        "scene", "background", "room", "indoor", "indoors", "wall", "floor"}
_SCENE_WORDS = {"studio", "kitchen", "office", "bedroom", "bathroom", "living room",
                "scene", "room", "workshop", "pattern", "background"}


def clean_name(raw: str) -> str:
    """Normalise a VLM reply to a compact lowercase class noun."""
    if not raw:
        return ""
    s = raw.strip().lower()
    s = s.splitlines()[0] if s else s
    s = s.strip(" \t\"'`*.-:")
    s = re.sub(r"^(a|an|the)\s+", "", s)
    s = re.sub(r"[^\w\s/+-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if len(words) > 3:
        s = " ".join(words[:3])
    return s


def is_bad_name(name: str) -> bool:
    if not name or name in _BAD:
        return True
    return any(w in name for w in _SCENE_WORDS)


def to_pil(crop_rgb: np.ndarray, max_side: int = 0):
    """np.uint8 RGB array → PIL.Image, optionally thumbnailed to max_side."""
    from PIL import Image

    arr = np.asarray(crop_rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if max_side and max_side > 0:
        img.thumbnail((max_side, max_side))
    return img
