"""Open-vocabulary instance naming via the local Gemma-3-4b-it multimodal VLM.

Kept as a lightweight fallback to the (default) Qwen2.5-VL namer. Same interface:
``name_one(crops)`` / ``name_objects({gid: [crops]})``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from ._naming_utils import (PROMPT, clean_name, install_device_mesh_shim,
                            is_bad_name, to_pil)

logger = logging.getLogger(__name__)


class GemmaInstanceNamer:
    """Lazy-loaded Gemma-3-4b-it namer. Load once, reuse across videos."""

    def __init__(
        self,
        model_path: str = "rose/models/gemma-3-4b-it",
        device: str = "cuda",
        max_crops_per_object: int = 3,
        crop_resize: int = 224,
        max_new_tokens: int = 12,
    ):
        self.model_path = model_path
        self.device = device
        self.max_crops_per_object = max_crops_per_object
        self.crop_resize = crop_resize
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        install_device_mesh_shim()
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        logger.info("GemmaInstanceNamer: loading %s ...", self.model_path)
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map=self.device,
        ).eval()
        logger.info("GemmaInstanceNamer: ready.")

    def name_one(self, crops: List[np.ndarray]) -> str:
        self.load()
        import torch

        imgs = [to_pil(c, self.crop_resize) for c in crops[: self.max_crops_per_object]]
        if not imgs:
            return "object"
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": PROMPT})
        messages = [{"role": "user", "content": content}]
        raw_inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        dev = self._model.device
        inputs = {}
        for k, v in raw_inputs.items():
            if hasattr(v, "to"):
                v = v.to(dev)
                if v.is_floating_point():
                    v = v.to(torch.bfloat16)
            inputs[k] = v
        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        out = self._processor.decode(gen[0][in_len:], skip_special_tokens=True)
        name = clean_name(out)
        return "object" if is_bad_name(name) else name

    def name_objects(self, gid_to_crops: Dict[int, List[np.ndarray]]) -> Dict[int, str]:
        self.load()
        names: Dict[int, str] = {}
        for gid in sorted(gid_to_crops.keys()):
            try:
                names[gid] = self.name_one(gid_to_crops[gid])
            except Exception as e:
                logger.warning("Gemma naming failed for gid=%s: %r", gid, e)
                names[gid] = "object"
        return names


def make_namer(provider: str, model_path: str, device: str = "cuda",
               max_crops_per_object: int = 3, max_new_tokens: int = 16):
    """Factory → an instance namer for any HF image-text-to-text VLM.

    Routes through the generic HFInstanceNamer (AutoModelForImageTextToText), which
    handles Gemma-3, Qwen2.5-VL, Qwen3-VL and DeepSeek-VL uniformly. ``provider`` is
    advisory (the model_path selects the actual model). A close-up COCO + ROSE-crop
    bake-off (2026-06-04) found Qwen3-VL-4B best (73.8% vs Qwen2.5-VL 72.5%, Gemma
    65%, DeepSeek-VL 56%); Qwen3-VL-4B is the default.
    """
    from .hf_namer import HFInstanceNamer
    return HFInstanceNamer(
        model_path=model_path, device=device,
        max_crops_per_object=max_crops_per_object, max_new_tokens=max_new_tokens,
    )
