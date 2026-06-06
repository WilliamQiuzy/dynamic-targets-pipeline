"""Generic open-vocabulary instance namer over any HF image-text-to-text VLM.

Works for Gemma-3, Qwen2.5-VL, Qwen3-VL, DeepSeek-VL, etc. via
``AutoModelForImageTextToText`` + ``AutoProcessor`` and the unified chat template
(transformers 5.9 routes the model-specific image handling). One class, swappable
by model path — used by the namer bake-off and the dynamic-targets export.

Interface matches Gemma/Qwen namers: ``name_one(crops)`` / ``name_objects({gid:[crops]})``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ._naming_utils import (PROMPT, clean_name, install_device_mesh_shim,
                            is_bad_name, to_pil)

logger = logging.getLogger(__name__)


class HFInstanceNamer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_crops_per_object: int = 3,
        max_new_tokens: int = 16,
        crop_resize: int = 0,
        min_pixels: Optional[int] = 128 * 28 * 28,
        max_pixels: Optional[int] = 768 * 28 * 28,
    ):
        self.model_path = model_path
        self.device = device
        self.max_crops_per_object = max_crops_per_object
        self.max_new_tokens = max_new_tokens
        self.crop_resize = crop_resize
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        install_device_mesh_shim()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("HFInstanceNamer: loading %s ...", self.model_path)
        proc_kw = {}
        # Qwen-family processors accept pixel budgets to cap vision tokens on small crops.
        if "qwen" in self.model_path.lower() and self.min_pixels:
            proc_kw = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True, **proc_kw)
        except Exception:
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map=self.device,
            trust_remote_code=True,
        ).eval()
        logger.info("HFInstanceNamer: ready (%s).", type(self._model).__name__)

    def _gen(self, inputs, in_len):
        import torch
        with torch.inference_mode():
            gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._processor.batch_decode(
            gen[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def name_one(self, crops: List[np.ndarray]) -> str:
        self.load()
        import torch

        imgs = [to_pil(c, self.crop_resize) for c in crops[: self.max_crops_per_object]]
        if not imgs:
            return "object"
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": PROMPT})
        messages = [{"role": "user", "content": content}]
        # Unified path: processor builds ids + pixel tensors from the chat template.
        raw = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        dev = self._model.device
        inputs = {}
        for k, v in raw.items():
            if hasattr(v, "to"):
                v = v.to(dev)
                if v.is_floating_point():
                    v = v.to(torch.bfloat16)
            inputs[k] = v
        in_len = inputs["input_ids"].shape[-1]
        out = self._gen(inputs, in_len)
        name = clean_name(out)
        return "object" if is_bad_name(name) else name

    def name_voted(self, crops: List[np.ndarray]) -> str:
        """Name each of the top-K (sharpest) crops individually and majority-vote.

        More robust than one multi-image prompt: a single blurry/partial/occluded
        view can't dominate, and diverse frames reinforce the true class. Ties are
        broken toward the FIRST (sharpest) crop's vote.
        """
        from collections import Counter
        cs = crops[: self.max_crops_per_object]
        if not cs:
            return "object"
        if len(cs) == 1:
            return self.name_one(cs)
        votes = []
        for c in cs:
            n = self.name_one([c])
            if n and n != "object":
                votes.append(n)
        if not votes:
            return "object"
        cnt = Counter(votes)
        top = cnt.most_common(1)[0][1]
        winners = [v for v in votes if cnt[v] == top]  # preserve sharpness order on ties
        return winners[0]

    def name_objects(self, gid_to_crops: Dict[int, List[np.ndarray]],
                     vote: bool = True) -> Dict[int, str]:
        self.load()
        names: Dict[int, str] = {}
        for gid in sorted(gid_to_crops.keys()):
            try:
                names[gid] = self.name_voted(gid_to_crops[gid]) if vote else self.name_one(gid_to_crops[gid])
            except Exception as e:
                logger.warning("naming failed for gid=%s: %r", gid, e)
                names[gid] = "object"
        return names
