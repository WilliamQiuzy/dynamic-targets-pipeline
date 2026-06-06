"""Open-vocabulary instance naming via a local Qwen2.5-VL model.

Qwen2.5-VL is a strong multimodal VLM with excellent fine-grained recognition,
which the client requested over Gemma. Same interface as GemmaInstanceNamer:
``name_one(crops)`` / ``name_objects({gid: [crops]})``.

Default weights: rose/models/qwen2.5-vl-7b-instruct (7B — quality/runnability
balance; the 32B variant can be dropped in via config.qwen_model_path).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from ._naming_utils import (PROMPT, clean_name, install_device_mesh_shim,
                            is_bad_name, to_pil)

logger = logging.getLogger(__name__)


class QwenInstanceNamer:
    """Lazy-loaded Qwen2.5-VL namer. Load once, reuse across videos."""

    def __init__(
        self,
        model_path: str = "rose/models/qwen2.5-vl-7b-instruct",
        device: str = "cuda",
        max_crops_per_object: int = 3,
        crop_resize: int = 0,  # 0 = let Qwen's processor handle resolution
        max_new_tokens: int = 16,
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
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("QwenInstanceNamer: loading %s ...", self.model_path)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        # min/max pixels keep small crops from being upsampled into huge token counts.
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28
        )
        logger.info("QwenInstanceNamer: ready.")

    def name_one(self, crops: List[np.ndarray]) -> str:
        self.load()
        import torch

        imgs = [to_pil(c, self.crop_resize) for c in crops[: self.max_crops_per_object]]
        if not imgs:
            return "object"
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": PROMPT})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            image_inputs, video_inputs = imgs, None
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)
        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        out = self._processor.batch_decode(
            gen[:, in_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        name = clean_name(out)
        return "object" if is_bad_name(name) else name

    def name_objects(self, gid_to_crops: Dict[int, List[np.ndarray]]) -> Dict[int, str]:
        self.load()
        names: Dict[int, str] = {}
        for gid in sorted(gid_to_crops.keys()):
            try:
                names[gid] = self.name_one(gid_to_crops[gid])
            except Exception as e:
                logger.warning("Qwen naming failed for gid=%s: %r", gid, e)
                names[gid] = "object"
        return names
