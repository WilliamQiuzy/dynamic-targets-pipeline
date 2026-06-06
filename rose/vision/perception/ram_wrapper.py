"""RAM++ wrapper for ROSE pipeline.

Implements Step 1: per-frame image tagging for object discovery.
Output: list of tag strings. Tags are used solely to trigger SAM3 runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from rose.engine.config.rose_config import RAMPlusConfig

logger = logging.getLogger(__name__)


class RAMPlusWrapper:
    """Wrapper around RAM++ for per-frame image tagging."""

    def __init__(self, config: Optional[RAMPlusConfig] = None):
        self.config = config or RAMPlusConfig()
        self._model = None
        self._transform = None

    def load(self) -> None:
        """Load the RAM++ model. Called lazily on first inference."""
        if self._model is not None:
            return

        ram_src = str(Path("rose/vision/recognize-anything").resolve())
        if ram_src not in sys.path:
            sys.path.insert(0, ram_src)

        import torch
        from ram.models import ram_plus
        from ram import get_transform

        self._transform = get_transform(image_size=384)

        # Resolve checkpoint path
        if self.config.checkpoint_path:
            pretrained_path = str(Path(self.config.checkpoint_path).resolve())
            if not Path(pretrained_path).is_file():
                raise FileNotFoundError(
                    f"RAM++ checkpoint not found: {pretrained_path}"
                )
        else:
            weight_dir = Path(self.config.model_path)
            weight_candidates = sorted(weight_dir.glob("*.pth"))
            if not weight_candidates:
                raise FileNotFoundError(
                    f"No .pth files found in {weight_dir}. "
                    f"Set ram_plus.checkpoint_path or place weights in {weight_dir}."
                )
            pretrained_path = str(weight_candidates[0])
            logger.info("Auto-selected RAM++ checkpoint: %s", pretrained_path)

        self._model = ram_plus(
            pretrained=pretrained_path,
            image_size=384,
            vit="swin_l",
        )
        self._model.eval()
        self._model = self._model.to(self.config.device)
        logger.info("RAM++ model loaded on %s", self.config.device)

    def infer(self, image: Union[np.ndarray, str, Path]) -> List[str]:
        """Run RAM++ on a single image.

        Args:
            image: RGB image as (H, W, 3) uint8 ndarray, or path to image file.

        Returns:
            List of tag strings (e.g. ["car", "person", "tree"]).
        """
        self.load()

        import torch
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            pil_img = PILImage.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image)
        else:
            pil_img = image

        image_tensor = self._transform(pil_img).unsqueeze(0).to(self.config.device)

        with torch.inference_mode():
            tags_str, _ = self._model.generate_tag(image_tensor)

        # tags_str is a list with one pipe-separated string per batch element
        raw = tags_str[0] if isinstance(tags_str, list) else tags_str
        tags = [t.strip() for t in raw.split("|") if t.strip()]
        return tags
