"""YOLO bbox detection wrapper for ROSE pipeline (Step 1).

Ported from the validated eval script (assets/eval_yolo_bbox_sam3_masks.py).
Provides normalized bounding boxes for SAM3 bbox prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from rose.engine.config.rose_config import YOLOConfig


@dataclass
class YoloDetection:
    """Single YOLO detection with normalized coordinates."""
    class_name: str
    score: float
    frame_idx: int
    bbox_xyxy: Tuple[float, float, float, float]      # normalized [0,1]
    bbox_xywh_norm: Tuple[float, float, float, float]  # normalized [0,1], SAM3 format


class YoloBBoxDetector:
    """Thin wrapper around Ultralytics YOLO for frame-wise bbox proposals.

    Lazily loads the model on first ``detect()`` call.  Returns detections
    with coordinates normalized to [0, 1] for direct use as SAM3 bbox prompts.
    """

    def __init__(self, config: Optional[YOLOConfig] = None) -> None:
        config = config or YOLOConfig()
        self.model_path = config.model_path
        self.conf = config.conf_threshold
        self.iou = config.iou_threshold
        self.imgsz = config.imgsz
        self.device = config.device
        self.max_det = config.max_det
        self._model = None

    def load(self) -> None:
        """Load YOLO model (lazy, called automatically on first detect)."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Install via: pip install ultralytics"
            ) from exc

        self._model = YOLO(self.model_path)

    def detect(self, rgb: np.ndarray, frame_idx: int) -> List[YoloDetection]:
        """Run YOLO on an RGB frame and return normalized detections.

        Args:
            rgb: Image array (H, W, 3) in RGB format.
            frame_idx: Frame index for metadata.

        Returns:
            Detections sorted by confidence (descending), with coordinates
            normalized to [0, 1].
        """
        if self._model is None:
            self.load()

        results = self._model(
            source=rgb,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
        cls_idx = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.empty((0,), dtype=int)
        names = result.names or {}

        h, w = rgb.shape[:2]
        detections: List[YoloDetection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            name = str(names.get(int(cls_idx[i]), int(cls_idx[i]) if i < len(cls_idx) else ""))
            score = float(scores[i]) if i < len(scores) else 0.0

            # Normalize to [0, 1] with bounds clamping
            x1n = float(max(0.0, min(1.0, x1 / w)))
            y1n = float(max(0.0, min(1.0, y1 / h)))
            x2n = float(max(0.0, min(1.0, x2 / w)))
            y2n = float(max(0.0, min(1.0, y2 / h)))
            ww = max(0.0, x2n - x1n)
            hh = max(0.0, y2n - y1n)

            if ww <= 0 or hh <= 0:
                continue

            detections.append(
                YoloDetection(
                    class_name=name,
                    score=score,
                    frame_idx=frame_idx,
                    bbox_xyxy=(x1n, y1n, x2n, y2n),
                    bbox_xywh_norm=(x1n, y1n, ww, hh),
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
