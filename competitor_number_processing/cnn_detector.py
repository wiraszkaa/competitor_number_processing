"""
YOLOv8-based bib detector with the same interface as PersonDetector.

Usage:
    from competitor_number_processing.cnn_detector import YOLOv8BibDetector
    detector = YOLOv8BibDetector()
    detections = detector.detect(image_bgr)  # List[DetectedPerson]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from competitor_number_processing.detector import DetectedPerson

DEFAULT_WEIGHTS = Path("cache/runs/bib_yolov8n/weights/best.pt")


@dataclass(frozen=True)
class YOLODetectionConfig:
    weights_path: Path = DEFAULT_WEIGHTS
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    min_width: int = 20
    min_height: int = 15
    device: str = "cpu"
    # Filter to specific class indices during inference.
    # The dataset has 2 classes: 0=competitor, 1=number.
    # Default: detect only 'number' (class 1) since those regions go to OCR.
    # Use None to detect all classes.
    classes: Optional[Tuple[int, ...]] = (1,)


class YOLOv8BibDetector:
    """
    YOLOv8 bib detector returning List[DetectedPerson].

    Same interface as PersonDetector — drop-in replacement.
    Model is lazy-loaded on first detect() call.
    """

    def __init__(self, config: Optional[YOLODetectionConfig] = None):
        self.config = config or YOLODetectionConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        weights = Path(self.config.weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights}\n"
                "Run scripts/train_yolo.py first."
            )
        self._model = YOLO(str(weights))

    def detect(self, image: np.ndarray) -> List[DetectedPerson]:
        """Detect bib regions in a BGR image. Returns List[DetectedPerson]."""
        self._load_model()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._model.predict(
            source=image_rgb,
            imgsz=self.config.imgsz,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            classes=list(self.config.classes) if self.config.classes is not None else None,
            verbose=False,
        )
        detections: List[DetectedPerson] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                if w < self.config.min_width or h < self.config.min_height:
                    continue
                detections.append(DetectedPerson(
                    x=int(x1), y=int(y1),
                    width=int(w), height=int(h),
                    confidence=round(conf, 4),
                ))
        return detections

    def detect_from_file(self, image_path: Path) -> List[DetectedPerson]:
        image = cv2.imread(str(Path(image_path)))
        if image is None:
            raise ValueError(f"Failed to load: {image_path}")
        return self.detect(image)

    def extract_regions(
        self,
        image: np.ndarray,
        detections: List[DetectedPerson],
        padding: int = 5,
    ) -> List[np.ndarray]:
        h, w = image.shape[:2]
        return [
            image[
                max(0, d.y - padding):min(h, d.y + d.height + padding),
                max(0, d.x - padding):min(w, d.x + d.width + padding),
            ].copy()
            for d in detections
        ]

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[DetectedPerson],
        color: Tuple[int, int, int] = (0, 128, 255),
        thickness: int = 2,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        result = image.copy()
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d.x, d.y, d.x + d.width, d.y + d.height
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            parts = [f"{d.confidence:.2f}"]
            if labels and i < len(labels) and labels[i]:
                parts.append(f"#{labels[i]}")
            label = " ".join(parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return result
