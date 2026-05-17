"""
EasyOCR wrapper for reading digit sequences from bib region crops.

Usage:
    from competitor_number_processing.ocr import BibOCR
    ocr = BibOCR()
    number = ocr.read_number(bib_region_bgr)  # "1234" or None
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class OCRResult:
    raw_text: str
    number: Optional[str]
    confidence: float


_DEFAULT_FINETUNED = Path("cache/runs/bib_ocr/ocr_finetuned.pth")


class BibOCR:
    """EasyOCR wrapper tuned for bib digit recognition."""

    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = False,
        min_confidence: float = 0.3,
        weights_path: Optional[Path] = None,
    ):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.min_confidence = min_confidence
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_FINETUNED
        self._reader = None  # lazy init (~5s on first call, downloads model weights)

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        import easyocr
        # Use quantize=False when loading fine-tuned weights — trained model is not quantized
        use_quantize = not self.weights_path.exists()
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False,
                                      quantize=use_quantize)
        if self.weights_path.exists():
            import torch
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            self._reader.recognizer.load_state_dict(state)
            print(f"[OK] Loaded fine-tuned OCR weights from {self.weights_path}")
        return self._reader

    @staticmethod
    def _preprocess(region: np.ndarray, target_height: int = 64) -> np.ndarray:
        """Upscale small crops and apply CLAHE contrast normalisation."""
        h, w = region.shape[:2]
        if h < target_height:
            scale = target_height / h
            region = cv2.resize(region, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read_number(self, region: np.ndarray) -> Optional[str]:
        """Return digit string (e.g. '1234') or None if unreadable."""
        return self.read_number_detailed(region).number

    def read_number_detailed(self, region: np.ndarray) -> OCRResult:
        if region is None or region.size == 0:
            return OCRResult(raw_text="", number=None, confidence=0.0)

        preprocessed = self._preprocess(region)
        raw = self._get_reader().readtext(
            preprocessed,
            allowlist="0123456789",   # restrict to digits — biggest accuracy boost
            detail=1,
            paragraph=False,
            min_size=10,
        )
        confident = [(text, conf) for (_, text, conf) in raw if conf >= self.min_confidence]
        if not confident:
            return OCRResult(raw_text="", number=None, confidence=0.0)

        all_text = "".join(t for t, _ in confident)
        mean_conf = sum(c for _, c in confident) / len(confident)
        digits = re.sub(r"\D", "", all_text)
        return OCRResult(
            raw_text=all_text,
            number=digits or None,
            confidence=round(mean_conf, 4),
        )

    def read_batch(self, regions: List[np.ndarray]) -> List[Optional[str]]:
        return [self.read_number(r) for r in regions]
