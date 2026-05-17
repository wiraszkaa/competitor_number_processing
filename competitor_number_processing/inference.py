"""Inference pipeline: detect bib numbers in one image or a directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2

from competitor_number_processing.preprocess import PreprocessConfig, preprocess_image
from competitor_number_processing.cnn_detector import YOLOv8BibDetector, YOLODetectionConfig
from competitor_number_processing.ocr import BibOCR
from pipeline.config import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in _SUPPORTED)
    raise FileNotFoundError(f"Path not found: {input_path}")


def _process_one(
    image_path: Path,
    detector: YOLOv8BibDetector,
    ocr: BibOCR,
    preprocess_cfg: PreprocessConfig,
    out_dir: Optional[Path],
) -> dict:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.warning(f"Cannot read {image_path.name} — skipping")
        return {"file": image_path.name, "error": "cannot read image"}

    tmp_dir = Path("cache/temp_process")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = preprocess_image(image_path, preprocess_cfg, save_debug_to=tmp_dir, prefix=image_path.stem)
    preprocessed = cv2.imread(str(paths["final"]))

    detections = detector.detect(preprocessed)
    regions = detector.extract_regions(preprocessed, detections, padding=5)
    numbers = ocr.read_batch(regions)

    result = {
        "file": image_path.name,
        "bibs": [
            {
                "confidence": det.confidence,
                "number": num,
                "bbox": {"x": det.x, "y": det.y, "w": det.width, "h": det.height},
            }
            for det, num in zip(detections, numbers)
        ],
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = detector.visualize_detections(preprocessed, detections, labels=numbers)
        cv2.imwrite(str(out_dir / f"{image_path.stem}_result.png"), vis)

    return result


def run(args: Any) -> None:
    """Entry point called from main.process_entry_point."""
    input_path = Path(args.input)
    images = _collect_images(input_path)
    if not images:
        print(f"No supported images found in {input_path}")
        sys.exit(1)

    logger.info(f"Processing {len(images)} image(s)")
    preprocess_cfg = PreprocessConfig(max_long_edge=1280, autocontrast=True)

    use_hog = getattr(args, "hog", False)
    if use_hog:
        from competitor_number_processing.detector import PersonDetector, DetectionConfig
        detector = PersonDetector(DetectionConfig(threshold=args.conf - 1.0))
        logger.info("Using HOG+SVM detector")
    else:
        detector = YOLOv8BibDetector(YOLODetectionConfig(confidence_threshold=args.conf))
        logger.info("Using YOLOv8 detector")

    ocr = BibOCR(gpu=False)

    results = [_process_one(img, detector, ocr, preprocess_cfg, args.output) for img in images]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['file']}: ERROR — {r['error']}")
                continue
            bibs = r["bibs"]
            if not bibs:
                print(f"{r['file']}: no bibs detected")
            else:
                nums = [b["number"] or "(unread)" for b in bibs]
                print(f"{r['file']}: {', '.join(nums)}")
