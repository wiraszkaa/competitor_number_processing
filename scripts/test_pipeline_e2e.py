"""
Smoke test: preprocess → detect bibs (YOLOv8) → read numbers (EasyOCR) → visualise.

Run:
    uv run python scripts/test_pipeline_e2e.py --image cache/raw/1024875576.jpg
    uv run python scripts/test_pipeline_e2e.py  # uses first image in cache/raw/
"""

import argparse
import sys
from pathlib import Path

import cv2

from competitor_number_processing.preprocess import PreprocessConfig, preprocess_image
from competitor_number_processing.cnn_detector import YOLOv8BibDetector, YOLODetectionConfig
from competitor_number_processing.ocr import BibOCR
from pipeline.config import get_pipeline_logger

logger = get_pipeline_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, type=Path)
    p.add_argument("--output", default="cache/test_output", type=Path)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Bib detection confidence threshold")
    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image)
    else:
        raw_dir = Path("cache/raw")
        candidates = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
        if not candidates:
            logger.error("No images in cache/raw/")
            sys.exit(1)
        image_path = candidates[0]

    logger.info(f"Processing: {image_path.name}")

    # Stage 1: Preprocess
    cfg = PreprocessConfig(max_long_edge=1280, autocontrast=True)
    paths = preprocess_image(image_path, cfg, save_debug_to=out_dir, prefix=image_path.stem)
    final_path = paths["final"]
    logger.info(f"Preprocessed → {final_path.name}")

    # Stage 2: Detect bibs (YOLOv8)
    detector = YOLOv8BibDetector(YOLODetectionConfig(confidence_threshold=args.conf))
    image_bgr = cv2.imread(str(final_path))
    detections = detector.detect(image_bgr)
    logger.info(f"Detected {len(detections)} bib(s)")
    for i, d in enumerate(detections):
        logger.info(f"  Bib {i}: bbox=({d.x},{d.y},{d.width},{d.height}) conf={d.confidence:.3f}")

    # Stage 3: OCR
    ocr = BibOCR(gpu=False)
    regions = detector.extract_regions(image_bgr, detections, padding=5)
    numbers = ocr.read_batch(regions)
    logger.info(f"OCR results: {numbers}")

    # Stage 4: Visualise
    vis = detector.visualize_detections(image_bgr, detections, labels=numbers)
    vis_path = out_dir / f"{image_path.stem}_result.png"
    cv2.imwrite(str(vis_path), vis)
    logger.info(f"Saved → {vis_path}")

    print("\n=== Pipeline Result ===")
    for i, (det, num) in enumerate(zip(detections, numbers)):
        print(f"  Bib {i+1}: conf={det.confidence:.3f}  number={num or '(unread)'}")
    if not detections:
        print("  No bibs detected. Try lowering --conf (e.g. --conf 0.10)")


if __name__ == "__main__":
    main()
