"""
Run the full pipeline on every validation image and save annotated results.

For each image: detects bib regions, reads the number with OCR, saves a PNG
with the bib box + OCR text overlaid so results can be visually inspected.

Run:
    uv run python scripts/validate_ocr.py
    uv run python scripts/validate_ocr.py --conf 0.15  # lower threshold to catch more bibs

Output: cache/val_results/<image_name>_result.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from competitor_number_processing.cnn_detector import YOLOv8BibDetector, YOLODetectionConfig
from competitor_number_processing.ocr import BibOCR
from pipeline.config import get_pipeline_logger

logger = get_pipeline_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", default="cache/roboflow_datasets/competitor-numbers_v1/valid/images",
                   type=Path)
    p.add_argument("--output", default="cache/val_results", type=Path)
    p.add_argument("--conf", type=float, default=0.20)
    return p.parse_args()


def draw_result(image: np.ndarray, bbox, number: str | None, conf: float) -> np.ndarray:
    x, y, w, h = bbox
    color = (0, 200, 80) if number else (0, 100, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    label = f"#{number}" if number else f"? ({conf:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(image, label, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def main(args=None):
    if args is None:
        args = parse_args()
    val_dir = Path(args.val_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))
    if not images:
        logger.error(f"No images found in {val_dir}")
        return

    logger.info(f"Validating {len(images)} images → {out_dir}")

    detector = YOLOv8BibDetector(YOLODetectionConfig(confidence_threshold=args.conf))
    ocr = BibOCR(gpu=False)

    total = detected = ocr_read = 0
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        total += 1

        dets = detector.detect(image)
        regions = detector.extract_regions(image, dets, padding=5)
        numbers = ocr.read_batch(regions)

        detected += len(dets)
        ocr_read += sum(1 for n in numbers if n)

        vis = image.copy()
        for det, number in zip(dets, numbers):
            draw_result(vis, (det.x, det.y, det.width, det.height), number, det.confidence)

        if not dets:
            label = "no bib detected"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

        out_path = out_dir / f"{img_path.stem}_result.png"
        cv2.imwrite(str(out_path), vis)

    logger.info(f"Done. {total} images | {detected} bibs detected | {ocr_read} numbers read")
    logger.info(f"Detection rate: {detected/total:.1%} avg bibs/image: {detected/total:.2f}")
    logger.info(f"OCR success rate (of detected bibs): "
                f"{ocr_read/detected:.1%}" if detected else "OCR: N/A")
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
