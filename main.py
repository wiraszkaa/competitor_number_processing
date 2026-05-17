#!/usr/bin/env python3
"""
  uv run preprocess              Drive sync, preprocess images, check Roboflow status
  uv run preprocess --validate   Also verify preprocessed files exist on Drive

  uv run train                   Download latest Roboflow dataset and retrain YOLOv8n
  uv run train --skip-download   Retrain on existing cached dataset
  uv run train --epochs N --batch N --model yolov8n.pt

  uv run process <path>          Extract bib numbers from image file or directory
  uv run process <path> --output DIR --conf 0.15 --json
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools" / "drive_manager"))
sys.path.insert(0, str(Path(__file__).parent / "tools" / "roboflow_manager"))

from pipeline.dataset_preparation import main as prepare_main


def _validate_dataset(dataset_dir: Path) -> None:
    """Log dataset structure: class names, image counts, missing labels per split."""
    import yaml
    from pipeline.config import get_pipeline_logger
    logger = get_pipeline_logger(__name__)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml missing in {dataset_dir}")
    with open(data_yaml) as f:
        meta = yaml.safe_load(f)
    logger.info(f"Classes ({meta['nc']}): {meta['names']}")
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            logger.warning(f"Split '{split}' not found — skipping")
            continue
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        missing = [i.name for i in images if not (lbl_dir / (i.stem + ".txt")).exists()]
        logger.info(f"  {split}: {len(images)} images, {len(missing)} missing labels")
        if missing:
            logger.warning(f"    Missing labels: {missing[:5]}")


def _parse_preprocess_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drive sync, preprocess images, check Roboflow annotation status",
    )
    p.add_argument("--validate", action="store_true",
                   help="Verify preprocessed files exist on Drive")
    return p.parse_args()


def preprocess_entry_point() -> None:
    args = _parse_preprocess_args()
    try:
        results = prepare_main(validate=args.validate)
        sys.exit(0 if not results.get("error") else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _parse_train_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download latest Roboflow dataset and fine-tune YOLOv8n",
    )
    p.add_argument("--skip-download", action="store_true",
                   help="Skip dataset download, retrain on existing cache")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=8, help="Reduce to 4 if out of memory")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--skip-ocr", action="store_true",
                   help="Skip OCR fine-tuning step after YOLO training")
    p.add_argument("--ocr-only", action="store_true",
                   help="Skip dataset download and YOLO training, only fine-tune OCR")
    return p.parse_args()


def train_entry_point() -> None:
    args = _parse_train_args()
    try:
        from pipeline.config import load_config
        from roboflow_manager import RoboflowClient
        from competitor_number_processing.training import (
            train_yolo, extract_bib_crops, find_dataset_dir,
            load_labels_annotations, merge_correct_numbers, upload_labels_to_drive,
        )

        config = load_config()

        if args.ocr_only:
            from competitor_number_processing.ocr_training import fine_tune_ocr
            fine_tune_ocr(
                csv_path=Path("cache/bib_crops/labels.csv"),
                crop_dir=Path("cache/bib_crops"),
                output_path=Path("cache/runs/bib_ocr/ocr_finetuned.pth"),
                epochs=20,
                lr=1e-4,
                batch_size=16,
            )
            return

        drive_manager = None
        bib_crops_folder_id = config.get("google_drive", {}).get("bib_crops_drive_folder_id", "")
        if bib_crops_folder_id and bib_crops_folder_id != "YOUR_DRIVE_FOLDER_ID_FOR_BIB_CROPS":
            from drive_manager.manager import DriveManager
            drive_manager = DriveManager(config["google_drive"]["credentials_path"])

        if not args.skip_download:
            rf = config["roboflow"]
            client = RoboflowClient(
                api_key=rf["api_key"],
                workspace=rf["workspace"],
                project=rf["project"],
                version=rf["version"],
            )
            dataset_dir = client.download_dataset(
                output_dir=Path("cache/roboflow_datasets"),
                format="yolov8",
                extract=True,
            )
            _validate_dataset(dataset_dir)

            annotations = {}
            if drive_manager and bib_crops_folder_id:
                annotations = load_labels_annotations(drive_manager, bib_crops_folder_id)

            extract_bib_crops(dataset_dir)

            csv_path = Path("cache/bib_crops/labels.csv")
            if annotations and csv_path.exists():
                merge_correct_numbers(csv_path, annotations)

            if drive_manager and bib_crops_folder_id and csv_path.exists():
                upload_labels_to_drive(drive_manager, bib_crops_folder_id, csv_path)
        else:
            dataset_dir = find_dataset_dir(config)

        train_args = types.SimpleNamespace(
            model=args.model,
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            dataset_dir=str(dataset_dir),
        )
        train_yolo(train_args)

        if not args.skip_ocr:
            from competitor_number_processing.ocr_training import fine_tune_ocr
            fine_tune_ocr(
                csv_path=Path("cache/bib_crops/labels.csv"),
                crop_dir=Path("cache/bib_crops"),
                output_path=Path("cache/runs/bib_ocr/ocr_finetuned.pth"),
                epochs=20,
                lr=1e-4,
                batch_size=16,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _parse_process_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract bib numbers from image file or directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run process race.jpg\n"
            "  uv run process photos/ --output results/ --conf 0.15\n"
            "  uv run process race.jpg --json"
        ),
    )
    p.add_argument("input", type=Path, help="Image file or directory")
    p.add_argument("--output", type=Path, default=None,
                   help="Save annotated PNGs here (optional)")
    p.add_argument("--conf", type=float, default=0.20,
                   help="Detection confidence threshold (default: 0.20)")
    p.add_argument("--json", action="store_true", help="Print results as JSON")
    p.add_argument("--hog", action="store_true",
                   help="Use classical HOG+SVM person detection instead of YOLOv8")
    return p.parse_args()


def process_entry_point() -> None:
    args = _parse_process_args()
    try:
        from competitor_number_processing.inference import run
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    preprocess_entry_point()
