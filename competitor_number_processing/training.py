"""YOLOv8n training and OCR crop extraction for the bib number pipeline."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import cv2

from pipeline.config import load_config, get_pipeline_logger

logger = get_pipeline_logger(__name__)


def load_labels_annotations(drive_manager: Any, folder_id: str) -> dict[str, str]:
    """Download labels.csv from Drive and return {filename: correct_number} for non-empty entries."""
    try:
        original_folder = drive_manager.folder_id
        drive_manager.folder_id = folder_id
        files = drive_manager.list_files_in_folder(folder_id=folder_id)
        labels_file = next((f for f in files if f["name"] == "labels.csv"), None)
        if not labels_file:
            drive_manager.folder_id = original_folder
            logger.info("No labels.csv on Drive yet")
            return {}
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        drive_manager.download_file(labels_file["id"], tmp_path, skip_if_exists=False, check_hash=False)
        drive_manager.folder_id = original_folder
        annotations: dict[str, str] = {}
        with open(tmp_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("correct_number", "").strip():
                    annotations[row["file"]] = row["correct_number"].strip()
        tmp_path.unlink(missing_ok=True)
        logger.info(f"[OK] Loaded {len(annotations)} existing annotations from Drive")
        return annotations
    except Exception as e:
        logger.warning(f"Could not load labels.csv from Drive: {e}")
        return {}


def merge_correct_numbers(csv_path: Path, annotations: dict[str, str]) -> None:
    """Restore correct_number values from Drive into a freshly generated labels.csv."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["file", "predicted_ocr", "correct_number"]
        for row in reader:
            if row["file"] in annotations:
                row["correct_number"] = annotations[row["file"]]
            rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[OK] Merged {sum(1 for r in rows if r.get('correct_number'))} annotations into labels.csv")


def upload_labels_to_drive(drive_manager: Any, folder_id: str, csv_path: Path) -> None:
    """Upload labels.csv to Drive, replacing any existing copy."""
    try:
        original_folder = drive_manager.folder_id
        drive_manager.folder_id = folder_id
        files = drive_manager.list_files_in_folder(folder_id=folder_id)
        for f in files:
            if f["name"] == "labels.csv":
                drive_manager.delete_file(f["id"])
        file_id = drive_manager.upload_file(csv_path, file_name="labels.csv")
        drive_manager.folder_id = original_folder
        if file_id:
            logger.info("[OK] labels.csv uploaded to Drive")
        else:
            logger.warning("Failed to upload labels.csv to Drive")
    except Exception as e:
        logger.warning(f"Could not upload labels.csv to Drive: {e}")


def find_dataset_dir(config: dict) -> Path:
    rf = config["roboflow"]
    base = Path("cache/roboflow_datasets")
    candidate = base / f"{rf['project']}_v{rf['version']}"
    if candidate.exists() and (candidate / "data.yaml").exists():
        return candidate
    matches = sorted(
        [d for d in base.iterdir() if (d / "data.yaml").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No dataset with data.yaml found under {base}. "
        "Run `uv run train` to download it first."
    )


def _ensure_splits(dataset_dir: Path) -> Path:
    """Return path to a data.yaml where val/test fall back to train if missing."""
    import shutil
    import yaml

    data_yaml = dataset_dir / "data.yaml"
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    train_rel = cfg.get("train", "../train/images")
    train_abs = (dataset_dir / train_rel).resolve()
    changed = False

    for split in ("val", "test"):
        rel = cfg.get(split)
        if rel:
            abs_path = (dataset_dir / rel).resolve()
            if not abs_path.exists():
                cfg[split] = train_rel
                changed = True
                logger.warning(f"Split '{split}' not found — falling back to train for validation")

    if not changed:
        return data_yaml

    patched = dataset_dir / "data_patched.yaml"
    with open(patched, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return patched


def train_yolo(args: Any) -> None:
    """Fine-tune YOLOv8n on the bib dataset.

    args: SimpleNamespace with model/epochs/imgsz/batch/dataset_dir fields.
    """
    from ultralytics import YOLO

    config = load_config()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else find_dataset_dir(config)
    data_yaml = _ensure_splits(dataset_dir)
    logger.info(f"Dataset: {data_yaml}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(Path("cache/runs").absolute()),
        name="bib_yolov8n",
        exist_ok=True,
        patience=20,
        freeze=10,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        verbose=True, save=True, save_period=10, val=True, plots=True,
    )

    best = Path("cache/runs/bib_yolov8n/weights/best.pt")
    mAP = results.results_dict.get("metrics/mAP50(B)", "N/A")
    logger.info(f"Training complete. Best weights: {best}")
    logger.info(f"Best mAP50: {mAP:.4f}" if isinstance(mAP, float) else f"Best mAP50: {mAP}")


def extract_bib_crops(dataset_dir: Path) -> None:
    """Extract class-1 (number) crops from training images for future OCR fine-tuning.

    Saves PNGs to cache/bib_crops/ and writes labels.csv with columns:
        file, predicted_ocr, correct_number
    User fills in correct_number to enable future OCR fine-tuning.
    """
    from competitor_number_processing.ocr import BibOCR

    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"
    crop_dir = Path("cache/bib_crops")
    crop_dir.mkdir(parents=True, exist_ok=True)
    csv_path = crop_dir / "labels.csv"

    ocr = BibOCR(gpu=False)
    rows: list[dict] = []

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in all_images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        with open(lbl_path) as f:
            lines = [line.strip().split() for line in f if line.strip()]

        idx = 0
        for parts in lines:
            if len(parts) != 5 or parts[0] != "1":
                continue
            cx, cy, bw, bh = (float(x) for x in parts[1:])
            x1 = max(0, int((cx - bw / 2) * w) - 4)
            y1 = max(0, int((cy - bh / 2) * h) - 4)
            x2 = min(w, int((cx + bw / 2) * w) + 4)
            y2 = min(h, int((cy + bh / 2) * h) + 4)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_name = f"{img_path.stem}_{idx}.png"
            cv2.imwrite(str(crop_dir / crop_name), crop)
            predicted = ocr.read_number(crop) or ""
            rows.append({"file": crop_name, "predicted_ocr": predicted, "correct_number": ""})
            idx += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "predicted_ocr", "correct_number"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} bib crops → {crop_dir}/")
    logger.info(f"Fill 'correct_number' in {csv_path} to enable future OCR fine-tuning")
