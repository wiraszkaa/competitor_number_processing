# Competitor Number Processing System

Extracts and reads competitor bib numbers from sports event photos (marathons, cycling, triathlon). The full pipeline covers image collection → preprocessing → annotation → model training → inference.

## Pipeline Overview

```
uv run images_collector     # 1. Collect training images → Google Drive
uv run deduplicator         # 2. Remove duplicates from Drive
uv run preprocess           # 3. Preprocess + upload to Drive + upload to Roboflow
                            #    (annotate images in Roboflow UI)
uv run train                # 4. Download annotated dataset + train YOLO + fine-tune OCR
uv run process <image>      # 5. Detect and read bib numbers
```

## Setup

### Install

```bash
# Install UV
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh               # macOS/Linux

# Clone and install
git clone <repository-url>
cd analiza_danych_multimedialnych
uv sync
```

### Configure

```bash
cp secrets/config.example.json secrets/config.json
```

Edit `secrets/config.json`:

```json
{
  "google_custom_search": {
    "api_key": "YOUR_GOOGLE_API_KEY",
    "search_engine_id": "YOUR_SEARCH_ENGINE_ID",
    "num_results": 10
  },
  "google_drive": {
    "credentials_path": "secrets/client_secret.json",
    "raw_folder_id": "YOUR_RAW_FOLDER_ID",
    "preprocessed_folder_id": "YOUR_PREPROCESSED_FOLDER_ID",
    "tracking_drive_folder_id": "YOUR_TRACKING_FOLDER_ID",
    "bib_crops_drive_folder_id": "YOUR_BIB_CROPS_FOLDER_ID",
    "download_dir_raw": "cache/raw",
    "download_dir_preprocessed": "cache/preprocessed"
  },
  "cache": {
    "directory": "cache",
    "max_image_size_mb": 10
  },
  "tracking": {
    "file": "tracking.json"
  },
  "roboflow": {
    "api_key": "YOUR_ROBOFLOW_API_KEY",
    "workspace": "your_workspace",
    "project": "your_project",
    "version": 1,
    "format": "yolo",
    "download_dir": "cache/roboflow_datasets"
  }
}
```

**Drive folder IDs** — create four folders in Google Drive and paste their IDs:
- `raw_folder_id` — raw photos uploaded by the collector
- `preprocessed_folder_id` — preprocessed images ready for Roboflow
- `tracking_drive_folder_id` — stores `tracking.json` (shared state across machines)
- `bib_crops_drive_folder_id` — stores `labels.csv` with OCR ground truth annotations

### Google Drive OAuth

1. [Google Cloud Console](https://console.cloud.google.com/) → Enable **Google Drive API**
2. Credentials → Create → **OAuth Client ID** (Desktop app) → download JSON
3. Save as `secrets/client_secret.json`
4. Add your email to "Test Users" in the OAuth consent screen

On first run a browser window opens for authentication; token is saved automatically.

### Google Custom Search (for image collector)

1. [Google Cloud Console](https://console.cloud.google.com/) → Enable **Custom Search API** → create API key
2. [Programmable Search Engine](https://programmablesearchengine.google.com/) → create engine → enable "Search the entire web" + "Image Search"
3. Copy the Search Engine ID into config

---

## Commands

### `uv run images_collector`

PyQt6 GUI — search Google Images for sports event photos and upload selected ones to Drive's raw folder.

```bash
uv run images_collector
```

### `uv run deduplicator`

Remove duplicate images from Google Drive using MD5 checksums (no downloads needed).

```bash
uv run deduplicator                # dry run — report only
uv run deduplicator --delete       # delete with confirmation
uv run deduplicator --delete-confirm  # delete immediately
```

### `uv run preprocess`

Downloads raw images from Drive, preprocesses them (resize to 1280px, autocontrast, noise reduction), uploads to Drive's preprocessed folder, and auto-uploads to Roboflow. Syncs `tracking.json` to/from Drive so team members share state.

```bash
uv run preprocess
```

After this step, go to [roboflow.com](https://roboflow.com) and annotate the uploaded images.

### `uv run train`

Downloads the annotated dataset from Roboflow, fine-tunes **YOLOv8n** for bib detection, then fine-tunes **EasyOCR**'s recognition model on labeled bib crops.

```bash
uv run train                        # full run: download + YOLO + OCR
uv run train --skip-download        # skip dataset download, retrain on cached data
uv run train --ocr-only             # skip YOLO, only fine-tune OCR
uv run train --skip-ocr             # skip OCR fine-tuning, YOLO only
uv run train --epochs 100 --batch 4 # custom YOLO hyperparameters
uv run train --model yolov8s.pt     # larger base model
```

**OCR ground truth** — after `uv run train` runs for the first time, a `cache/bib_crops/labels.csv` file is created with all detected bib crops and an empty `correct_number` column. Fill this in manually, then re-run `uv run train --ocr-only` to fine-tune the OCR model. The CSV is synced to Drive (via `bib_crops_drive_folder_id`) so team members share annotations.

Output files:
- `cache/runs/bib_yolov8n/weights/best.pt` — YOLO bib detector
- `cache/runs/bib_ocr/ocr_finetuned.pth` — fine-tuned OCR model (auto-loaded at inference)

### `uv run process <path>`

Runs the full inference pipeline: preprocess → detect bibs (YOLO) → read digits (EasyOCR).

```bash
uv run process race.jpg                          # single image
uv run process photos/                           # directory of images
uv run process race.jpg --output results/        # save annotated PNG
uv run process race.jpg --conf 0.15              # lower detection threshold
uv run process race.jpg --json                   # print JSON output
uv run process race.jpg --hog                    # use classical HOG+SVM instead of YOLO
```

Example output:
```
race.jpg: 1547, 203, (unread)
```

`--hog` uses the classical HOG+SVM person detector instead of YOLOv8 — useful for comparing approaches.

---

## Architecture

```
Raw Image
    │
    ├─ preprocess (resize 1280px, autocontrast, CLAHE)
    │
    ├─ YOLOv8n bib detector  →  cache/runs/bib_yolov8n/weights/best.pt
    │       detects class 0 (competitor) and class 1 (number bib)
    │
    ├─ EasyOCR (digits only, allowlist="0123456789")
    │       auto-loads fine-tuned weights if cache/runs/bib_ocr/ocr_finetuned.pth exists
    │
    └─ result: bib regions + digit strings
```

**Detection modes** (`uv run process`):
- Default: YOLOv8n directly detects bib number regions → OCR
- `--hog`: HOG+SVM detects person regions → OCR on whole-person crops

---

## Project Structure

```
├── main.py                              # All CLI entry points
├── pyproject.toml
├── secrets/
│   ├── config.json                      # Local config (not committed)
│   ├── config.example.json
│   └── client_secret.json               # Google OAuth (not committed)
├── cache/                               # Git-ignored, auto-generated
│   ├── raw/                             # Downloaded raw images
│   ├── bib_crops/                       # Extracted bib crops + labels.csv
│   ├── roboflow_datasets/               # Downloaded Roboflow datasets
│   └── runs/
│       ├── bib_yolov8n/                 # YOLO training output & weights
│       └── bib_ocr/                     # Fine-tuned OCR weights
├── pipeline/
│   ├── dataset_preparation.py           # preprocess pipeline orchestration
│   └── config.py
├── competitor_number_processing/
│   ├── preprocess.py                    # image preprocessing
│   ├── detector.py                      # HOG+SVM person detector (classical)
│   ├── cnn_detector.py                  # YOLOv8n bib detector
│   ├── ocr.py                           # EasyOCR wrapper (auto-loads fine-tuned weights)
│   ├── ocr_training.py                  # EasyOCR fine-tuning on bib crops
│   ├── training.py                      # YOLO training + bib crop extraction
│   ├── inference.py                     # inference pipeline (process command)
│   └── tracking.py                      # pipeline state tracker
└── tools/
    ├── images_collector/                # PyQt6 GUI
    ├── drive_manager/                   # Google Drive API wrapper
    ├── images_deduplicator/             # MD5-based deduplication
    └── roboflow_manager/                # Roboflow API client
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No bibs detected | Lower `--conf 0.10`; if still nothing, retrain with more data |
| OOM during YOLO training | Use `--batch 4` |
| mAP50 < 0.60 | Try `--model yolov8s.pt --epochs 120` |
| OCR reads nothing | Bib crop too small — lower `--conf` to get larger boxes |
| OAuth error | Delete `secrets/token.json` and re-authenticate |
| `Module not found` | Run `uv sync` |

---

Educational project — WSB University
