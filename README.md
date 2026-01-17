# Competitor Number Processing System

A multimedia data analysis project for extracting and processing competitor numbers from sports event images. The system includes tools for collecting training images from the web, managing them on Google Drive, and **collaborative preprocessing pipeline** for team-based image processing.

## Project Overview

**Main Purpose**: Extract and analyze competitor numbers from sports event images (marathons, cycling races, triathlons, etc.) for participant identification and tracking.

**Core Pipeline**:

- **Main Pipeline** ([main.py](main.py)): Collaborative preprocessing workflow
  - Downloads raw images from Google Drive
  - Preprocesses images (resize, contrast, noise reduction)
  - Uploads processed images to shared folder
  - Syncs with team members' processed images
  - Tracks processing status in tracking.json

**Supporting Tools**:

- **competitor_number_processing**: Core preprocessing pipeline for competitor number images
- **images_collector**: GUI tool for searching and collecting training images from the web
- **drive_manager**: Bidirectional Google Drive file management
- **images_deduplicator**: Remove duplicate images from training datasets
- **image_search**: Google Custom Search API integration with duplicate tracking

This is an educational project for multimedia data analysis at WSB University.

## Quick Start

### 1. Install UV (Python Package Manager)

If you don't have UV installed:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd analiza_danych_multimedialnych

# Install all dependencies
uv sync
```

### 3. Configure API Credentials

Copy the example config and edit it:

```bash
# Copy example config
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
    "folder_id": "YOUR_DRIVE_FOLDER_ID",
    "download_dir": "cache/downloads"
  },
  "cache": {
    "directory": "cache",
    "max_image_size_mb": 10
  },
  "tracking": {
    "file": "tracking.json"
  }
}
```

### 4. Setup Google Drive OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable "Google Drive API"
3. Create **OAuth Client ID** (Desktop app)
4. Download credentials and save as `secrets/client_secret.json`
5. Add your email to "Test Users" in OAuth consent screen

**Important**: After setup, delete `secrets/token.json` if it exists (scope changed to full Drive access)

### 5. Configure Preprocessing Folders

For collaborative preprocessing, create two folders in Google Drive:

- **Raw folder**: Contains original, unprocessed images
- **Preprocessed folder**: Contains processed images from the team

Update `secrets/config.json` with both folder IDs and your processor ID:

```json
"google_drive": {
  "raw_folder_id": "YOUR_RAW_FOLDER_ID",
  "preprocessed_folder_id": "YOUR_PREPROCESSED_FOLDER_ID"
},
"preprocessing": {
  "processor_id": "YourName"
}
```

To get folder IDs: Open the folder in Google Drive and copy the ID from the URL.

### 6. Run the Preprocessing Pipeline

```bash
# Run the pipeline
uv run .\main.py
```

This will:

1. Download raw images not yet processed
2. Preprocess them locally
3. Upload to shared Drive folder
4. Download teammates' processed images

## Available Tools

### Core: Competitor Number Processing

The main package for preprocessing competitor number images before detection/recognition:

```python
from competitor_number_processing.preprocess import preprocess_image, PreprocessConfig

# Preprocess image for better competitor number detection
config = PreprocessConfig(
    max_long_edge=1280,      # Resize large images
    autocontrast=True,       # Normalize illumination
    median_filter_size=3,    # Noise reduction
    sharpness=1.2           # Enhance details
)

processed_img = preprocess_image("sports_event_photo.jpg", config)
processed_img.save("processed_event_photo.jpg")
```

**Features**:

- Adaptive resizing (maintains aspect ratio)
- Contrast and illumination normalization
- Noise reduction filters
- Sharpness enhancement
- Configurable preprocessing pipeline

### Supporting Tools

#### 1. Image Collector (GUI)

Search and upload images to Google Drive with manual selection.

```bash
uv run images_collector
```

**Features:**

- Search for sports event/competitor number images using Google Custom Search API
- Preview thumbnails and full-resolution images
- Manual selection/rejection of training images
- Automatic duplicate detection (URL & content hash)
- Session persistence
- Batch upload to Google Drive for training dataset

#### 2. Image Deduplicator (CLI)

Remove duplicate images from Google Drive.

```bash
# Dry run - scan and report only (safe)
uv run deduplicator

# Delete with confirmation prompt
uv run deduplicator --delete

# Delete immediately (dangerous!)
uv run deduplicator --delete-confirm

# Show help
uv run deduplicator --help
```

**Features:**

- MD5-based duplicate detection (no downloads needed)
- Detailed duplicate reports with file sizes
- Space savings calculation
- Safe dry-run mode by default
- Keeps oldest file in each duplicate group
- Essential for maintaining clean training datasets

## Typical Workflow

1. **Collect Training Images**

   ```bash
   uv run images_collector
   ```

   Search for "marathon competitor number", "cycling race number", "triathlon bib", etc. Select quality images and upload to Drive.

2. **Remove Duplicates**

   ```bash
   uv run deduplicator
   ```

   Clean your training dataset by removing duplicate images.

3. **Download Images for Processing**

   ```python
   from drive_manager import DriveManager

   manager = DriveManager(credentials_path, folder_id)
   manager.download_all_from_folder(Path("training_images/"))
   ```

4. **Preprocess Images**

   ```python
   from competitor_number_processing.preprocess import preprocess_image, PreprocessConfig
   from pathlib import Path

   config = PreprocessConfig(
       max_long_edge=1280,
       autocontrast=True,
       median_filter_size=3
   )

   for img_path in Path("training_images/").glob("*.jpg"):
       processed = preprocess_image(str(img_path), config)
       processed.save(f"processed/{img_path.name}")
   ```

5. **Competitor Number Detection** (Future implementation)
   - OCR / Text detection
   - Number recognition
   - Participant tracking

## API Setup Guide

### Google Custom Search API

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **Custom Search API**
4. Go to **Credentials** → **Create Credentials** → **API Key**
5. Create a Custom Search Engine at [Programmable Search Engine](https://programmablesearchengine.google.com/)
6. Enable "Image Search" and "Search the entire web"
7. Copy the **Search Engine ID**

### Google Drive API (OAuth 2.0)

1. In the same Google Cloud project, enable **Google Drive API**
2. Go to **APIs & Services** → **Credentials**
3. Click **Create Credentials** → **OAuth Client ID**
4. Choose **Desktop app** as application type
5. Download the JSON file
6. Save it as `secrets/client_secret.json`
7. Configure **OAuth consent screen**:
   - Add your email to "Test Users" if in Testing mode
   - Scopes needed: `https://www.googleapis.com/auth/drive`

**First Run**: When you run a tool that uses Google Drive, a browser will open for authentication. Grant permissions and the token will be saved automatically.

## Project Structure

```
analiza_danych_multimedialnych/
├── pyproject.toml              # Main project config with all dependencies
├── README.md                   # This file
├── tracking.json               # Image tracking database
├── secrets/
│   ├── config.json            # Your configuration (not committed)
│   ├── config.example.json    # Configuration template
│   ├── client_secret.json     # Google OAuth credentials (not committed)
│   └── token.json             # OAuth token (auto-generated, not committed)
├── cache/                      # Downloaded images cache
│   └── downloads/             # Downloaded images from Drive
├── competitor_number_processing/  # Main package
│   ├── __init__.py
│   └── preprocess.py
└── tools/                      # Modular tools
    ├── images_collector/       # PyQt6 GUI application
    │   ├── pyproject.toml
    │   └── images_collector/
    │       ├── __init__.py
    │       ├── main.py         # GUI entry point
    │       └── gui.py          # Main application window
    ├── drive_manager/          # Google Drive management
    │   ├── pyproject.toml
    │   └── drive_manager/
    │       ├── __init__.py
    │       └── manager.py      # DriveManager class
    ├── images_deduplicator/    # Duplicate removal tool
    │   ├── pyproject.toml
    │   ├── README.md
    │   └── images_deduplicator/
    │       ├── __init__.py
    │       ├── deduplicator.py # Core deduplication logic
    │       └── cli.py          # CLI entry point
    └── image_search/           # Search & tracking
        ├── pyproject.toml
        └── image_search/
            ├── __init__.py
            ├── searcher.py     # Google Search integration
            └── tracker.py      # Image tracking system
```

## Available Scripts

Registered in `pyproject.toml`:

```bash
# Image collector GUI
uv run images_collector

# Image deduplicator CLI
uv run deduplicator [--delete|--delete-confirm|--help]
```

## Tools Documentation

### Drive Manager

Bidirectional file management for Google Drive:

```python
from drive_manager import DriveManager

manager = DriveManager(credentials_path, folder_id)

# Upload
file_id = manager.upload_file(Path("image.jpg"))

# List files
files = manager.list_files_in_folder()

# Download (with hash verification)
manager.download_file(file_id, Path("local/image.jpg"), check_hash=True)

# Download all
manager.download_all_from_folder(Path("downloads/"))

# Delete
manager.delete_file(file_id)
manager.delete_files([id1, id2, id3])
```

### Image Deduplicator

Programmatic usage:

```python
from images_deduplicator import ImageDeduplicator
from drive_manager import DriveManager

manager = DriveManager(credentials_path, folder_id)
deduplicator = ImageDeduplicator(manager, dry_run=True)

# Find duplicates
duplicates = deduplicator.find_duplicates()

# Show report
deduplicator.show_duplicates_report(limit=10)

# Get statistics
stats = deduplicator.get_statistics()
print(f"Space wasted: {stats['space_wasted_mb']:.2f} MB")

# Delete (when dry_run=False)
successful, failed = deduplicator.delete_duplicates(
    confirm=True,
    keep_strategy="oldest"
)
```

## Duplicate Detection & Prevention

The system uses multiple strategies to prevent and remove duplicates from training datasets:

### During Collection (images_collector)

- **URL hash**: Prevents re-downloading from same URL
- **File hash (SHA-256)**: Detects identical content from different sources
- **Tracking database**: Stores all metadata in `tracking.json`

### On Google Drive (images_deduplicator)

- **MD5 hash**: Uses Google Drive's built-in checksums
- **No downloads needed**: Analyzes metadata only
- **Smart keeping**: Preserves oldest file by modification time

This ensures your training dataset contains only unique, high-quality competitor number images from various sports events.

## Development

### Adding a New Tool

1. Create directory in `tools/your_tool/`
2. Add `pyproject.toml` with dependencies
3. Create package `tools/your_tool/your_tool/`
4. Add to main `pyproject.toml`:

   ```toml
   dependencies = ["your_tool"]

   [tool.uv.sources]
   your_tool = { path = "./tools/your_tool", editable = true }
   ```

5. Run `uv sync`

## Troubleshooting

### "Module not found" errors

```bash
# Re-sync dependencies
uv sync

# Check installed packages
uv pip list
```

### OAuth errors / Permission denied

```bash
# Delete old token (scope changed)
rm secrets/token.json

# Re-authenticate on next run
```

### Pylance import errors in VS Code

The project includes `.vscode/settings.json` with proper paths. If imports still show errors:

1. Reload VS Code window
2. Check Python interpreter is using the UV virtual environment
3. Run `uv sync` to ensure all packages are installed

## License

Educational project for WSB University
