"""
Image Search & Upload Application

A PyQt6 application for searching images using Google Custom Search API,
allowing manual selection, and uploading to Google Drive.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Configure API credentials in `config.json`:

   - Google Custom Search API key and search engine ID
   - Google Drive OAuth 2.0 Client Secret path
   - Drive folder ID for uploads

3. Run the application:
   ```bash
   uv run main.py
   ```

## Features

- **Search**: Search for images using Google Custom Search API
- **Pagination**: "Load More" button to fetch additional results (automatically skips duplicates)
- **Preview**: Display thumbnails and full-resolution previews
- **Selection**: Select images for upload or Reject unwanted ones
- **Persistence**: Automatically loads previous session images (pending/failed) on startup
- **Tracking**: Prevents duplicates using URL and file content hashing
- **Upload**: Upload selected images to your personal Google Drive via OAuth 2.0

## API Setup

### Google Custom Search API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable "Custom Search API"
3. Create API credentials (API Key)
4. Create a Custom Search Engine at [Programmable Search Engine](https://programmablesearchengine.google.com/)
5. Enable "Image Search" and "Search the entire web"
6. Copy the Search Engine ID

### Google Drive API (OAuth 2.0)

1. In Google Cloud Console, enable "Google Drive API"
2. Go to **APIs & Services > Credentials**
3. Create **OAuth Client ID** (Application type: Desktop app)
4. Download the JSON file and rename it to `client_secret.json`
5. Place `client_secret.json` in the project root
6. **Important**: Add your email to "Test Users" in "OAuth consent screen" if the app is in Testing mode.

## Configuration

Edit `config.json`:

```json
{
  "google_custom_search": {
    "api_key": "YOUR_API_KEY",
    "search_engine_id": "YOUR_SEARCH_ENGINE_ID",
    "num_results": 10
  },
  "google_drive": {
    "credentials_path": "D:\\Path\\To\\client_secret.json",
    "folder_id": "YOUR_DRIVE_FOLDER_ID"
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

## Project Structure

```
.
├── main.py                 # Entry point
├── config.json            # Configuration file
├── client_secret.json     # OAuth 2.0 credentials (not committed)
├── tracking.json          # Image tracking database
├── pyproject.toml         # Project dependencies
├── cache/                 # Downloaded images cache
└── src/
    ├── main_gui.py        # Main PyQt6 application
    ├── tracker.py         # JSON-based image tracker
    ├── search.py          # Google Custom Search integration
    └── drive_uploader.py  # Google Drive upload functionality (OAuth)
```

## Usage

1. Enter search query in the search box
2. Click "Search" or press Enter
3. Wait for thumbnails to load
4. Click thumbnails to preview full-resolution images
5. Check boxes next to images you want to upload
6. Click "Upload Selected to Drive"
7. Wait for upload completion

## Duplicate Prevention

The application tracks images using two methods:

- URL hash: Prevents re-downloading the same URL
- File hash (SHA-256): Detects duplicate image content

Tracking data is stored in `tracking.json` with the following information:

- URL and file hashes
- Search query
- Selection status (pending/selected/rejected/uploaded)
- Timestamps
- Google Drive file IDs (after upload)

## License

Educational project for WSB University
