# Image Deduplicator

Remove duplicate images from Google Drive based on MD5 hash comparison.

## Features

- **Safe by default**: Dry-run mode shows what would be deleted without making changes
- **MD5-based detection**: Uses Google Drive's built-in MD5 checksums (no downloads needed)
- **Smart keep strategy**: Keeps the oldest file in each duplicate group by default
- **Detailed reporting**: Shows duplicate groups with file sizes and modification dates
- **Space analysis**: Calculates wasted storage space from duplicates
- **Batch deletion**: Efficiently removes multiple duplicates at once

## Usage

### 1. Scan for duplicates (Safe - No deletion)

```bash
uv run python deduplicate.py
```

This will:

- List all files in your configured Google Drive folder
- Group files by MD5 hash
- Show a detailed report of duplicates found
- Calculate space wasted by duplicates
- **Not delete anything**

### 2. Delete duplicates with confirmation

```bash
uv run python deduplicate.py --delete
```

This will:

- Find all duplicates
- Prepare deletion list (keeping oldest file in each group)
- **Require explicit confirmation** before deleting

### 3. Delete duplicates immediately (Dangerous!)

```bash
uv run python deduplicate.py --delete-confirm
```

⚠️ **WARNING**: This will delete files immediately without asking for confirmation!

## How it works

1. **Scanning**: Lists all files in the configured Google Drive folder
2. **Hashing**: Groups files by their MD5 checksum (provided by Google Drive API)
3. **Duplicate detection**: Identifies groups where the same hash appears multiple times
4. **Keep strategy**: For each duplicate group, keeps the file with the earliest `modifiedTime`
5. **Deletion**: Removes all other files in the duplicate group

## Keep Strategies

The deduplicator supports different strategies for choosing which file to keep:

- `oldest` (default): Keep the file with earliest modification time
- `newest`: Keep the file with latest modification time
- `first`: Keep the first file encountered (arbitrary)

## Example Output

```
================================================================================
GOOGLE DRIVE IMAGE DEDUPLICATOR
================================================================================
Mode: DRY RUN (safe)
Folder ID: 1oHeMvKvBAfRBiw3Hz5-tBxAHeyH-aONg
================================================================================

Scanning Google Drive for files...
Found 150 files in folder 1oHeMvKvBAfRBiw3Hz5-tBxAHeyH-aONg
Found 150 files. Analyzing for duplicates...

============================================================
DUPLICATE DETECTION RESULTS
============================================================
Total files scanned: 150
Duplicate groups found: 12
Total duplicate files: 28
Files to keep: 12
Files that can be deleted: 16
============================================================

DETAILED DUPLICATES REPORT
================================================================================

Group 1 - MD5: a3b5c9d2e1f4g6h7...
Files (3 duplicates):
  [KEEP  ] image_001.jpg                          |   2.45 MB | 2025-01-10 | ID: abc123
  [DELETE] image_001_copy.jpg                     |   2.45 MB | 2025-01-11 | ID: def456
  [DELETE] image_001_final.jpg                    |   2.45 MB | 2025-01-12 | ID: ghi789

💾 Space wasted by duplicates: 38.50 MB
📊 16 files can be safely deleted
```

## Configuration

The deduplicator uses the same `secrets/config.json` as other tools:

```json
{
  "google_drive": {
    "credentials_path": "secrets/client_secret.json",
    "folder_id": "YOUR_DRIVE_FOLDER_ID"
  }
}
```

## Safety Features

- **Dry-run by default**: Won't delete anything unless explicitly told to
- **Confirmation required**: `--delete` mode still requires confirmation
- **Detailed preview**: See exactly what will be deleted before it happens
- **Keep strategy**: Always keeps at least one copy of each unique file
- **MD5-based**: Only considers files with identical MD5 hashes as duplicates

## Integration

The deduplicator is automatically integrated with your project:

```python
from images_deduplicator import ImageDeduplicator
from drive_manager import DriveManager

# Initialize
drive_manager = DriveManager(credentials_path, folder_id)
deduplicator = ImageDeduplicator(drive_manager, dry_run=True)

# Find duplicates
duplicates = deduplicator.find_duplicates()

# Show report
deduplicator.show_duplicates_report()

# Delete (if dry_run=False)
deduplicator.delete_duplicates(confirm=True, keep_strategy="oldest")
```

## Requirements

- Python 3.12+
- Google Drive API credentials with `drive` scope
- `drive_manager` tool installed
