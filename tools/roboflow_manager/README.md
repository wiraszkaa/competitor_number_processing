# Roboflow Manager

A standalone tool for downloading and managing annotated datasets from Roboflow.

## Features

- **Download datasets** in multiple formats (YOLO, COCO, etc.)
- **Automatic extraction** of downloaded zip files
- **Metadata tracking** with JSON export
- **Version management** - easily switch between project versions
- **Error handling** with informative messages
- **API validation** - test credentials before downloading

## Installation

This tool is included as a dependency in the main project. No separate installation needed.

## Configuration

Add your Roboflow credentials to `secrets/config.json`:

```json
{
  "roboflow": {
    "api_key": "YOUR_ROBOFLOW_API_KEY",
    "workspace": "YOUR_WORKSPACE_NAME",
    "project": "YOUR_PROJECT_NAME",
    "version": 1,
    "format": "yolo",
    "download_dir": "cache/roboflow_datasets"
  }
}
```

## Quick Start

```python
from roboflow_manager import RoboflowClient
from pathlib import Path

# Initialize client
client = RoboflowClient(
    api_key="your_api_key",
    workspace="your_workspace",
    project="your_project",
    version=1
)

# Download dataset
dataset_dir = client.download_dataset(
    output_dir=Path("cache/roboflow_datasets"),
    format="yolo",
    extract=True
)

# Save metadata
client.save_metadata(output_dir=dataset_dir)
```

## API Reference

### RoboflowClient

#### `__init__(api_key, workspace, project, version=1)`

Initialize the client and validate credentials.

**Parameters:**

- `api_key` (str): Your Roboflow API key
- `workspace` (str): Workspace name
- `project` (str): Project name
- `version` (int): Project version (default: 1)

#### `download_dataset(output_dir, format='yolo', extract=True)`

Download dataset from Roboflow.

**Parameters:**

- `output_dir` (Path): Where to save the dataset
- `format` (str): Export format - 'yolo', 'coco', etc. (default: 'yolo')
- `extract` (bool): Whether to extract zip file (default: True)

**Returns:** Path to extracted dataset directory

#### `get_dataset_info()`

Get metadata about the dataset (classes, image count, etc.).

**Returns:** Dictionary with dataset information

#### `list_available_versions()`

List all available versions of the project.

**Returns:** List of version dictionaries

#### `save_metadata(output_dir, metadata=None)`

Save dataset metadata to JSON file.

**Parameters:**

- `output_dir` (Path): Where to save metadata
- `metadata` (dict, optional): Additional custom metadata

**Returns:** Path to saved metadata file

## Output Structure

After downloading with YOLO format, the dataset structure will be:

```
{project}_v{version}/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   ├── img_002.txt
│   │   └── ...
│   ├── val/
│   └── test/
├── data.yaml
└── roboflow_metadata.json
```

## Error Handling

The client provides clear error messages for:

- Invalid API credentials
- Project not found
- Connection errors
- Extraction failures

## Example Usage

See `scripts/example_roboflow_download.py` for a complete example.
