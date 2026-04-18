"""
Example usage of RoboflowClient for downloading and managing datasets.
"""

from pathlib import Path
from roboflow_manager import RoboflowClient
import json


def main():
    # Load configuration
    config_path = Path("secrets/config.json")
    with open(config_path) as f:
        config = json.load(f)

    roboflow_config = config["roboflow"]

    # Initialize client
    client = RoboflowClient(
        api_key=roboflow_config["api_key"],
        workspace=roboflow_config["workspace"],
        project=roboflow_config["project"],
        version=roboflow_config["version"],
    )

    # Download dataset
    output_dir = Path(roboflow_config["download_dir"])
    dataset_dir = client.download_dataset(
        output_dir=output_dir,
        format=roboflow_config.get("format", "yolo"),
        extract=True,
    )

    print(f"\nDataset downloaded to: {dataset_dir}")

    # Save metadata
    client.save_metadata(
        output_dir=dataset_dir,
        metadata={
            "download_date": "2024-04-18",
            "classes": ["person", "number"],
        },
    )

    # List directory structure
    print("\nDataset structure:")
    for item in sorted(dataset_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(dataset_dir)
            print(f"  {rel_path}")


if __name__ == "__main__":
    main()
