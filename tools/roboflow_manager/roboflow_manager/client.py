"""
Roboflow client module - handles authentication and dataset downloads from Roboflow API.
"""

import json
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests


class RoboflowClient:
    """Manage datasets from Roboflow using API key authentication"""

    BASE_URL = "https://api.roboflow.com"

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int = 1,
    ):
        """
        Initialize Roboflow client

        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Roboflow project name
            version: Project version number (default: 1)
        """
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.session = requests.Session()
        self.credentials_valid = False
        self._validate_credentials()

    def _validate_credentials(self):
        """Validate API credentials by checking project exists (doesn't require published version)"""
        try:
            # Check if we can access the project at all (no version needed for this)
            url = f"{self.BASE_URL}/{self.workspace}/{self.project}"
            params = {"api_key": self.api_key}

            response = self.session.head(url, params=params, timeout=10)

            if response.status_code == 404:
                raise ValueError(f"Project not found: {self.workspace}/{self.project}")
            elif response.status_code == 401:
                raise ValueError("Invalid Roboflow API key")

            # Check if published version exists (not required, just informational)
            version_url = (
                f"{self.BASE_URL}/{self.workspace}/{self.project}/{self.version}"
            )
            version_response = self.session.head(version_url, params=params, timeout=10)

            if version_response.status_code == 200:
                print(
                    f"[OK] Successfully authenticated with Roboflow (v{self.version} available)"
                )
            else:
                print(
                    f"[OK] Successfully authenticated with Roboflow (v{self.version} not yet published)"
                )

            self.credentials_valid = True
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Connection error to Roboflow API: {e}")

    def download_dataset(
        self,
        output_dir: Path,
        format: str = "yolo",
        extract: bool = True,
    ) -> Path:
        """
        Download dataset from Roboflow

        Args:
            output_dir: Directory to save dataset
            format: Export format (yolo, coco, etc.)
            extract: Whether to extract zip file (default: True)

        Returns:
            Path to extracted dataset directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading {self.project} v{self.version} ({format} format)...")

            # Step 1: Request export and poll until Roboflow finishes generating it.
            # The format must be a path segment, not a query parameter.
            export_url = (
                f"{self.BASE_URL}/{self.workspace}/{self.project}"
                f"/{self.version}/{format}"
            )
            params = {"api_key": self.api_key}

            import time
            download_link = None
            for attempt in range(20):
                export_response = self.session.get(export_url, params=params, timeout=60)
                export_response.raise_for_status()
                export_data = export_response.json()
                download_link = export_data.get("export", {}).get("link")
                if download_link:
                    break
                progress = export_data.get("progress", 0)
                print(f"  Export generating... progress={progress} (attempt {attempt + 1}/20)")
                time.sleep(3)

            if not download_link:
                raise RuntimeError(
                    f"Timed out waiting for Roboflow export. Last response: {export_data}"
                )

            # Step 2: Download the actual zip from the signed URL
            zip_response = self.session.get(download_link, timeout=300, stream=True)
            zip_response.raise_for_status()

            zip_path = output_dir / f"{self.project}_v{self.version}.zip"
            with open(zip_path, "wb") as f:
                for chunk in zip_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"[OK] Downloaded {zip_path.name}")

            if extract:
                return self._extract_dataset(zip_path, output_dir)

            return zip_path

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    def _extract_dataset(self, zip_path: Path, output_dir: Path) -> Path:
        """
        Extract dataset zip file

        Args:
            zip_path: Path to zip file
            output_dir: Directory to extract to

        Returns:
            Path to extracted dataset directory
        """
        try:
            # Find the root folder name in the zip
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Get the first folder in the zip (usually the project name)
                names = zip_ref.namelist()
                if not names:
                    raise ValueError("Empty zip file")

                # Extract to temp location to find root folder
                temp_extract_dir = output_dir / "temp_extract"
                zip_ref.extractall(temp_extract_dir)

            # Find the actual dataset directory
            # Usually it's a single folder at the root of the zip
            extracted_items = list(temp_extract_dir.iterdir())

            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                dataset_dir = extracted_items[0]
            else:
                # If multiple items, assume extraction is correct
                dataset_dir = temp_extract_dir

            # Move to final location with version naming
            final_dir = output_dir / f"{self.project}_v{self.version}"
            if final_dir.exists():
                import shutil

                shutil.rmtree(final_dir)

            dataset_dir.rename(final_dir)

            # Clean up temp and zip
            if temp_extract_dir.exists():
                import shutil

                shutil.rmtree(temp_extract_dir)
            zip_path.unlink()

            print(f"[OK] Extracted to {final_dir.name}")
            return final_dir

        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information from Roboflow

        Returns:
            Dictionary with dataset metadata (classes, images count, etc.)
        """
        try:
            url = f"{self.BASE_URL}/{self.workspace}/{self.project}/{self.version}/info"
            params = {"api_key": self.api_key}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get dataset info: {e}")

    def list_available_versions(self) -> List[Dict[str, Any]]:
        """
        List all available versions of the project

        Returns:
            List of version dictionaries with metadata
        """
        try:
            url = f"{self.BASE_URL}/{self.workspace}/{self.project}/versions"
            params = {"api_key": self.api_key}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            return response.json().get("versions", [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to list versions: {e}")
            return []

    def get_annotation_status(self) -> Dict[str, Any]:
        """
        Get current annotation status of images in the project (without publishing).

        This checks the ACTIVE images in the project, not published versions.

        Returns:
            Dictionary with annotation status:
            - total_images: Total images in project
            - annotated: Number of annotated images
            - pending: Number of images awaiting annotation
            - annotation_progress: Percentage complete
            - status: Overall status message
        """
        try:
            # Get project information endpoint
            url = f"{self.BASE_URL}/{self.workspace}/{self.project}"
            params = {"api_key": self.api_key}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                project_info = data.get("project", {})

                total_images = project_info.get("images", 0)
                unannotated = project_info.get("unannotated", 0)
                annotated = total_images - unannotated
                progress = round(
                    (annotated / total_images * 100) if total_images > 0 else 0, 1
                )

                return {
                    "total_images": total_images,
                    "annotated": annotated,
                    "pending": unannotated,
                    "annotation_progress": progress,
                    "status": f"{annotated}/{total_images} images annotated ({progress}%)",
                }

            # Fallback: return structure showing no data available
            return {
                "total_images": 0,
                "annotated": 0,
                "pending": 0,
                "annotation_progress": 0.0,
                "status": "Unable to fetch real-time annotation status",
                "note": "Check roboflow.com: Log in → Project → Dashboard",
            }

        except requests.exceptions.RequestException as e:
            # Return helpful error response
            return {
                "total_images": 0,
                "annotated": 0,
                "pending": 0,
                "annotation_progress": 0.0,
                "status": "Error checking annotation status",
                "error": str(e),
                "note": "Check roboflow.com manually",
            }

    def get_images_list(self) -> Dict[str, Any]:
        """
        Get images information from the project.

        Returns:
            Dictionary containing project images data
        """
        try:
            url = f"{self.BASE_URL}/{self.workspace}/{self.project}"
            params = {"api_key": self.api_key}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                project_info = data.get("project", {})
                return {
                    "total_images": project_info.get("images", 0),
                    "annotated": project_info.get("images", 0)
                    - project_info.get("unannotated", 0),
                    "unannotated": project_info.get("unannotated", 0),
                    "classes": project_info.get("classes", {}),
                }

            return {
                "error": "Unable to fetch images",
                "status_code": response.status_code,
            }

        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
            }

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list versions: {e}")

    def upload_image(self, file_path: Path) -> bool:
        """
        Upload a single image to the Roboflow project.

        Args:
            file_path: Local path to the image file

        Returns:
            True on success, False on failure
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            print("❌ roboflow package not installed. Run: uv add roboflow")
            return False

        try:
            # Cache the project object so workspace/project aren't re-fetched per image
            if not hasattr(self, "_rf_project"):
                rf = Roboflow(api_key=self.api_key)
                self._rf_project = rf.workspace(self.workspace).project(self.project)
            self._rf_project.upload(image_path=str(file_path), num_retry_uploads=3)
            return True
        except Exception as e:
            print(f"⚠️  Failed to upload {file_path.name} to Roboflow: {e}")
            return False

    def save_metadata(
        self,
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save dataset metadata to JSON file

        Args:
            output_dir: Directory to save metadata
            metadata: Additional metadata to include

        Returns:
            Path to saved metadata file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            dataset_info = self.get_dataset_info()

            # Combine with additional metadata
            full_metadata = {
                "api_metadata": dataset_info,
                "download_config": {
                    "workspace": self.workspace,
                    "project": self.project,
                    "version": self.version,
                    "format": "yolo",
                },
            }

            if metadata:
                full_metadata["custom_metadata"] = metadata

            metadata_path = output_dir / "roboflow_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(full_metadata, f, indent=2)

            print(f"[OK] Saved metadata to {metadata_path.name}")
            return metadata_path

        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
            return None
