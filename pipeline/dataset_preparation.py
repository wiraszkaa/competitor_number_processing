"""
Dataset Preparation Pipeline - First step of the processing pipeline.

This pipeline step handles:
1. Download raw images from Google Drive (if not already downloaded)
2. Check preprocessing status
3. Preprocess images that haven't been processed yet
4. Upload preprocessed images to Drive
5. Check annotation status via Roboflow
6. Simplify tracking (no processor_id tracking)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "drive_manager"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "roboflow_manager"))

from drive_manager.manager import DriveManager
from roboflow_manager.client import RoboflowClient

from competitor_number_processing.preprocess import PreprocessConfig, preprocess_image
from competitor_number_processing.tracking import PreprocessingTracker
from pipeline.config import load_config, get_pipeline_logger

logger = get_pipeline_logger(__name__)


class DatasetPreparationPipeline:
    """Orchestrates dataset download, preprocessing, and annotation checking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, validate: bool = False):
        """
        Initialize the dataset preparation pipeline.

        Args:
            config: Configuration dictionary (loaded from file if not provided)
            validate: Whether to validate preprocessed files exist on Drive
        """
        self.config = config or load_config()
        self.validate = validate
        self.tracker: Optional[PreprocessingTracker] = None
        self.drive_manager: Optional[DriveManager] = None
        self.roboflow_client: Optional[RoboflowClient] = None

    def run(self) -> Dict[str, Any]:
        """
        Run the complete dataset preparation pipeline.

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 70)
        logger.info("Dataset Preparation Pipeline")
        logger.info("=" * 70)

        results = {}

        try:
            # Step 1: Initialize managers
            logger.info("\n📋 Initializing managers...")
            self._initialize_managers()
            results["initialization"] = "success"

            # Step 2: Download raw images
            logger.info("\n📥 Step 1: Download Raw Images")
            logger.info("=" * 70)
            raw_files = self._download_raw_images()
            results["raw_images_downloaded"] = len(raw_files)

            # Step 2b: Sync preprocessed files from Drive
            logger.info("\n🔄 Step 1b: Sync Preprocessed Files from Drive")
            logger.info("=" * 70)
            synced_files = self._sync_preprocessed_files()
            results["preprocessed_files_synced"] = len(synced_files)

            # Step 2c: [OPTIONAL] Validate preprocessed files exist on Drive
            # Set validate=True to enable this check (helps catch accidental deletions)
            if self.validate:
                logger.info("\n🔍 Step 1c: Validate Preprocessed Files")
                logger.info("=" * 70)
                validation_results = self._validate_preprocessed_files(validate=True)
                results["validation"] = validation_results
            else:
                logger.info(
                    "\n⏭️  Step 1c: Skipping Validation (use --validate to enable)"
                )
                results["validation"] = {"skipped": True}

            # Step 3: Preprocess images
            logger.info("\n⚙️  Step 2: Preprocess Images")
            logger.info("=" * 70)
            preprocessing_results = self._preprocess_images(raw_files)
            results["preprocessing"] = preprocessing_results

            # Step 4: Check annotation status
            logger.info("\n📝 Step 3: Check Annotation Status")
            logger.info("=" * 70)
            annotation_results = self._check_annotation_status()
            results["annotation"] = annotation_results

            # Step 5: Summary
            logger.info("\n" + "=" * 70)
            logger.info("Dataset Preparation Complete")
            logger.info("=" * 70)
            self._print_summary(results)

            return results

        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            results["error"] = str(e)
            return results

    def _initialize_managers(self) -> None:
        """Initialize Drive and Roboflow managers."""
        # Initialize tracking
        tracking_file = Path(self.config["tracking"]["file"])
        self.tracker = PreprocessingTracker(tracking_file)
        self.tracker.print_summary()

        # Initialize Drive manager
        credentials_path = self.config["google_drive"]["credentials_path"]
        self.drive_manager = DriveManager(credentials_path)
        logger.info("✓ Drive manager initialized")

        # Initialize Roboflow client (if configured)
        roboflow_config = self.config.get("roboflow", {})
        if (
            roboflow_config.get("api_key")
            and roboflow_config.get("api_key") != "YOUR_ROBOFLOW_API_KEY"
        ):
            try:
                self.roboflow_client = RoboflowClient(
                    api_key=roboflow_config["api_key"],
                    workspace=roboflow_config["workspace"],
                    project=roboflow_config["project"],
                    version=roboflow_config.get("version", 1),
                )
                logger.info("✓ Roboflow client initialized")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize Roboflow: {e}")
                self.roboflow_client = None
        else:
            logger.info("ℹ️  Roboflow not configured")

    def _download_raw_images(self) -> List[Path]:
        """
        Download raw images from Google Drive that haven't been processed yet.

        Returns:
            List of downloaded image paths
        """
        if not self.drive_manager:
            logger.warning("Drive manager not initialized")
            return []

        drive_config = self.config["google_drive"]
        raw_folder_id = drive_config["raw_folder_id"]
        download_dir = Path(drive_config["download_dir_raw"])
        download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📂 Checking raw folder: {raw_folder_id}")

        # List all files in raw folder
        try:
            drive_files = self.drive_manager.list_files_in_folder(
                folder_id=raw_folder_id
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not access Drive: {e}")
            return []

        if not drive_files:
            logger.info("✓ No files in raw folder")
            return []

        logger.info(f"Found {len(drive_files)} files in raw folder")

        # Filter image files and check if already processed
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        files_to_download = []
        downloaded_paths = []

        for drive_file in drive_files:
            file_id = drive_file["id"]
            file_name = drive_file["name"]
            local_path = download_dir / file_name

            # Skip preprocessed markers
            if any(
                marker in file_name
                for marker in [
                    "__final",
                    "__resized",
                    "__brightness",
                    "__contrast",
                    "__denoise",
                    "__grass",
                ]
            ):
                continue

            # Skip non-image files
            if Path(file_name).suffix.lower() not in image_extensions:
                continue

            # Check if already downloaded and processed
            should_download = True
            if local_path.exists():
                record = self.tracker.get_record_by_file(local_path)
                if record and record.preprocessing_status == "completed":
                    # File already processed, preprocessed version should be on Drive
                    logger.info(f"  ✓ {file_name} - already completed")
                    should_download = False

            if should_download:
                files_to_download.append((file_id, file_name, local_path))

        logger.info(f"\n📥 Downloading {len(files_to_download)} files...")

        for file_id, file_name, local_path in tqdm(
            files_to_download,
            desc="Downloading",
            unit="file",
            disable=len(files_to_download) == 0,
        ):
            success = self.drive_manager.download_file(file_id, local_path)
            if success:
                self.tracker.add_or_update_record(
                    local_path, drive_raw_id=file_id, status="pending"
                )
                downloaded_paths.append(local_path)
            else:
                logger.error(f"❌ Failed to download: {file_name}")

        # Check for existing files that might be pending
        logger.info(f"\n📂 Checking for existing files needing processing...")
        for drive_file in tqdm(
            drive_files,
            desc="Scanning",
            unit="file",
            disable=len(drive_files) == 0,
        ):
            file_name = drive_file["name"]
            file_id = drive_file["id"]
            local_path = download_dir / file_name

            if local_path.exists():
                record = self.tracker.get_record_by_file(local_path)
                if not record:
                    self.tracker.add_or_update_record(
                        local_path, drive_raw_id=file_id, status="pending"
                    )

        self.tracker.save()
        return downloaded_paths

    def _sync_preprocessed_files(self) -> List[Path]:
        """
        Download preprocessed files from Google Drive to keep local cache in sync.

        This creates a local mirror of the Drive's preprocessed folder.
        The cache/preprocessed directory is AUTHORITATIVE - only files from Drive are stored here.
        Locally created preprocessed files are stored in cache/temp_preprocessed/ instead.

        Returns:
            List of synced preprocessed file paths
        """
        if not self.drive_manager:
            logger.warning("Drive manager not initialized")
            return []

        drive_config = self.config["google_drive"]
        preprocessed_folder_id = drive_config.get(
            "preprocessed_folder_id", "PLACEHOLDER_FOR_PREPROCESSED_FOLDER"
        )

        if preprocessed_folder_id == "PLACEHOLDER_FOR_PREPROCESSED_FOLDER":
            logger.info("ℹ️  Preprocessed folder not configured")
            return []

        cache_dir = Path(self.config["cache"]["directory"])
        preprocessed_dir = cache_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"📂 Checking preprocessed folder for sync: {preprocessed_folder_id}"
        )

        # List all files in preprocessed folder
        try:
            drive_files = self.drive_manager.list_files_in_folder(
                folder_id=preprocessed_folder_id
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not access Drive: {e}")
            return []

        if not drive_files:
            logger.info("✓ No preprocessed files on Drive")
            return []

        logger.info(f"Found {len(drive_files)} preprocessed files on Drive")

        # Filter to only __final.png files
        files_to_download = []
        synced_paths = []

        for drive_file in drive_files:
            file_id = drive_file["id"]
            file_name = drive_file["name"]

            # Only download final preprocessed images
            if not file_name.endswith("__final.png"):
                continue

            local_path = preprocessed_dir / file_name

            # Skip if already exists locally
            if local_path.exists():
                synced_paths.append(local_path)
                continue

            files_to_download.append((file_id, file_name, local_path))

        logger.info(
            f"\n📥 Syncing {len(files_to_download)} preprocessed files from Drive..."
        )

        for file_id, file_name, local_path in tqdm(
            files_to_download,
            desc="Syncing",
            unit="file",
            disable=len(files_to_download) == 0,
        ):
            try:
                original_folder = self.drive_manager.folder_id
                self.drive_manager.folder_id = preprocessed_folder_id

                success = self.drive_manager.download_file(file_id, local_path)

                self.drive_manager.folder_id = original_folder

                if success:
                    logger.info(f"   ✓ Synced: {file_name}")
                    synced_paths.append(local_path)
                else:
                    logger.warning(f"   ⚠️  Failed to sync: {file_name}")
            except Exception as e:
                logger.warning(f"   ⚠️  Error syncing {file_name}: {e}")
                self.drive_manager.folder_id = original_folder

        return synced_paths

    def _validate_preprocessed_files(self, validate: bool = False) -> Dict[str, Any]:
        """
        Validate that preprocessed files marked as 'completed' still exist on Drive.

        This is an optional step that checks for accidental deletions and updates tracking
        to mark missing files as pending for reprocessing.

        Args:
            validate: If False, skips validation (default). Set to True to enable validation.

        Returns:
            Dictionary with validation results
        """
        results = {
            "checked": 0,
            "missing_on_drive": 0,
            "reset_to_pending": 0,
        }

        if not validate:
            logger.info("ℹ️  File validation disabled (set validate=True to enable)")
            return results

        if not self.tracker or not self.drive_manager:
            logger.warning("Tracker or Drive manager not initialized")
            return results

        drive_config = self.config["google_drive"]
        preprocessed_folder_id = drive_config.get(
            "preprocessed_folder_id", "PLACEHOLDER_FOR_PREPROCESSED_FOLDER"
        )

        if preprocessed_folder_id == "PLACEHOLDER_FOR_PREPROCESSED_FOLDER":
            logger.info("ℹ️  Preprocessed folder not configured, skipping validation")
            return results

        logger.info(
            f"📂 Validating preprocessed files in Drive: {preprocessed_folder_id}"
        )

        # Get all files currently on Drive
        try:
            drive_files = self.drive_manager.list_files_in_folder(
                folder_id=preprocessed_folder_id
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not access Drive for validation: {e}")
            return results

        if not drive_files:
            logger.warning("⚠️  No preprocessed files found on Drive!")

        # Create set of preprocessed file names on Drive
        drive_file_names = {f["name"] for f in drive_files}
        logger.info(f"Found {len(drive_file_names)} preprocessed files on Drive")

        # Check each completed record
        completed_records = self.tracker.get_completed_files()
        logger.info(f"Checking {len(completed_records)} completed records...")

        for record in completed_records:
            results["checked"] += 1

            if not record.drive_preprocessed_id:
                # No Drive ID recorded, can't validate
                continue

            # Expected preprocessed file name (file_name stem + __final.png)
            file_stem = Path(record.file_name).stem
            expected_file_name = f"{file_stem}__final.png"

            # Check if file exists on Drive
            if expected_file_name not in drive_file_names:
                logger.warning(
                    f"  ⚠️  Missing on Drive: {expected_file_name} (original: {record.file_name})"
                )
                results["missing_on_drive"] += 1

                # Reset to pending for reprocessing
                try:
                    raw_path = (
                        Path(self.config["google_drive"]["download_dir_raw"])
                        / record.file_name
                    )
                    self.tracker.add_or_update_record(raw_path, status="pending")
                    results["reset_to_pending"] += 1
                    logger.info(f"     ↻ Reset to pending: {record.file_name}")
                except Exception as e:
                    logger.error(f"     ✗ Failed to reset: {e}")

        self.tracker.save()

        logger.info(f"\n✓ Validation complete:")
        logger.info(f"  • Checked: {results['checked']} completed files")
        logger.info(f"  • Missing on Drive: {results['missing_on_drive']}")
        logger.info(f"  • Reset to pending: {results['reset_to_pending']}")

        return results

    def _preprocess_images(self, downloaded_files: List[Path] = None) -> Dict[str, Any]:
        """
        Preprocess images from the downloaded set (or all pending if not specified).

        Files are preprocessed to a temporary location (temp_preprocessed), then uploaded to Drive.
        The authoritative cache/preprocessed folder contains only synced files from Drive.

        Args:
            downloaded_files: List of files downloaded in this run (if None, processes all pending)

        Returns:
            Dictionary with preprocessing results
        """
        if not self.tracker or not self.drive_manager:
            logger.warning("Tracker or Drive manager not initialized")
            return {}

        cache_dir = Path(self.config["cache"]["directory"])
        # Use temp directory for local preprocessing (not authoritative)
        # cache/preprocessed is reserved for synced files from Drive only
        output_dir = cache_dir / "temp_preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration for preprocessing
        preprocess_config = PreprocessConfig(
            max_long_edge=1280,
            autocontrast=True,
            gamma=1.0,
            brightness=1.0,
            median_filter_size=3,
            gaussian_blur_radius=0.0,
            contrast=1.0,
            enable_grass_preprocessing=False,
            grass_edge_enhancement=False,
            grass_sharpening=False,
        )

        # Get all raw files
        raw_dir = Path(self.config["google_drive"]["download_dir_raw"])
        if not raw_dir.exists():
            logger.info("✓ No raw directory yet")
            return {"processed": 0, "uploaded": 0, "skipped": 0, "failed": 0}

        all_raw_files = [
            f
            for f in raw_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        ]

        # Filter to only pending files from downloaded set (or all if none specified)
        # If downloaded_files is provided, only process those files
        files_to_check = downloaded_files if downloaded_files else all_raw_files

        pending_files = []
        for f in files_to_check:
            record = self.tracker.get_record_by_file(f)
            if not record:
                # New file not in tracker
                pending_files.append(f)
            elif record.preprocessing_status in ["pending", "failed"]:
                # Already in tracker but not completed
                pending_files.append(f)
            elif record.preprocessing_status == "completed":
                # Already processed - skip it
                # Preprocessed version is on Drive and will be synced
                logger.info(f"  ✓ {f.name} - already completed")
                # Skip it

        logger.info(f"\n🖼️  Processing {len(pending_files)} image(s)")

        results = {
            "processed": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
        }

        drive_config = self.config["google_drive"]
        preprocessed_folder_id = drive_config.get(
            "preprocessed_folder_id", "PLACEHOLDER_FOR_PREPROCESSED_FOLDER"
        )

        for image_path in tqdm(
            pending_files,
            desc="Preprocessing",
            unit="image",
            disable=len(pending_files) == 0,
        ):

            try:
                # Mark as in progress
                self.tracker.mark_as_in_progress(image_path)
                self.tracker.save()

                result_paths = preprocess_image(
                    input_path=image_path,
                    cfg=preprocess_config,
                    save_debug_to=output_dir,
                    prefix=image_path.stem,
                )

                final_path = result_paths["final"]
                results["processed"] += 1

                # Upload to Drive
                if preprocessed_folder_id != "PLACEHOLDER_FOR_PREPROCESSED_FOLDER":
                    original_folder = self.drive_manager.folder_id
                    self.drive_manager.folder_id = preprocessed_folder_id

                    file_id = self.drive_manager.upload_file(final_path)

                    self.drive_manager.folder_id = original_folder

                    if file_id:
                        self.tracker.mark_as_completed(
                            image_path, drive_preprocessed_id=file_id
                        )
                        results["uploaded"] += 1
                    else:
                        logger.error(
                            f"❌ Upload failed for {image_path.name} - marking as failed"
                        )
                        self.tracker.mark_as_failed(
                            image_path, "Upload to Drive failed"
                        )
                        results["failed"] += 1
                else:
                    logger.error(
                        f"❌ Preprocessed folder not configured - cannot mark as completed"
                    )
                    self.tracker.mark_as_failed(
                        image_path,
                        "Preprocessed folder not configured in config",
                    )

                self.tracker.save()

            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                self.tracker.mark_as_failed(image_path, str(e))
                self.tracker.save()
                results["failed"] += 1

        return results

    def _check_annotation_status(self) -> Dict[str, Any]:
        """
        Check annotation status via Roboflow (real-time, without publishing).

        This shows:
        - Current annotation progress in Roboflow
        - Number of images annotated vs pending
        - Overall progress percentage

        Returns:
            Dictionary with annotation status
        """
        roboflow_config = self.config.get("roboflow", {})

        if not roboflow_config.get("api_key"):
            logger.info("ℹ️  Roboflow not configured")
            return {
                "total_images": 0,
                "annotated": 0,
                "pending": 0,
                "annotation_progress": 0,
                "status": "not_configured",
            }

        if not self.roboflow_client:
            logger.info(
                "⚠️  Roboflow client not initialized - checking configuration..."
            )
            logger.info(
                f"   Project: {roboflow_config.get('workspace')}/{roboflow_config.get('project')}"
            )
            return {
                "total_images": 0,
                "annotated": 0,
                "pending": 0,
                "annotation_progress": 0,
                "status": "client_not_initialized",
            }

        try:
            logger.info("📝 Checking real-time annotation status in Roboflow...")

            # Get current annotation status (no need to publish)
            annotation_status = self.roboflow_client.get_annotation_status()

            total = annotation_status.get("total_images", 0)
            annotated = annotation_status.get("annotated", 0)
            pending = annotation_status.get("pending", 0)
            progress = annotation_status.get("annotation_progress", 0)

            if total > 0:
                logger.info(f"\n✅ Roboflow Annotation Status:")
                logger.info(f"   • Total images: {total}")
                logger.info(f"   • ✓ Annotated: {annotated}")
                logger.info(f"   • ⏳ Pending: {pending}")
                logger.info(f"   • 📊 Progress: {progress}%")

                if pending == 0:
                    logger.info(f"\n🎉 All images annotated! Ready to publish version.")
                else:
                    logger.info(
                        f"\n📌 {pending} images still need annotation in Roboflow"
                    )
            else:
                logger.info(
                    f"\n   {annotation_status.get('status', 'No images in project yet')}"
                )
                logger.info(f"   {annotation_status.get('note', '')}")

            return annotation_status

        except Exception as e:
            logger.error(f"❌ Error checking annotation status: {e}")
            return {
                "total_images": 0,
                "annotated": 0,
                "pending": 0,
                "annotation_progress": 0,
                "status": "error",
                "error": str(e),
            }

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary of pipeline results."""
        logger.info("\n📊 Pipeline Results:")
        logger.info(
            f"  Raw images downloaded: {results.get('raw_images_downloaded', 0)}"
        )

        preprocessing = results.get("preprocessing", {})
        if preprocessing:
            logger.info(f"\n  Preprocessing:")
            logger.info(f"    • Processed: {preprocessing.get('processed', 0)}")
            logger.info(f"    • Uploaded: {preprocessing.get('uploaded', 0)}")
            logger.info(f"    • Skipped: {preprocessing.get('skipped', 0)}")
            logger.info(f"    • Failed: {preprocessing.get('failed', 0)}")

        annotation = results.get("annotation", {})
        if annotation and annotation.get("status") == "checked":
            logger.info(f"\n  Annotation Status:")
            logger.info(f"    • Total images: {annotation.get('total_images', 0)}")
            logger.info(f"    • Annotated: {annotation.get('annotated', 0)}")
            logger.info(f"    • Pending: {annotation.get('pending', 0)}")

        # Final tracker summary
        if self.tracker:
            logger.info("\n📈 Updated Tracking Status:")
            self.tracker.load()
            self.tracker.print_summary()

        logger.info("\n✅ Dataset preparation complete!")


def main(validate: bool = False):
    """Entry point for dataset preparation pipeline.

    Args:
        validate: Whether to validate preprocessed files exist on Drive
    """
    config = load_config()
    pipeline = DatasetPreparationPipeline(config, validate=validate)
    results = pipeline.run()
    return results


if __name__ == "__main__":
    main()
