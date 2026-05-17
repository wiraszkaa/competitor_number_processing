"""
Dataset Preparation Pipeline.

Drive raw_folder is the single source of truth for images to process.
Drive preprocessed_folder (matching filename stem) is the source of truth for what's done.
tracking.json lives on Drive and only tracks Roboflow upload state.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "drive_manager"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "roboflow_manager"))

from drive_manager.manager import DriveManager
from roboflow_manager.client import RoboflowClient

from competitor_number_processing.preprocess import PreprocessConfig, preprocess_image
from competitor_number_processing.tracking import PipelineTracker
from pipeline.config import load_config, get_pipeline_logger

logger = get_pipeline_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
PREPROCESS_CFG = PreprocessConfig(
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


class DatasetPreparationPipeline:
    """Orchestrates dataset preprocessing and Roboflow upload."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.tracker: Optional[PipelineTracker] = None
        self.drive_manager: Optional[DriveManager] = None
        self.roboflow_client: Optional[RoboflowClient] = None

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("Dataset Preparation Pipeline")
        logger.info("=" * 70)

        results: Dict[str, Any] = {}

        try:
            logger.info("\n📋 Initializing managers...")
            self._initialize_managers()

            logger.info("\n☁️  Syncing tracking.json from Drive...")
            self._sync_tracking_from_drive()

            logger.info("\n📂 Reading Drive state...")
            raw_files, preprocessed_drive_files = self._get_drive_state()
            preprocessed_stems: Set[str] = {
                Path(f["name"]).stem for f in preprocessed_drive_files
            }
            logger.info(
                f"  Raw: {len(raw_files)} files | Preprocessed: {len(preprocessed_stems)} files"
            )

            logger.info("\n⚙️  Processing new images...")
            proc_results = self._process_new_images(raw_files, preprocessed_stems)
            results["preprocessing"] = proc_results

            # Refresh preprocessed Drive listing after new uploads
            if proc_results.get("uploaded", 0) > 0:
                logger.info("\n🔄 Refreshing preprocessed folder listing...")
                _, preprocessed_drive_files = self._get_drive_state()

            logger.info("\n📤 Uploading to Roboflow...")
            roboflow_results = self._upload_to_roboflow(preprocessed_drive_files)
            results["roboflow_upload"] = roboflow_results

            logger.info("\n📝 Checking annotation status...")
            results["annotation"] = self._check_annotation_status()

            logger.info("\n💾 Saving tracking to Drive...")
            self._save_tracking_to_drive()

            logger.info("\n" + "=" * 70)
            logger.info("Dataset Preparation Complete")
            logger.info("=" * 70)
            self._print_summary(results)

        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)

        return results

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    def _initialize_managers(self) -> None:
        drive_config = self.config["google_drive"]
        credentials_path = drive_config["credentials_path"]
        self.drive_manager = DriveManager(credentials_path)
        logger.info("✓ Drive manager initialized")

        tracking_file = Path(self.config["tracking"]["file"])
        self.tracker = PipelineTracker(tracking_file)

        roboflow_config = self.config.get("roboflow", {})
        if (
            roboflow_config.get("api_key")
            and roboflow_config["api_key"] != "YOUR_ROBOFLOW_API_KEY"
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
        else:
            logger.info("ℹ️  Roboflow not configured")

    # ------------------------------------------------------------------ #
    # Tracking sync                                                        #
    # ------------------------------------------------------------------ #

    def _sync_tracking_from_drive(self) -> None:
        """Download tracking.json from Drive if it exists."""
        assert self.drive_manager is not None
        assert self.tracker is not None
        folder_id = self.config["google_drive"].get("tracking_drive_folder_id", "")
        if not folder_id or folder_id == "YOUR_DRIVE_FOLDER_ID_FOR_TRACKING":
            logger.info("ℹ️  tracking_drive_folder_id not configured — using local tracking.json")
            return

        try:
            drive_files = self.drive_manager.list_files_in_folder(folder_id=folder_id)
        except Exception as e:
            logger.warning(f"⚠️  Could not list tracking folder: {e}")
            return

        tracking_on_drive = next(
            (f for f in drive_files if f["name"] == "tracking.json"), None
        )
        if not tracking_on_drive:
            logger.info("ℹ️  No tracking.json on Drive yet — starting fresh")
            return

        local_path = Path(self.config["tracking"]["file"])
        original_folder = self.drive_manager.folder_id
        self.drive_manager.folder_id = folder_id
        success = self.drive_manager.download_file(
            tracking_on_drive["id"], local_path, skip_if_exists=False, check_hash=False
        )
        self.drive_manager.folder_id = original_folder

        if success:
            self.tracker.load()
            logger.info(f"✓ Downloaded tracking.json from Drive")
        else:
            logger.warning("⚠️  Failed to download tracking.json from Drive")

    def _save_tracking_to_drive(self) -> None:
        """Save tracking.json locally and upload to Drive."""
        assert self.drive_manager is not None
        assert self.tracker is not None
        self.tracker.save()

        folder_id = self.config["google_drive"].get("tracking_drive_folder_id", "")
        if not folder_id or folder_id == "YOUR_DRIVE_FOLDER_ID_FOR_TRACKING":
            logger.info("ℹ️  tracking_drive_folder_id not configured — tracking.json saved locally only")
            return

        local_path = Path(self.config["tracking"]["file"])
        original_folder = self.drive_manager.folder_id
        self.drive_manager.folder_id = folder_id

        # Delete old tracking.json on Drive before uploading new one
        try:
            drive_files = self.drive_manager.list_files_in_folder(folder_id=folder_id)
            for f in drive_files:
                if f["name"] == "tracking.json":
                    self.drive_manager.delete_file(f["id"])
        except Exception:
            pass

        file_id = self.drive_manager.upload_file(local_path, file_name="tracking.json")
        self.drive_manager.folder_id = original_folder

        if file_id:
            logger.info("✓ tracking.json uploaded to Drive")
        else:
            logger.warning("⚠️  Failed to upload tracking.json to Drive")

    # ------------------------------------------------------------------ #
    # Drive state                                                          #
    # ------------------------------------------------------------------ #

    def _get_drive_state(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """List raw and preprocessed folders on Drive."""
        assert self.drive_manager is not None
        drive_config = self.config["google_drive"]
        raw_folder_id = drive_config["raw_folder_id"]
        preprocessed_folder_id = drive_config.get("preprocessed_folder_id", "")

        try:
            all_raw = self.drive_manager.list_files_in_folder(folder_id=raw_folder_id)
        except Exception as e:
            logger.warning(f"⚠️  Could not list raw folder: {e}")
            all_raw = []

        # Only image files (skip any accidentally uploaded non-images)
        raw_files = [
            f for f in all_raw
            if Path(f["name"]).suffix.lower() in IMAGE_EXTENSIONS
        ]

        preprocessed_files: List[Dict[str, Any]] = []
        if preprocessed_folder_id and preprocessed_folder_id != "YOUR_DRIVE_FOLDER_ID":
            try:
                preprocessed_files = self.drive_manager.list_files_in_folder(
                    folder_id=preprocessed_folder_id
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not list preprocessed folder: {e}")

        return raw_files, preprocessed_files

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _process_new_images(
        self, raw_files: List[Dict[str, Any]], preprocessed_stems: Set[str]
    ) -> Dict[str, Any]:
        """Download, preprocess, and upload images not yet in preprocessed_folder."""
        assert self.drive_manager is not None
        drive_config = self.config["google_drive"]
        preprocessed_folder_id = drive_config.get("preprocessed_folder_id", "")

        raw_dir = Path(drive_config["download_dir_raw"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(self.config["cache"]["directory"]) / "temp_preprocessed"
        temp_dir.mkdir(parents=True, exist_ok=True)

        to_process = [
            f for f in raw_files
            if Path(f["name"]).stem not in preprocessed_stems
        ]

        logger.info(f"  {len(to_process)} image(s) need preprocessing")

        results = {"processed": 0, "uploaded": 0, "skipped": 0, "failed": 0}

        if not to_process:
            return results

        if not preprocessed_folder_id or preprocessed_folder_id == "YOUR_DRIVE_FOLDER_ID":
            logger.error("❌ preprocessed_folder_id not configured — cannot upload")
            return results

        for drive_file in tqdm(to_process, desc="Processing", unit="image"):
            file_name = drive_file["name"]
            file_id = drive_file["id"]
            stem = Path(file_name).stem
            local_raw = raw_dir / file_name
            output_name = f"{stem}.png"

            try:
                # Download raw
                success = self.drive_manager.download_file(file_id, local_raw)
                if not success:
                    logger.error(f"❌ Could not download {file_name}")
                    results["failed"] += 1
                    continue

                # Preprocess
                result_paths = preprocess_image(
                    input_path=local_raw,
                    cfg=PREPROCESS_CFG,
                    save_debug_to=temp_dir,
                    prefix=stem,
                )
                results["processed"] += 1

                # Upload preprocessed with same name (stem.png)
                final_path = result_paths["final"]
                original_folder = self.drive_manager.folder_id
                self.drive_manager.folder_id = preprocessed_folder_id
                uploaded_id = self.drive_manager.upload_file(
                    final_path, file_name=output_name
                )
                self.drive_manager.folder_id = original_folder

                if uploaded_id:
                    results["uploaded"] += 1
                    logger.info(f"  ✓ {file_name} → {output_name}")
                else:
                    logger.error(f"❌ Upload failed for {output_name}")
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"❌ Error processing {file_name}: {e}")
                results["failed"] += 1

        return results

    # ------------------------------------------------------------------ #
    # Roboflow upload                                                      #
    # ------------------------------------------------------------------ #

    def _upload_to_roboflow(self, preprocessed_drive_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload preprocessed files to Roboflow that haven't been uploaded yet."""
        assert self.drive_manager is not None
        assert self.tracker is not None
        results = {"uploaded": 0, "skipped": 0, "failed": 0}

        if not self.roboflow_client:
            logger.info("ℹ️  Roboflow not configured — skipping upload")
            return results

        preprocessed_folder_id = self.config["google_drive"].get("preprocessed_folder_id", "")

        to_upload = [
            f for f in preprocessed_drive_files
            if not self.tracker.is_roboflow_uploaded(f["name"])
        ]

        logger.info(f"  {len(to_upload)} file(s) to upload to Roboflow")

        if not to_upload:
            return results

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for drive_file in tqdm(to_upload, desc="Roboflow upload", unit="image"):
                file_name = drive_file["name"]
                file_id = drive_file["id"]
                local_path = tmp_dir / file_name

                try:
                    # Download from Drive preprocessed_folder
                    original_folder = self.drive_manager.folder_id
                    self.drive_manager.folder_id = preprocessed_folder_id
                    success = self.drive_manager.download_file(
                        file_id, local_path, skip_if_exists=False, check_hash=False
                    )
                    self.drive_manager.folder_id = original_folder

                    if not success:
                        logger.warning(f"⚠️  Could not download {file_name} for Roboflow upload")
                        results["failed"] += 1
                        continue

                    ok = self.roboflow_client.upload_image(local_path)
                    if ok:
                        self.tracker.mark_roboflow_uploaded(file_name)
                        results["uploaded"] += 1
                    else:
                        results["failed"] += 1

                except Exception as e:
                    logger.error(f"❌ Roboflow upload error for {file_name}: {e}")
                    results["failed"] += 1

        return results

    # ------------------------------------------------------------------ #
    # Annotation status                                                    #
    # ------------------------------------------------------------------ #

    def _check_annotation_status(self) -> Dict[str, Any]:
        if not self.roboflow_client:
            logger.info("ℹ️  Roboflow not configured")
            return {"status": "not_configured"}

        try:
            status = self.roboflow_client.get_annotation_status()
            total = status.get("total_images", 0)
            annotated = status.get("annotated", 0)
            pending = status.get("pending", 0)
            progress = status.get("annotation_progress", 0)

            if total > 0:
                logger.info(f"  Total: {total} | Annotated: {annotated} | Pending: {pending} | {progress}%")
                if pending == 0:
                    logger.info("  🎉 All images annotated!")
            else:
                logger.info(f"  {status.get('status', 'No images in project yet')}")

            return status
        except Exception as e:
            logger.error(f"❌ Error checking annotation status: {e}")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def _print_summary(self, results: Dict[str, Any]) -> None:
        logger.info("\n📊 Pipeline Results:")

        proc = results.get("preprocessing", {})
        logger.info(f"  Preprocessing: {proc.get('processed', 0)} processed, {proc.get('uploaded', 0)} uploaded to Drive, {proc.get('failed', 0)} failed")

        rf = results.get("roboflow_upload", {})
        logger.info(f"  Roboflow:      {rf.get('uploaded', 0)} uploaded, {rf.get('failed', 0)} failed")

        ann = results.get("annotation", {})
        if ann.get("total_images", 0) > 0:
            logger.info(f"  Annotation:    {ann.get('annotated', 0)}/{ann.get('total_images', 0)} ({ann.get('annotation_progress', 0)}%)")

        assert self.tracker is not None
        self.tracker.print_summary()
        logger.info("\n✅ Done!")


def main(validate: bool = False):  # validate kept for CLI back-compat, unused
    config = load_config()
    pipeline = DatasetPreparationPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    main()
