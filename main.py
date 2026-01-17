"""
Main processing pipeline for competitor number extraction.

This script implements a collaborative preprocessing workflow:
1. Download raw images that haven't been preprocessed yet
2. Preprocess images locally
3. Upload preprocessed images to shared Drive folder
4. Download preprocessed images from teammates
5. Track all processing status in tracking.json
6. Ready for further processing steps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from competitor_number_processing.preprocess import PreprocessConfig, preprocess_image
from competitor_number_processing.tracking import PreprocessingTracker
from competitor_number_processing.detector import PersonDetector, DetectionConfig

# Import drive_manager - need to add to path first
sys.path.insert(0, str(Path(__file__).parent / "tools" / "drive_manager"))
from drive_manager.manager import DriveManager


def load_config() -> Dict[str, Any]:
    """Load configuration from secrets/config.json."""
    config_path = Path(__file__).parent / "secrets" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please create it based on secrets/config.example.json"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_preprocessed(image_path: Path, output_dir: Path) -> bool:
    """Check if an image has already been preprocessed locally."""
    stem = image_path.stem
    final_file = output_dir / f"{stem}__final.png"
    return final_file.exists()


def verify_and_update_preprocessing_status(
    tracker: PreprocessingTracker,
    manager: DriveManager,
    preprocessed_folder_id: str,
) -> int:
    """
    Verify that completed preprocessing records still have files on Drive.
    If preprocessed files are missing from Drive, reset status to pending.

    Returns:
        Number of records reset to pending
    """
    print("🔍 Verifying preprocessing status against Drive...")

    # Get all completed records
    completed_records = tracker.get_completed_files()

    if not completed_records:
        print("   No completed records to verify")
        return 0

    print(f"   Checking {len(completed_records)} completed records...")

    # Get list of files in preprocessed folder
    try:
        drive_files = manager.list_files_in_folder(folder_id=preprocessed_folder_id)
        drive_file_ids = {f["id"] for f in drive_files}
    except Exception as e:
        print(f"   ⚠️  Could not access Drive folder: {e}")
        return 0

    reset_count = 0
    for record in completed_records:
        if (
            record.drive_preprocessed_id
            and record.drive_preprocessed_id not in drive_file_ids
        ):
            # File was removed from Drive - reset to pending
            print(f"   ⚠️  {record.file_name}: preprocessed file missing from Drive")

            # Update the record directly in tracker.records
            record.preprocessing_status = "pending"
            record.drive_preprocessed_id = None
            record.processed_by = None
            record.processed_timestamp = None
            tracker.records[record.file_hash] = record

            reset_count += 1

    if reset_count > 0:
        tracker.save()
        print(f"   ✅ Reset {reset_count} record(s) to pending for reprocessing")
    else:
        print(f"   ✅ All completed records verified")

    return reset_count


def download_raw_images_not_yet_preprocessed(
    config: Dict[str, Any], tracker: PreprocessingTracker
) -> Tuple[List[Path], DriveManager]:
    """
    Download raw images from Drive that haven't been preprocessed yet.

    Returns:
        Tuple of (list of downloaded file paths, drive manager instance)
    """
    drive_config = config["google_drive"]
    credentials_path = drive_config["credentials_path"]
    raw_folder_id = drive_config["raw_folder_id"]
    download_dir = Path(drive_config["download_dir_raw"])

    print(f"📥 Initializing Drive Manager...")
    manager = DriveManager(credentials_path)

    print(f"📂 Checking raw files in Drive folder: {raw_folder_id}")
    download_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the raw folder
    drive_files = manager.list_files_in_folder(folder_id=raw_folder_id)

    if not drive_files:
        print("⚠️  No files found in raw folder")
        return [], manager

    print(f"Found {len(drive_files)} files in raw folder")

    # Filter files that need to be processed
    files_to_download = []
    for drive_file in drive_files:
        file_id = drive_file["id"]
        file_name = drive_file["name"]

        # Skip preprocessed files in raw folder (they shouldn't be there)
        if "__final" in file_name or any(
            marker in file_name
            for marker in ["__resized", "__brightness", "__contrast", "__denoise"]
        ):
            print(f"  ⚠️  {file_name} - preprocessed file in raw folder, skipping")
            # If it's in tracker as pending, mark it as completed since it's already processed
            local_path = download_dir / file_name
            if local_path.exists():
                record = tracker.get_record_by_file(local_path)
                if record and record.preprocessing_status == "pending":
                    # Find the original raw file name (remove __final.png suffix)
                    original_name = file_name.replace("__final.png", "").replace(
                        "__final.jpg", ""
                    )
                    print(f"      Marking as completed (already preprocessed)")
                    tracker.add_or_update_record(
                        local_path,
                        drive_raw_id=file_id,
                        status="completed",
                        local_preprocessed_path=local_path,
                        drive_preprocessed_id=file_id,
                    )
            continue

        # Check if already processed in tracking
        # We'll download and check hash to be sure
        local_path = download_dir / file_name

        # Download if not exists or always verify
        should_download = True
        if local_path.exists():
            # Check if this file is already marked as completed
            record = tracker.get_record_by_file(local_path)
            if record and record.preprocessing_status == "completed":
                print(f"  ⏭️  {file_name} - already completed, skipping download")
                should_download = False

        if should_download:
            files_to_download.append((file_id, file_name, local_path))

    print(f"\n📥 Need to download {len(files_to_download)} raw files...")

    downloaded_paths = []
    for file_id, file_name, local_path in files_to_download:
        # Download the file
        success = manager.download_file(file_id, local_path)
        if success:
            print(f"  ✓ Downloaded: {file_name}")
            # Add to tracker
            tracker.add_or_update_record(
                local_path, drive_raw_id=file_id, status="pending"
            )
            downloaded_paths.append(local_path)
        else:
            print(f"  ❌ Failed to download: {file_name}")

    # Also check for files that already exist locally but might be pending
    print(f"\n📂 Checking for existing raw files that need processing...")
    for drive_file in drive_files:
        file_id = drive_file["id"]
        file_name = drive_file["name"]
        local_path = download_dir / file_name

        if local_path.exists():
            record = tracker.get_record_by_file(local_path)
            # Add to tracker if not there or if pending/failed
            if not record:
                print(f"  📝 Adding to tracker: {file_name}")
                tracker.add_or_update_record(
                    local_path, drive_raw_id=file_id, status="pending"
                )
            elif record.preprocessing_status in ["pending", "failed"]:
                # Ensure drive_raw_id is set
                if not record.drive_raw_id:
                    record.drive_raw_id = file_id
                    tracker.records[record.file_hash] = record

    tracker.save()
    return downloaded_paths, manager


def download_preprocessed_images_from_team(
    config: Dict[str, Any], manager: DriveManager, tracker: PreprocessingTracker
) -> List[Path]:
    """
    Download preprocessed images uploaded by teammates.

    Returns:
        List of downloaded preprocessed file paths
    """
    drive_config = config["google_drive"]
    preprocessed_folder_id = drive_config["preprocessed_folder_id"]
    download_dir = Path(drive_config["download_dir_preprocessed"])

    if preprocessed_folder_id == "PLACEHOLDER_FOR_PREPROCESSED_FOLDER":
        print("\n⚠️  Preprocessed folder ID not configured. Skipping team downloads.")
        print("   Please update 'preprocessed_folder_id' in config.json")
        return []

    print(f"\n📥 Downloading preprocessed images from team...")
    print(f"📂 Preprocessed folder: {preprocessed_folder_id}")
    download_dir.mkdir(parents=True, exist_ok=True)

    # List all files in preprocessed folder
    drive_files = manager.list_files_in_folder(folder_id=preprocessed_folder_id)

    if not drive_files:
        print("   No preprocessed files from team yet")
        return []

    print(f"   Found {len(drive_files)} preprocessed files in Drive")

    downloaded_paths = []
    for drive_file in drive_files:
        file_id = drive_file["id"]
        file_name = drive_file["name"]
        local_path = download_dir / file_name

        # Check if already downloaded
        if local_path.exists():
            print(f"  ⏭️  {file_name} - already exists locally")
            continue

        # Download
        success = manager.download_file(file_id, local_path)
        if success:
            print(f"  ✓ Downloaded: {file_name}")
            downloaded_paths.append(local_path)
        else:
            print(f"  ❌ Failed to download: {file_name}")

    return downloaded_paths


def preprocess_and_upload_images(
    image_files: List[Path],
    output_dir: Path,
    preprocess_config: PreprocessConfig,
    manager: DriveManager,
    tracker: PreprocessingTracker,
    config: Dict[str, Any],
) -> Dict[str, List[Path]]:
    """
    Preprocess images locally and upload to Drive.
    - Uploads non-grass version to Drive (for number detection)
    - Keeps grass-enhanced version locally (for person detection)

    Returns:
        Dict with 'processed', 'uploaded', 'skipped', 'failed', 'grass_enhanced' keys.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    drive_config = config["google_drive"]
    preprocessed_folder_id = drive_config["preprocessed_folder_id"]

    processed = []
    uploaded = []
    skipped = []
    failed = []
    grass_enhanced = []

    # Filter only image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]

    # Get pending files from tracker
    pending_records = tracker.get_pending_files()
    pending_hashes = {record.file_hash for record in pending_records}

    # Filter to only process pending files
    files_to_process = []
    for img_path in image_files:
        record = tracker.get_record_by_file(img_path)
        if record and record.file_hash in pending_hashes:
            files_to_process.append(img_path)
        elif not record:
            # New file not in tracker - add it
            tracker.add_or_update_record(img_path, status="pending")
            files_to_process.append(img_path)

    total = len(files_to_process)
    print(f"\n🖼️  Processing {total} pending image(s)")

    for idx, image_path in enumerate(files_to_process, 1):
        # Skip debug/processed files
        if any(
            marker in image_path.stem
            for marker in [
                "__resized",
                "__brightness",
                "__contrast",
                "__denoise",
                "__final",
            ]
        ):
            continue

        print(f"\n[{idx}/{total}] Processing: {image_path.name}")

        # Check if already completed
        record = tracker.get_record_by_file(image_path)
        if record and record.preprocessing_status == "completed":
            print(f"  ⏭️  Already preprocessed, skipping")
            skipped.append(image_path)
            continue

        try:
            # Mark as in progress
            tracker.mark_as_in_progress(image_path)
            tracker.save()

            print(f"  ⚙️  Preprocessing...")
            result_paths = preprocess_image(
                input_path=image_path,
                cfg=preprocess_config,
                save_debug_to=output_dir,
                prefix=image_path.stem,
            )

            final_path = result_paths["final"]
            print(f"  ✓ Preprocessed (non-grass): {final_path.name}")
            processed.append(image_path)

            # Track grass-enhanced version if created
            if "grass_enhanced" in result_paths:
                grass_path = result_paths["grass_enhanced"]
                print(f"  ✓ Grass-enhanced (local): {grass_path.name}")
                grass_enhanced.append(grass_path)

            # Upload NON-GRASS version to Drive (for number detection)
            if preprocessed_folder_id != "PLACEHOLDER_FOR_PREPROCESSED_FOLDER":
                print(f"  ☁️  Uploading non-grass version to Drive...")
                # Temporarily set folder for upload
                original_folder = manager.folder_id
                manager.folder_id = preprocessed_folder_id

                file_id = manager.upload_file(final_path)

                manager.folder_id = original_folder  # Restore

                if file_id:
                    print(f"  ✓ Uploaded to Drive (ID: {file_id})")
                    uploaded.append(final_path)

                    # Mark as completed in tracker
                    tracker.mark_as_completed(
                        image_path, final_path, drive_preprocessed_id=file_id
                    )
                else:
                    print(f"  ⚠️  Upload failed, but file is preprocessed locally")
                    tracker.mark_as_completed(image_path, final_path)
            else:
                # Just mark as completed locally
                tracker.mark_as_completed(image_path, final_path)
                print(f"  ℹ️  Preprocessed folder not configured - saved locally only")

            tracker.save()

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed.append(image_path)
            tracker.mark_as_failed(image_path, str(e))
            tracker.save()

    return {
        "processed": processed,
        "uploaded": uploaded,
        "skipped": skipped,
        "failed": failed,
        "grass_enhanced": grass_enhanced,
    }


def main():
    """Main entry point for collaborative preprocessing pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Competitor Number Preprocessing and Detection Pipeline"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing steps and only run detection on existing grass-enhanced images",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading raw images from Drive",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading preprocessed images to Drive",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Collaborative Competitor Number Preprocessing Pipeline")
    print("=" * 70)

    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_config()

    # Initialize tracking
    tracking_file = Path(config["tracking"]["file"])

    # Get processor ID from config
    processor_id = config.get("preprocessing", {}).get("processor_id", "default_user")
    print(f"   Processor ID: {processor_id}")

    tracker = PreprocessingTracker(tracking_file, processor_id=processor_id)
    tracker.print_summary()

    # Initialize Drive manager
    credentials_path = config["google_drive"]["credentials_path"]
    drive_manager = DriveManager(credentials_path)

    # Verify preprocessing status against Drive (check if files were removed)
    if not args.skip_preprocessing:
        preprocessed_folder_id = config["google_drive"]["preprocessed_folder_id"]
        if preprocessed_folder_id != "PLACEHOLDER_FOR_PREPROCESSED_FOLDER":
            reset_count = verify_and_update_preprocessing_status(
                tracker, drive_manager, preprocessed_folder_id
            )
            if reset_count > 0:
                print(f"   🔄 {reset_count} image(s) will be reprocessed")
                tracker.print_summary()

    # Step 1: Download raw images that need preprocessing
    if not args.skip_preprocessing and not args.skip_download:
        print("\n" + "=" * 70)
        print("Step 1: Download Raw Images (Not Yet Preprocessed)")
        print("=" * 70)

        raw_files, drive_manager = download_raw_images_not_yet_preprocessed(
            config, tracker
        )

        if raw_files:
            print(f"\n✓ Downloaded {len(raw_files)} raw files that need preprocessing")
        else:
            print("\n✓ All raw files already downloaded or processed")
    else:
        if args.skip_preprocessing:
            print("\n⏭️  Skipping download (preprocessing disabled)")
        else:
            print("\n⏭️  Skipping download (--skip-download flag)")

    # Step 2: Preprocess images locally and upload
    results = {
        "processed": [],
        "uploaded": [],
        "skipped": [],
        "failed": [],
        "grass_enhanced": [],
    }

    if not args.skip_preprocessing:
        print("\n" + "=" * 70)
        print("Step 2: Preprocess Images & Upload to Drive")
        print("=" * 70)

        cache_dir = Path(config["cache"]["directory"])
        output_dir = cache_dir / "processed_local"
        preprocess_config = PreprocessConfig(
            max_long_edge=1280,
            autocontrast=True,
            gamma=1.0,
            brightness=1.0,
            median_filter_size=3,
            gaussian_blur_radius=0.0,
            contrast=1.0,
            enable_grass_preprocessing=True,  # Enable grass-aware preprocessing
            grass_edge_enhancement=True,  # Add strong edges between grass and people
            grass_sharpening=True,  # Sharpen non-grass regions
        )

        # Get all raw files to check
        raw_dir = Path(config["google_drive"]["download_dir_raw"])
        all_raw_files = (
            [
                f
                for f in raw_dir.iterdir()
                if f.is_file()
                and f.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            ]
            if raw_dir.exists()
            else []
        )

        results = preprocess_and_upload_images(
            all_raw_files, output_dir, preprocess_config, drive_manager, tracker, config
        )
    else:
        print("\n⏭️  Skipping preprocessing (--skip-preprocessing flag)")
        cache_dir = Path(config["cache"]["directory"])
        output_dir = cache_dir / "processed_local"
        # Find existing grass-enhanced images
        existing_grass = list(output_dir.glob("*__grass_enhanced.png"))
        results["grass_enhanced"] = existing_grass
        print(f"   Found {len(existing_grass)} existing grass-enhanced images")

    # Step 3: Detect people on grass-enhanced images
    print("\n" + "=" * 70)
    print("Step 3: Detect People on Grass-Enhanced Images")
    print("=" * 70)

    if results["grass_enhanced"]:
        print(
            f"\n🔍 Detecting people in {len(results['grass_enhanced'])} grass-enhanced images..."
        )

        # Initialize person detector with more sensitive settings
        detection_config = DetectionConfig(
            scale=1.03,  # Smaller scale for more detection attempts
            min_neighbors=0,  # More lenient grouping
            min_size=(30, 60),  # Smaller minimum size to catch distant people
            threshold=-1.0,  # More lenient threshold
            use_contour_detection=True,
            min_contour_area=1500,  # Lower area threshold
        )
        detector = PersonDetector(detection_config)

        # Create output directory for detection results
        detections_dir = output_dir / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)

        total_detections = 0
        for idx, grass_img_path in enumerate(results["grass_enhanced"], 1):
            print(
                f"\n[{idx}/{len(results['grass_enhanced'])}] Processing: {grass_img_path.name}"
            )

            # Get corresponding final image path (non-grass version)
            final_img_path = output_dir / grass_img_path.name.replace(
                "__grass_enhanced.png", "__final.png"
            )

            if not final_img_path.exists():
                print(f"  ⚠️  Final image not found: {final_img_path.name}")
                continue

            try:
                # Detect people on grass-enhanced image
                detections = detector.detect_from_file(grass_img_path)
                print(f"  ✓ Found {len(detections)} person(s)")
                total_detections += len(detections)

                if detections:
                    # Visualize on FINAL image (non-grass version for better visualization)
                    output_path = (
                        detections_dir / f"{grass_img_path.stem}__detected.png"
                    )
                    detector.save_visualized_detections(
                        final_img_path,  # Use final image instead of grass-enhanced
                        output_path,
                        detections=detections,
                    )
                    print(
                        f"  💾 Saved visualization on final image: {output_path.name}"
                    )

                    # Print detection details
                    for i, person in enumerate(detections, 1):
                        print(
                            f"     Person {i}: bbox=({person.x}, {person.y}, {person.width}, {person.height}), confidence={person.confidence:.2f}"
                        )

            except Exception as e:
                print(f"  ❌ Detection failed: {e}")

        print(f"\n✓ Total people detected: {total_detections}")
        print(f"📁 Detection visualizations saved to: {detections_dir}")
    else:
        print("\n⏭️  No grass-enhanced images to process")

    # Step 4: Download preprocessed images from teammates
    if not args.skip_preprocessing:
        print("\n" + "=" * 70)
        print("Step 4: Download Preprocessed Images from Team")
        print("=" * 70)

        team_preprocessed = download_preprocessed_images_from_team(
            config, drive_manager, tracker
        )

        if team_preprocessed:
            print(
                f"\n✓ Downloaded {len(team_preprocessed)} preprocessed files from team"
            )
        else:
            print("\n✓ No new preprocessed files from team")
    else:
        team_preprocessed = []
        print("\n⏭️  Skipping team download (preprocessing disabled)")

    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)

    if not args.skip_preprocessing:
        print(f"\n📊 This Session Summary:")
        print(f"  • Newly preprocessed:  {len(results['processed'])}")
        print(f"  • Uploaded to Drive:   {len(results['uploaded'])}")
        print(f"  • Grass-enhanced:      {len(results['grass_enhanced'])}")
        print(f"  • Already done:        {len(results['skipped'])}")
        print(f"  • Failed:              {len(results['failed'])}")
        print(f"  • From team:           {len(team_preprocessed)}")
    else:
        print(f"\n📊 Detection Summary:")
        print(f"  • Grass-enhanced images: {len(results['grass_enhanced'])}")
        print(
            f"  • People detected:       {total_detections if results['grass_enhanced'] else 0}"
        )

    # Updated tracker summary
    tracker.load()  # Reload to get latest
    tracker.print_summary()

    if results["failed"]:
        print(f"\n❌ Failed files:")
        for f in results["failed"]:
            print(f"  - {f.name}")

    print(f"\n📁 Local preprocessed files: {output_dir}")

    preprocessed_dir = Path(config["google_drive"]["download_dir_preprocessed"])
    if preprocessed_dir.exists() and any(preprocessed_dir.iterdir()):
        print(f"📁 Team preprocessed files: {preprocessed_dir}")

    print(f"\n🎯 Ready for next steps!")
    print("\nNext steps you can implement:")
    print("  1. Competitor number detection on preprocessed images")
    print("  2. OCR and number extraction")
    print("  3. Result validation and export")


if __name__ == "__main__":
    main()
