"""Synchronize raw and preprocessed images - verify tracking matches disk state."""

import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import load_config, get_pipeline_logger

logger = get_pipeline_logger(__name__)


def analyze_directories():
    """Analyze raw and preprocessed directories."""
    config = load_config()

    raw_dir = Path(config["google_drive"]["download_dir_raw"])
    preprocessed_dir = Path(config["google_drive"]["download_dir_preprocessed"])
    tracking_file = Path(config["tracking"]["file"])

    # Get actual files on disk
    raw_files = sorted(
        [
            f.name
            for f in raw_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        ]
    )

    preprocessed_files = sorted(
        [
            f.name
            for f in preprocessed_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".png"
        ]
    )

    # Load tracking
    tracking_data = {}
    if tracking_file.exists():
        with open(tracking_file, encoding="utf-8") as f:
            tracking_data = json.load(f)

    preprocessing_records = tracking_data.get("preprocessing", {})

    logger.info("=" * 70)
    logger.info("📊 SYNC ANALYSIS")
    logger.info("=" * 70)

    logger.info(f"\n📂 Raw images on disk: {len(raw_files)}")
    logger.info(f"📦 Preprocessed images on disk: {len(preprocessed_files)}")
    logger.info(
        f"📋 Preprocessing records in tracking.json: {len(preprocessing_records)}"
    )

    # Check for orphaned preprocessed files
    logger.info("\n🔍 Checking for orphaned preprocessed files...")
    orphaned = []
    for preprocessed_name in preprocessed_files:
        # Extract original filename from preprocessed name
        # Format: original_name__final.png or hash__final.png
        if "__final.png" in preprocessed_name:
            original_part = preprocessed_name.replace("__final.png", "")

            # Check if it's in tracking
            found_in_tracking = False
            for file_hash, record in preprocessing_records.items():
                if record.get("local_preprocessed_path"):
                    preprocessed_path = Path(record["local_preprocessed_path"])
                    if preprocessed_path.name == preprocessed_name:
                        found_in_tracking = True
                        break

            if not found_in_tracking:
                orphaned.append(preprocessed_name)

    if orphaned:
        logger.warning(
            f"⚠️  Found {len(orphaned)} orphaned preprocessed files (in disk but not in tracking)"
        )
        for f in orphaned[:10]:
            logger.warning(f"   - {f}")
        if len(orphaned) > 10:
            logger.warning(f"   ... and {len(orphaned) - 10} more")
    else:
        logger.info("✓ No orphaned files found")

    # Check for missing preprocessed files
    logger.info("\n🔍 Checking for missing preprocessed files...")
    missing = []
    for file_hash, record in preprocessing_records.items():
        if record.get("preprocessing_status") == "completed":
            if record.get("local_preprocessed_path"):
                preprocessed_path = Path(record["local_preprocessed_path"])
                if not preprocessed_path.exists():
                    missing.append(preprocessed_path.name)

    if missing:
        logger.warning(
            f"⚠️  Found {len(missing)} preprocessed files marked 'completed' but missing on disk"
        )
        for f in missing[:10]:
            logger.warning(f"   - {f}")
        if len(missing) > 10:
            logger.warning(f"   ... and {len(missing) - 10} more")
    else:
        logger.info("✓ All completed files exist on disk")

    # Check for extra raw files
    logger.info("\n🔍 Checking for extra raw files...")
    extra_raw = []
    tracked_raw = {}
    for file_hash, record in preprocessing_records.items():
        if record.get("local_raw_path"):
            raw_path = Path(record["local_raw_path"])
            tracked_raw[raw_path.name] = file_hash

    for raw_name in raw_files:
        if raw_name not in tracked_raw:
            extra_raw.append(raw_name)

    if extra_raw:
        logger.warning(f"⚠️  Found {len(extra_raw)} raw files not in tracking records")
        for f in extra_raw[:10]:
            logger.warning(f"   - {f}")
        if len(extra_raw) > 10:
            logger.warning(f"   ... and {len(extra_raw) - 10} more")
    else:
        logger.info("✓ All raw files are tracked")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Raw images: {len(raw_files)} on disk, {len(tracked_raw)} tracked")
    logger.info(
        f"Preprocessed: {len(preprocessed_files)} on disk, {len(preprocessing_records)} tracked"
    )
    logger.info(
        f"Missing: {len(missing)}, Orphaned: {len(orphaned)}, Extra raw: {len(extra_raw)}"
    )

    if missing or orphaned or extra_raw:
        logger.warning("\n⚠️  SYNC ISSUES DETECTED - Run cleanup/sync script to fix")
    else:
        logger.info("\n✅ Directories are synchronized!")

    return {
        "raw_on_disk": len(raw_files),
        "preprocessed_on_disk": len(preprocessed_files),
        "tracking_records": len(preprocessing_records),
        "missing": len(missing),
        "orphaned": len(orphaned),
        "extra_raw": len(extra_raw),
    }


if __name__ == "__main__":
    stats = analyze_directories()
