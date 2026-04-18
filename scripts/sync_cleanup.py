"""Clean up and synchronize preprocessing - remove orphaned files and fix tracking."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import load_config, get_pipeline_logger

logger = get_pipeline_logger(__name__)


def cleanup_and_sync():
    """Remove orphaned preprocessed files and reset tracking for missing files."""
    config = load_config()

    raw_dir = Path(config["google_drive"]["download_dir_raw"])
    preprocessed_dir = Path(config["google_drive"]["download_dir_preprocessed"])
    tracking_file = Path(config["tracking"]["file"])

    # Get actual files on disk
    raw_files = {
        f.name
        for f in raw_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    }

    preprocessed_files = {
        f.name
        for f in preprocessed_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    }

    # Load tracking
    with open(tracking_file, encoding="utf-8") as f:
        tracking_data = json.load(f)

    preprocessing_records = tracking_data.get("preprocessing", {})

    logger.info("=" * 70)
    logger.info("🧹 CLEANUP & SYNC")
    logger.info("=" * 70)

    # Find orphaned preprocessed files (in disk but not in tracking)
    logger.info(f"\n🔍 Finding orphaned preprocessed files...")
    orphaned_files = []
    tracked_preprocessed = set()

    for file_hash, record in preprocessing_records.items():
        if record.get("local_preprocessed_path"):
            preprocessed_path = Path(record["local_preprocessed_path"])
            tracked_preprocessed.add(preprocessed_path.name)

    for preprocessed_name in preprocessed_files:
        if preprocessed_name not in tracked_preprocessed:
            orphaned_files.append(preprocessed_name)

    if orphaned_files:
        logger.warning(f"Found {len(orphaned_files)} orphaned files")
        logger.info(f"\n🗑️  Deleting orphaned files...")

        for orphaned_name in orphaned_files:
            orphaned_path = preprocessed_dir / orphaned_name
            try:
                orphaned_path.unlink()
                logger.info(f"  ✓ Deleted: {orphaned_name}")
            except Exception as e:
                logger.error(f"  ❌ Failed to delete {orphaned_name}: {e}")
    else:
        logger.info("✓ No orphaned files found")

    # Reset tracking for missing preprocessed files
    logger.info(f"\n🔍 Checking for missing preprocessed files...")
    reset_count = 0

    for file_hash, record in list(preprocessing_records.items()):
        if record.get("preprocessing_status") == "completed":
            if record.get("local_preprocessed_path"):
                preprocessed_path = Path(record["local_preprocessed_path"])
                if not preprocessed_path.exists():
                    logger.warning(
                        f"Missing file marked as completed: {preprocessed_path.name}"
                    )
                    record["preprocessing_status"] = "pending"
                    record["local_preprocessed_path"] = None
                    reset_count += 1
            else:
                # No preprocessed path recorded but marked completed - reset to pending
                logger.warning(f"Completed but no preprocessed path: {file_hash}")
                record["preprocessing_status"] = "pending"
                reset_count += 1

    if reset_count > 0:
        logger.info(
            f"\n🔄 Reset {reset_count} missing/invalid files to 'pending' status"
        )
    else:
        logger.info("✓ All completed files exist on disk")

    # Save updated tracking
    logger.info(f"\n💾 Saving updated tracking.json...")
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ CLEANUP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Deleted orphaned files: {len(orphaned_files)}")
    logger.info(f"Reset to pending: {reset_count}")

    # Re-analyze
    raw_count = len(
        {
            f.name
            for f in raw_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        }
    )
    preprocessed_count = len(
        {
            f.name
            for f in preprocessed_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".png"
        }
    )

    logger.info(f"\n📊 Final State:")
    logger.info(f"   Raw images: {raw_count}")
    logger.info(f"   Preprocessed images: {preprocessed_count}")

    # Count by status
    completed = sum(
        1
        for r in preprocessing_records.values()
        if r.get("preprocessing_status") == "completed"
    )
    pending = sum(
        1
        for r in preprocessing_records.values()
        if r.get("preprocessing_status") == "pending"
    )

    logger.info(f"   Tracking - Completed: {completed}, Pending: {pending}")

    if raw_count == preprocessed_count == completed:
        logger.info(
            f"\n✅ SYNCHRONIZED! All {raw_count} images are preprocessed and tracked"
        )
    else:
        logger.warning(
            f"\n⚠️  Still out of sync. Raw: {raw_count}, Preprocessed: {preprocessed_count}, Tracking: {completed}"
        )


if __name__ == "__main__":
    cleanup_and_sync()
