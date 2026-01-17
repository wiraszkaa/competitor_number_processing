"""
CLI entry point for image deduplicator
"""

import json
import sys
from pathlib import Path

from drive_manager import DriveManager
from images_deduplicator import ImageDeduplicator


def main():
    """Main deduplication workflow"""
    # Load config
    config_path = Path("secrets/config.json")
    if not config_path.exists():
        print("Error: secrets/config.json not found")
        print("Please copy secrets/config.example.json and configure it")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    # Parse command line args for mode
    dry_run = True
    confirm_delete = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "--delete":
            dry_run = False
            print("\n⚠️  DELETE MODE ACTIVATED")
            print("Files will be permanently deleted from Google Drive!\n")
        elif sys.argv[1] == "--delete-confirm":
            dry_run = False
            confirm_delete = True
            print("\n⚠️  DELETE MODE WITH CONFIRMATION ACTIVATED")
            print("Files will be permanently deleted from Google Drive!\n")
        elif sys.argv[1] == "--help":
            print("Usage: uv run deduplicator [--delete|--delete-confirm|--help]")
            print("\nModes:")
            print("  (no args)         : Dry run - scan and report only (safe)")
            print("  --delete          : Enable deletion but require confirmation")
            print("  --delete-confirm  : Delete duplicates immediately (dangerous!)")
            print("  --help            : Show this help message")
            return 0

    print("=" * 80)
    print("GOOGLE DRIVE IMAGE DEDUPLICATOR")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (safe)' if dry_run else 'DELETE MODE (destructive)'}")
    print(f"Folder ID: {config['google_drive']['folder_id']}")
    print("=" * 80)
    print()

    # Initialize DriveManager
    print("Authenticating with Google Drive...")
    drive_manager = DriveManager(
        credentials_path=config["google_drive"]["credentials_path"],
        folder_id=config["google_drive"]["folder_id"],
    )

    # Create deduplicator
    deduplicator = ImageDeduplicator(drive_manager, dry_run=dry_run)

    # Find duplicates
    duplicates = deduplicator.find_duplicates()

    if not duplicates:
        print("\n✓ No duplicates found! Your Drive folder is clean.")
        return 0

    # Show detailed report
    deduplicator.show_duplicates_report(limit=10)

    # Show statistics
    stats = deduplicator.get_statistics()
    print(f"\n💾 Space wasted by duplicates: {stats['space_wasted_mb']:.2f} MB")
    print(f"📊 {stats['files_to_delete']} files can be safely deleted\n")

    # Handle deletion
    if dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No files were deleted")
        print("=" * 80)
        print("\nTo delete duplicates, run:")
        print("  uv run deduplicator --delete")
        print("\nOr to delete without confirmation:")
        print("  uv run deduplicator --delete-confirm")
    else:
        # Attempt deletion
        successful, failed = deduplicator.delete_duplicates(
            confirm=confirm_delete, keep_strategy="oldest"
        )

        if successful > 0 or failed > 0:
            print(f"\n✓ Deleted {successful} duplicate files")
            if failed > 0:
                print(f"✗ Failed to delete {failed} files")
        else:
            print("\nNo files were deleted (confirmation required)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
