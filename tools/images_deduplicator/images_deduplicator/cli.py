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
    folder_id = config["google_drive"]["raw_folder_id"]  # Default to raw_folder_id

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--delete":
            dry_run = False
            print("\n⚠️  DELETE MODE ACTIVATED")
            print("Files will be permanently deleted from Google Drive!\n")
        elif arg == "--delete-confirm":
            dry_run = False
            confirm_delete = True
            print("\n⚠️  DELETE MODE WITH CONFIRMATION ACTIVATED")
            print("Files will be permanently deleted from Google Drive!\n")
        elif arg == "--folder-id":
            if i + 1 < len(sys.argv):
                folder_id = sys.argv[i + 1]
                i += 1  # Skip next arg since we consumed it
            else:
                print("Error: --folder-id requires a folder ID argument")
                return 1
        elif arg == "--help":
            print(
                "Usage: uv run deduplicator [--delete|--delete-confirm|--folder-id <id>|--help]"
            )
            print("\nModes:")
            print("  (no args)         : Dry run - scan and report only (safe)")
            print("  --delete          : Enable deletion but require confirmation")
            print("  --delete-confirm  : Delete duplicates immediately (dangerous!)")
            print(
                "  --folder-id <id>  : Specify Google Drive folder ID (default: raw_folder_id)"
            )
            print("  --help            : Show this help message")
            return 0
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Use --help for usage information")
            return 1
        i += 1

    print("=" * 80)
    print("GOOGLE DRIVE IMAGE DEDUPLICATOR")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (safe)' if dry_run else 'DELETE MODE (destructive)'}")
    print(f"Folder ID: {folder_id}")
    print("=" * 80)
    print()

    # Initialize DriveManager
    print("Authenticating with Google Drive...")
    drive_manager = DriveManager(
        credentials_path=config["google_drive"]["credentials_path"],
        folder_id=folder_id,
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
