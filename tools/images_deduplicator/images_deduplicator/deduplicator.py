"""
Image Deduplicator - Remove duplicate images from Google Drive based on MD5 hash
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from drive_manager import DriveManager


class ImageDeduplicator:
    """Remove duplicate images from Google Drive using MD5 hash comparison"""

    def __init__(self, drive_manager: DriveManager, dry_run: bool = True):
        """
        Initialize deduplicator

        Args:
            drive_manager: Configured DriveManager instance
            dry_run: If True, only report duplicates without deleting (default: True)
        """
        self.drive_manager = drive_manager
        self.dry_run = dry_run
        self.duplicates_found: Dict[str, List[Dict[str, Any]]] = {}
        self.files_to_delete: List[str] = []

    def find_duplicates(
        self, folder_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find duplicate files in a Google Drive folder based on MD5 hash

        Args:
            folder_id: Drive folder ID to scan (default: drive_manager's folder_id)

        Returns:
            Dictionary mapping MD5 hash to list of file metadata for duplicates
        """
        print("Scanning Google Drive for files...")
        files = self.drive_manager.list_files_in_folder(folder_id)

        if not files:
            print("No files found in the folder")
            return {}

        print(f"Found {len(files)} files. Analyzing for duplicates...")

        # Group files by MD5 hash
        hash_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for file in files:
            # Skip files without MD5 (folders, Google Docs, etc.)
            md5 = file.get("md5Checksum")
            if not md5:
                continue

            hash_groups[md5].append(file)

        # Filter to only duplicates (hash appears more than once)
        duplicates = {
            md5: files_list
            for md5, files_list in hash_groups.items()
            if len(files_list) > 1
        }

        self.duplicates_found = duplicates

        # Print summary
        total_duplicate_files = sum(len(files) for files in duplicates.values())
        total_groups = len(duplicates)

        print(f"\n{'='*60}")
        print(f"DUPLICATE DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Total files scanned: {len(files)}")
        print(f"Duplicate groups found: {total_groups}")
        print(f"Total duplicate files: {total_duplicate_files}")
        print(f"Files to keep: {total_groups}")
        print(f"Files that can be deleted: {total_duplicate_files - total_groups}")
        print(f"{'='*60}\n")

        return duplicates

    def show_duplicates_report(self, limit: Optional[int] = None) -> None:
        """
        Print detailed report of duplicate files

        Args:
            limit: Maximum number of duplicate groups to show (default: all)
        """
        if not self.duplicates_found:
            print("No duplicates found. Run find_duplicates() first.")
            return

        print("\nDETAILED DUPLICATES REPORT")
        print("=" * 80)

        groups_shown = 0
        for md5_hash, files in self.duplicates_found.items():
            if limit and groups_shown >= limit:
                remaining = len(self.duplicates_found) - limit
                print(f"\n... and {remaining} more duplicate groups")
                break

            groups_shown += 1
            print(f"\nGroup {groups_shown} - MD5: {md5_hash[:16]}...")
            print(f"Files ({len(files)} duplicates):")

            # Sort by modification time (oldest first = will be kept)
            sorted_files = sorted(
                files, key=lambda f: f.get("modifiedTime", ""), reverse=False
            )

            for idx, file in enumerate(sorted_files):
                size_mb = int(file.get("size", 0)) / (1024 * 1024)
                modified = file.get("modifiedTime", "unknown")[:10]
                status = "KEEP" if idx == 0 else "DELETE"

                print(
                    f"  [{status:6}] {file['name']:40} | {size_mb:6.2f} MB | {modified} | ID: {file['id']}"
                )

        print("=" * 80)
        print(f"\nStrategy: Keep the OLDEST file in each group (earliest modifiedTime)")
        print("=" * 80)

    def prepare_deletion_list(
        self, keep_strategy: str = "oldest"
    ) -> List[Tuple[str, str]]:
        """
        Prepare list of files to delete based on keep strategy

        Args:
            keep_strategy: Which file to keep ('oldest', 'newest', or 'first')
                - 'oldest': Keep file with earliest modifiedTime
                - 'newest': Keep file with latest modifiedTime
                - 'first': Keep first file in the list (arbitrary)

        Returns:
            List of tuples (file_id, file_name) to delete
        """
        if not self.duplicates_found:
            print("No duplicates found. Run find_duplicates() first.")
            return []

        files_to_delete: List[Tuple[str, str]] = []

        for _, files in self.duplicates_found.items():
            if keep_strategy == "oldest":
                # Sort by modification time (oldest first)
                sorted_files = sorted(
                    files, key=lambda f: f.get("modifiedTime", ""), reverse=False
                )
            elif keep_strategy == "newest":
                # Sort by modification time (newest first)
                sorted_files = sorted(
                    files, key=lambda f: f.get("modifiedTime", ""), reverse=True
                )
            else:  # 'first'
                sorted_files = files

            # Keep first file, delete the rest
            for file in sorted_files[1:]:
                files_to_delete.append((file["id"], file["name"]))

        self.files_to_delete = [fid for fid, _ in files_to_delete]

        print(f"\nPrepared {len(files_to_delete)} files for deletion")
        print(f"Strategy: Keep {keep_strategy} file in each duplicate group\n")

        return files_to_delete

    def delete_duplicates(
        self, confirm: bool = False, keep_strategy: str = "oldest"
    ) -> Tuple[int, int]:
        """
        Delete duplicate files from Google Drive

        Args:
            confirm: Must be True to actually delete files (safety check)
            keep_strategy: Which file to keep ('oldest', 'newest', or 'first')

        Returns:
            Tuple of (successful_deletions, failed_deletions)
        """
        if not self.duplicates_found:
            print("No duplicates found. Run find_duplicates() first.")
            return (0, 0)

        # Prepare deletion list
        files_to_delete = self.prepare_deletion_list(keep_strategy)

        if not files_to_delete:
            print("No files to delete")
            return (0, 0)

        # Safety checks
        if self.dry_run:
            print("\n" + "!" * 80)
            print("DRY RUN MODE: No files will actually be deleted")
            print("!" * 80)
            print("\nFiles that would be deleted:")
            for file_id, file_name in files_to_delete[:10]:
                print(f"  - {file_name} (ID: {file_id})")
            if len(files_to_delete) > 10:
                print(f"  ... and {len(files_to_delete) - 10} more files")
            print("\nTo actually delete files:")
            print("  1. Create ImageDeduplicator with dry_run=False")
            print("  2. Call delete_duplicates(confirm=True)")
            return (0, 0)

        if not confirm:
            print("\n" + "!" * 80)
            print("CONFIRMATION REQUIRED")
            print("!" * 80)
            print(
                f"\nYou are about to delete {len(files_to_delete)} files from Google Drive!"
            )
            print("This action CANNOT be undone.")
            print("\nTo proceed, call delete_duplicates(confirm=True)")
            return (0, 0)

        # Actual deletion
        print("\n" + "=" * 80)
        print(f"DELETING {len(files_to_delete)} DUPLICATE FILES")
        print("=" * 80)

        successful = 0
        failed = 0

        for idx, (file_id, file_name) in enumerate(files_to_delete, 1):
            print(f"[{idx}/{len(files_to_delete)}] Deleting: {file_name}", end="...")

            if self.drive_manager.delete_file(file_id):
                successful += 1
                print(" ✓")
            else:
                failed += 1
                print(" ✗ FAILED")

        print("=" * 80)
        print(f"DELETION COMPLETE")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print("=" * 80)

        return (successful, failed)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about duplicates found

        Returns:
            Dictionary with statistics
        """
        if not self.duplicates_found:
            return {
                "total_files_scanned": 0,
                "duplicate_groups": 0,
                "total_duplicates": 0,
                "files_to_keep": 0,
                "files_to_delete": 0,
                "space_wasted_mb": 0,
            }

        total_duplicates = sum(len(files) for files in self.duplicates_found.values())
        duplicate_groups = len(self.duplicates_found)

        # Calculate wasted space
        space_wasted = 0
        for files in self.duplicates_found.values():
            if files:
                # All duplicates have same size, multiply by (count - 1)
                file_size = int(files[0].get("size", 0))
                space_wasted += file_size * (len(files) - 1)

        return {
            "duplicate_groups": duplicate_groups,
            "total_duplicates": total_duplicates,
            "files_to_keep": duplicate_groups,
            "files_to_delete": total_duplicates - duplicate_groups,
            "space_wasted_mb": space_wasted / (1024 * 1024),
        }


def main():
    """Example usage of ImageDeduplicator"""
    import json
    from pathlib import Path

    # Load config
    config_path = Path("secrets/config.json")
    if not config_path.exists():
        print("Error: secrets/config.json not found")
        return

    with open(config_path) as f:
        config = json.load(f)

    # Initialize DriveManager
    drive_manager = DriveManager(
        credentials_path=config["google_drive"]["credentials_path"],
        folder_id=config["google_drive"]["folder_id"],
    )

    # Create deduplicator in dry-run mode (safe)
    deduplicator = ImageDeduplicator(drive_manager, dry_run=True)

    # Find duplicates
    duplicates = deduplicator.find_duplicates()

    if duplicates:
        # Show detailed report (limit to first 5 groups)
        deduplicator.show_duplicates_report(limit=5)

        # Show statistics
        stats = deduplicator.get_statistics()
        print(f"\nSpace wasted by duplicates: {stats['space_wasted_mb']:.2f} MB")

        # Prepare deletion list (but don't delete - dry run mode)
        deduplicator.delete_duplicates(confirm=False, keep_strategy="oldest")
    else:
        print("No duplicates found!")


if __name__ == "__main__":
    main()
