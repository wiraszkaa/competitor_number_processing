"""
Preprocessing tracking module.

This module handles tracking the preprocessing status of images across the team.
It maintains a shared tracking file to coordinate who has processed which images.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class PreprocessingRecord:
    """Record of an image's preprocessing status."""

    file_name: str
    file_hash: str  # SHA256 hash of the raw file
    drive_raw_id: Optional[str] = None  # File ID in raw folder
    drive_preprocessed_id: Optional[str] = None  # File ID in preprocessed folder
    preprocessing_status: str = "pending"  # pending, in_progress, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingRecord:
        """Create from dictionary."""
        return cls(**data)


class PreprocessingTracker:
    """Manages preprocessing tracking across the team."""

    def __init__(self, tracking_file: Path):
        """
        Initialize the tracker.

        Args:
            tracking_file: Path to the JSON tracking file
        """
        self.tracking_file = Path(tracking_file)
        self.records: Dict[str, PreprocessingRecord] = {}
        self.load()

    def load(self) -> None:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load preprocessing records
                if "preprocessing" in data:
                    for file_hash, record_data in data["preprocessing"].items():
                        self.records[file_hash] = PreprocessingRecord.from_dict(
                            record_data
                        )

                print(
                    f"✓ Loaded {len(self.records)} preprocessing records from {self.tracking_file}"
                )
            except Exception as e:
                print(f"Warning: Could not load tracking file: {e}")
                self.records = {}
        else:
            print(f"Creating new tracking file at {self.tracking_file}")
            self.records = {}

    def save(self) -> None:
        """Save tracking data to file."""
        # Load existing data to preserve other sections
        existing_data = {}
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                pass

        # Update preprocessing section
        existing_data["preprocessing"] = {
            file_hash: record.to_dict() for file_hash, record in self.records.items()
        }

        # Save back
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def add_or_update_record(
        self,
        file_path: Path,
        drive_raw_id: Optional[str] = None,
        drive_preprocessed_id: Optional[str] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> PreprocessingRecord:
        """
        Add or update a preprocessing record.

        Args:
            file_path: Path to the raw file
            drive_raw_id: Google Drive ID of raw file
            drive_preprocessed_id: Google Drive ID of preprocessed file
            status: New status
            error: Error message if failed

        Returns:
            The created or updated record
        """
        file_hash = self.calculate_file_hash(file_path)

        # Get or create record
        if file_hash in self.records:
            record = self.records[file_hash]
        else:
            record = PreprocessingRecord(
                file_name=file_path.name,
                file_hash=file_hash,
            )

        # Update fields
        if drive_raw_id is not None:
            record.drive_raw_id = drive_raw_id
        if drive_preprocessed_id is not None:
            record.drive_preprocessed_id = drive_preprocessed_id
        if status is not None:
            record.preprocessing_status = status
        if error is not None:
            record.error_message = error
            record.preprocessing_status = "failed"

        self.records[file_hash] = record
        return record

    def get_record_by_hash(self, file_hash: str) -> Optional[PreprocessingRecord]:
        """Get a record by file hash."""
        return self.records.get(file_hash)

    def get_record_by_file(self, file_path: Path) -> Optional[PreprocessingRecord]:
        """Get a record by file path."""
        file_hash = self.calculate_file_hash(file_path)
        return self.records.get(file_hash)

    def get_pending_files(self) -> List[PreprocessingRecord]:
        """Get all files that need preprocessing."""
        return [
            record
            for record in self.records.values()
            if record.preprocessing_status in ["pending", "failed"]
        ]

    def get_completed_files(self) -> List[PreprocessingRecord]:
        """Get all successfully preprocessed files."""
        return [
            record
            for record in self.records.values()
            if record.preprocessing_status == "completed"
        ]

    def get_in_progress_files(self) -> List[PreprocessingRecord]:
        """Get files currently being processed."""
        return [
            record
            for record in self.records.values()
            if record.preprocessing_status == "in_progress"
        ]

    def mark_as_in_progress(self, file_path: Path) -> PreprocessingRecord:
        """Mark a file as being processed."""
        return self.add_or_update_record(file_path, status="in_progress")

    def mark_as_completed(
        self,
        file_path: Path,
        preprocessed_path: Optional[Path] = None,
        drive_preprocessed_id: Optional[str] = None,
    ) -> PreprocessingRecord:
        """Mark a file as successfully preprocessed."""
        return self.add_or_update_record(
            file_path,
            status="completed",
            drive_preprocessed_id=drive_preprocessed_id,
        )

    def mark_as_failed(self, file_path: Path, error: str) -> PreprocessingRecord:
        """Mark a file as failed preprocessing."""
        return self.add_or_update_record(file_path, error=error)

    def print_summary(self) -> None:
        """Print a summary of preprocessing status."""
        pending = len(self.get_pending_files())
        in_progress = len(self.get_in_progress_files())
        completed = len(self.get_completed_files())
        failed = len(
            [r for r in self.records.values() if r.preprocessing_status == "failed"]
        )

        print("\n📊 Preprocessing Status Summary:")
        print(f"  • Pending:     {pending}")
        print(f"  • In Progress: {in_progress}")
        print(f"  • Completed:   {completed}")
        print(f"  • Failed:      {failed}")
        print(f"  • Total:       {len(self.records)}")
