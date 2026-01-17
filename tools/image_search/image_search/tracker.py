"""
Image tracker module - manages JSON file for tracking downloaded images
and their selection status to prevent duplicates.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ImageTracker:
    """Tracks images using JSON file with dual hashing (URL and file content)"""

    def __init__(self, tracking_file: str = "tracking.json"):
        self.tracking_file = Path(tracking_file)
        self.data = self._load()
        self._recalculate_statistics()

    def _recalculate_statistics(self):
        """Recalculate statistics based on current images"""
        stats = {
            "totalImages": len(self.data["images"]),
            "selected": 0,
            "rejected": 0,
            "uploaded": 0,
        }

        for img in self.data["images"]:
            status = img.get("status")
            if status in stats:
                stats[status] += 1

        self.data["statistics"] = stats

    def _load(self) -> Dict[str, Any]:
        """Load tracking data from JSON file"""
        if self.tracking_file.exists():
            with open(self.tracking_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "images": [],
            "statistics": {
                "totalImages": 0,
                "selected": 0,
                "rejected": 0,
                "uploaded": 0,
            },
        }

    def save(self):
        """Save tracking data to JSON file safely"""
        temp_file = self.tracking_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.tracking_file)
        except Exception as e:
            print(f"Error saving tracking data: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

    @staticmethod
    def hash_url(url: str) -> str:
        """Generate hash from URL"""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_file(file_path: Path) -> str:
        """Generate hash from file content"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_duplicate(self, url: str, file_path: Optional[Path] = None) -> bool:
        """Check if image is already tracked (by URL or file hash)"""
        url_hash = self.hash_url(url)

        # Check URL hash
        for img in self.data["images"]:
            if img.get("urlHash") == url_hash:
                return True

        # Check file hash if file is provided
        if file_path and file_path.exists():
            file_hash = self.hash_file(file_path)
            for img in self.data["images"]:
                if img.get("fileHash") == file_hash:
                    return True

        return False

    def add_image(
        self,
        url: str,
        search_query: str,
        file_path: Optional[Path] = None,
        status: str = "pending",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add new image to tracking"""
        url_hash = self.hash_url(url)
        file_hash = (
            self.hash_file(file_path) if file_path and file_path.exists() else None
        )

        image_entry: Dict[str, Any] = {
            "urlHash": url_hash,
            "fileHash": file_hash,
            "url": url,
            "searchQuery": search_query,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "localPath": str(file_path) if file_path else None,
            "metadata": metadata or {},
        }

        self.data["images"].append(image_entry)
        self.data["statistics"]["totalImages"] += 1
        self.save()

        return url_hash

    def update_status(
        self, url_hash: str, status: str, drive_file_id: Optional[str] = None
    ):
        """Update image status (selected, rejected, uploaded)"""
        for img in self.data["images"]:
            if img["urlHash"] == url_hash:
                old_status = img["status"]
                img["status"] = status
                img["lastUpdated"] = datetime.now().isoformat()

                if drive_file_id:
                    img["driveFileId"] = drive_file_id

                # Update statistics
                if old_status in self.data["statistics"]:
                    self.data["statistics"][old_status] = max(
                        0, self.data["statistics"][old_status] - 1
                    )
                if status in self.data["statistics"]:
                    self.data["statistics"][status] += 1

                self.save()
                return True
        return False

    def update_file_hash(self, url_hash: str, file_path: Path):
        """Update file hash after download"""
        file_hash = self.hash_file(file_path)
        for img in self.data["images"]:
            if img["urlHash"] == url_hash:
                img["fileHash"] = file_hash
                img["localPath"] = str(file_path)
                self.save()
                return True
        return False

    def get_images_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all images with specific status"""
        return [img for img in self.data["images"] if img["status"] == status]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return self.data["statistics"]

    def update_metadata(self, url_hash: str, patch: Dict[str, Any]) -> bool:
        """
        Merge provided fields into image["metadata"] for a given url_hash.
        Creates metadata dict if it does not exist.
        """
        for img in self.data["images"]:
            if img["urlHash"] == url_hash:
                md: Dict[str, Any] = img.get("metadata") or {}
                md.update(patch or {})
                img["metadata"] = md
                img["lastUpdated"] = datetime.now().isoformat()
                self.save()
                return True
        return False

    def get_image_by_url_hash(self, url_hash: str) -> Optional[Dict[str, Any]]:
        """Get image entry by URL hash"""
        for img in self.data["images"]:
            if img["urlHash"] == url_hash:
                return img
        return None

    def get_image_by_file_hash(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get image entry by file hash"""
        if not file_path.exists():
            return None

        file_hash = self.hash_file(file_path)
        for img in self.data["images"]:
            if img.get("fileHash") == file_hash:
                return img
        return None
