"""
Pipeline tracking module.

Tracks which preprocessed images have been uploaded to Roboflow.
Drive folders are the source of truth for raw/preprocessed state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set


class PipelineTracker:
    """Tracks Roboflow upload state. Persists to tracking.json alongside image_collector data."""

    def __init__(self, tracking_file: Path):
        self.tracking_file = Path(tracking_file)
        self.roboflow_uploaded: Set[str] = set()
        self._other_sections: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not self.tracking_file.exists():
            print(f"Creating new tracking file at {self.tracking_file}")
            return
        try:
            with open(self.tracking_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.roboflow_uploaded = set(data.get("roboflow_uploaded", []))
            # Preserve other sections (e.g. "images" from image_collector)
            # Drop "preprocessing" — replaced by Drive folder state
            self._other_sections = {
                k: v for k, v in data.items()
                if k not in ("roboflow_uploaded", "preprocessing")
            }
            print(f"[OK] Loaded tracking: {len(self.roboflow_uploaded)} Roboflow-uploaded files")
        except Exception as e:
            print(f"Warning: Could not load tracking file: {e}")

    def save(self) -> None:
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self._other_sections)
        data["roboflow_uploaded"] = sorted(self.roboflow_uploaded)
        with open(self.tracking_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def is_roboflow_uploaded(self, filename: str) -> bool:
        return filename in self.roboflow_uploaded

    def mark_roboflow_uploaded(self, filename: str) -> None:
        self.roboflow_uploaded.add(filename)

    def get_roboflow_uploaded(self) -> List[str]:
        return sorted(self.roboflow_uploaded)

    def print_summary(self) -> None:
        print(f"\nTracking Status:")
        print(f"  Uploaded to Roboflow: {len(self.roboflow_uploaded)}")
