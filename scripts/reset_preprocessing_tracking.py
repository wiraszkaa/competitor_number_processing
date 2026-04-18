#!/usr/bin/env python3
"""
Clean and reset the preprocessing tracking section.

This script clears the preprocessing tracking to start fresh.
Useful when preprocessed images have been manually deleted or tracking is corrupted.
"""

import json
from pathlib import Path


def reset_preprocessing_tracking():
    """Reset preprocessing section in tracking.json."""
    tracking_file = Path(__file__).parent.parent / "tracking.json"

    if not tracking_file.exists():
        print(f"✗ Tracking file not found: {tracking_file}")
        return False

    # Load existing tracking
    with open(tracking_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if there's old data to preserve
    old_preprocessing_count = len(data.get("preprocessing", {}))

    # Reset preprocessing section
    data["preprocessing"] = {}

    # Save back
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Preprocessing tracking reset!")
    print(f"   • Removed {old_preprocessing_count} old preprocessing records")
    print(f"   • Preserved other sections (images, etc)")
    print(f"   • File: {tracking_file}")

    return True


if __name__ == "__main__":
    reset_preprocessing_tracking()
