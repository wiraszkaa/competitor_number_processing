#!/usr/bin/env python3
"""
Image Search & Upload Application
Main entry point
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from main_gui import main

if __name__ == "__main__":
    main()
