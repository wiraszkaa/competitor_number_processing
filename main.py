#!/usr/bin/env python3
"""
Entry point for Dataset Preparation Pipeline.

Usage:
    python main.py                  # Run pipeline without validation
    python main.py --validate       # Run pipeline with Drive validation
"""

import argparse
import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent / "tools" / "drive_manager"))
sys.path.insert(0, str(Path(__file__).parent / "tools" / "roboflow_manager"))

from pipeline.dataset_preparation import main


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset Preparation Pipeline - Download, preprocess, and sync images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run process                          Run pipeline without validation
  uv run process --validate               Run pipeline and validate Drive sync
  python main.py                          Direct script execution
  python main.py --validate               Direct script with validation
        """,
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that completed preprocessed files exist on Drive (helps catch missing files)",
    )

    return parser.parse_args()


def entry_point():
    """Main entry point that parses arguments and runs the pipeline.

    This function is called both by the pyproject.toml script entry point
    and by the __main__ block.
    """
    args = parse_args()

    try:
        results = main(validate=args.validate)
        sys.exit(0 if not results.get("error") else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    entry_point()
