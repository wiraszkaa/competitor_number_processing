"""Upload preprocessed images to Roboflow for annotation."""

import json
from pathlib import Path
import sys

from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import load_config, get_pipeline_logger

logger = get_pipeline_logger(__name__)


def upload_images_to_roboflow():
    """Upload all preprocessed images to Roboflow project."""

    config = load_config()
    roboflow_config = config.get("roboflow", {})

    # Check configuration
    if (
        not roboflow_config.get("api_key")
        or roboflow_config.get("api_key") == "YOUR_ROBOFLOW_API_KEY"
    ):
        logger.error("❌ Roboflow API key not configured")
        return False

    if not roboflow_config.get("workspace") or not roboflow_config.get("project"):
        logger.error("❌ Roboflow workspace/project not configured")
        return False

    # Get preprocessed images
    preprocessed_dir = Path(config["google_drive"]["download_dir_preprocessed"])
    if not preprocessed_dir.exists():
        logger.error(f"❌ Preprocessed directory not found: {preprocessed_dir}")
        return False

    # Find all preprocessed images (__final.png files)
    preprocessed_files = sorted(
        [f for f in preprocessed_dir.iterdir() if f.name.endswith("__final.png")]
    )

    if not preprocessed_files:
        logger.warning("⚠️  No preprocessed images found")
        return False

    logger.info(f"📤 Found {len(preprocessed_files)} preprocessed images to upload")
    logger.info(f"   API Key: {roboflow_config['api_key'][:10]}...")
    logger.info(f"   Workspace: {roboflow_config['workspace']}")
    logger.info(f"   Project: {roboflow_config['project']}")

    try:
        from roboflow import Roboflow

        # Initialize Roboflow
        logger.info("\n🔐 Authenticating with Roboflow...")
        rf = Roboflow(api_key=roboflow_config["api_key"])

        # Get project
        project = rf.workspace(roboflow_config["workspace"]).project(
            roboflow_config["project"]
        )

        # Upload images
        logger.info(f"\n📤 Uploading {len(preprocessed_files)} images...")

        # Batch upload (recommended by Roboflow)
        uploaded_count = 0
        failed_count = 0

        for image_path in tqdm(preprocessed_files, desc="Uploading", unit="image"):
            try:
                # Upload image
                project.upload(image_path=str(image_path), num_retry_uploads=3)
                uploaded_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to upload {image_path.name}: {e}")
                failed_count += 1

        logger.info(f"\n✅ Upload complete!")
        logger.info(f"   Uploaded: {uploaded_count}")
        if failed_count > 0:
            logger.warning(f"   Failed: {failed_count}")

        logger.info(f"\n📝 Next steps:")
        logger.info(
            f"   1. Go to: https://app.roboflow.com/{roboflow_config['workspace']}/{roboflow_config['project']}"
        )
        logger.info(f"   2. Annotate the uploaded images (mark jersey numbers)")
        logger.info(f"   3. Publish as Version 1 when done")
        logger.info(
            f"   4. Run: uv run python run_dataset_preparation.py (to check annotation status)"
        )

        return uploaded_count > 0

    except ImportError:
        logger.error("❌ roboflow package not installed")
        logger.info("   Install with: uv add roboflow")
        return False
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = upload_images_to_roboflow()
    sys.exit(0 if success else 1)
