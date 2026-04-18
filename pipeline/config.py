"""
Pipeline configuration and utilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict


def load_config() -> Dict[str, Any]:
    """
    Load configuration from secrets/config.json.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
    """
    config_path = Path(__file__).parent.parent / "secrets" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please create it based on secrets/config.example.json"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pipeline_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for pipeline steps.

    Args:
        name: Logger name (e.g., __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
