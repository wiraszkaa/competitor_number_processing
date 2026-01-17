"""
Competitor number processing package for extracting numbers from sports event images.
"""

from competitor_number_processing.preprocess import PreprocessConfig, preprocess_image
from competitor_number_processing.detector import (
    DetectionConfig,
    DetectedPerson,
    PersonDetector,
    detect_people,
    extract_competitor_regions,
)

__all__ = [
    "PreprocessConfig",
    "preprocess_image",
    "DetectionConfig",
    "DetectedPerson",
    "PersonDetector",
    "detect_people",
    "extract_competitor_regions",
]
