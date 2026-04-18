"""
Pipeline module - orchestrates multi-step data processing pipeline.

Pipeline steps:
1. dataset_preparation - Download images, preprocess, check annotations
2. (Future) model_training - Train ML models
3. (Future) inference - Run inference on data
"""

from pipeline.config import load_config, get_pipeline_logger
from pipeline.dataset_preparation import DatasetPreparationPipeline

__all__ = [
    "load_config",
    "get_pipeline_logger",
    "DatasetPreparationPipeline",
]
