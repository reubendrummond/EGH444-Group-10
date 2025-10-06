"""
Modular training framework for segmentation models.
"""

from .config import TrainingConfig, TrainingHistory
from .trainer import SegmentationTrainer

__all__ = ["TrainingConfig", "TrainingHistory", "SegmentationTrainer"]