"""
Modular training framework for segmentation models.
"""

from .config import TrainingConfig, TrainingHistory
from .trainer import SegmentationTrainer
from .early_stopping import EarlyStopping

__all__ = ["TrainingConfig", "TrainingHistory", "SegmentationTrainer", "EarlyStopping"]