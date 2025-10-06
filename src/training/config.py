"""
Training configuration dataclass for segmentation models.
"""

from dataclasses import dataclass, field
from typing import Tuple, Literal, List
from pathlib import Path


@dataclass
class TrainingHistory:
    """Training history storage."""
    train_loss: List[float] = field(default_factory=list)
    val_miou: List[float] = field(default_factory=list)
    val_pixacc: List[float] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration for segmentation model training."""

    # Dataset configuration
    data_root: str = "./datasets/NYUDepthv2"
    num_classes: int = 40
    label_mode: Literal["nyu40", "raw"] = "nyu40"

    # Input modality configuration
    use_rgb: bool = True
    use_depth: bool = False

    # Model configuration
    pretrained_backbone: bool = True
    freeze_backbone: bool = False

    # Training hyperparameters
    epochs: int = 2
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-4

    # Data augmentation
    size: Tuple[int, int] = (240, 320)
    hflip: float = 0.5

    # System configuration
    device: str = "auto"
    workers: int = 2
    seed: int = 42

    # Loss and evaluation
    ignore_index: int = 255

    # Logging and saving
    save_path: str = "checkpoint.pt"
    log_every: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of use_rgb or use_depth must be True")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if not Path(self.data_root).exists():
            raise ValueError(f"data_root path does not exist: {self.data_root}")

    @property
    def input_channels(self) -> int:
        """Calculate number of input channels based on modality flags."""
        channels = 0
        if self.use_rgb:
            channels += 3
        if self.use_depth:
            channels += 1
        return channels

    @property
    def size_hw(self) -> Tuple[int, int]:
        """Get size as (height, width) tuple."""
        return self.size
