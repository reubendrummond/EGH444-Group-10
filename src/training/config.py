"""
Training configuration dataclass for segmentation models.
"""

from dataclasses import dataclass, field
from typing import Tuple, Literal, List, Optional
from pathlib import Path


@dataclass
class TrainingHistory:
    """Training history storage for comprehensive experiment tracking."""

    # Training metrics (per epoch)
    train_loss: List[float] = field(default_factory=list)
    train_miou: List[float] = field(default_factory=list)
    train_pixacc: List[float] = field(default_factory=list)

    # Validation metrics (per epoch)
    val_loss: List[float] = field(default_factory=list)
    val_miou: List[float] = field(default_factory=list)
    val_pixacc: List[float] = field(default_factory=list)

    # Learning rate tracking
    learning_rates: List[float] = field(default_factory=list)

    # Timing information
    epoch_times: List[float] = field(default_factory=list)

    def add_epoch(self, epoch_data: dict) -> None:
        """Add metrics for a completed epoch."""
        self.train_loss.append(epoch_data.get("train_loss", 0.0))
        self.train_miou.append(epoch_data.get("train_miou", 0.0))
        self.train_pixacc.append(epoch_data.get("train_pixacc", 0.0))

        self.val_loss.append(epoch_data.get("val_loss", 0.0))
        self.val_miou.append(epoch_data.get("val_miou", 0.0))
        self.val_pixacc.append(epoch_data.get("val_pixacc", 0.0))

        self.learning_rates.append(epoch_data.get("learning_rate", 0.0))
        self.epoch_times.append(epoch_data.get("epoch_time", 0.0))

    def get_best_epoch(self, metric: str = "val_miou") -> tuple[int, float]:
        """Get the epoch number and value of the best performance."""
        if metric not in ["val_miou", "val_pixacc", "train_miou", "train_pixacc"]:
            raise ValueError(f"Unsupported metric: {metric}")

        values = getattr(self, metric)
        if not values:
            return 0, 0.0

        best_idx = max(range(len(values)), key=lambda i: values[i])
        return best_idx + 1, values[best_idx]  # 1-indexed epoch

    def __len__(self) -> int:
        """Return number of completed epochs."""
        return len(self.train_loss)


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

    # Checkpoint and resumption
    save_path: str = "checkpoint.pt"
    save_best_path: str = "checkpoint_best.pt"
    save_latest_path: str = "checkpoint_latest.pt"
    save_every_n_epochs: int = 5  # Save checkpoint every N epochs (0 to disable)
    resume_from: Optional[str] = None  # Path to resume training from
    auto_resume: bool = True  # Automatically resume from latest if exists

    # Logging and evaluation
    log_every: int = 1

    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_miou"

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

    def validate_for_dual_encoder(self):
        """Validate configuration for dual-encoder models."""
        if not (self.use_rgb and self.use_depth):
            raise ValueError(
                "Dual-encoder models require both use_rgb=True and use_depth=True. "
                f"Current: use_rgb={self.use_rgb}, use_depth={self.use_depth}"
            )

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
