"""
Early stopping implementation for PyTorch training.
"""

import numpy as np


class EarlyStopping:
    """Early stopping to stop training when monitored metric doesn't improve."""

    def __init__(self, patience: int = 10, metric: str = "val_miou", min_delta: float = 1e-6):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement before stopping
            metric: Metric to monitor (e.g., 'val_miou', 'val_loss', 'val_pixacc')
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta

        # Auto-determine mode based on metric name
        self.mode = 'min' if 'loss' in metric.lower() else 'max'

        # Initialize tracking variables
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0

        print(f"EarlyStopping: monitoring '{metric}' with patience={patience} (mode={self.mode})")

    def __call__(self, epoch: int, score: float) -> bool:
        """
        Check if training should stop early.

        Args:
            epoch: Current epoch number
            score: Current metric score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                print(f"EarlyStopping: Stopping early at epoch {epoch}")
                print(f"Best {self.metric}: {self.best_score:.6f}")
                return True

        return False

    def _is_better(self, score: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            return score > (self.best_score + self.min_delta)

    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0