"""
Modular segmentation trainer that works with any model and dataset configuration.
"""
import os
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .config import TrainingConfig, TrainingHistory

# Use absolute imports that work when src is in the path
from src.datasets.nyu_depth import NYUDepthDataset
from src.datasets.transforms import SegTrainTransform, SegEvalTransform
from src.utils.metrics import evaluate
from src.utils.common import set_seeds


class SegmentationTrainer:
    """Generic trainer for segmentation models."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration object
        """
        self.config = config
        self.device = self._setup_device()
        self.history = TrainingHistory()

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        print(f"Using device: {device}")
        return device

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        # Initialize transforms with modality configuration
        train_tf = SegTrainTransform(
            size_hw=self.config.size_hw,
            hflip_p=self.config.hflip,
            label_mode=self.config.label_mode,
            use_rgb=self.config.use_rgb,
            use_depth=self.config.use_depth
        )
        val_tf = SegEvalTransform(
            size_hw=self.config.size_hw,
            label_mode=self.config.label_mode,
            use_rgb=self.config.use_rgb,
            use_depth=self.config.use_depth
        )

        # Initialize datasets
        train_ds = NYUDepthDataset(self.config.data_root, split="train", transform=train_tf)
        val_ds = NYUDepthDataset(self.config.data_root, split="test", transform=val_tf)

        # Initialize data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=True,
        )

        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")

        return train_loader, val_loader

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for training."""
        model = model.to(self.device)

        # Print parameter information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def _create_optimizer_and_criterion(self, model: nn.Module) -> tuple[optim.Optimizer, nn.Module]:
        """Create optimizer and loss criterion."""
        criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer, criterion

    def _extract_input_tensor(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract appropriate input tensor based on modality configuration."""
        tensors = []

        if self.config.use_rgb:
            tensors.append(batch["image"])

        if self.config.use_depth:
            tensors.append(batch["depth"])

        if len(tensors) == 1:
            return tensors[0]
        else:
            return torch.cat(tensors, dim=1)

    def train(self, model: nn.Module) -> nn.Module:
        """
        Train the provided model and return the trained model.

        Args:
            model: PyTorch model to train

        Returns:
            Trained model
        """
        # Set seeds for reproducibility
        set_seeds(self.config.seed)

        print(f"Training configuration:")
        print(f"  Use RGB: {self.config.use_rgb}")
        print(f"  Use Depth: {self.config.use_depth}")
        print(f"  Input channels: {self.config.input_channels}")
        print(f"  Label mode: {self.config.label_mode}")
        print(f"  Num classes: {self.config.num_classes}")

        # Prepare components
        train_loader, val_loader = self._create_data_loaders()
        model = self._prepare_model(model)
        optimizer, criterion = self._create_optimizer_and_criterion(model)

        # Initialize mixed precision scaler
        scaler = GradScaler(self.device.type, enabled=(self.device.type == "cuda"))

        best_miou = -1.0

        print(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(1, self.config.epochs + 1):
            # Training phase
            model.train()
            running_loss = 0.0
            epoch_start = time.time()

            for it, batch in enumerate(train_loader, start=1):
                # Extract input based on modality configuration
                inputs = self._extract_input_tensor(batch).to(self.device, non_blocking=True)
                targets = batch["mask"].to(self.device, non_blocking=True)

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass with mixed precision
                with autocast(self.device.type, enabled=(self.device.type == "cuda")):
                    outputs = model(inputs)
                    # Handle both dict and tensor outputs
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]
                    loss = criterion(outputs, targets)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                # Log training progress
                if it % self.config.log_every == 0:
                    avg_loss = running_loss / self.config.log_every
                    print(f"[Epoch {epoch} | Iter {it:05d}] loss={avg_loss:.4f}")
                    running_loss = 0.0

            # Validation phase
            metrics = evaluate(
                model, val_loader, self.config.num_classes,
                self.config.ignore_index, device=self.device,
                input_extractor=self._extract_input_tensor
            )

            epoch_time = time.time() - epoch_start
            print(
                f"[Epoch {epoch}] mIoU={metrics['mIoU']:.4f}  "
                f"PixelAcc={metrics['PixelAcc']:.4f}  "
                f"(epoch time {epoch_time:.1f}s)"
            )

            # Update history
            self.history.val_miou.append(metrics['mIoU'])
            self.history.val_pixacc.append(metrics['PixelAcc'])

            # Save best checkpoint
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": self.config,
                    "metrics": metrics,
                    "history": self.history,
                }
                torch.save(checkpoint, self.config.save_path)
                print(f"✓ Saved best checkpoint to {self.config.save_path} (mIoU={best_miou:.4f})")

        print("Training complete!")
        print(f"Best mIoU: {best_miou:.4f}")

        # Load best model for final evaluation
        if os.path.isfile(self.config.save_path):
            checkpoint = torch.load(self.config.save_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state"])
            final_metrics = evaluate(
                model, val_loader, self.config.num_classes,
                self.config.ignore_index, device=self.device,
                input_extractor=self._extract_input_tensor
            )
            print(f"Final validation metrics: mIoU={final_metrics['mIoU']:.4f}, "
                  f"PixelAcc={final_metrics['PixelAcc']:.4f}")

        return model