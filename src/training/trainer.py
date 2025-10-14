"""
Modular segmentation trainer that works with any model and dataset configuration.
"""

import os
import time
import inspect
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from .config import TrainingConfig, TrainingHistory
from .early_stopping import EarlyStopping

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

        # Initialize early stopping if enabled
        self.early_stopping = None
        if config.early_stopping_enabled:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                metric=config.early_stopping_metric,
            )

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        print(f"Using device: {device}")
        return device

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        # Initialize transforms with modality configuration
        train_tf = SegTrainTransform(
            size_hw=self.config.size_hw,
            # hflip_p=self.config.hflip,
            label_mode=self.config.label_mode,
            use_rgb=self.config.use_rgb,
            use_depth=self.config.use_depth,
            hflip_p=0.3,
            depth_noise_p=0.2,
            depth_noise_std=0.025,  # Moderate noise
            depth_dropout_p=0.2,  # Moderate dropout probability
            depth_dropout_rate=0.04,  # Visible but not excessive
            rotation_p=0.4,
            rotation_angle=8.0,  # Reduced rotation angle
            crop_zoom_p=0.3,
            crop_scale_range=(0.85, 0.95),
            color_jitter_p=0.6,  # Keep color effects visible
            brightness_range=0.25,  # Still noticeable brightness
            contrast_range=0.2,  # Good contrast variation
            saturation_range=0.25,  # Visible saturation changes
            hue_range=0.06,  # Subtle but visible hue shifts
        )
        val_tf = SegEvalTransform(
            size_hw=self.config.size_hw,
            label_mode=self.config.label_mode,
            use_rgb=self.config.use_rgb,
            use_depth=self.config.use_depth,
        )

        # Initialize datasets
        train_ds = NYUDepthDataset(
            self.config.data_root, split="train", transform=train_tf
        )
        test_ds = NYUDepthDataset(self.config.data_root, split="test", transform=val_tf)
        val_ds = Subset(test_ds, range(min(len(test_ds), 100)))  # first 100 samples

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

    def _create_optimizer_and_criterion(
        self, model: nn.Module
    ) -> tuple[optim.Optimizer, nn.Module]:
        """Create optimizer and loss criterion."""
        criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        optimizer = optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        return optimizer, criterion

    def _is_dual_input_model(self, model: nn.Module) -> bool:
        """
        Detect if model expects dual inputs (RGB and depth separately).

        Args:
            model: The model to inspect

        Returns:
            True if model expects dual inputs, False for single input
        """
        # Get the forward method signature
        forward_signature = inspect.signature(model.forward)
        params = list(forward_signature.parameters.keys())

        # Skip 'self' parameter
        input_params = [p for p in params if p != "self"]

        # Dual-input models should have 2 input parameters (e.g., 'rgb', 'depth')
        # Single-input models should have 1 input parameter (e.g., 'x', 'input')
        return len(input_params) >= 2

    def _extract_input_tensor(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract appropriate input tensor based on modality configuration (for single-input models)."""
        tensors = []

        if self.config.use_rgb:
            tensors.append(batch["image"])

        if self.config.use_depth:
            tensors.append(batch["depth"])

        if len(tensors) == 1:
            return tensors[0]
        else:
            return torch.cat(tensors, dim=1)

    def _forward_model(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through model, handling both single-input and dual-input models.

        Args:
            model: The model to run forward pass
            batch: Batch dictionary containing input data

        Returns:
            Model output tensor or dict
        """
        if self._is_dual_input_model(model):
            # Dual-input model: pass RGB and depth separately
            if not (self.config.use_rgb and self.config.use_depth):
                raise ValueError(
                    "Dual-input models require both use_rgb=True and use_depth=True"
                )

            rgb = batch["image"]
            depth = batch["depth"]
            return model(rgb, depth)
        else:
            # Single-input model: use concatenated tensor
            inputs = self._extract_input_tensor(batch)
            return model(inputs)

    @torch.no_grad()
    def _compute_comprehensive_metrics(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        phase_name: str = "eval",
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics (loss + mIoU + pixel accuracy) for any data loader.

        Args:
            model: Model to evaluate
            loader: DataLoader to evaluate on
            criterion: Loss function
            phase_name: Name for logging (e.g., "train", "val")

        Returns:
            Dictionary with loss, mIoU, PixelAcc, and per-class IoU
        """
        model.eval()

        # Initialize metrics tracking
        total_loss = 0.0
        cm = np.zeros(
            (self.config.num_classes, self.config.num_classes), dtype=np.int64
        )
        total_pixels, correct_pixels = 0, 0
        num_batches = 0

        for batch in loader:
            # Extract targets and move to device
            targets = batch["mask"].to(self.device, non_blocking=True)

            # Move batch data to device
            batch_on_device = {
                k: (
                    v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }

            # Forward pass with mixed precision
            with autocast(self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self._forward_model(model, batch_on_device)
                # Handle both dict and tensor outputs
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                loss = criterion(outputs, targets)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            predictions = torch.argmax(outputs, dim=1)

            # Move to CPU for metrics computation
            predictions_np = predictions.cpu().numpy().astype(np.int64)
            targets_np = targets.cpu().numpy().astype(np.int64)

            # Update confusion matrix (excluding ignore_index)
            valid_mask = targets_np != self.config.ignore_index
            valid_preds = predictions_np[valid_mask]
            valid_targets = targets_np[valid_mask]

            if len(valid_preds) > 0:
                # Update confusion matrix
                cm_batch = np.bincount(
                    self.config.num_classes * valid_targets + valid_preds,
                    minlength=self.config.num_classes**2,
                ).reshape(self.config.num_classes, self.config.num_classes)
                cm += cm_batch

                # Update pixel accuracy
                total_pixels += len(valid_preds)
                correct_pixels += np.sum(valid_preds == valid_targets)

        # Compute final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute IoU per class
        iou_per_class = []
        for i in range(self.config.num_classes):
            intersection = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - intersection
            if union > 0:
                iou_per_class.append(intersection / union)
            else:
                iou_per_class.append(0.0)

        # Compute mean IoU
        miou = np.mean(iou_per_class)

        # Compute pixel accuracy
        pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        return {
            "loss": avg_loss,
            "mIoU": miou,
            "PixelAcc": pixel_acc,
            "IoU_per_class": iou_per_class,
        }

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

        # Check for resumption
        start_epoch = 1
        best_miou = -1.0
        if self._should_resume():
            start_epoch, best_miou = self._load_checkpoint_for_resumption(
                model, optimizer, scaler
            )

        print(
            f"Starting training for {self.config.epochs} epochs (from epoch {start_epoch})..."
        )

        for epoch in range(start_epoch, self.config.epochs + 1):
            # Training phase
            model.train()
            total_train_loss = 0.0
            num_batches = 0
            epoch_start = time.time()
            train_start = time.time()

            for it, batch in enumerate(train_loader, start=1):
                # Extract input based on modality configuration
                # Extract targets and move to device
                targets = batch["mask"].to(self.device, non_blocking=True)

                # Move batch data to device
                batch_on_device = {
                    k: (
                        v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass with mixed precision
                with autocast(self.device.type, enabled=(self.device.type == "cuda")):
                    outputs = self._forward_model(model, batch_on_device)
                    # Handle both dict and tensor outputs
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]
                    loss = criterion(outputs, targets)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                num_batches += 1

                # Log training progress
                if it % self.config.log_every == 0:
                    recent_avg = total_train_loss / num_batches
                    print(f"[Epoch {epoch} | Iter {it:05d}] loss={recent_avg:.4f}")

            # Calculate comprehensive training metrics
            train_time = time.time() - train_start

            # Always compute training metrics (loss + mIoU + pixel accuracy)
            train_eval_start = time.time()
            train_metrics = self._compute_comprehensive_metrics(
                model, train_loader, criterion, "train"
            )
            train_eval_time = time.time() - train_eval_start

            # Validation phase - compute comprehensive metrics (loss + mIoU + pixel accuracy)
            val_start = time.time()
            val_metrics = self._compute_comprehensive_metrics(
                model, val_loader, criterion, "val"
            )
            val_time = time.time() - val_start

            epoch_time = time.time() - epoch_start

            # Calculate performance metrics
            train_samples = len(train_loader.dataset)
            val_samples = len(val_loader.dataset)
            train_speed = train_samples / train_time if train_time > 0 else 0
            val_speed = val_samples / val_time if val_time > 0 else 0

            # Print comprehensive epoch summary with all metrics
            print(
                f"[Epoch {epoch}] Train: loss={train_metrics['loss']:.4f} mIoU={train_metrics['mIoU']:.4f} pixacc={train_metrics['PixelAcc']:.4f} "
                f"({train_time:.1f}s+{train_eval_time:.1f}s, {train_speed:.1f} smp/s) | "
                f"Val: loss={val_metrics['loss']:.4f} mIoU={val_metrics['mIoU']:.4f} pixacc={val_metrics['PixelAcc']:.4f} "
                f"({val_time:.1f}s, {val_speed:.1f} smp/s)"
            )

            # Update history with comprehensive metrics
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_data = {
                "train_loss": train_metrics["loss"],
                "train_miou": train_metrics["mIoU"],
                "train_pixacc": train_metrics["PixelAcc"],
                "val_loss": val_metrics["loss"],
                "val_miou": val_metrics["mIoU"],
                "val_pixacc": val_metrics["PixelAcc"],
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
            self.history.add_epoch(epoch_data)

            # Save checkpoints
            is_best = val_metrics["mIoU"] > best_miou
            if is_best:
                best_miou = val_metrics["mIoU"]

            # Save checkpoints with timing
            save_start = time.time()
            self._save_checkpoints(
                model, optimizer, scaler, epoch, val_metrics, is_best
            )
            save_time = time.time() - save_start

            # Print comprehensive timing summary
            total_time = time.time() - epoch_start
            time_breakdown = {
                "train": train_time,
                "train_eval": train_eval_time,
                "val": val_time,
                "save": save_time,
                "total": total_time,
            }

            # Calculate percentages
            train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
            val_pct = (val_time / total_time) * 100 if total_time > 0 else 0
            save_pct = (save_time / total_time) * 100 if total_time > 0 else 0

            print(
                f"⏱️  Timing: Save({save_time:.1f}s) | Total({total_time:.1f}s) | Breakdown: Train({train_pct:.1f}%) Val({val_pct:.1f}%) Save({save_pct:.1f}%)"
            )

            # Check early stopping
            if self.early_stopping is not None:
                # Get the metric value to monitor
                if self.config.early_stopping_metric == "val_miou":
                    metric_value = val_metrics["mIoU"]
                elif self.config.early_stopping_metric == "val_loss":
                    metric_value = val_metrics["loss"]
                elif self.config.early_stopping_metric == "val_pixacc":
                    metric_value = val_metrics["PixelAcc"]
                else:
                    raise ValueError(
                        f"Unsupported early stopping metric: {self.config.early_stopping_metric}"
                    )

                if self.early_stopping(epoch, metric_value):
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    break

            # Overfitting detection warnings
            train_val_gap_miou = train_metrics["mIoU"] - val_metrics["mIoU"]
            train_val_gap_loss = val_metrics["loss"] - train_metrics["loss"]

            if train_val_gap_miou > 0.1:
                print(
                    f"⚠️  Potential overfitting: Train mIoU ({train_metrics['mIoU']:.4f}) >> Val mIoU ({val_metrics['mIoU']:.4f}), gap: {train_val_gap_miou:.4f}"
                )
            if train_val_gap_loss > 0.5:
                print(
                    f"⚠️  Potential overfitting: Val loss ({val_metrics['loss']:.4f}) >> Train loss ({train_metrics['loss']:.4f}), gap: {train_val_gap_loss:.4f}"
                )

        print("Training complete!")
        print(f"Best mIoU: {best_miou:.4f}")

        # Load best model for final evaluation
        if os.path.isfile(self.config.save_path):
            print("\n🔄 Loading best model for final evaluation...")
            load_start = time.time()
            checkpoint = torch.load(
                self.config.save_path, map_location=self.device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
            load_time = time.time() - load_start

            # Create criterion for final evaluation
            final_criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

            eval_start = time.time()
            final_metrics = self._compute_comprehensive_metrics(
                model, val_loader, final_criterion, "final"
            )
            eval_time = time.time() - eval_start

            print(
                f"Final validation metrics: loss={final_metrics['loss']:.4f} mIoU={final_metrics['mIoU']:.4f}, "
                f"PixelAcc={final_metrics['PixelAcc']:.4f} (load: {load_time:.1f}s, eval: {eval_time:.1f}s)"
            )

        return model

    def _apply_config_to_optimizer(self, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """
        Apply current config parameters to optimizer, overriding loaded checkpoint values.

        Args:
            optimizer: Optimizer to update

        Returns:
            Dictionary of old vs new parameter values for logging
        """
        changes = {}

        # Get current parameters from first param group
        param_group = optimizer.param_groups[0]

        # Store old values and apply new ones
        old_lr = param_group["lr"]
        old_weight_decay = param_group["weight_decay"]

        # Apply current config values
        param_group["lr"] = self.config.lr
        param_group["weight_decay"] = self.config.weight_decay

        # Track changes
        if old_lr != self.config.lr:
            changes["learning_rate"] = {"old": old_lr, "new": self.config.lr}
        if old_weight_decay != self.config.weight_decay:
            changes["weight_decay"] = {
                "old": old_weight_decay,
                "new": self.config.weight_decay,
            }

        return changes

    def _should_resume(self) -> bool:
        """Check if training should be resumed from a checkpoint."""
        if self.config.resume_from:
            return os.path.isfile(self.config.resume_from)

        if self.config.auto_resume:
            return os.path.isfile(self.config.save_latest_path)

        return False

    def _load_checkpoint_for_resumption(
        self, model: nn.Module, optimizer: optim.Optimizer, scaler: GradScaler
    ) -> tuple[int, float]:
        """
        Load checkpoint for training resumption.

        Returns:
            tuple of (start_epoch, best_miou)
        """
        checkpoint_path = self.config.resume_from or self.config.save_latest_path

        print(f"Resuming training from: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Load model state
            model.load_state_dict(checkpoint["model_state"])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Apply current config to override checkpoint optimizer parameters
            param_changes = self._apply_config_to_optimizer(optimizer)

            # Load scaler state if available
            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])

            # Restore training history
            if "history" in checkpoint:
                self.history = checkpoint["history"]

            start_epoch = checkpoint.get("epoch", 0) + 1

            # Calculate best mIoU from history
            best_miou = max(self.history.val_miou) if self.history.val_miou else -1.0

            print(f"✓ Resumed from epoch {checkpoint.get('epoch', 0)}")
            print(f"✓ Training history loaded with {len(self.history)} epochs")
            print(f"✓ Best mIoU so far: {best_miou:.4f}")

            # Log parameter changes if any
            if param_changes:
                print("🔧 Applied config overrides:")
                for param_name, change in param_changes.items():
                    print(f"  {param_name}: {change['old']:.6f} → {change['new']:.6f}")
            else:
                print("✓ No optimizer parameter changes needed")

            return start_epoch, best_miou

        except Exception as e:
            print(f"❌ Failed to load checkpoint for resumption: {e}")
            print("Starting training from scratch...")
            return 1, -1.0

    def _save_checkpoints(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool,
    ) -> None:
        """Save checkpoints according to configuration with detailed timing."""
        # Prepare checkpoint data
        prep_start = time.time()
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "history": self.history,
        }
        prep_time = time.time() - prep_start

        # Always save latest checkpoint
        latest_start = time.time()
        torch.save(checkpoint, self.config.save_latest_path)
        latest_time = time.time() - latest_start

        best_time = 0.0
        periodic_time = 0.0

        # Save best checkpoint if this is the best performance
        if is_best:
            best_start = time.time()
            torch.save(checkpoint, self.config.save_best_path)
            # Also save to the legacy save_path for backward compatibility
            torch.save(checkpoint, self.config.save_path)
            best_time = time.time() - best_start
            print(
                f"✓ Saved best checkpoint (mIoU={metrics['mIoU']:.4f}) to {self.config.save_best_path} ({best_time:.1f}s)"
            )

        # Save periodic checkpoint
        if (
            self.config.save_every_n_epochs > 0
            and epoch % self.config.save_every_n_epochs == 0
        ):
            periodic_start = time.time()
            periodic_path = f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, periodic_path)
            periodic_time = time.time() - periodic_start
            print(
                f"✓ Saved periodic checkpoint to {periodic_path} ({periodic_time:.1f}s)"
            )

        # Calculate checkpoint size for reference
        import os

        try:
            checkpoint_size_mb = os.path.getsize(self.config.save_latest_path) / (
                1024 * 1024
            )
        except:
            checkpoint_size_mb = 0

        # Print timing summary
        total_save_time = prep_time + latest_time + best_time + periodic_time
        print(
            f"💾 Checkpoint timing: prep({prep_time:.1f}s) latest({latest_time:.1f}s) size({checkpoint_size_mb:.1f}MB) total({total_save_time:.1f}s)"
        )
