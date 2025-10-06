"""
Evaluation metrics for segmentation extracted from EGH444_Assessment2_Template.py
"""
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    ignore_index: int = 255,
    device: torch.device = None,
    input_extractor: callable = None,
) -> Dict[str, Any]:
    """
    Evaluate segmentation model performance.

    Args:
        model: PyTorch segmentation model
        loader: DataLoader for evaluation data
        num_classes: Number of semantic classes
        ignore_index: Label value to ignore in evaluation
        device: Device to run evaluation on
        input_extractor: Function to extract input tensor from batch

    Returns:
        Dictionary containing:
            - mIoU: Mean Intersection over Union
            - PixelAcc: Pixel accuracy
            - IoU_per_class: IoU for each class
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total, correct = 0, 0

    for batch in loader:
        if input_extractor:
            images = input_extractor(batch).to(device, non_blocking=True)
        else:
            images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast(device.type, enabled=(device.type == "cuda")):
            out = model(images)
            # Handle both dict and tensor outputs
            if isinstance(out, dict):
                out = out["out"]
        pred = out.argmax(1)  # [B,H,W]

        # Calculate pixel accuracy
        valid = targets != ignore_index
        total += valid.sum().item()
        correct += (pred[valid] == targets[valid]).sum().item()

        # Update confusion matrix
        p = pred[valid].view(-1).cpu().numpy()
        t = targets[valid].view(-1).cpu().numpy()
        for i in range(p.shape[0]):
            if 0 <= t[i] < num_classes and 0 <= p[i] < num_classes:
                cm[t[i], p[i]] += 1

    # Calculate IoU metrics
    inter = np.diag(cm).astype(np.float64)
    gt = cm.sum(1).astype(np.float64)
    pr = cm.sum(0).astype(np.float64)
    union = gt + pr - inter
    iou = inter / np.maximum(union, 1)
    miou = float(np.nanmean(iou)) if iou.size > 0 else 0.0
    pixacc = float(correct / max(total, 1))

    return {"mIoU": miou, "PixelAcc": pixacc, "IoU_per_class": iou}


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute IoU per class for batch of predictions.

    Args:
        pred: Predicted class labels [B, H, W]
        target: Ground truth labels [B, H, W]
        num_classes: Number of classes

    Returns:
        IoU per class as tensor
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore index 255
    valid = target != 255
    pred = pred[valid]
    target = target[valid]

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return torch.tensor(ious)