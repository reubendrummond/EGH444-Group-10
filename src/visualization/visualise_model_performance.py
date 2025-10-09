"""
Model performance visualization for semantic segmentation.

This module provides functionality to visualize model predictions alongside
original images and ground truth labels for semantic segmentation tasks.
"""

from typing import Optional, Tuple
import warnings
import inspect

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from ..datasets.nyu_depth import NYUDepthDataset
from ..datasets.transforms import SegEvalTransform

# Use clean visualization style
mplstyle.use(["seaborn-v0_8-whitegrid"])


def _detect_model_input_type(model: nn.Module) -> str:
    """
    Detect whether model expects RGB-only or RGB+depth inputs.

    Args:
        model: PyTorch model to inspect

    Returns:
        'rgb' for RGB-only models, 'rgbd' for dual-encoder models
    """
    # Check forward method signature for dual-encoder models
    forward_signature = inspect.signature(model.forward)
    param_names = list(forward_signature.parameters.keys())

    # If forward method takes both 'rgb' and 'depth' parameters, it's a dual-encoder
    if "rgb" in param_names and "depth" in param_names:
        return "rgbd"

    # Otherwise, check the first convolutional layer for input channels
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            if in_channels == 3:
                return "rgb"
            elif in_channels == 4:
                return "rgbd"
            else:
                warnings.warn(
                    f"Unexpected input channels: {in_channels}. Assuming RGB."
                )
                return "rgb"

    # Fallback to RGB if no conv layer found
    warnings.warn("No Conv2d layer found. Assuming RGB input.")
    return "rgb"


def _apply_colormap_to_segmentation(
    seg_mask: np.ndarray, num_classes: int = 40
) -> np.ndarray:
    """
    Apply colormap to segmentation mask for visualization.

    Args:
        seg_mask: Segmentation mask (H, W) with class indices
        num_classes: Number of classes in the dataset

    Returns:
        RGB image (H, W, 3) with colors for each class
    """
    # Create a colormap with distinct colors for each class
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, 20))
    if num_classes > 20:
        cmap2 = plt.get_cmap("Set3")
        colors2 = cmap2(np.linspace(0, 1, num_classes - 20))
        colors = np.vstack([colors, colors2])

    # Convert to 0-255 RGB
    colors = (colors[:, :3] * 255).astype(np.uint8)

    # Apply colormap
    colored = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
    for i in range(min(num_classes, len(colors))):
        mask = seg_mask == i
        colored[mask] = colors[i]

    return colored


def visualise_model_performance(
    model: nn.Module,
    dataset_path: str = "datasets/NYUDepthv2",
    samples: int = 6,
    split: str = "test",
    figsize: Tuple[int, int] = (15, 25),
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_classes: int = 40,
) -> None:
    """
    Visualize model performance by showing original images, ground truth, and predictions.

    Args:
        model: PyTorch model for semantic segmentation
        dataset_path: Path to dataset root directory
        samples: Number of samples to visualize
        split: Dataset split to use ('train', 'test', 'val')
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        device: Device to run inference on (auto-detected if None)
        num_classes: Number of classes in the dataset
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Detect model input requirements
    input_type = _detect_model_input_type(model)
    use_depth = input_type == "rgbd"

    print(f"Detected model input type: {input_type.upper()}")

    # Setup dataset and transforms
    transform = SegEvalTransform(use_rgb=True, use_depth=use_depth)
    dataset = NYUDepthDataset(base_dir=dataset_path, split=split, transform=transform)

    # Sample indices for visualization
    if len(dataset) < samples:
        print(
            f"Warning: Dataset has only {len(dataset)} samples, showing all available."
        )
        samples = len(dataset)

    # Use evenly spaced indices to get diverse samples
    indices = np.linspace(0, len(dataset) - 1, samples, dtype=int)

    # Create subplot grid
    _, axes = plt.subplots(samples, 3, figsize=figsize, facecolor="white")
    if samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx]
            rgb = sample["image"]  # Already normalized by transform
            label = sample["mask"]

            # Prepare input for model and get prediction
            if use_depth:
                depth = sample["depth"]
                # Check if model expects separate RGB and depth inputs (dual-encoder)
                forward_signature = inspect.signature(model.forward)
                param_names = list(forward_signature.parameters.keys())

                if "rgb" in param_names and "depth" in param_names:
                    # Dual-encoder model: pass RGB and depth separately
                    rgb_input = rgb.unsqueeze(0).to(device)
                    depth_input = depth.unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        logits = model(rgb_input, depth_input)
                else:
                    # Concatenated input model
                    model_input = torch.cat([rgb, depth], dim=0).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        logits = model(model_input)
            else:
                # RGB-only model
                model_input = rgb.unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(model_input)

            # Handle different model output formats
            if isinstance(logits, dict):
                # For models that return a dictionary (e.g., dual-encoder models)
                logits = logits["out"] if "out" in logits else list(logits.values())[0]

            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            # Convert tensors to numpy for visualization
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            label_np = label.cpu().numpy()

            # Denormalize RGB for display (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_display = (rgb_np * std + mean).clip(0, 1)

            # Apply colormaps to segmentation masks
            label_colored = _apply_colormap_to_segmentation(label_np, num_classes)
            pred_colored = _apply_colormap_to_segmentation(pred, num_classes)

            # Plot RGB image
            axes[i, 0].imshow(rgb_display)
            axes[i, 0].set_title(f"Original RGB (Sample {idx})")
            axes[i, 0].axis("off")

            # Plot ground truth
            axes[i, 1].imshow(label_colored)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            # Plot prediction
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

    plt.tight_layout(pad=1.0, h_pad=2.5, w_pad=2.6)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Visualization saved to: {save_path}")

    plt.show()

    # Print summary
    print(f"\n✓ Visualized {samples} samples from {split} set")
    print(f"  Model input type: {input_type.upper()}")
    print(f"  Dataset: {dataset_path}")
