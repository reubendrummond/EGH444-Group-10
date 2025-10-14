#!/usr/bin/env python3
"""
Simple Augmentation Sample Visualization

Shows sample outputs from training transforms in a clean grid format:
- RGB: original | sample 1 | sample 2 | ...
- Depth: original | sample 1 | sample 2 | ...

Usage:
    python scripts/visualize_augmentation_samples.py --samples 5
"""

import argparse
import random
from typing import Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

# Import project modules
import sys

sys.path.append(".")
from src.datasets.nyu_depth import NYUDepthDataset
from src.datasets.transforms import (
    SegTrainTransform,
    SegEvalTransform,
    ImageNetTransform,
)

# Use clean visualization style
mplstyle.use(["seaborn-v0_8-whitegrid"])


def denormalize_rgb_for_display(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized RGB tensor to displayable numpy array."""
    return ImageNetTransform.denormalize_for_display(rgb_tensor)


def denormalize_depth_for_display(depth_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized depth tensor to displayable numpy array."""
    depth_np = depth_tensor.squeeze().cpu().numpy()
    # Convert back to visual range for display
    depth_display = (depth_np * 255).astype(np.uint8)
    return depth_display


def apply_colormap_to_segmentation(
    seg_mask: np.ndarray, num_classes: int = 40
) -> np.ndarray:
    """Apply colormap to segmentation mask for visualization."""
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


def visualize_augmentation_samples(
    dataset_path: str = "datasets/NYUDepthv2",
    samples: int = 5,
    split: str = "train",
    figsize: Tuple[int, int] = (15, 9),
    save_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Visualize augmentation samples in a grid format.

    Args:
        dataset_path: Path to dataset root directory
        samples: Number of augmented samples to show
        split: Dataset split to use ('train', 'test')
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        seed: Random seed for reproducible results
    """
    print(f"🎨 Visualizing augmentation samples from {dataset_path}")
    print(f"   Samples: {samples}, Split: {split}, Seed: {seed}")

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset without transforms
    dataset = NYUDepthDataset(base_dir=dataset_path, split=split, transform=None)

    # Pick a single sample to augment multiple times
    sample_idx = len(dataset) // 2  # Use middle sample
    raw_sample = dataset[sample_idx]

    print(f"   Using sample {sample_idx} for augmentation demonstration")

    # Create transforms
    original_transform = SegEvalTransform(
        size_hw=(240, 320), use_rgb=True, use_depth=True
    )
    augment_transform = SegTrainTransform(
        size_hw=(240, 320),
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
        use_rgb=True,
        use_depth=True,
    )

    # Generate original and augmented samples
    original_data = original_transform(raw_sample)
    augmented_samples = []

    for i in range(samples):
        # Re-seed for each sample to get variety
        random.seed(seed + i)
        np.random.seed(seed + i)
        augmented = augment_transform(raw_sample)
        augmented_samples.append(augmented)

    # Reset seed to original
    random.seed(seed)
    np.random.seed(seed)

    # Create figure with 3 rows (RGB, Depth, and Mask)
    _, axes = plt.subplots(3, samples + 1, figsize=figsize, facecolor="white")

    # RGB row
    # Original RGB
    rgb_orig = denormalize_rgb_for_display(original_data["image"])
    axes[0, 0].imshow(rgb_orig)
    axes[0, 0].set_title("RGB\nOriginal")
    axes[0, 0].axis("off")

    # Augmented RGB samples
    for i, aug_sample in enumerate(augmented_samples, 1):
        rgb_aug = denormalize_rgb_for_display(aug_sample["image"])
        axes[0, i].imshow(rgb_aug)
        axes[0, i].set_title(f"Sample {i}")
        axes[0, i].axis("off")

    # Depth row
    # Original Depth
    depth_orig = denormalize_depth_for_display(original_data["depth"])
    axes[1, 0].imshow(depth_orig, cmap="plasma")
    axes[1, 0].set_title("Depth\nOriginal")
    axes[1, 0].axis("off")

    # Augmented Depth samples
    for i, aug_sample in enumerate(augmented_samples, 1):
        depth_aug = denormalize_depth_for_display(aug_sample["depth"])
        axes[1, i].imshow(depth_aug, cmap="plasma")
        axes[1, i].set_title(f"Sample {i}")
        axes[1, i].axis("off")

    # Mask row
    # Original Mask
    mask_orig = apply_colormap_to_segmentation(original_data["mask"].numpy())
    axes[2, 0].imshow(mask_orig)
    axes[2, 0].set_title("Mask\nOriginal")
    axes[2, 0].axis("off")

    # Augmented Mask samples
    for i, aug_sample in enumerate(augmented_samples, 1):
        mask_aug = apply_colormap_to_segmentation(aug_sample["mask"].numpy())
        axes[2, i].imshow(mask_aug)
        axes[2, i].set_title(f"Sample {i}")
        axes[2, i].axis("off")

    # Add row labels
    axes[0, 0].text(
        -0.1,
        0.5,
        "RGB",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.1,
        0.5,
        "Depth",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=14,
        fontweight="bold",
    )
    axes[2, 0].text(
        -0.1,
        0.5,
        "Mask",
        transform=axes[2, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=0.5)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ Visualization saved to: {save_path}")

    if save_path:
        print("📊 Saving visualization...")
    else:
        plt.show()

    # Print transform configuration
    print("\n⚙️  Enhanced Transform Configuration:")
    print(f"   Horizontal flip probability: {augment_transform.hflip_p}")
    print(
        f"   Depth noise probability: {augment_transform.depth_noise_p} (std: {augment_transform.depth_noise_std})"
    )
    print(
        f"   Depth dropout probability: {augment_transform.depth_dropout_p} (rate: {augment_transform.depth_dropout_rate})"
    )
    print(
        f"   Rotation probability: {augment_transform.rotation_p} (±{augment_transform.rotation_angle}°)"
    )
    print(
        f"   Crop+zoom probability: {augment_transform.crop_zoom_p} (scale: {augment_transform.crop_scale_range})"
    )
    print(f"   Color jitter probability: {augment_transform.color_jitter_p}")
    print(f"     Brightness range: ±{augment_transform.brightness_range}")
    print(f"     Contrast range: ±{augment_transform.contrast_range}")
    print(f"     Saturation range: ±{augment_transform.saturation_range}")
    print(f"     Hue range: ±{augment_transform.hue_range}")
    print("\n💡 Enhanced for better visibility of augmentation effects!")

    print(
        f"\n✅ Visualized {samples} augmented samples from dataset sample {sample_idx}"
    )


def main():
    """Command-line interface for the augmentation sample visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize augmentation samples in grid format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/NYUDepthv2",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of augmented samples to show"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[15, 9],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save the visualization"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible results"
    )

    args = parser.parse_args()

    visualize_augmentation_samples(
        dataset_path=args.dataset,
        samples=args.samples,
        split=args.split,
        figsize=tuple(args.figsize),
        save_path=args.save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
