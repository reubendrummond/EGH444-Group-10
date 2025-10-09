"""
Training history visualization for segmentation models.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

# Use a clean, publication-ready style
mplstyle.use(['seaborn-v0_8-whitegrid'])


def load_training_history(checkpoint_path: str) -> Dict:
    """
    Load training history from a PyTorch checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing training history and metadata

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint doesn't contain expected history data
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        # Load checkpoint with weights_only=False to access custom objects
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract training history
        if 'history' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'history' key")

        history = checkpoint['history']

        # Extract history data (support both old and new format)
        result = {
            'train_loss': getattr(history, 'train_loss', []),
            'train_miou': getattr(history, 'train_miou', []),
            'train_pixacc': getattr(history, 'train_pixacc', []),
            'val_loss': getattr(history, 'val_loss', []),
            'val_miou': getattr(history, 'val_miou', []),
            'val_pixacc': getattr(history, 'val_pixacc', []),
            'learning_rates': getattr(history, 'learning_rates', []),
            'epoch_times': getattr(history, 'epoch_times', []),
            'final_epoch': checkpoint.get('epoch', len(getattr(history, 'val_miou', []))),
            'best_miou': max(getattr(history, 'val_miou', [0])) if getattr(history, 'val_miou', []) else 0,
            'final_metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', None)
        }

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to load training history from {checkpoint_path}: {e}")


def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 300,
    show_best: bool = True
) -> plt.Figure:
    """
    Plot comprehensive training curves comparing training vs validation metrics.

    Args:
        history: Training history dictionary from load_training_history()
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figures
        show_best: Whether to highlight best performance

    Returns:
        matplotlib Figure object
    """
    # Determine available metrics
    has_train_loss = bool(history['train_loss'])
    has_train_miou = bool(history['train_miou'])
    has_train_pixacc = bool(history['train_pixacc'])
    has_val_loss = bool(history['val_loss'])
    has_val_metrics = bool(history['val_miou']) or bool(history['val_pixacc']) or has_val_loss

    if not has_val_metrics and not has_train_loss:
        raise ValueError("No training metrics found in training history")

    # Define consistent colors for training and validation across all subplots
    train_color = '#1f77b4'  # Blue
    val_color = '#ff7f0e'    # Orange

    # Create epochs array
    max_epochs = max(
        len(history['train_loss']),
        len(history['val_loss']),
        len(history['val_miou']),
        len(history['val_pixacc']),
        len(history['train_miou']),
        len(history['train_pixacc'])
    )
    epochs = list(range(1, max_epochs + 1))

    # Create subplot layout
    n_plots = 0
    if has_train_loss or has_val_loss: n_plots += 1
    if history['val_miou'] or has_train_miou: n_plots += 1
    if history['val_pixacc'] or has_train_pixacc: n_plots += 1

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')
    axes = axes.flatten()

    plot_idx = 0

    # Plot 1: Training and Validation Loss
    if has_train_loss or history['val_loss']:
        ax = axes[plot_idx]
        if has_train_loss:
            ax.plot(epochs[:len(history['train_loss'])], history['train_loss'],
                   color=train_color, linewidth=2, marker='o', markersize=3, label='Training Loss')
        if history['val_loss']:
            ax.plot(epochs[:len(history['val_loss'])], history['val_loss'],
                   color=val_color, linewidth=2, marker='s', markersize=3, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss: Training vs Validation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1

    # Plot 2: mIoU Comparison
    if history['val_miou'] or has_train_miou:
        ax = axes[plot_idx]
        if has_train_miou and history['train_miou']:
            ax.plot(epochs[:len(history['train_miou'])], history['train_miou'],
                   color=train_color, linestyle='--', linewidth=2, marker='s', markersize=3, alpha=0.7, label='Training mIoU')
        if history['val_miou']:
            ax.plot(epochs[:len(history['val_miou'])], history['val_miou'],
                   color=val_color, linewidth=2, marker='o', markersize=3, label='Validation mIoU')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU')
        ax.set_title('Mean IoU: Training vs Validation')
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on actual data range for better visibility
        all_miou_values = []
        if history['val_miou']:
            all_miou_values.extend(history['val_miou'])
        if has_train_miou and history['train_miou']:
            all_miou_values.extend(history['train_miou'])

        if all_miou_values:
            min_miou = min(all_miou_values)
            max_miou = max(all_miou_values)
            # Add 10% padding above and below the data range
            padding = (max_miou - min_miou) * 0.1 if max_miou > min_miou else 0.05
            ax.set_ylim(max(0, min_miou - padding), min(1, max_miou + padding))
        else:
            ax.set_ylim(0, 1)

        # Highlight best validation performance
        if show_best and history['val_miou']:
            best_epoch = np.argmax(history['val_miou']) + 1
            best_miou = max(history['val_miou'])
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.plot(best_epoch, best_miou, 'ro', markersize=6)

        ax.legend()
        plot_idx += 1

    # Plot 3: Pixel Accuracy Comparison
    if history['val_pixacc'] or has_train_pixacc:
        ax = axes[plot_idx]
        if has_train_pixacc and history['train_pixacc']:
            ax.plot(epochs[:len(history['train_pixacc'])], history['train_pixacc'],
                   color=train_color, linestyle='--', linewidth=2, marker='s', markersize=3, alpha=0.7, label='Training Pixel Acc')
        if history['val_pixacc']:
            ax.plot(epochs[:len(history['val_pixacc'])], history['val_pixacc'],
                   color=val_color, linewidth=2, marker='o', markersize=3, label='Validation Pixel Acc')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pixel Accuracy')
        ax.set_title('Pixel Accuracy: Training vs Validation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Highlight best validation performance
        if show_best and history['val_pixacc']:
            best_epoch = np.argmax(history['val_pixacc']) + 1
            best_pixacc = max(history['val_pixacc'])
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.plot(best_epoch, best_pixacc, 'ro', markersize=6)

        ax.legend()
        plot_idx += 1

    # Plot 4: Learning Rate (if available)
    if history['learning_rates'] and plot_idx < 4:
        ax = axes[plot_idx]
        ax.plot(epochs[:len(history['learning_rates'])], history['learning_rates'],
               'purple', linewidth=2, marker='^', markersize=3, label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.legend()
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Training curves saved to: {save_path}")

    return fig


def plot_combined_metrics(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    show_best: bool = True
) -> plt.Figure:
    """
    Plot both metrics on the same figure with dual y-axes.

    Args:
        history: Training history dictionary from load_training_history()
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figures
        show_best: Whether to highlight best performance

    Returns:
        matplotlib Figure object
    """
    val_miou = history['val_miou']
    val_pixacc = history['val_pixacc']

    if not val_miou and not val_pixacc:
        raise ValueError("No validation metrics found in training history")

    # Create epochs array
    epochs = list(range(1, max(len(val_miou), len(val_pixacc)) + 1))

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize, facecolor='white')

    # Plot mIoU on primary axis
    if val_miou:
        color1 = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean IoU', color=color1)
        line1 = ax1.plot(epochs[:len(val_miou)], val_miou, color=color1, linewidth=2,
                        marker='o', markersize=4, label='Validation mIoU')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Highlight best mIoU
        if show_best:
            best_epoch = np.argmax(val_miou) + 1
            best_miou = max(val_miou)
            ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax1.plot(best_epoch, best_miou, 'ro', markersize=8)

    # Plot Pixel Accuracy on secondary axis
    if val_pixacc:
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Pixel Accuracy', color=color2)
        line2 = ax2.plot(epochs[:len(val_pixacc)], val_pixacc, color=color2, linewidth=2,
                        marker='s', markersize=4, label='Validation Pixel Acc')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 1)

        # Highlight best pixel accuracy
        if show_best:
            best_epoch = np.argmax(val_pixacc) + 1
            best_pixacc = max(val_pixacc)
            ax2.plot(best_epoch, best_pixacc, 'ro', markersize=8)

    # Add title and legend
    plt.title('Training Progress: Validation Metrics')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    if val_pixacc:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='lower right')
    else:
        ax1.legend(loc='lower right')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Combined metrics plot saved to: {save_path}")

    return fig


def print_training_summary(history: Dict) -> None:
    """
    Print a summary of training progress.

    Args:
        history: Training history dictionary from load_training_history()
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    final_epoch = history['final_epoch']
    print(f"Final Epoch: {final_epoch}")

    # Training metrics summary
    if history['train_loss']:
        final_train_loss = history['train_loss'][-1]
        min_train_loss = min(history['train_loss'])
        min_loss_epoch = history['train_loss'].index(min_train_loss) + 1
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Min Training Loss: {min_train_loss:.4f} (epoch {min_loss_epoch})")

    # Validation loss summary
    if history['val_loss']:
        final_val_loss = history['val_loss'][-1]
        min_val_loss = min(history['val_loss'])
        min_val_loss_epoch = history['val_loss'].index(min_val_loss) + 1
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Min Validation Loss: {min_val_loss:.4f} (epoch {min_val_loss_epoch})")

    # Validation metrics summary
    if history['val_miou']:
        best_miou = max(history['val_miou'])
        best_miou_epoch = np.argmax(history['val_miou']) + 1
        final_miou = history['val_miou'][-1]
        print(f"Best Val mIoU: {best_miou:.4f} (epoch {best_miou_epoch})")
        print(f"Final Val mIoU: {final_miou:.4f}")

    if history['val_pixacc']:
        best_pixacc = max(history['val_pixacc'])
        best_pixacc_epoch = np.argmax(history['val_pixacc']) + 1
        final_pixacc = history['val_pixacc'][-1]
        print(f"Best Val Pixel Accuracy: {best_pixacc:.4f} (epoch {best_pixacc_epoch})")
        print(f"Final Val Pixel Accuracy: {final_pixacc:.4f}")

    # Training vs Validation comparison (overfitting detection)
    if history['train_miou'] and history['val_miou'] and len(history['train_miou']) > 0:
        final_train_miou = history['train_miou'][-1]
        final_val_miou = history['val_miou'][-1]
        overfitting_gap = final_train_miou - final_val_miou
        print(f"Train-Val mIoU Gap: {overfitting_gap:.4f} {'(possible overfitting)' if overfitting_gap > 0.1 else ''}")

    # Print configuration if available
    if history['config']:
        config = history['config']
        print("\nTraining Configuration:")
        print(f"  Epochs: {getattr(config, 'epochs', 'N/A')}")
        print(f"  Batch Size: {getattr(config, 'batch_size', 'N/A')}")
        print(f"  Learning Rate: {getattr(config, 'lr', 'N/A')}")
        print(f"  Use RGB: {getattr(config, 'use_rgb', 'N/A')}")
        print(f"  Use Depth: {getattr(config, 'use_depth', 'N/A')}")

    print("="*60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Visualize training history from PyTorch checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file (.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
        help="Output directory for plots (default: ./plots)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "jpg"],
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plots (default: 300)"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[12, 5],
        metavar=("W", "H"),
        help="Figure size in inches (default: 12 5)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (only save)"
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only create combined metrics plot"
    )

    args = parser.parse_args()

    try:
        # Load training history
        print(f"Loading training history from: {args.checkpoint}")
        history = load_training_history(args.checkpoint)

        # Print summary
        print_training_summary(history)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate base filename
        checkpoint_name = Path(args.checkpoint).stem

        if args.combined_only:
            # Create combined plot only
            save_path = output_dir / f"{checkpoint_name}_combined.{args.format}"
            fig = plot_combined_metrics(
                history,
                save_path=str(save_path),
                figsize=tuple(args.figsize),
                dpi=args.dpi
            )

            if not args.no_show:
                plt.show()
        else:
            # Create separate plots
            save_path = output_dir / f"{checkpoint_name}_curves.{args.format}"
            fig1 = plot_training_curves(
                history,
                save_path=str(save_path),
                figsize=tuple(args.figsize),
                dpi=args.dpi
            )

            # Create combined plot
            save_path = output_dir / f"{checkpoint_name}_combined.{args.format}"
            fig2 = plot_combined_metrics(
                history,
                save_path=str(save_path),
                figsize=(10, 6),
                dpi=args.dpi
            )

            if not args.no_show:
                plt.show()

        print(f"\n✓ Visualization complete! Plots saved in: {output_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())