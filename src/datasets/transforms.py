"""
Data transforms for segmentation extracted from EGH444_Assessment2_Template.py
"""

import random
from typing import Dict, Any, Tuple, Literal

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.models import ResNet50_Weights


class ImageNetTransform:
    """ImageNet normalization and denormalization using official PyTorch constants."""

    # Class-level constants from official PyTorch ResNet50 weights
    _resnet_transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
    _mean = _resnet_transforms.mean
    _std = _resnet_transforms.std

    @classmethod
    def normalize(cls, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor with ImageNet mean and std.

        Args:
            tensor: Input tensor in range [0, 1] with shape [C, H, W] or [B, C, H, W]

        Returns:
            Normalized tensor
        """
        mean = torch.tensor(cls._mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(cls._std, dtype=tensor.dtype, device=tensor.device)

        if tensor.dim() == 3:  # [C, H, W]
            mean = mean[:, None, None]
            std = std[:, None, None]
        elif tensor.dim() == 4:  # [B, C, H, W]
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

        return (tensor - mean) / std

    @classmethod
    def denormalize(cls, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor from ImageNet normalization.

        Args:
            tensor: Normalized tensor with shape [C, H, W] or [B, C, H, W]

        Returns:
            Denormalized tensor in range [0, 1]
        """
        mean = torch.tensor(cls._mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(cls._std, dtype=tensor.dtype, device=tensor.device)

        if tensor.dim() == 3:  # [C, H, W]
            mean = mean[:, None, None]
            std = std[:, None, None]
        elif tensor.dim() == 4:  # [B, C, H, W]
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

        return (tensor * std + mean).clamp(0, 1)

    @classmethod
    def denormalize_for_display(cls, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor and convert to uint8 numpy array for visualization.

        Args:
            tensor: Normalized tensor with shape [C, H, W]

        Returns:
            Denormalized image as HWC uint8 numpy array
        """
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor [C, H, W], got {tensor.dim()}D")

        denorm_tensor = cls.denormalize(tensor)
        img = denorm_tensor.permute(1, 2, 0).cpu().numpy()
        return (img * 255).astype(np.uint8)

# Type alias for label modes
LabelMode = Literal["nyu40", "raw"]


def to_nyu40_ids(mask_np: np.ndarray) -> np.ndarray:
    """
    Ensure mask uses NYUv2-40 label space [0..39].
    Any pixel outside [0..39] is set to 255 (ignore).
    If your masks are already 0..39, this is a no-op; if not, out-of-range is safely ignored.
    """
    out = mask_np.astype(np.int64, copy=True)
    bad = (out < 0) | (out > 39)
    out[bad] = 255
    return out


class SegTrainTransform:
    """Training transform with data augmentation for segmentation."""

    def __init__(
        self,
        size_hw: Tuple[int, int] = (240, 320),
        hflip_p: float = 0.5,
        label_mode: LabelMode = "nyu40",
        use_rgb: bool = True,
        use_depth: bool = False,
    ) -> None:
        """
        Initialize training transform.

        Args:
            size_hw: Target size as (height, width)
            hflip_p: Probability of horizontal flip augmentation
            label_mode: Label processing mode ('nyu40' or 'raw')
            use_rgb: Whether to include RGB data in output
            use_depth: Whether to include depth data in output
        """
        self.H, self.W = size_hw
        self.hflip_p = hflip_p
        self.label_mode = label_mode
        self.use_rgb = use_rgb
        self.use_depth = use_depth

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of use_rgb or use_depth must be True")

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply training transforms to a sample.

        Args:
            sample: Dictionary with keys 'rgb', 'depth', 'mask', 'id'

        Returns:
            Transformed sample with tensor data
        """
        rgb, depth, mask, sid = (
            sample["rgb"],
            sample["depth"],
            sample["mask"],
            sample["id"],
        )

        # Resize all components
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # Convert to NYUv2-40 format if needed
        if self.label_mode == "nyu40":
            mask = to_nyu40_ids(mask)

        # Random horizontal flip
        if random.random() < self.hflip_p:
            rgb = np.ascontiguousarray(np.flip(rgb, axis=1))
            depth = np.ascontiguousarray(np.flip(depth, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))

        # Convert mask to tensor (always needed)
        mask_t = torch.from_numpy(mask).long()

        # Build output dictionary based on modality flags
        result = {"mask": mask_t, "id": sid}

        if self.use_rgb:
            # Convert RGB to tensor and normalize with ImageNet stats
            rgb_t = ImageNetTransform.normalize(TF.to_tensor(rgb))
            result["image"] = rgb_t

        if self.use_depth:
            # Simple per-image depth normalization to [0,1]
            valid = depth > 0
            if valid.any():
                dmin, dmax = float(depth[valid].min()), float(depth[valid].max())
                depth = (depth - dmin) / max(dmax - dmin, 1e-6)
            else:
                depth[:] = 0.0
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()
            result["depth"] = depth_t

        return result


class SegEvalTransform:
    """Evaluation transform without augmentation for segmentation."""

    def __init__(
        self,
        size_hw: Tuple[int, int] = (240, 320),
        label_mode: LabelMode = "nyu40",
        use_rgb: bool = True,
        use_depth: bool = False,
    ) -> None:
        """
        Initialize evaluation transform.

        Args:
            size_hw: Target size as (height, width)
            label_mode: Label processing mode ('nyu40' or 'raw')
            use_rgb: Whether to include RGB data in output
            use_depth: Whether to include depth data in output
        """
        self.H, self.W = size_hw
        self.label_mode = label_mode
        self.use_rgb = use_rgb
        self.use_depth = use_depth

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of use_rgb or use_depth must be True")

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply evaluation transforms to a sample.

        Args:
            sample: Dictionary with keys 'rgb', 'depth', 'mask', 'id'

        Returns:
            Transformed sample with tensor data
        """
        rgb, depth, mask, sid = (
            sample["rgb"],
            sample["depth"],
            sample["mask"],
            sample["id"],
        )

        # Resize all components (no augmentation for eval)
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # Convert to NYUv2-40 format if needed
        if self.label_mode == "nyu40":
            mask = to_nyu40_ids(mask)

        # Convert mask to tensor (always needed)
        mask_t = torch.from_numpy(mask).long()

        # Build output dictionary based on modality flags
        result = {"mask": mask_t, "id": sid}

        if self.use_rgb:
            # Convert RGB to tensor and normalize with ImageNet stats
            rgb_t = ImageNetTransform.normalize(TF.to_tensor(rgb))
            result["image"] = rgb_t

        if self.use_depth:
            # Depth normalization (same as training)
            valid = depth > 0
            if valid.any():
                dmin, dmax = float(depth[valid].min()), float(depth[valid].max())
                depth = (depth - dmin) / max(dmax - dmin, 1e-6)
            else:
                depth[:] = 0.0
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()
            result["depth"] = depth_t

        return result
