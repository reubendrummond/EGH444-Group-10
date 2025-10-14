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


# Global depth normalization constants for NYU Depth v2 dataset
# These values preserve global depth relationships across the dataset
class DepthNorm:
    """Global depth normalization constants for consistent depth processing."""

    MIN_DEPTH = 1.0  # Minimum valid depth value in dataset
    MAX_DEPTH = 255.0  # Maximum depth value (99.9th percentile to handle outliers)

    @classmethod
    def normalize_global(cls, depth: np.ndarray) -> np.ndarray:
        """
        Apply global depth normalization preserving relative depth relationships.

        Args:
            depth: Raw depth array with values in [0, 255] range

        Returns:
            Normalized depth array in [0, 1] range with global consistency
        """
        # Clip outliers and normalize to [0, 1]
        depth_clipped = np.clip(depth, cls.MIN_DEPTH, cls.MAX_DEPTH)
        depth_normalized = (depth_clipped - cls.MIN_DEPTH) / (
            cls.MAX_DEPTH - cls.MIN_DEPTH
        )

        # Set invalid pixels (originally 0) back to 0
        depth_normalized[depth == 0] = 0.0

        return depth_normalized.astype(np.float32)


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
        size_hw: Tuple[int, int],
        hflip_p: float = 0.5,
        label_mode: LabelMode = "nyu40",
        use_rgb: bool = True,
        use_depth: bool = False,
        depth_noise_p: float = 0.3,
        depth_noise_std: float = 0.02,
        depth_dropout_p: float = 0.1,
        depth_dropout_rate: float = 0.05,
        rotation_p: float = 0.3,
        rotation_angle: float = 15.0,
        crop_zoom_p: float = 0.3,
        crop_scale_range: Tuple[float, float] = (0.8, 1.0),
        color_jitter_p: float = 0.3,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.2,
        hue_range: float = 0.1,
    ) -> None:
        """
        Initialize training transform.

        Args:
            size_hw: Target size as (height, width)
            hflip_p: Probability of horizontal flip augmentation
            label_mode: Label processing mode ('nyu40' or 'raw')
            use_rgb: Whether to include RGB data in output
            use_depth: Whether to include depth data in output
            depth_noise_p: Probability of adding Gaussian noise to depth
            depth_noise_std: Standard deviation of depth noise
            depth_dropout_p: Probability of applying depth dropout
            depth_dropout_rate: Fraction of depth pixels to randomly zero out
            rotation_p: Probability of applying rotation augmentation
            rotation_angle: Maximum rotation angle in degrees (±rotation_angle)
            crop_zoom_p: Probability of applying random crop with zoom
            crop_scale_range: Scale range for random crop (min_scale, max_scale)
            color_jitter_p: Probability of applying color jittering
            brightness_range: Range for brightness adjustment (±brightness_range)
            contrast_range: Range for contrast adjustment (±contrast_range)
            saturation_range: Range for saturation adjustment (±saturation_range)
            hue_range: Range for hue adjustment (±hue_range)
        """
        self.H, self.W = size_hw
        self.hflip_p = hflip_p
        self.label_mode = label_mode
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.depth_noise_p = depth_noise_p
        self.depth_noise_std = depth_noise_std
        self.depth_dropout_p = depth_dropout_p
        self.depth_dropout_rate = depth_dropout_rate
        self.rotation_p = rotation_p
        self.rotation_angle = rotation_angle
        self.crop_zoom_p = crop_zoom_p
        self.crop_scale_range = crop_scale_range
        self.color_jitter_p = color_jitter_p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of use_rgb or use_depth must be True")

    def _augment_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply depth-specific augmentations.

        Args:
            depth: Depth array in [0, 1] range (after global normalization)

        Returns:
            Augmented depth array
        """
        # Create mask for valid depth pixels
        valid_mask = depth > 0

        # Gaussian noise augmentation
        if random.random() < self.depth_noise_p and valid_mask.any():
            noise = np.random.normal(0, self.depth_noise_std, depth.shape).astype(
                np.float32
            )
            depth = depth + noise
            # Clamp to valid range and preserve invalid pixels
            depth = np.clip(depth, 0, 1)
            depth[~valid_mask] = 0

        # Depth dropout augmentation (simulate sensor holes/failures)
        if random.random() < self.depth_dropout_p and valid_mask.any():
            dropout_mask = np.random.random(depth.shape) < self.depth_dropout_rate
            # Only drop out valid pixels
            dropout_mask = dropout_mask & valid_mask
            depth[dropout_mask] = 0

        return depth

    def _apply_spatial_augmentations(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply spatial augmentations to RGB, depth, and mask simultaneously.

        Args:
            rgb: RGB image array (H, W, 3)
            depth: Depth array (H, W)
            mask: Mask array (H, W)

        Returns:
            Tuple of augmented (rgb, depth, mask)
        """
        # Random crop with zoom
        if random.random() < self.crop_zoom_p:
            # Get original dimensions
            orig_h, orig_w = rgb.shape[:2]

            # Random crop scale
            scale = random.uniform(self.crop_scale_range[0], self.crop_scale_range[1])
            crop_w = int(orig_w * scale)
            crop_h = int(orig_h * scale)

            # Random crop position
            max_x = orig_w - crop_w
            max_y = orig_h - crop_h
            x = random.randint(0, max(0, max_x))
            y = random.randint(0, max(0, max_y))

            # Crop all components
            rgb = rgb[y:y+crop_h, x:x+crop_w]
            depth = depth[y:y+crop_h, x:x+crop_w]
            mask = mask[y:y+crop_h, x:x+crop_w]

        # Random rotation
        if random.random() < self.rotation_p:
            # Random rotation angle
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)

            # Get image center and rotation matrix
            center = (rgb.shape[1] // 2, rgb.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation to all components
            rgb = cv2.warpAffine(rgb, rotation_matrix, (rgb.shape[1], rgb.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            depth = cv2.warpAffine(depth, rotation_matrix, (depth.shape[1], depth.shape[0]),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return rgb, depth, mask

    def _apply_color_augmentations(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply color augmentations to RGB tensor.

        Args:
            rgb_tensor: RGB tensor in [0, 1] range with shape [C, H, W]

        Returns:
            Color-augmented RGB tensor
        """
        if random.random() < self.color_jitter_p:
            # Apply brightness adjustment
            if self.brightness_range > 0:
                brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
                rgb_tensor = TF.adjust_brightness(rgb_tensor, brightness_factor)

            # Apply contrast adjustment
            if self.contrast_range > 0:
                contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
                rgb_tensor = TF.adjust_contrast(rgb_tensor, contrast_factor)

            # Apply saturation adjustment
            if self.saturation_range > 0:
                saturation_factor = 1 + random.uniform(-self.saturation_range, self.saturation_range)
                rgb_tensor = TF.adjust_saturation(rgb_tensor, saturation_factor)

            # Apply hue adjustment
            if self.hue_range > 0:
                hue_factor = random.uniform(-self.hue_range, self.hue_range)
                rgb_tensor = TF.adjust_hue(rgb_tensor, hue_factor)

            # Clamp values to valid range
            rgb_tensor = torch.clamp(rgb_tensor, 0, 1)

        return rgb_tensor

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

        # Apply spatial augmentations first (on original resolution)
        rgb, depth, mask = self._apply_spatial_augmentations(rgb, depth, mask)

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
            # Convert RGB to tensor
            rgb_t = TF.to_tensor(rgb)
            # Apply color augmentations before normalization
            rgb_t = self._apply_color_augmentations(rgb_t)
            # Normalize with ImageNet stats
            rgb_t = ImageNetTransform.normalize(rgb_t)
            result["image"] = rgb_t

        if self.use_depth:
            # Global depth normalization preserving cross-scene relationships
            depth = DepthNorm.normalize_global(depth)
            # Apply depth-specific augmentations during training
            depth = self._augment_depth(depth)
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
            # Global depth normalization (same as training)
            depth = DepthNorm.normalize_global(depth)
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()
            result["depth"] = depth_t

        return result
