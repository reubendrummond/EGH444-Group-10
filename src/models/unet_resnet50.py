"""
U-Net with ResNet50 encoder for semantic segmentation.

Implementation based on:
- U-Net: https://arxiv.org/abs/1505.04597
- ResNet encoder: https://www.researchgate.net/publication/351575653_Building_segmentation_from_satellite_imagery_using_U-Net_with_ResNet_encoder
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Double convolution block used in U-Net decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize ConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution block."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection for U-Net decoder."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """
        Initialize UpBlock.

        Args:
            in_channels: Number of input channels from previous layer
            skip_channels: Number of channels from skip connection
            out_channels: Number of output channels
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(
            in_channels // 2 + skip_channels, out_channels
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with upsampling and skip connection.

        Args:
            x: Input tensor from previous decoder layer
            skip: Skip connection tensor from encoder

        Returns:
            Upsampled and processed tensor
        """
        x = self.upsample(x)

        # Handle size mismatch between upsampled and skip features
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class ResNet50Encoder(nn.Module):
    """ResNet50 encoder for feature extraction."""

    def __init__(self, pretrained: bool = True, freeze: bool = False) -> None:
        """
        Initialize ResNet50 encoder.

        Args:
            pretrained: Whether to use ImageNet pretrained weights (strict - will fail if unavailable)
            freeze: Whether to freeze encoder parameters (no gradient updates)
        """
        super().__init__()

        # Load ResNet50 - strict pretrained loading
        if pretrained:
            print("Loading ImageNet pretrained ResNet50 weights...")
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("✓ Successfully loaded ImageNet pretrained ResNet50 weights")
        else:
            resnet = models.resnet50(weights=None)
            print("Using ResNet50 with random initialization")

        # Extract layers for feature pyramid
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels, 1/4 resolution
        self.layer2 = resnet.layer2  # 512 channels, 1/8 resolution
        self.layer3 = resnet.layer3  # 1024 channels, 1/16 resolution
        self.layer4 = resnet.layer4  # 2048 channels, 1/32 resolution

        # Freeze encoder parameters if requested
        if freeze:
            self.freeze_encoder()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input RGB tensor [B, 3, H, W]

        Returns:
            List of feature maps at different scales
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = x  # 64 channels, 1/2 resolution

        x = self.maxpool(x)
        c1 = self.layer1(x)  # 256 channels, 1/4 resolution
        c2 = self.layer2(c1)  # 512 channels, 1/8 resolution
        c3 = self.layer3(c2)  # 1024 channels, 1/16 resolution
        c4 = self.layer4(c3)  # 2048 channels, 1/32 resolution

        return [c0, c1, c2, c3, c4]

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters to prevent gradient updates."""
        for param in self.parameters():
            param.requires_grad = False
        print("Encoder frozen: parameters will not be updated during training")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters to allow gradient updates."""
        for param in self.parameters():
            param.requires_grad = True
        print("Encoder unfrozen: parameters will be updated during training")


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections."""

    def __init__(self, encoder_channels: List[int], num_classes: int) -> None:
        """
        Initialize U-Net decoder.

        Args:
            encoder_channels: List of encoder channel counts [c0, c1, c2, c3, c4]
            num_classes: Number of output classes
        """
        super().__init__()

        # Decoder layers with skip connections
        self.up4 = UpBlock(encoder_channels[4], encoder_channels[3], 512)  # 2048 + 1024 -> 512
        self.up3 = UpBlock(512, encoder_channels[2], 256)                  # 512 + 512 -> 256
        self.up2 = UpBlock(256, encoder_channels[1], 128)                  # 256 + 256 -> 128
        self.up1 = UpBlock(128, encoder_channels[0], 64)                   # 128 + 64 -> 64

        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            features: List of encoder features [c0, c1, c2, c3, c4]

        Returns:
            Segmentation output tensor
        """
        c0, c1, c2, c3, c4 = features

        # Upsampling path with skip connections
        x = self.up4(c4, c3)  # 1/16 resolution
        x = self.up3(x, c2)   # 1/8 resolution
        x = self.up2(x, c1)   # 1/4 resolution
        x = self.up1(x, c0)   # 1/2 resolution

        # Final upsampling to original resolution
        x = self.final_up(x)  # Full resolution
        x = self.final_conv(x)

        return x


class UNetResNet50(nn.Module):
    """Complete U-Net with ResNet50 encoder for semantic segmentation."""

    def __init__(self, num_classes: int = 40, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        """
        Initialize U-Net with ResNet50 backbone.

        Args:
            num_classes: Number of segmentation classes
            pretrained: Whether to use ImageNet pretrained ResNet50 (strict - will fail if unavailable)
            freeze_backbone: Whether to freeze backbone parameters for fine-tuning
        """
        super().__init__()

        self.num_classes = num_classes

        # Initialize encoder and decoder
        self.encoder = ResNet50Encoder(pretrained=pretrained, freeze=freeze_backbone)

        # ResNet50 channel sizes: [64, 256, 512, 1024, 2048]
        encoder_channels = [64, 256, 512, 1024, 2048]
        self.decoder = UNetDecoder(encoder_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through U-Net.

        Args:
            x: Input RGB tensor [B, 3, H, W]

        Returns:
            Dictionary with 'out' key containing segmentation logits [B, num_classes, H, W]
        """
        # Extract multi-scale features
        features = self.encoder(x)

        # Decode with skip connections
        output = self.decoder(features)

        return {"out": output}

    def freeze_backbone(self) -> None:
        """Freeze backbone encoder parameters."""
        self.encoder.freeze_encoder()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone encoder parameters."""
        self.encoder.unfreeze_encoder()

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet_resnet50(num_classes: int, pretrained_backbone: bool = True, freeze_backbone: bool = False) -> UNetResNet50:
    """
    Build U-Net with ResNet50 encoder.

    Args:
        num_classes: Number of segmentation classes
        pretrained_backbone: Whether to use ImageNet pretrained backbone (strict - will fail if unavailable)
        freeze_backbone: Whether to freeze backbone parameters for fine-tuning

    Returns:
        UNetResNet50 model
    """
    return UNetResNet50(num_classes=num_classes, pretrained=pretrained_backbone, freeze_backbone=freeze_backbone)