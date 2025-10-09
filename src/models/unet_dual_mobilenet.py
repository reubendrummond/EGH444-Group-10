"""
FuseNet-style Dual-Encoder U-Net with MobileNetV3 encoders for RGB-D semantic segmentation.

Implementation based on:
- FuseNet: https://link.springer.com/chapter/10.1007/978-3-319-54181-5_14
- U-Net: https://arxiv.org/abs/1505.04597
- MobileNetV3: https://arxiv.org/abs/1905.02244

This implements a dual-encoder architecture where:
- RGB encoder: MobileNetV3 for semantic features
- Depth encoder: MobileNetV3 adapted for single-channel depth input
- Fusion strategy: FuseNet approach - depth features fused into RGB features at each level
"""
from typing import Dict, List, Literal

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


class FusionBlock(nn.Module):
    """FuseNet-style fusion block to combine RGB and depth features."""

    def __init__(self, rgb_channels: int, depth_channels: int, out_channels: int) -> None:
        """
        Initialize FusionBlock.

        Args:
            rgb_channels: Number of RGB feature channels
            depth_channels: Number of depth feature channels
            out_channels: Number of output channels after fusion
        """
        super().__init__()
        total_channels = rgb_channels + depth_channels

        # Fusion convolution to combine and reduce channels
        self.fusion_conv = nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Optional: Add attention mechanism for better fusion
        self.attention = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 4, total_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and depth features.

        Args:
            rgb_feat: RGB features [B, rgb_channels, H, W]
            depth_feat: Depth features [B, depth_channels, H, W]

        Returns:
            Fused features [B, out_channels, H, W]
        """
        # Ensure spatial dimensions match
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            depth_feat = F.interpolate(depth_feat, size=rgb_feat.shape[-2:],
                                     mode='bilinear', align_corners=False)

        # Concatenate features
        combined = torch.cat([rgb_feat, depth_feat], dim=1)

        # Apply attention weighting
        attention_weights = self.attention(combined)
        combined = combined * attention_weights

        # Fusion convolution
        fused = self.fusion_conv(combined)
        fused = self.relu(self.bn(fused))

        return fused


class MobileNetV3DepthEncoder(nn.Module):
    """MobileNetV3 encoder adapted for single-channel depth input."""

    def __init__(self, variant: Literal['small', 'large'] = 'small', pretrained: bool = True, freeze: bool = False) -> None:
        """
        Initialize MobileNetV3 depth encoder.

        Args:
            variant: MobileNetV3 variant ('small' or 'large')
            pretrained: Whether to use ImageNet pretrained weights
            freeze: Whether to freeze encoder parameters
        """
        super().__init__()

        self.variant = variant

        # Load MobileNetV3
        if variant == 'small':
            if pretrained:
                print("Loading ImageNet pretrained MobileNetV3-Small weights for depth encoder...")
                mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                print("✓ Successfully loaded ImageNet pretrained MobileNetV3-Small weights")
            else:
                mobilenet = models.mobilenet_v3_small(weights=None)
                print("Using MobileNetV3-Small with random initialization for depth encoder")
        else:  # large
            if pretrained:
                print("Loading ImageNet pretrained MobileNetV3-Large weights for depth encoder...")
                mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                print("✓ Successfully loaded ImageNet pretrained MobileNetV3-Large weights")
            else:
                mobilenet = models.mobilenet_v3_large(weights=None)
                print("Using MobileNetV3-Large with random initialization for depth encoder")

        # Extract feature extraction layers
        self.features = mobilenet.features

        # Adapt first layer for single-channel depth input
        self._adapt_first_layer_for_depth(pretrained)

        # Define feature extraction points based on variant
        if variant == 'small':
            self.feature_indices = [0, 1, 3, 8, 11]
            self.feature_channels = [16, 16, 24, 48, 96]
        else:  # large
            self.feature_indices = [0, 2, 4, 9, 15]
            self.feature_channels = [16, 24, 40, 80, 160]

        # Freeze encoder parameters if requested
        if freeze:
            self.freeze_encoder()

    def _adapt_first_layer_for_depth(self, pretrained: bool) -> None:
        """Adapt the first convolution layer for single-channel depth input."""
        original_conv = self.features[0][0]  # First layer in the features

        # Create new conv layer with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        if pretrained:
            # Average the weights across RGB channels for depth initialization
            with torch.no_grad():
                # Original weight shape: [out_channels, 3, kernel_h, kernel_w]
                # New weight shape: [out_channels, 1, kernel_h, kernel_w]
                averaged_weights = original_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(averaged_weights)

                if original_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(original_conv.bias)

            print("✓ Adapted first conv layer for depth input using averaged RGB weights")

        # Replace the first conv layer
        self.features[0][0] = new_conv

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through depth encoder.

        Args:
            x: Input depth tensor [B, 1, H, W]

        Returns:
            List of feature maps at different scales
        """
        features = []

        # Extract features at specified indices
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Save features at extraction points
            if i in self.feature_indices:
                features.append(x)

        # Ensure we have 5 feature maps
        assert len(features) == 5, f"Expected 5 feature maps, got {len(features)}"

        return features

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters to prevent gradient updates."""
        for param in self.parameters():
            param.requires_grad = False
        print(f"MobileNetV3-{self.variant} depth encoder frozen: parameters will not be updated during training")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters to allow gradient updates."""
        for param in self.parameters():
            param.requires_grad = True
        print(f"MobileNetV3-{self.variant} depth encoder unfrozen: parameters will be updated during training")

    def get_feature_channels(self) -> List[int]:
        """Get the number of channels for each feature map."""
        return self.feature_channels


class DualEncoderDecoder(nn.Module):
    """U-Net decoder that handles fused features from dual encoders."""

    def __init__(self, fused_channels: List[int], num_classes: int) -> None:
        """
        Initialize dual-encoder decoder.

        Args:
            fused_channels: List of fused feature channel counts [c0, c1, c2, c3, c4]
            num_classes: Number of output classes
        """
        super().__init__()

        c0, c1, c2, c3, c4 = fused_channels

        # Decoder upsampling blocks
        self.up4 = self._make_up_block(c4, c3, 256)
        self.up3 = self._make_up_block(256, c2, 128)
        self.up2 = self._make_up_block(128, c1, 64)
        self.up1 = self._make_up_block(64, c0, 32)

        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def _make_up_block(self, in_channels: int, skip_channels: int, out_channels: int) -> nn.Sequential:
        """Create an upsampling block with skip connection."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            ConvBlock(in_channels // 2 + skip_channels, out_channels)
        )

    def forward(self, fused_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            fused_features: List of fused encoder features [c0, c1, c2, c3, c4]

        Returns:
            Segmentation output tensor
        """
        c0, c1, c2, c3, c4 = fused_features

        # Upsampling path with skip connections
        x = self.up4[0](c4)  # Upsample
        if x.shape[-2:] != c3.shape[-2:]:
            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.up4[1](x)  # Conv block

        x = self.up3[0](x)  # Upsample
        if x.shape[-2:] != c2.shape[-2:]:
            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.up3[1](x)  # Conv block

        x = self.up2[0](x)  # Upsample
        if x.shape[-2:] != c1.shape[-2:]:
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.up2[1](x)  # Conv block

        x = self.up1[0](x)  # Upsample
        if x.shape[-2:] != c0.shape[-2:]:
            x = F.interpolate(x, size=c0.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c0], dim=1)
        x = self.up1[1](x)  # Conv block

        # Final upsampling to original resolution
        x = self.final_up(x)
        x = self.final_conv(x)

        return x


class DualEncoderMobileNetUNet(nn.Module):
    """FuseNet-style Dual-Encoder U-Net with MobileNetV3 encoders for RGB-D semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 40,
        variant: Literal['small', 'large'] = 'small',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ) -> None:
        """
        Initialize Dual-Encoder U-Net with MobileNetV3 backbones.

        Args:
            num_classes: Number of segmentation classes
            variant: MobileNetV3 variant ('small' or 'large')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()

        self.num_classes = num_classes
        self.variant = variant

        # Import the existing MobileNetV3Encoder for RGB
        from .unet_mobilenet import MobileNetV3Encoder

        # RGB encoder (existing implementation)
        self.rgb_encoder = MobileNetV3Encoder(variant=variant, pretrained=pretrained, freeze=freeze_backbone)

        # Depth encoder (new implementation)
        self.depth_encoder = MobileNetV3DepthEncoder(variant=variant, pretrained=pretrained, freeze=freeze_backbone)

        # Get channel configurations
        rgb_channels = self.rgb_encoder.get_feature_channels()
        depth_channels = self.depth_encoder.get_feature_channels()

        # Create fusion blocks for each level
        fused_channels = []
        self.fusion_blocks = nn.ModuleList()

        for i, (rgb_ch, depth_ch) in enumerate(zip(rgb_channels, depth_channels)):
            # Fused channel count (can be customized per level)
            fused_ch = rgb_ch  # Keep same as RGB for simplicity
            fused_channels.append(fused_ch)

            fusion_block = FusionBlock(rgb_ch, depth_ch, fused_ch)
            self.fusion_blocks.append(fusion_block)

        # Decoder that handles fused features
        self.decoder = DualEncoderDecoder(fused_channels, num_classes)

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Dual-Encoder UNet-MobileNetV3-{variant.upper()} initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  RGB encoder channels: {rgb_channels}")
        print(f"  Depth encoder channels: {depth_channels}")
        print(f"  Fused channels: {fused_channels}")

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-encoder U-Net.

        Args:
            rgb: Input RGB tensor [B, 3, H, W]
            depth: Input depth tensor [B, 1, H, W]

        Returns:
            Dictionary with 'out' key containing segmentation logits [B, num_classes, H, W]
        """
        # Extract features from both encoders
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)

        # Fuse features at each level using FuseNet strategy
        fused_features = []
        for rgb_feat, depth_feat, fusion_block in zip(rgb_features, depth_features, self.fusion_blocks):
            fused_feat = fusion_block(rgb_feat, depth_feat)
            fused_features.append(fused_feat)

        # Decode with fused features
        output = self.decoder(fused_features)

        return {"out": output}

    def freeze_backbone(self) -> None:
        """Freeze both encoder backbones."""
        self.rgb_encoder.freeze_encoder()
        self.depth_encoder.freeze_encoder()

    def unfreeze_backbone(self) -> None:
        """Unfreeze both encoder backbones."""
        self.rgb_encoder.unfreeze_encoder()
        self.depth_encoder.unfreeze_encoder()

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_dual_encoder_mobilenet_unet(
    num_classes: int,
    variant: Literal['small', 'large'] = 'small',
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False
) -> DualEncoderMobileNetUNet:
    """
    Build FuseNet-style Dual-Encoder U-Net with MobileNetV3 encoders.

    Args:
        num_classes: Number of segmentation classes
        variant: MobileNetV3 variant ('small' ~16M params, 'large' ~30M params)
        pretrained_backbone: Whether to use ImageNet pretrained backbones
        freeze_backbone: Whether to freeze backbone parameters for fine-tuning

    Returns:
        DualEncoderMobileNetUNet model

    Example:
        >>> # Lightweight dual-encoder model (~16M parameters)
        >>> model = build_dual_encoder_mobilenet_unet(num_classes=40, variant='small')
        >>>
        >>> # Forward pass with RGB and depth
        >>> rgb = torch.randn(1, 3, 240, 320)
        >>> depth = torch.randn(1, 1, 240, 320)
        >>> output = model(rgb, depth)
        >>> print(output["out"].shape)  # [1, 40, 240, 320]
    """
    return DualEncoderMobileNetUNet(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained_backbone,
        freeze_backbone=freeze_backbone
    )