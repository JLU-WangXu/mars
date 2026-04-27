"""
Feature Pyramid Network for Multi-scale Feature Fusion

This module implements a Feature Pyramid Network (FPN) architecture for
combining semantic information from deep layers with spatial details
from shallow layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class ConvBlock(nn.Module):
    """Convolution block with batch normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class LateralConnection(nn.Module):
    """Lateral connection module for FPN to extract features from backbone."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.align_conv = ConvBlock(out_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lateral = self.lateral_conv(x)
        return self.align_conv(lateral)


class FeaturePyramidBlock(nn.Module):
    """Single pyramid block with top-down pathway and lateral connection."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.lateral_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.output_conv = ConvBlock(out_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Upsampled features from higher pyramid level (top-down)
            lateral: Features from lateral connection (bottom-up)
        Returns:
            Fused features for this pyramid level
        """
        if x is not None:
            h, w = lateral.shape[2:]
            x = F.interpolate(x, size=(h, w), mode='nearest')
            fused = x + self.lateral_conv(lateral)
        else:
            fused = self.lateral_conv(lateral)
        return self.output_conv(fused)


class SpatialAttention(nn.Module):
    """Spatial attention module for balancing semantic and spatial information."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module for emphasizing important feature channels."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        return x * attention


class SemanticSpatialBalance(nn.Module):
    """
    Balances semantic and spatial information across pyramid levels.
    Deep levels have stronger semantic attention, shallow levels preserve spatial details.
    """

    def __init__(self, channels: int, level: int, total_levels: int):
        super().__init__()
        self.level = level
        self.total_levels = total_levels

        # Deeper levels (higher level index) get stronger channel attention (semantic)
        # Shallower levels get stronger spatial attention (spatial details)
        semantic_weight = level / max(1, total_levels - 1)
        spatial_weight = 1.0 - semantic_weight

        self.channel_attention = ChannelAttention(channels, reduction=16)
        self.spatial_attention = SpatialAttention(channels)

        self.semantic_weight = nn.Parameter(torch.tensor(semantic_weight))
        self.spatial_weight = nn.Parameter(torch.tensor(spatial_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_feat = self.channel_attention(x)
        spatial_feat = self.spatial_attention(x)
        # Weighted combination based on pyramid level
        w_s = torch.sigmoid(self.semantic_weight)
        w_sp = torch.sigmoid(self.spatial_weight)
        return w_s * channel_feat + w_sp * spatial_feat


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.

    Architecture:
    - Bottom-up pathway: Extract features from backbone network
    - Top-down pathway: Build pyramid from deep to shallow levels
    - Lateral connections: Combine features from both pathways
    - Balanced attention: Adaptively balance semantic and spatial information
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_levels: Optional[int] = None,
        extra_levels: int = 0,
        fusion_method: str = 'add'
    ):
        """
        Args:
            in_channels_list: List of input channels from backbone feature maps
            out_channels: Output channels for each pyramid level
            num_levels: Number of pyramid levels (defaults to len(in_channels_list))
            extra_levels: Number of extra pyramid levels to add (for detection heads)
            fusion_method: Feature fusion method ('add' or 'concat')
        """
        super().__init__()
        self.out_channels = out_channels
        self.num_levels = num_levels or len(in_channels_list)
        self.extra_levels = extra_levels
        self.fusion_method = fusion_method

        # Lateral connections for each input level
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(LateralConnection(in_ch, out_channels))

        # Top-down pathway blocks
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_levels):
            self.fpn_blocks.append(FeaturePyramidBlock(out_channels))

        # Semantic-spatial balance modules
        self.balance_modules = nn.ModuleList()
        total = self.num_levels + extra_levels
        for i in range(self.num_levels):
            self.balance_modules.append(SemanticSpatialBalance(out_channels, i, total))

        # Extra pyramid levels (P5 -> P6, P6 -> P7)
        self.extra_convs = nn.ModuleList()
        for _ in range(extra_levels):
            self.extra_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Output convolutions for each level
        self.output_convs = nn.ModuleList()
        for _ in range(self.num_levels + extra_levels):
            self.output_convs.append(ConvBlock(out_channels, out_channels, kernel_size=3))

    def forward(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through FPN.

        Args:
            backbone_features: List of feature maps from backbone [C2, C3, C4, C5, ...]

        Returns:
            List of pyramid feature maps [P2, P3, P4, P5, P6, P7]
        """
        assert len(backbone_features) >= self.num_levels, \
            f"Expected at least {self.num_levels} backbone features, got {len(backbone_features)}"

        # Extract lateral features
        laterals = []
        for i, feat in enumerate(backbone_features[:self.num_levels]):
            laterals.append(self.lateral_convs[i](feat))

        # Build pyramid (top-down pathway)
        pyramid_features = [None] * self.num_levels
        for i in range(self.num_levels - 1, -1, -1):
            if i == self.num_levels - 1:
                pyramid_features[i] = self.fpn_blocks[i](None, laterals[i])
            else:
                pyramid_features[i] = self.fpn_blocks[i](pyramid_features[i + 1], laterals[i])

        # Apply semantic-spatial balance
        balanced_features = []
        for i, feat in enumerate(pyramid_features):
            balanced = self.balance_modules[i](feat)
            balanced_features.append(self.output_convs[i](balanced))

        # Add extra pyramid levels
        for i in range(self.extra_levels):
            extra_feat = F.max_pool2d(balanced_features[-1], kernel_size=2, stride=2)
            balanced_features.append(self.output_convs[self.num_levels + i](extra_feat))

        return balanced_features


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module that combines features from different scales
    using learned fusion weights.
    """

    def __init__(
        self,
        channels: int,
        num_scales: int,
        fusion_type: str = 'adaptive'
    ):
        """
        Args:
            channels: Number of feature channels
            num_scales: Number of input scales to fuse
            fusion_type: Type of fusion ('adaptive', 'additive', 'concatenation')
        """
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.fusion_type = fusion_type

        if fusion_type == 'adaptive':
            # Learn fusion weights for each scale
            self.fusion_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.fusion_conv = ConvBlock(channels * num_scales, channels)
        elif fusion_type == 'additive':
            self.scale_convs = nn.ModuleList([
                ConvBlock(channels, channels) for _ in range(num_scales)
            ])
        elif fusion_type == 'concatenation':
            self.fusion_conv = nn.Sequential(
                ConvBlock(channels * num_scales, channels * 2),
                ConvBlock(channels * 2, channels)
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features.

        Args:
            features: List of feature maps at different scales

        Returns:
            Fused feature map
        """
        if not features:
            raise ValueError("Empty feature list")

        # Align all features to the largest resolution
        max_h = max(f.shape[2] for f in features)
        max_w = max(f.shape[3] for f in features)

        aligned_features = []
        for f in features:
            if f.shape[2] != max_h or f.shape[3] != max_w:
                f = F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False)
            aligned_features.append(f)

        if self.fusion_type == 'adaptive':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, aligned_features))
            fused = self.fusion_conv(torch.cat(aligned_features, dim=1))
            return fused
        elif self.fusion_type == 'additive':
            transformed = [self.scale_convs[i](f) for i, f in enumerate(aligned_features)]
            return sum(transformed) / len(transformed)
        elif self.fusion_type == 'concatenation':
            return self.fusion_conv(torch.cat(aligned_features, dim=1))


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for aggregating contextual information
    at multiple scales.
    """

    def __init__(self, in_channels: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        self.fusion = ConvBlock(in_channels * 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pool_features = []

        for stage in self.stages:
            pool_feat = stage(x)
            pool_feat = F.interpolate(pool_feat, size=(h, w), mode='bilinear', align_corners=False)
            pool_features.append(pool_feat)

        return self.fusion(torch.cat([x] + pool_features, dim=1))


class SemanticPyramid(nn.Module):
    """
    High-level semantic pyramid that enhances feature representations
    across multiple scales with progressive semantic refinement.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 4,
        semantic_dim: int = 256
    ):
        super().__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels

        # Semantic extraction branches
        self.semantic_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, semantic_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels)
        ])

        # Feature refinement
        self.refine_convs = nn.ModuleList([
            ConvBlock(in_channels + semantic_dim, out_channels)
            for _ in range(num_levels)
        ])

        # Level-specific attention
        self.level_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> List[torch.Tensor]:
        """
        Generate semantic pyramid features.

        Args:
            x: Input feature map
            target_size: Optional target spatial size for output

        Returns:
            List of refined feature maps at different scales
        """
        if target_size is None:
            target_size = (x.shape[2], x.shape[3])

        outputs = []
        for i in range(self.num_levels):
            # Extract semantic features
            semantic = self.semantic_branches[i](x)
            semantic = semantic.expand(-1, -1, x.shape[2], x.shape[3])

            # Concatenate and refine
            combined = torch.cat([x, semantic], dim=1)
            refined = self.refine_convs[i](combined)

            # Apply level-specific attention
            attention = self.level_attention[i](refined)
            refined = refined * attention

            # Resize to target size
            refined = F.interpolate(refined, size=target_size, mode='bilinear', align_corners=False)
            outputs.append(refined)

        return outputs


def build_feature_pyramid(
    backbone_features: List[torch.Tensor],
    out_channels: int = 256,
    extra_levels: int = 0,
    return_all_levels: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to build a feature pyramid from backbone features.

    Args:
        backbone_features: Feature maps from backbone network
        out_channels: Output channels per pyramid level
        extra_levels: Number of extra levels to add
        return_all_levels: If True, return all levels as dict

    Returns:
        Dictionary mapping level names to feature maps
    """
    num_levels = len(backbone_features)
    in_channels = [f.shape[1] for f in backbone_features]

    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels,
        out_channels=out_channels,
        num_levels=num_levels,
        extra_levels=extra_levels
    )

    pyramid_features = fpn(backbone_features)

    if return_all_levels:
        level_names = ['p2', 'p3', 'p4', 'p5'] + [f'p{6 + i}' for i in range(extra_levels)]
        return {name: feat for name, feat in zip(level_names, pyramid_features)}

    return {'p5': pyramid_features[0]}


class FeaturePyramidWithNeck(nn.Module):
    """
    Complete feature pyramid neck combining FPN and multi-scale fusion.
    Suitable for detection and segmentation heads.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: int = 5,
        pyramid_cfg: Optional[Dict] = None
    ):
        super().__init__()
        pyramid_cfg = pyramid_cfg or {}

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            num_levels=len(in_channels_list),
            extra_levels=max(0, num_outs - len(in_channels_list))
        )

        # Multi-scale fusion for final outputs
        self.msff = MultiScaleFeatureFusion(
            channels=out_channels,
            num_scales=min(3, len(in_channels_list)),
            fusion_type=pyramid_cfg.get('fusion_type', 'adaptive')
        )

        # Optional context module
        if pyramid_cfg.get('use_context', False):
            self.context = PyramidPoolingModule(out_channels)

    def forward(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        pyramid = self.fpn(backbone_features)

        if hasattr(self, 'context'):
            # Apply context module to highest level feature
            pyramid[-1] = self.context(pyramid[-1])

        return pyramid
