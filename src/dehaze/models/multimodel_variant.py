"""Lightweight multi-model feature-fusion dehazing architecture.

All tensors use RGB values in ``[0, 1]`` and ``(N, C, H, W)`` layout.  The
atmospheric-context and layer-separation paths are *learned feature
approximations*: they do not estimate physically calibrated transmission,
airlight, or intrinsic image layers.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    """A compact residual feature block without normalization layers."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x), inplace=True)


class MultiScaleFeatureExtractor(nn.Module):
    """Produces full-, half-, and quarter-resolution feature tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.full = ResidualUnit(channels)
        self.down_half = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU(inplace=True), ResidualUnit(channels))
        self.down_quarter = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU(inplace=True), ResidualUnit(channels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.full(x)
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        return full, half, quarter


class AdaptiveAttentionFeatureFusion(nn.Module):
    """Learns global scale weights and fuses all scales at full resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels, 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, full: torch.Tensor, half: torch.Tensor, quarter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = full.shape[-2:]
        aligned = (
            full,
            F.interpolate(half, size=size, mode="bilinear", align_corners=False),
            F.interpolate(quarter, size=size, mode="bilinear", align_corners=False),
        )
        weights = torch.softmax(self.attention(torch.cat(aligned, dim=1)), dim=1)
        fused = sum(weights[:, i : i + 1] * feature for i, feature in enumerate(aligned))
        return self.project(fused), weights


class AtmosphericContextBranch(nn.Module):
    """Learns low-frequency atmospheric-context features, not physical airlight."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            ResidualUnit(channels, dilation=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        return self.mix(torch.cat([x, smooth], dim=1))


class LayerSeparationBranch(nn.Module):
    """Learns base/detail features using a local smoothing approximation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.ReLU(inplace=True), ResidualUnit(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = x - base
        return self.mix(torch.cat([base, detail], dim=1))


class ContextualFeatureBranch(nn.Module):
    """Uses dilated residual convolutions for a broader learned context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(ResidualUnit(channels, dilation=2), ResidualUnit(channels, dilation=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AdaptiveMultiModelFusion(nn.Module):
    """Pixel-wise softmax fusion across the three learned branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 3, channels, 1), nn.ReLU(inplace=True), nn.Conv2d(channels, 3, 1))
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, atmospheric: torch.Tensor, layer: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = (atmospheric, layer, context)
        weights = torch.softmax(self.gate(torch.cat(features, dim=1)), dim=1)
        fused = sum(weights[:, i : i + 1] * feature for i, feature in enumerate(features))
        return self.project(fused), weights


class EdgeAwareRefinement(nn.Module):
    """Applies learned feature refinement gated by fixed Sobel edge magnitude."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("kx", torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer("ky", torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.gate = nn.Sequential(nn.Conv2d(channels + 1, channels, 1), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1))

    def edge_map(self, rgb: torch.Tensor) -> torch.Tensor:
        luminance = 0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]
        gx = F.conv2d(luminance, self.kx.to(dtype=rgb.dtype), padding=1)
        gy = F.conv2d(luminance, self.ky.to(dtype=rgb.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    def forward(self, features: torch.Tensor, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge = self.edge_map(rgb)
        gate = self.gate(torch.cat([features, edge], dim=1))
        return F.relu(features + gate * self.refine(features), inplace=True), edge


class ColorTextureRefinement(nn.Module):
    """Separates channel mixing (colour) from local detail (texture) features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.color = nn.Conv2d(channels, channels, 1)
        self.texture = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.color(x) + self.texture(x), inplace=True)


class MultiModelDehazeVariant(nn.Module):
    """Lightweight multi-scale, edge-gated residual image-dehazing model."""

    def __init__(self, residual_scale: float = 0.2, base_channels: int = 24) -> None:
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        self.residual_scale = float(residual_scale)
        self.base_channels = int(base_channels)
        self.stem = nn.Sequential(nn.Conv2d(3, base_channels, 3, padding=1), nn.ReLU(inplace=True), ResidualUnit(base_channels))
        self.multiscale = MultiScaleFeatureExtractor(base_channels)
        self.aafm = AdaptiveAttentionFeatureFusion(base_channels)
        self.atmospheric = AtmosphericContextBranch(base_channels)
        self.layer_separation = LayerSeparationBranch(base_channels)
        self.contextual = ContextualFeatureBranch(base_channels)
        self.branch_fusion = AdaptiveMultiModelFusion(base_channels)
        self.edge_refinement = EdgeAwareRefinement(base_channels)
        self.color_texture = ColorTextureRefinement(base_channels)
        self.reconstruction = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        if hazy.ndim != 4 or hazy.shape[1] != 3:
            raise ValueError(f"Expected RGB tensor with shape (N, 3, H, W), got {tuple(hazy.shape)}")
        features = self.stem(hazy)
        scales = self.multiscale(features)
        fused_scales, _ = self.aafm(*scales)
        atmospheric = self.atmospheric(fused_scales)
        layer = self.layer_separation(fused_scales)
        context = self.contextual(fused_scales)
        fused_branches, _ = self.branch_fusion(atmospheric, layer, context)
        refined, _ = self.edge_refinement(fused_branches, hazy)
        refined = self.color_texture(refined)
        residual = torch.tanh(self.reconstruction(refined)) * self.residual_scale
        return (hazy + residual).clamp(0.0, 1.0)

