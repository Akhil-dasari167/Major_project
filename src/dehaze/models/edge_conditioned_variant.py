"""Edge-Conditioned Head Model Variant for Stage 3.8.

Extracts parameter-free Sobel gradient magnitude from input luminance and
concatenates it as a 4th channel to condition early feature representation on edges.
"""
import torch
from torch import nn
import torch.nn.functional as F

from dehaze.models.residual_scale_variant import ResidualBlock


class EdgeConditionedVariant(nn.Module):
    """Residual CNN with edge-conditioned 4-channel input head."""
    def __init__(self, residual_scale: float = 0.2, base_channels: int = 32, residual_blocks: int = 4):
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        
        # Fixed Sobel kernels for parameter-free edge extraction
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        
        # Head modified to accept 4 channels: RGB (3) + Sobel Magnitude (1)
        self.head = nn.Sequential(
            nn.Conv2d(4, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(residual_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, 3, padding=1)

    def compute_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge magnitude from RGB input luminance in [0, 1]."""
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        if self.kx.device != x.device or self.kx.dtype != x.dtype:
            self.kx = self.kx.to(device=x.device, dtype=x.dtype)
            self.ky = self.ky.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        if hazy.dim() != 4 or hazy.size(1) != 3:
            raise ValueError(f"Expected 4D tensor with 3 channels, got shape {hazy.shape}")
        edge_map = self.compute_edge_map(hazy)
        x_in = torch.cat([hazy, edge_map], dim=1)
        feat = self.head(x_in)
        feat = self.blocks(feat)
        residual = torch.tanh(self.tail(feat)) * self.residual_scale
        return (hazy + residual).clamp(0.0, 1.0)
