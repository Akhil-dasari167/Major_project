"""Controlled V0 residual-scale study model; V0 source remains unchanged."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.body(x))


class ResidualScaleVariant(nn.Module):
    """V0-equivalent CNN with only the output residual scale made explicit."""
    def __init__(self, residual_scale: float, base_channels: int = 32, residual_blocks: int = 4):
        super().__init__()
        if residual_scale <= 0: raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        self.head = nn.Sequential(nn.Conv2d(3, base_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(residual_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.tail(self.blocks(self.head(hazy)))) * self.residual_scale
        return (hazy + residual).clamp(0.0, 1.0)
