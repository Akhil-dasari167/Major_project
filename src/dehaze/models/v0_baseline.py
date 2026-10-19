"""V0: lightweight residual CNN baseline for paired image dehazing."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__(); self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.relu(x + self.body(x))


class V0ResidualCNN(nn.Module):
    """Predicts a bounded residual correction and adds it to the hazy input."""
    def __init__(self, base_channels: int = 32, residual_blocks: int = 4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, base_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(residual_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, 3, padding=1)
    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.tail(self.blocks(self.head(hazy)))) * 0.1
        return (hazy + residual).clamp(0.0, 1.0)


def parameter_count(model: nn.Module) -> int: return sum(p.numel() for p in model.parameters() if p.requires_grad)
