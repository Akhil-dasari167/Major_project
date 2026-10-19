"""Differentiable SSIM and composite L1+SSIM loss modules in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def create_window_2d(window_size: int, sigma: float, channel: int) -> torch.Tensor:
    k1d = gaussian_kernel_1d(window_size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    w = k2d.expand(channel, 1, window_size, window_size).contiguous()
    return w


class DifferentiableSSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, in_channels: int = 3, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = in_channels
        self.data_range = data_range
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2
        window = create_window_2d(window_size, sigma, in_channels)
        self.register_buffer("window", window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = self.window.to(device=img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        cs_map = (2.0 * sigma12 + self.c2) / (sigma1_sq + sigma2_sq + self.c2)
        ssim_map = ((2.0 * mu1_mu2 + self.c1) / (mu1_sq + mu2_sq + self.c1)) * cs_map

        return ssim_map.mean(dim=(1, 2, 3))


class L1SSIMLoss(nn.Module):
    """Composite loss: L_total = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    def __init__(self, alpha: float = 0.8, window_size: int = 11, sigma: float = 1.5, in_channels: int = 3):
        super().__init__()
        self.alpha = alpha
        self.ssim_module = DifferentiableSSIM(window_size=window_size, sigma=sigma, in_channels=in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target)
        ssim_val = self.ssim_module(pred, target).mean()
        loss_ssim = 1.0 - ssim_val
        total_loss = self.alpha * l1 + (1.0 - self.alpha) * loss_ssim
        return total_loss, {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "loss_ssim": loss_ssim.item(),
            "total": total_loss.item()
        }
