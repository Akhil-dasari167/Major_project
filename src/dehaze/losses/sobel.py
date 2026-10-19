"""Differentiable Sobel edge loss module in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """Computes L1 difference between predicted and target Sobel edge gradients."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        
        weight_x = kx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        weight_y = ky.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)
        self.in_channels = in_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        if self.weight_x.device != pred.device or self.weight_x.dtype != pred.dtype:
            self.weight_x = self.weight_x.to(device=pred.device, dtype=pred.dtype)
            self.weight_y = self.weight_y.to(device=pred.device, dtype=pred.dtype)
        
        gx_pred = F.conv2d(pred, self.weight_x, padding=1, groups=self.in_channels)
        gy_pred = F.conv2d(pred, self.weight_y, padding=1, groups=self.in_channels)
        
        gx_target = F.conv2d(target, self.weight_x, padding=1, groups=self.in_channels)
        gy_target = F.conv2d(target, self.weight_y, padding=1, groups=self.in_channels)
        
        loss_x = F.l1_loss(gx_pred, gx_target)
        loss_y = F.l1_loss(gy_pred, gy_target)
        return 0.5 * (loss_x + loss_y)


class L1SSIMSobelLoss(nn.Module):
    """Composite loss: L_total = w_l1 * L1 + w_ssim * (1 - SSIM) + w_sobel * L_Sobel."""
    def __init__(
        self,
        weight_l1: float = 0.8,
        weight_ssim: float = 0.15,
        weight_sobel: float = 0.05,
        window_size: int = 11,
        sigma: float = 1.5,
        in_channels: int = 3,
    ):
        super().__init__()
        from dehaze.losses.ssim import DifferentiableSSIM
        self.weight_l1 = weight_l1
        self.weight_ssim = weight_ssim
        self.weight_sobel = weight_sobel
        self.ssim_module = DifferentiableSSIM(window_size=window_size, sigma=sigma, in_channels=in_channels)
        self.sobel_module = SobelEdgeLoss(in_channels=in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target)
        ssim_val = self.ssim_module(pred, target).mean()
        loss_ssim = 1.0 - ssim_val
        loss_sobel = self.sobel_module(pred, target)
        
        total_loss = (
            self.weight_l1 * l1
            + self.weight_ssim * loss_ssim
            + self.weight_sobel * loss_sobel
        )
        return total_loss, {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_sobel": loss_sobel.item(),
            "total": total_loss.item(),
        }
