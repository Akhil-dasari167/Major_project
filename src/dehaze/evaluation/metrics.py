"""Full-reference image metrics for normalized RGB tensors."""
import math
import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction.detach().float().cpu() - target.detach().float().cpu()) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    truth = target.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return float(structural_similarity(truth, pred, channel_axis=2, data_range=1.0))
