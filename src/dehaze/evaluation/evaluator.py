"""Inference, metric, timing, parameter, and Conv2d-MAC utilities."""
import time
import torch
from torch import nn
from dehaze.evaluation.metrics import psnr, ssim


@torch.inference_mode()
def evaluate_loader(model: nn.Module, loader, device: torch.device) -> tuple[list[dict], float]:
    model.eval(); records, elapsed = [], 0.0
    for batch in loader:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); outputs = model(inputs)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        for i, image_id in enumerate(batch["id"]): records.append({"id": image_id, "psnr": psnr(outputs[i], targets[i]), "ssim": ssim(outputs[i], targets[i])})
    return records, elapsed


def conv2d_macs(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> int:
    macs, hooks = [], []
    def hook(module, inputs, output):
        batch, out_channels, height, width = output.shape; kh, kw = module.kernel_size
        macs.append(batch * out_channels * height * width * (module.in_channels // module.groups) * kh * kw)
    for module in model.modules():
        if isinstance(module, nn.Conv2d): hooks.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode(): model(torch.zeros(input_shape, device=device))
    finally:
        for handle in hooks: handle.remove()
    return int(sum(macs))
