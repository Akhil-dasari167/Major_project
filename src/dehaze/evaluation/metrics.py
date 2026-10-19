"""Full-reference image metrics for normalized RGB tensors and arrays."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_numpy_hwc(tensor_or_array: Any) -> np.ndarray:
    """Normalize input tensor, PIL image, or ndarray to float32 NumPy array in HWC [0, 1] format."""
    if hasattr(tensor_or_array, "convert"):
        # PIL Image
        np_arr = np.asarray(tensor_or_array.convert("RGB"), dtype=np.float32)
    elif isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().float().cpu()
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected single-image tensor, got batch of size {arr.shape[0]}")
            arr = arr.squeeze(0)
        if arr.ndim == 3:
            # If shape is (C, H, W) where C in (1, 3, 4) and W > C, permute to (H, W, C)
            if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
                arr = arr.permute(1, 2, 0)
        elif arr.ndim == 2:
            arr = arr.unsqueeze(-1)
        np_arr = arr.numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        np_arr = tensor_or_array.astype(np.float32)
        if np_arr.ndim == 4 and np_arr.shape[0] == 1:
            np_arr = np_arr[0]
        if np_arr.ndim == 3 and np_arr.shape[0] in (1, 3, 4) and np_arr.shape[2] not in (1, 3, 4):
            np_arr = np.transpose(np_arr, (1, 2, 0))
        elif np_arr.ndim == 2:
            np_arr = np_arr[:, :, np.newaxis]
    else:
        raise TypeError(f"Unsupported image input type: {type(tensor_or_array)}")

    # Check for [0, 255] range and normalize if necessary
    if np_arr.size > 0 and np_arr.max() > 1.0:
        np_arr = np_arr / 255.0

    return np.clip(np_arr, 0.0, 1.0).astype(np.float32)


def psnr(prediction: Any, target: Any) -> float:
    """Peak Signal-to-Noise Ratio (dB) for RGB images in [0, 1].
    
    Returns float('inf') for identical images.
    """
    pred = _to_numpy_hwc(prediction)
    truth = _to_numpy_hwc(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch in PSNR: prediction {pred.shape} vs target {truth.shape}")
    mse = float(np.mean((pred - truth) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def ssim(prediction: Any, target: Any) -> float:
    """Full-reference Mean Structural Similarity Index (SSIM) in range [-1.0, 1.0]."""
    pred = _to_numpy_hwc(prediction)
    truth = _to_numpy_hwc(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch in SSIM: prediction {pred.shape} vs target {truth.shape}")

    if pred.shape[2] == 1:
        score = structural_similarity(
            truth[:, :, 0],
            pred[:, :, 0],
            data_range=1.0,
        )
    else:
        score = structural_similarity(
            truth,
            pred,
            channel_axis=2,
            data_range=1.0,
        )
    return float(score)
