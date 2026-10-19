"""Regression and unit tests for evaluation metrics (PSNR and SSIM)."""
import math
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehaze.evaluation.metrics import psnr, ssim


class MetricTests(unittest.TestCase):
    def test_identical_images(self):
        """Identical images must yield infinite PSNR and exactly 1.0 SSIM."""
        x = torch.rand(3, 64, 64)
        p = psnr(x, x)
        s = ssim(x, x)
        self.assertEqual(p, float("inf"))
        self.assertAlmostEqual(s, 1.0, places=5)

    def test_black_and_white_images(self):
        """Black vs white images must produce finite PSNR and valid SSIM in [-1, 1]."""
        black = torch.zeros(3, 32, 32)
        white = torch.ones(3, 32, 32)
        p = psnr(black, white)
        s = ssim(black, white)
        self.assertAlmostEqual(p, 0.0, places=4)
        self.assertGreaterEqual(s, -1.0)
        self.assertLessEqual(s, 1.0)

    def test_shape_compatibility(self):
        """Metrics must seamlessly support 3D (CHW), 3D (HWC), and 4D (1, C, H, W)."""
        base = torch.rand(3, 40, 40)
        noisy = base + torch.randn(3, 40, 40) * 0.05

        # 3D CHW
        p_chw = psnr(base, noisy)
        s_chw = ssim(base, noisy)

        # 4D BCHW (1, 3, 40, 40)
        p_bchw = psnr(base.unsqueeze(0), noisy.unsqueeze(0))
        s_bchw = ssim(base.unsqueeze(0), noisy.unsqueeze(0))

        # 3D HWC (40, 40, 3)
        base_hwc = base.permute(1, 2, 0)
        noisy_hwc = noisy.permute(1, 2, 0)
        p_hwc = psnr(base_hwc, noisy_hwc)
        s_hwc = ssim(base_hwc, noisy_hwc)

        self.assertAlmostEqual(p_chw, p_bchw, places=4)
        self.assertAlmostEqual(p_chw, p_hwc, places=4)
        self.assertAlmostEqual(s_chw, s_bchw, places=4)
        self.assertAlmostEqual(s_chw, s_hwc, places=4)

    def test_numpy_and_pil_support(self):
        """Metrics must support numpy ndarrays and PIL Image objects."""
        img1 = Image.new("RGB", (32, 32), color=(120, 150, 180))
        img2 = Image.new("RGB", (32, 32), color=(125, 145, 185))

        p_pil = psnr(img1, img2)
        s_pil = ssim(img1, img2)
        self.assertGreater(p_pil, 30.0)
        self.assertGreater(s_pil, 0.90)
        self.assertLessEqual(s_pil, 1.0)

        # NumPy arrays in [0, 255]
        arr1 = np.asarray(img1)
        arr2 = np.asarray(img2)
        p_np = psnr(arr1, arr2)
        s_np = ssim(arr1, arr2)
        self.assertAlmostEqual(p_pil, p_np, places=3)
        self.assertAlmostEqual(s_pil, s_np, places=3)

    def test_ssim_bounded_range(self):
        """SSIM must always return a scalar float strictly in [-1.0, 1.0]."""
        for _ in range(5):
            x = torch.rand(3, 48, 48)
            y = torch.rand(3, 48, 48)
            val = ssim(x, y)
            self.assertIsInstance(val, float)
            self.assertFalse(math.isnan(val))
            self.assertFalse(math.isinf(val))
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)

    def test_known_sample_pair(self):
        """Verification of the sample NH-HAZE pair produces valid metrics without anomalous values."""
        hazy_path = Path(__file__).resolve().parents[1] / "sample_images" / "sample_hazy.png"
        gt_path = Path(__file__).resolve().parents[1] / "sample_images" / "sample_clean_GT.png"
        if hazy_path.is_file() and gt_path.is_file():
            hazy = Image.open(hazy_path).convert("RGB")
            gt = Image.open(gt_path).convert("RGB")
            p = psnr(hazy, gt)
            s = ssim(hazy, gt)
            # Raw hazy image vs Clean GT has PSNR ~9.34 dB and SSIM ~0.41
            self.assertGreater(p, 8.0)
            self.assertLess(p, 15.0)
            self.assertGreater(s, 0.35)
            self.assertLess(s, 0.50)

    def test_mismatched_shapes_raise_error(self):
        """Different image dimensions must raise ValueError."""
        x = torch.rand(3, 32, 32)
        y = torch.rand(3, 48, 48)
        with self.assertRaises(ValueError):
            psnr(x, y)
        with self.assertRaises(ValueError):
            ssim(x, y)


if __name__ == "__main__":
    unittest.main()
