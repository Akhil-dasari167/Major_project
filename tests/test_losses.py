import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dehaze.losses.ssim import DifferentiableSSIM, L1SSIMLoss
from dehaze.losses.sobel import SobelEdgeLoss, L1SSIMSobelLoss


class LossTests(unittest.TestCase):
    def test_ssim_loss_identical(self):
        loss_fn = L1SSIMLoss(alpha=0.8)
        x = torch.rand(2, 3, 64, 64)
        loss, metrics = loss_fn(x, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=5)
        self.assertAlmostEqual(metrics["loss_ssim"], 0.0, places=5)

    def test_ssim_loss_differentiable(self):
        loss_fn = L1SSIMLoss(alpha=0.8)
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)
        loss, metrics = loss_fn(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0.0)

    def test_sobel_edge_loss_identical(self):
        sobel_fn = SobelEdgeLoss(in_channels=3)
        x = torch.rand(2, 3, 64, 64)
        loss = sobel_fn(x, x)
        self.assertEqual(loss.item(), 0.0)

    def test_sobel_edge_loss_gradients(self):
        sobel_fn = SobelEdgeLoss(in_channels=3)
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)
        loss = sobel_fn(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0.0)

    def test_composite_l1_ssim_sobel_loss(self):
        comp_fn = L1SSIMSobelLoss(weight_l1=0.8, weight_ssim=0.15, weight_sobel=0.05)
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)
        total, metrics = comp_fn(x, y)
        
        self.assertIn("l1", metrics)
        self.assertIn("ssim", metrics)
        self.assertIn("loss_ssim", metrics)
        self.assertIn("loss_sobel", metrics)
        self.assertIn("total", metrics)
        
        # Verify weight formula exactly
        expected = 0.8 * metrics["l1"] + 0.15 * metrics["loss_ssim"] + 0.05 * metrics["loss_sobel"]
        self.assertAlmostEqual(total.item(), expected, places=5)
        
        total.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
