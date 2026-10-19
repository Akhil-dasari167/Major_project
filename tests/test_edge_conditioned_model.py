import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from dehaze.models.edge_conditioned_variant import EdgeConditionedVariant


class EdgeConditionedModelTests(unittest.TestCase):
    def setUp(self):
        self.model = EdgeConditionedVariant(residual_scale=0.2)

    def test_parameter_count(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(count, 76035, f"Expected 76,035 parameters, got {count}")

    def test_negative_scale_raises(self):
        with self.assertRaises(ValueError):
            EdgeConditionedVariant(residual_scale=-0.1)
        with self.assertRaises(ValueError):
            EdgeConditionedVariant(residual_scale=0.0)

    def test_forward_shape_and_range(self):
        for shape in [(1, 3, 64, 64), (2, 3, 128, 128), (1, 3, 256, 256)]:
            x = torch.rand(*shape)
            out = self.model(x)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))
            self.assertTrue(torch.isfinite(out).all())

    def test_invalid_input_shape_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.rand(3, 64, 64))  # 3D tensor
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 4, 64, 64))  # 4 channels instead of 3

    def test_gradient_flow(self):
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        out = self.model(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, f"Gradient for {name} should not be None")
            self.assertTrue(torch.isfinite(p.grad).all(), f"Gradient for {name} is non-finite")

    def test_device_consistency_for_sobel_buffers(self):
        # Verify edge map buffers adapt if input is moved
        x = torch.rand(1, 3, 32, 32)
        edge_map = self.model.compute_edge_map(x)
        self.assertEqual(edge_map.shape, (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(edge_map).all())
        self.assertTrue(torch.all(edge_map >= 0.0))


if __name__ == "__main__":
    unittest.main()
