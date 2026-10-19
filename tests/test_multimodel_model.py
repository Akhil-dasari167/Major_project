import tempfile
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehaze.evaluation.checkpoints import load_checkpoint_model, model_from_checkpoint_payload
from dehaze.models.multimodel_variant import MultiModelDehazeVariant
from dehaze.models.v0_baseline import parameter_count
from dehaze.training.engine import save_checkpoint


class MultiModelVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MultiModelDehazeVariant(residual_scale=0.2, base_channels=12)

    def test_shape_range_and_finiteness_for_odd_image_size(self):
        x = torch.rand(2, 3, 65, 67)
        y = self.model(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.all((y >= 0) & (y <= 1)))

    def test_multiscale_and_fusion_dimensions(self):
        x = torch.rand(1, 3, 64, 64)
        features = self.model.stem(x)
        full, half, quarter = self.model.multiscale(features)
        self.assertEqual(full.shape[-2:], (64, 64))
        self.assertEqual(half.shape[-2:], (32, 32))
        self.assertEqual(quarter.shape[-2:], (16, 16))
        fused, scale_weights = self.model.aafm(full, half, quarter)
        self.assertEqual(fused.shape, full.shape)
        self.assertEqual(scale_weights.shape, (1, 3, 1, 1))
        self.assertTrue(torch.allclose(scale_weights.sum(dim=1), torch.ones_like(scale_weights[:, :1]), atol=1e-6))
        atmospheric = self.model.atmospheric(fused)
        layer = self.model.layer_separation(fused)
        context = self.model.contextual(fused)
        branches, branch_weights = self.model.branch_fusion(atmospheric, layer, context)
        self.assertEqual(branches.shape, fused.shape)
        self.assertEqual(branch_weights.shape, (1, 3, 64, 64))
        self.assertTrue(torch.allclose(branch_weights.sum(dim=1), torch.ones_like(branch_weights[:, :1]), atol=1e-6))

    def test_gradient_flow_and_parameter_count(self):
        x = torch.rand(1, 3, 48, 48, requires_grad=True)
        y = self.model(x)
        y.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(parameter_count(self.model), 0)

    def test_rgb_validation_and_edge_response(self):
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 1, 16, 16))
        flat = torch.zeros(1, 3, 16, 16)
        stepped = flat.clone()
        stepped[:, :, :, 8:] = 1.0
        flat_energy = self.model.edge_refinement.edge_map(flat).mean()
        stepped_energy = self.model.edge_refinement.edge_map(stepped).mean()
        self.assertGreater(stepped_energy, flat_energy)

    def test_checkpoint_factory_and_round_trip(self):
        config = {"model_type": "multimodel_dehaze_variant", "residual_scale": 0.2, "base_channels": 12}
        payload = {"model_state": self.model.state_dict(), "config": config}
        rebuilt, descriptor = model_from_checkpoint_payload(payload, torch.device("cpu"))
        self.assertIsInstance(rebuilt, MultiModelDehazeVariant)
        self.assertEqual(descriptor["base_channels"], 12)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            save_checkpoint(path, self.model, optimizer, 1, config, 0.5)
            loaded, loaded_descriptor, checkpoint = load_checkpoint_model(path, torch.device("cpu"))
            self.assertIsInstance(loaded, MultiModelDehazeVariant)
            self.assertEqual(loaded_descriptor, descriptor)
            self.assertEqual(checkpoint["epoch"], 1)
            with torch.inference_mode():
                self.assertEqual(loaded(torch.rand(1, 3, 32, 32)).shape, (1, 3, 32, 32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_execution(self):
        model = MultiModelDehazeVariant(base_channels=12).cuda()
        output = model(torch.rand(1, 3, 32, 32, device="cuda"))
        self.assertTrue(output.is_cuda)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
