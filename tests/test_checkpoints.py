import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dehaze.evaluation.checkpoints import model_from_checkpoint_payload
from dehaze.models.residual_scale_variant import ResidualScaleVariant
from dehaze.models.v0_baseline import V0ResidualCNN
from dehaze.models.multimodel_variant import MultiModelDehazeVariant


class CheckpointFactoryTests(unittest.TestCase):
    def test_v0_payload_reconstructs_v0(self):
        source = V0ResidualCNN(); model, descriptor = model_from_checkpoint_payload({"model_state": source.state_dict(), "config": {}}, torch.device("cpu"))
        self.assertIsInstance(model, V0ResidualCNN); self.assertEqual(descriptor["residual_scale"], 0.1)

    def test_scaled_payload_reconstructs_matching_scale(self):
        source = ResidualScaleVariant(0.2); model, descriptor = model_from_checkpoint_payload({"model_state": source.state_dict(), "config": {"residual_scale": 0.2}}, torch.device("cpu"))
        self.assertIsInstance(model, ResidualScaleVariant); self.assertEqual(model.residual_scale, 0.2); self.assertEqual(descriptor["residual_scale"], 0.2)

    def test_multimodel_payload_reconstructs_matching_configuration(self):
        source = MultiModelDehazeVariant(0.2, 12)
        payload = {"model_state": source.state_dict(), "config": {"model_type": "multimodel_dehaze_variant", "residual_scale": 0.2, "base_channels": 12}}
        model, descriptor = model_from_checkpoint_payload(payload, torch.device("cpu"))
        self.assertIsInstance(model, MultiModelDehazeVariant)
        self.assertEqual(model.base_channels, 12)
        self.assertEqual(descriptor["kind"], "multimodel_dehaze_variant")
