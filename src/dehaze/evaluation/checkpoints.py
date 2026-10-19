"""Checkpoint-config-driven model reconstruction."""
from pathlib import Path
import torch
from dehaze.models.v0_baseline import V0ResidualCNN
from dehaze.models.residual_scale_variant import ResidualScaleVariant
from dehaze.models.edge_conditioned_variant import EdgeConditionedVariant
from dehaze.models.multimodel_variant import MultiModelDehazeVariant


def model_from_checkpoint_payload(payload: dict, device: torch.device):
    config = payload.get("config", {})
    if config.get("model_type") == "multimodel_dehaze_variant":
        scale = float(config.get("residual_scale", 0.2))
        channels = int(config.get("base_channels", 24))
        model = MultiModelDehazeVariant(residual_scale=scale, base_channels=channels)
        descriptor = {"kind": "multimodel_dehaze_variant", "residual_scale": scale, "base_channels": channels}
    elif config.get("model_type") == "edge_conditioned_variant":
        scale = float(config.get("residual_scale", 0.2))
        model = EdgeConditionedVariant(residual_scale=scale)
        descriptor = {"kind": "edge_conditioned_variant", "residual_scale": scale}
    elif "residual_scale" in config:
        model = ResidualScaleVariant(float(config["residual_scale"]))
        descriptor = {"kind": "residual_scale_variant", "residual_scale": float(config["residual_scale"])}
    else:
        model = V0ResidualCNN()
        descriptor = {"kind": "v0_residual_cnn", "residual_scale": 0.1}
    model = model.to(device); model.load_state_dict(payload["model_state"]); model.eval()
    return model, descriptor


def load_checkpoint_model(path: str | Path, device: torch.device):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model, descriptor = model_from_checkpoint_payload(payload, device)
    return model, descriptor, payload
