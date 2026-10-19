"""Train the isolated multi-model variant using train/validation pairs only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dehaze.data.dataset import EpochShuffleSampler, ImagePair, PairedDehazeDataset
from dehaze.data.splits import load_split, save_split, validate_split
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.losses.ssim import L1SSIMLoss
from dehaze.models.multimodel_variant import MultiModelDehazeVariant
from dehaze.models.v0_baseline import parameter_count
from dehaze.training.engine import save_checkpoint, write_json_once
from dehaze.utils.reproducibility import resolve_device, seed_everything


def load_simple_yaml(path: Path) -> dict:
    """Read the project scalar YAML configuration without requiring PyYAML."""
    values: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = int(value)
            except ValueError:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = value
    return values


def pairs_for_ids(data_root: Path, ids: list[str]) -> list[ImagePair]:
    """Construct only requested pairs; deliberately avoids opening test images."""
    pairs = []
    for image_id in ids:
        hazy = data_root / f"{image_id}_hazy.png"
        target = data_root / f"{image_id}_GT.png"
        if not hazy.is_file() or not target.is_file():
            raise FileNotFoundError(f"Missing pair for image ID {image_id}")
        pairs.append(ImagePair(image_id, hazy, target))
    return pairs


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    totals = {"loss": 0.0, "l1": 0.0, "ssim_loss": 0.0, "count": 0}
    for batch in loader:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = criterion(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        count = inputs.shape[0]
        totals["loss"] += components["total"] * count
        totals["l1"] += components["l1"] * count
        totals["ssim_loss"] += components["loss_ssim"] * count
        totals["count"] += count
    return {key: totals[key] / totals["count"] for key in ("loss", "l1", "ssim_loss")}


@torch.inference_mode()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "ssim_loss": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0}
    for batch in loader:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        outputs = model(inputs)
        _, components = criterion(outputs, targets)
        count = inputs.shape[0]
        totals["loss"] += components["total"] * count
        totals["l1"] += components["l1"] * count
        totals["ssim_loss"] += components["loss_ssim"] * count
        for index in range(count):
            totals["psnr"] += psnr(outputs[index], targets[index])
            totals["ssim"] += ssim(outputs[index], targets[index])
        totals["count"] += count
    return {key: totals[key] / totals["count"] for key in ("loss", "l1", "ssim_loss", "psnr", "ssim")}


def write_plot(path: Path, history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="L1+SSIM loss", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [row["validation_psnr"] for row in history], label="PSNR (dB)")
    axes[1].plot(epochs, [row["validation_ssim"] * 25 for row in history], label="SSIM × 25")
    axes[1].set(xlabel="Epoch", title="Validation quality")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/multimodel_variant.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--history-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output}")
    config = load_simple_yaml(args.config)
    if args.smoke_test:
        config["epochs"] = 1
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    manifest = load_split(Path(str(config["split"])))
    validate_split(manifest)
    train_pairs = pairs_for_ids(Path(str(config["data_root"])), manifest.train)
    validation_pairs = pairs_for_ids(Path(str(config["data_root"])), manifest.validation)
    train_set = PairedDehazeDataset(train_pairs, int(config["patch_size"]), bool(config["augment_horizontal_flip"]), int(config["seed"]))
    validation_set = PairedDehazeDataset(validation_pairs, None, False, int(config["seed"]))
    sampler = EpochShuffleSampler(train_set, int(config["seed"]))
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    model = MultiModelDehazeVariant(float(config["residual_scale"]), int(config["base_channels"])).to(device)
    criterion = L1SSIMLoss(alpha=float(config["loss_alpha"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    run_config = dict(config, model_type="multimodel_dehaze_variant", optimizer="Adam", loss_formula=f"{config['loss_alpha']} * L1 + {1 - float(config['loss_alpha'])} * (1 - SSIM)", smoke_test=args.smoke_test, device_resolved=str(device), resumed_from=str(args.resume_from) if args.resume_from else None)
    args.output.mkdir(parents=True)
    save_split(manifest, args.output / "split.json")
    write_json_once(args.output / "config.json", run_config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best_l1, best_path, stalled, start_epoch = [], float("inf"), None, 0, 1
    if args.resume_from:
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_l1 = float(payload["validation_loss"])
        if args.history_from:
            with args.history_from.open(newline="", encoding="utf-8") as handle:
                history = [{key: float(value) if key != "epoch" else int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_set.set_epoch(epoch); sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = validate_epoch(model, validation_loader, criterion, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in validation_metrics.items()}, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if validation_metrics["l1"] < best_l1:
            best_l1, stalled = validation_metrics["l1"], 0
            best_path = args.output / f"best_epoch_{epoch:03d}.pt"
            save_checkpoint(best_path, model, optimizer, epoch, run_config, best_l1)
        else:
            stalled += 1
        print(json.dumps(row, sort_keys=True))
        if stalled >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}: no validation-L1 improvement for {stalled} epochs.")
            break
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    write_plot(args.output / "training_curves.png", history)
    summary = {"best_checkpoint": str(best_path), "best_validation_l1": best_l1, "epochs_completed": len(history), "runtime_seconds": time.perf_counter() - started, "parameters": parameter_count(model), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, "train_ids": manifest.train, "validation_ids": manifest.validation, "test_ids_accessed": False}
    write_json_once(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
