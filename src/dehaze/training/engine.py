"""Reusable V0 training and checkpoint utilities."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def run_epoch(model: nn.Module, loader, optimizer, loss_fn, device: torch.device, training: bool, max_steps: int | None = None) -> float:
    model.train(training); total, count = 0.0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss encountered.")
            if training:
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Non-finite gradient encountered.")
                optimizer.step()
            total += loss.item() * inputs.size(0); count += inputs.size(0)
            if max_steps is not None and count >= max_steps * loader.batch_size: break
    return total / max(count, 1)


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, config: dict, validation_loss: float) -> None:
    if path.exists(): raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "config": config, "validation_loss": validation_loss}, path)


def write_json_once(path: Path, values: dict) -> None:
    if path.exists(): raise FileExistsError(f"Refusing to overwrite file: {path}")
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def append_history(path: Path, row: dict) -> None:
    """Append one epoch record; never replace an existing training log."""
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_l1", "validation_l1"])
        if not exists: writer.writeheader()
        writer.writerow(row)


def save_history_plot(path: Path, rows: list[dict]) -> None:
    """Write a uniquely named loss plot for the current completed epoch."""
    if path.exists(): raise FileExistsError(f"Refusing to overwrite plot: {path}")
    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, [row["train_l1"] for row in rows], label="train L1")
    axis.plot(epochs, [row["validation_l1"] for row in rows], label="validation L1")
    axis.set(xlabel="epoch", ylabel="L1 loss", title="V0 training history")
    axis.legend(); axis.grid(True, alpha=0.3); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)
