"""Fixed-budget validation-only study of V0 residual-scale constraints."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch.utils.data import DataLoader

from dehaze.data.dataset import discover_nh_haze_pairs, EpochShuffleSampler, PairedDehazeDataset
from dehaze.data.splits import load_split, save_split, validate_split
from dehaze.models.residual_scale_variant import ResidualScaleVariant
from dehaze.training.engine import run_epoch, save_checkpoint, write_json_once
from dehaze.utils.io import load_rgb, tensor_to_pil
from dehaze.utils.reproducibility import resolve_device, seed_everything

ROOT = Path(__file__).resolve().parents[1]
SCALES = (0.1, 0.2)
EPOCHS = 5
SEED = 42


def correction_stats(model, loader, device, scale):
    values = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device); output = model(inputs); correction = output - inputs
            values.append({"mean_abs": correction.abs().mean().item(), "mean_signed": correction.mean().item(), "near_bound": (correction.abs() >= scale * .95).float().mean().item(), "clamped": ((output <= 0) | (output >= 1)).float().mean().item()})
    return {key: sum(item[key] for item in values) / len(values) for key in values[0]}


def save_contact_sheet(model, pairs, ids, output, device):
    width, height, label = 240, 180, 16; font = ImageFont.load_default(); rows = []
    model.eval()
    with torch.inference_mode():
        for image_id in ids:
            pair = pairs[image_id]; hazy = load_rgb(pair.hazy_path); target = load_rgb(pair.target_path)
            from dehaze.utils.io import pil_to_tensor
            prediction = tensor_to_pil(model(pil_to_tensor(hazy).unsqueeze(0).to(device))[0])
            row = Image.new("RGB", (width * 3, height + label), "white")
            for index, (name, image) in enumerate((("Hazy", hazy), ("Ground truth", target), ("Prediction", prediction))):
                panel = Image.new("RGB", (width, height + label), "white")
                ImageDraw.Draw(panel).text((3, 2), f"{image_id} {name}", fill="black", font=font)
                panel.paste(ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS), (0, label)); row.paste(panel, (index * width, 0))
            rows.append(row)
    sheet = Image.new("RGB", (width * 3, len(rows) * (height + label)), "white")
    for index, row in enumerate(rows): sheet.paste(row, (0, index * (height + label)))
    sheet.save(output)


def run_variant(scale, run_dir, pairs, manifest, device):
    if run_dir.exists(): raise FileExistsError(f"Refusing to overwrite experimental run: {run_dir}")
    run_dir.mkdir(parents=True); seed_everything(SEED); save_split(manifest, run_dir / "split.json")
    config = {"study": "residual_scale", "residual_scale": scale, "epochs": EPOCHS, "patch_size": 256, "batch_size": 2, "learning_rate": 0.0002, "seed": SEED, "device": str(device), "split_source": "outputs/v0_pilot_001/split.json"}
    write_json_once(run_dir / "config.json", config)
    train = PairedDehazeDataset([pairs[i] for i in manifest.train], patch_size=256, augment=True, seed=SEED)
    validation = PairedDehazeDataset([pairs[i] for i in manifest.validation], patch_size=None, augment=False, seed=SEED)
    sampler = EpochShuffleSampler(train, SEED); train_loader = DataLoader(train, batch_size=2, sampler=sampler, num_workers=0); val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
    model = ResidualScaleVariant(scale).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=0.0002); loss_fn = torch.nn.L1Loss(); history=[]; best=float("inf")
    for epoch in range(1, EPOCHS + 1):
        train.set_epoch(epoch); sampler.set_epoch(epoch)
        train_l1 = run_epoch(model, train_loader, optimizer, loss_fn, device, True)
        validation_l1 = run_epoch(model, val_loader, None, loss_fn, device, False)
        stats = correction_stats(model, val_loader, device, scale)
        row = {"epoch": epoch, "train_l1": train_l1, "validation_l1": validation_l1, **stats}; history.append(row)
        save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, config, validation_l1)
        if validation_l1 < best:
            best = validation_l1; save_checkpoint(run_dir / f"best_epoch_{epoch:03d}.pt", model, optimizer, epoch, config, validation_l1)
        print({"scale": scale, **row})
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    save_contact_sheet(model, pairs, manifest.validation, run_dir / "validation_contact_sheet.png", device)
    return history


def main():
    device = resolve_device("cuda")
    if device.type != "cuda": raise RuntimeError("This controlled study requires the approved CUDA device.")
    pairs = {pair.image_id: pair for pair in discover_nh_haze_pairs(ROOT / "data" / "NH-HAZE")}
    manifest = load_split(ROOT / "outputs" / "v0_pilot_001" / "split.json"); validate_split(manifest, list(pairs))
    runs = {scale: ROOT / "outputs" / f"residual_scale_study_{str(scale).replace('.', 'p')}" for scale in SCALES}
    histories = {scale: run_variant(scale, run_dir, pairs, manifest, device) for scale, run_dir in runs.items()}
    figure, axis = plt.subplots(figsize=(6, 4))
    for scale, history in histories.items(): axis.plot([r["epoch"] for r in history], [r["validation_l1"] for r in history], marker="o", label=f"scale {scale}")
    axis.set(xlabel="epoch", ylabel="validation L1", title="Residual-scale study (validation only)"); axis.grid(True, alpha=.3); axis.legend(); figure.tight_layout()
    analysis = ROOT / "analysis" / "residual_scale_study"; analysis.mkdir(parents=True, exist_ok=True)
    plot = analysis / "validation_loss_comparison.png"
    if plot.exists(): raise FileExistsError(f"Refusing to overwrite analysis plot: {plot}")
    figure.savefig(plot, dpi=160); plt.close(figure)
    summary = {str(scale): {"best": min(history, key=lambda r:r["validation_l1"]), "final": history[-1]} for scale, history in histories.items()}
    summary_path = analysis / "study_summary.json"
    if summary_path.exists(): raise FileExistsError(f"Refusing to overwrite analysis summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
