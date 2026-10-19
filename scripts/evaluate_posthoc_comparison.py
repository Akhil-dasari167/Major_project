"""Reproducible post-hoc held-out comparison of existing dehazing checkpoints.

This script performs inference only. It refuses to reuse an output directory and
never writes to a checkpoint, dataset, or existing experiment artifact.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dehaze.data.dataset import discover_nh_haze_pairs
from dehaze.data.splits import load_split, validate_split
from dehaze.evaluation.checkpoints import load_checkpoint_model
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.models.v0_baseline import parameter_count
from dehaze.utils.io import load_rgb, pil_to_tensor
from dehaze.utils.reproducibility import resolve_device


def l1_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(prediction.detach().float().cpu() - target.detach().float().cpu())).item())


def load_pair(pair, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_tensor = pil_to_tensor(load_rgb(pair.hazy_path)).unsqueeze(0).to(device)
    target_tensor = pil_to_tensor(load_rgb(pair.target_path)).unsqueeze(0).to(device)
    return input_tensor, target_tensor


def timed_evaluation(label: str, checkpoint: Path, pairs: dict, test_ids: list[str], device: torch.device) -> tuple[dict, list[dict]]:
    model, descriptor, payload = load_checkpoint_model(checkpoint, device)
    model.eval()

    # One full-resolution warm-up is excluded from timing for each model.
    warmup_input, _ = load_pair(pairs[test_ids[0]], device)
    with torch.inference_mode():
        model(warmup_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    records = []
    with torch.inference_mode():
        for image_id in test_ids:
            inputs, targets = load_pair(pairs[image_id], device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "model": label,
                    "id": image_id,
                    "psnr": psnr(outputs[0], targets[0]),
                    "ssim": ssim(outputs[0], targets[0]),
                    "l1": l1_error(outputs[0], targets[0]),
                    "inference_seconds": elapsed,
                }
            )

    summary = {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_descriptor": descriptor,
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_config": payload["config"],
        "parameters": parameter_count(model),
        "test_images": len(records),
        "mean_psnr": sum(row["psnr"] for row in records) / len(records),
        "mean_ssim": sum(row["ssim"] for row in records) / len(records),
        "mean_l1": sum(row["l1"] for row in records) / len(records),
        "inference_seconds_total": sum(row["inference_seconds"] for row in records),
        "inference_seconds_per_image": sum(row["inference_seconds"] for row in records) / len(records),
    }
    return summary, records


def report_markdown(summary: dict, records: list[dict], split: dict, device: torch.device) -> str:
    baseline, experiment_b = summary["models"]
    lines = [
        "# Stage 3.9 Post-hoc Held-out Comparison",
        "",
        "## Protocol status",
        "",
        "This is a **post-hoc held-out comparison**, not an untouched or blind final test. The eight test scenes had already been evaluated by earlier V0 and Stage 3.6 runs. No training, tuning, checkpoint selection, or optimization was performed in this evaluation.",
        "",
        "## Methodology",
        "",
        f"- Device: `{device}`" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
        "- Data: NH-HAZE paired RGB images, loaded through the existing Pillow-based `load_rgb` and converted to float tensors in [0, 1].",
        f"- Test IDs: `{', '.join(split['test'])}`.",
        "- Each model was reconstructed from its checkpoint metadata, run in `eval()` and `torch.inference_mode()` modes, and evaluated independently on the same original-resolution image pairs.",
        "- Metrics: PSNR (RGB, data range 1.0), scikit-image SSIM (RGB, channel axis, data range 1.0), and mean absolute error (L1).",
        "- Timing: one full-resolution warm-up image per model was excluded; each timed image used CUDA synchronization immediately before and after forward inference. Image loading and metric calculation were excluded from the timed interval.",
        "",
        "## Aggregate results",
        "",
        "| Model | Checkpoint | PSNR (dB) | SSIM | L1 | Parameters | Checkpoint size | Mean inference |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (baseline, experiment_b):
        lines.append(
            f"| {item['label']} | `{item['checkpoint']}` | {item['mean_psnr']:.4f} | {item['mean_ssim']:.4f} | {item['mean_l1']:.6f} | {item['parameters']:,} | {item['checkpoint_bytes']:,} bytes | {item['inference_seconds_per_image'] * 1000:.2f} ms |"
        )
    lines += ["", "## Per-image results", "", "| Model | ID | PSNR (dB) | SSIM | L1 | Inference (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['model']} | {row['id']} | {row['psnr']:.4f} | {row['ssim']:.4f} | {row['l1']:.6f} | {row['inference_seconds'] * 1000:.2f} |")
    historical_psnr, historical_ssim = 14.583583145284086, 0.5354610234498978
    preferred = (
        "Stage 3.7 Experiment B" if experiment_b["mean_psnr"] > baseline["mean_psnr"] and experiment_b["mean_ssim"] > baseline["mean_ssim"]
        else "Stage 3.6 baseline"
    )
    lines += [
        "",
        "## Historical baseline check",
        "",
        f"The prior Stage 3.6 baseline artifact reported {historical_psnr:.4f} dB PSNR and {historical_ssim:.4f} SSIM. This run's baseline result is {baseline['mean_psnr']:.4f} dB / {baseline['mean_ssim']:.4f}. Small timing differences are expected; metric differences, if any, are reported directly in `summary.json`.",
        "",
        "## Limitations and recommendation",
        "",
        f"The test split has already been examined, so these measurements must not be presented as a blind final benchmark. Based on the aggregate PSNR and SSIM in this run, the preferred existing checkpoint is **{preferred}**. Any new architecture work should use the validation split for model selection and reserve an independently acquired, unseen dataset for a future final evaluation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--experiment-b", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--data", default=Path("data/NH-HAZE"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output}")
    if not args.baseline.is_file() or not args.experiment_b.is_file():
        raise FileNotFoundError("Both checkpoint paths must exist.")

    device = resolve_device(args.device)
    pairs = {pair.image_id: pair for pair in discover_nh_haze_pairs(args.data)}
    manifest = load_split(args.split)
    validate_split(manifest, list(pairs))
    expected_test_ids = ["09", "15", "16", "18", "48", "02", "08", "41"]
    if manifest.test != expected_test_ids:
        raise ValueError(f"Unexpected test split: {manifest.test}")

    args.output.mkdir(parents=True)
    baseline_summary, baseline_records = timed_evaluation("Stage 3.6 baseline", args.baseline, pairs, manifest.test, device)
    experiment_summary, experiment_records = timed_evaluation("Stage 3.7 Experiment B", args.experiment_b, pairs, manifest.test, device)
    records = baseline_records + experiment_records
    summary = {
        "protocol": "post-hoc held-out comparison; test split was previously examined",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split": manifest.to_dict(),
        "models": [baseline_summary, experiment_summary],
        "historical_baseline": {"mean_psnr": 14.583583145284086, "mean_ssim": 0.5354610234498978},
        "baseline_delta_from_historical": {
            "mean_psnr": baseline_summary["mean_psnr"] - 14.583583145284086,
            "mean_ssim": baseline_summary["mean_ssim"] - 0.5354610234498978,
        },
    }
    with (args.output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "id", "psnr", "ssim", "l1", "inference_seconds"])
        writer.writeheader()
        writer.writerows(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(report_markdown(summary, records, manifest.to_dict(), device), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
