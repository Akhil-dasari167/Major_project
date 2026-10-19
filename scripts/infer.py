"""Metadata-driven single-image or folder inference for dehazing checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dehaze.evaluation.checkpoints import load_checkpoint_model
from dehaze.utils.io import load_rgb, pil_to_tensor, save_tensor_image
from dehaze.utils.reproducibility import resolve_device

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def input_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported image format: {source.suffix}")
        return [source]
    if source.is_dir():
        files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
        if files:
            return files
        raise ValueError(f"No supported images found in {source}")
    raise FileNotFoundError(f"Input path does not exist: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output path: {args.output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    files = input_files(args.input)
    if args.input.is_dir():
        args.output.mkdir(parents=True)
    device = resolve_device(args.device)
    model, descriptor, _ = load_checkpoint_model(args.checkpoint, device)
    records = []
    for source in files:
        try:
            image = load_rgb(source)
            inputs = pil_to_tensor(image).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model(inputs)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            destination = args.output / f"{source.stem}_dehazed.png" if args.input.is_dir() else args.output
            save_tensor_image(prediction, destination)
            records.append({"input": str(source), "output": str(destination), "width": image.width, "height": image.height, "inference_seconds": elapsed})
        except Exception as error:
            raise RuntimeError(f"Failed to process {source}: {error}") from error
    summary_path = (args.output / "inference_summary.json") if args.input.is_dir() else args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"checkpoint": str(args.checkpoint), "model": descriptor, "device": str(device), "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"model": descriptor, "images": len(records), "mean_inference_seconds": sum(row["inference_seconds"] for row in records) / len(records), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
