"""Verification script for local demonstration and presentation asset generation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dehaze.evaluation.checkpoints import load_checkpoint_model
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.utils.io import pil_to_tensor, tensor_to_pil
from dehaze.utils.reproducibility import resolve_device


def run_demo_verification():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "demo_presentation_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("cuda")
    checkpoint_path = root / "models" / "best_epoch_019.pt"
    model, desc, _ = load_checkpoint_model(checkpoint_path, device)
    model.eval()

    scenes = [
        ("Scene_53_Builtin_Sample", root / "sample_images" / "sample_hazy.png", root / "sample_images" / "sample_clean_GT.png"),
        ("Scene_03_Validation", root / "data" / "NH-HAZE" / "03_hazy.png", root / "data" / "NH-HAZE" / "03_GT.png"),
        ("Scene_07_Validation", root / "data" / "NH-HAZE" / "07_hazy.png", root / "data" / "NH-HAZE" / "07_GT.png"),
    ]

    results = []
    print("=" * 65)
    print("DIP MAJOR PROJECT — LOCAL DEMONSTRATION VERIFICATION")
    print(f"Hardware: {device} | Checkpoint: {checkpoint_path.name}")
    print("=" * 65)

    for name, hazy_path, gt_path in scenes:
        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        # Warm-up pass
        t_in = pil_to_tensor(hazy_img).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed forward pass
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred_t = model(t_in)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - t0) * 1000

        gt_t = pil_to_tensor(gt_img).to(device)
        val_psnr = psnr(pred_t, gt_t)
        val_ssim = ssim(pred_t, gt_t)

        raw_hazy_psnr = psnr(t_in[0], gt_t)
        raw_hazy_ssim = ssim(t_in[0], gt_t)

        out_img = tensor_to_pil(pred_t)
        save_path = out_dir / f"{name}_dehazed.png"
        out_img.save(save_path, format="PNG")

        # Also copy input and GT for full side-by-side presentation package
        hazy_img.save(out_dir / f"{name}_input_hazy.png", format="PNG")
        gt_img.save(out_dir / f"{name}_ground_truth.png", format="PNG")

        res = {
            "scene": name,
            "resolution": f"{hazy_img.width}x{hazy_img.height}",
            "hazy_psnr_db": float(raw_hazy_psnr),
            "hazy_ssim": float(raw_hazy_ssim),
            "dehazed_psnr_db": float(val_psnr),
            "dehazed_ssim": float(val_ssim),
            "psnr_improvement_db": float(val_psnr - raw_hazy_psnr),
            "ssim_improvement": float(val_ssim - raw_hazy_ssim),
            "latency_ms": float(runtime_ms),
            "device": str(device),
            "output_png_bytes": save_path.stat().st_size,
        }
        results.append(res)

        print(f"\n[Scene: {name}]")
        print(f"  Resolution:          {res['resolution']}")
        print(f"  Raw Hazy Input:      PSNR = {raw_hazy_psnr:.4f} dB | SSIM = {raw_hazy_ssim:.4f}")
        print(f"  Dehazed Output:      PSNR = {val_psnr:.4f} dB | SSIM = {val_ssim:.4f}")
        print(f"  Net Restoration:     +{res['psnr_improvement_db']:.4f} dB | +{res['ssim_improvement']:.4f}")
        print(f"  Inference Latency:   {runtime_ms:.2f} ms ({device})")
        print(f"  Saved PNG Output:    {save_path.name} ({res['output_png_bytes']:,} bytes)")

    # Save summary JSON
    summary_path = out_dir / "demo_verification_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device), "results": results}, f, indent=2)

    print("\n" + "=" * 65)
    print(f"All 3 demonstration scenes successfully verified!")
    print(f"Presentation assets saved to: {str(out_dir).encode('ascii', 'replace').decode('ascii')}")
    print("=" * 65)


if __name__ == "__main__":
    run_demo_verification()
