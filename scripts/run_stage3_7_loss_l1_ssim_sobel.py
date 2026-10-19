"""Stage 3.7 Experiment C: L1 + SSIM + Sobel Edge Loss training runner.

Loss: L_total = 0.8 * L1 + 0.15 * (1 - SSIM) + 0.05 * L_Sobel.
Architecture: ResidualScaleVariant (scale = 0.2).
Split: Canonical Seed 42 (39 train / 8 validation).
"""
import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dehaze.data.dataset import discover_nh_haze_pairs, EpochShuffleSampler, PairedDehazeDataset
from dehaze.data.splits import load_split, save_split, validate_split
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.losses.sobel import L1SSIMSobelLoss
from dehaze.models.residual_scale_variant import ResidualScaleVariant
from dehaze.training.engine import save_checkpoint, write_json_once
from dehaze.utils.io import load_rgb, pil_to_tensor, tensor_to_pil
from dehaze.utils.reproducibility import resolve_device, seed_everything

SCALE = 0.2
WEIGHT_L1 = 0.8
WEIGHT_SSIM = 0.15
WEIGHT_SOBEL = 0.05
EPOCHS = 20
SEED = 42
PATCH = 256
BATCH = 2
LR = 2e-4


# Sobel kernels for measuring edge gradient energy
SOBEL_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)


def compute_edge_energy(t: torch.Tensor, device: torch.device) -> float:
    # t: (B, 3, H, W)
    kx = SOBEL_KX.to(device)
    ky = SOBEL_KY.to(device)
    gray = 0.2989 * t[:, 0:1] + 0.5870 * t[:, 1:2] + 0.1140 * t[:, 2:3]
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8).mean().item()


def run_epoch(model, loader, optimizer, criterion, device, training, scale):
    model.train(training)
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_sobel_loss = 0.0
    samples = 0
    batches = 0
    pvals, svals, absv, signed, near, clipped, edge_energies = [], [], [], [], [], [], []
    
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            x, y = batch['input'].to(device), batch['target'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            
            output = model(x)
            loss, metrics = criterion(output, y)
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {metrics}")
            
            if training:
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Non-finite gradient encountered during backward pass")
                optimizer.step()
            
            b_size = x.size(0)
            total_loss += loss.item() * b_size
            total_l1 += metrics['l1'] * b_size
            total_ssim_loss += metrics['loss_ssim'] * b_size
            total_sobel_loss += metrics['loss_sobel'] * b_size
            samples += b_size
            batches += 1
            
            if not training:
                correction = output - x
                absv.append(correction.abs().mean().item())
                signed.append(correction.mean().item())
                near.append((correction.abs() >= scale * 0.95).float().mean().item())
                clipped.append(((output <= 0) | (output >= 1)).float().mean().item())
                edge_energies.append(compute_edge_energy(output, device))
                for i in range(b_size):
                    pvals.append(psnr(output[i], y[i]))
                    svals.append(ssim(output[i], y[i]))
    
    result = {
        'loss': total_loss / samples,
        'l1': total_l1 / samples,
        'loss_ssim': total_ssim_loss / samples,
        'loss_sobel': total_sobel_loss / samples,
        'samples': samples,
        'batches': batches,
    }
    if not training:
        result.update({
            'psnr': sum(pvals) / len(pvals),
            'ssim': sum(svals) / len(svals),
            'mean_abs_correction': sum(absv) / len(absv),
            'mean_signed_correction': sum(signed) / len(signed),
            'near_bound_fraction': sum(near) / len(near),
            'clamped_fraction': sum(clipped) / len(clipped),
            'mean_edge_energy': sum(edge_energies) / len(edge_energies),
        })
    return result


def generate_contact_sheet(model, pairs, ids, path, device):
    w, h, label = 240, 180, 20
    font = ImageFont.load_default()
    sheet = Image.new('RGB', (w * 3, (h + label) * len(ids)), 'white')
    model.eval()
    with torch.inference_mode():
        for r, image_id in enumerate(ids):
            pair = pairs[image_id]
            hazy = load_rgb(pair.hazy_path)
            target = load_rgb(pair.target_path)
            pred = tensor_to_pil(model(pil_to_tensor(hazy).unsqueeze(0).to(device))[0])
            for c, (name, img) in enumerate((('Hazy Input', hazy), ('Ground Truth', target), ('Prediction', pred))):
                panel = Image.new('RGB', (w, h + label), 'white')
                ImageDraw.Draw(panel).text((4, 3), f'{image_id} {name}', fill='black', font=font)
                panel.paste(ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS), (0, label))
                sheet.paste(panel, (c * w, r * (h + label)))
    sheet.save(path)


def run_experiment(out_dir: Path, pairs: dict, manifest, device: torch.device, epochs: int = EPOCHS, is_smoke: bool = False):
    if out_dir.exists() and not is_smoke:
        raise FileExistsError(f"Refusing to overwrite output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(SEED)
    save_split(manifest, out_dir / 'split.json')
    
    config = {
        'stage': '3.7',
        'experiment': 'C_l1_ssim_sobel',
        'model_type': 'residual_scale_variant',
        'residual_scale': SCALE,
        'loss_type': 'L1SSIMSobelLoss',
        'loss_formula': f'{WEIGHT_L1} * L1 + {WEIGHT_SSIM} * (1 - SSIM) + {WEIGHT_SOBEL} * L_Sobel',
        'weight_l1': WEIGHT_L1,
        'weight_ssim': WEIGHT_SSIM,
        'weight_sobel': WEIGHT_SOBEL,
        'epochs': epochs,
        'patch_size': PATCH,
        'batch_size': BATCH,
        'learning_rate': LR,
        'optimizer': 'Adam',
        'seed': SEED,
        'device': str(device),
        'split_source': 'outputs/v0_pilot_001/split.json',
        'expected_train_steps_per_epoch': 20,
        'expected_validation_steps_per_epoch': 8,
    }
    write_json_once(out_dir / 'config.json', config)
    
    train_dataset = PairedDehazeDataset([pairs[i] for i in manifest.train], PATCH, True, SEED)
    val_dataset = PairedDehazeDataset([pairs[i] for i in manifest.validation], None, False, SEED)
    sampler = EpochShuffleSampler(train_dataset, SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = ResidualScaleVariant(SCALE).to(device)
    criterion = L1SSIMSobelLoss(weight_l1=WEIGHT_L1, weight_ssim=WEIGHT_SSIM, weight_sobel=WEIGHT_SOBEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    rows = []
    best_val_l1 = float('inf')
    best_path = None
    started = time.perf_counter()
    
    print(f"Starting Stage 3.7 Experiment C (Epochs: {epochs}, Device: {device}, Smoke: {is_smoke})")
    
    for epoch_idx in range(1, epochs + 1):
        train_dataset.set_epoch(epoch_idx)
        sampler.set_epoch(epoch_idx)
        
        tr = run_epoch(model, train_loader, optimizer, criterion, device, True, SCALE)
        va = run_epoch(model, val_loader, None, criterion, device, False, SCALE)
        
        row = {
            'epoch': epoch_idx,
            'train_loss_total': tr['loss'],
            'train_l1': tr['l1'],
            'train_ssim_loss': tr['loss_ssim'],
            'train_sobel_loss': tr['loss_sobel'],
            'validation_loss_total': va['loss'],
            'validation_l1': va['l1'],
            'validation_psnr': va['psnr'],
            'validation_ssim': va['ssim'],
            'validation_sobel_loss': va['loss_sobel'],
            'validation_edge_energy': va['mean_edge_energy'],
            'mean_abs_correction': va['mean_abs_correction'],
            'mean_signed_correction': va['mean_signed_correction'],
            'near_bound_fraction': va['near_bound_fraction'],
            'clamped_fraction': va['clamped_fraction'],
            'learning_rate': LR,
        }
        rows.append(row)
        
        save_checkpoint(out_dir / f'epoch_{epoch_idx:03d}.pt', model, optimizer, epoch_idx, config, va['l1'])
        if va['l1'] < best_val_l1:
            best_val_l1 = va['l1']
            best_path = out_dir / f'best_epoch_{epoch_idx:03d}.pt'
            save_checkpoint(best_path, model, optimizer, epoch_idx, config, va['l1'])
        
        print(f"Epoch {epoch_idx:02d}/{epochs:02d} | Train: {tr['loss']:.4f} (L1: {tr['l1']:.4f}, Sobel: {tr['loss_sobel']:.4f}) | Val L1: {va['l1']:.6f} | Val PSNR: {va['psnr']:.2f} dB | Val SSIM: {va['ssim']:.4f} | Edge: {va['mean_edge_energy']:.4f}")
    
    if not is_smoke:
        with (out_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        
        # Best model contact sheet
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        contact_path = out_dir / 'validation_contact_sheet_best.png'
        generate_contact_sheet(model, pairs, manifest.validation, contact_path, device)
        print(f"Saved best validation contact sheet to: {contact_path}")
        
        # Loss curves plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        ep_list = [r['epoch'] for r in rows]
        
        axes[0, 0].plot(ep_list, [r['train_loss_total'] for r in rows], label='Train Total', color='blue')
        axes[0, 0].plot(ep_list, [r['validation_loss_total'] for r in rows], label='Val Total', color='orange')
        axes[0, 0].set(xlabel='Epoch', ylabel='Total Loss', title='Total Loss (0.8 L1 + 0.15 SSIM + 0.05 Sobel)')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(ep_list, [r['train_l1'] for r in rows], label='Train L1', color='blue', linestyle='--')
        axes[0, 1].plot(ep_list, [r['validation_l1'] for r in rows], label='Val L1', color='red')
        axes[0, 1].set(xlabel='Epoch', ylabel='L1 Loss', title='L1 Loss Trajectory')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(ep_list, [r['validation_psnr'] for r in rows], label='Val PSNR (dB)', color='green')
        axes[1, 0].plot(ep_list, [r['validation_ssim'] * 25 for r in rows], label='Val SSIM (x25)', color='purple')
        axes[1, 0].set(xlabel='Epoch', ylabel='Metric Value', title='Validation PSNR & SSIM')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(ep_list, [r['validation_edge_energy'] for r in rows], label='Val Edge Energy', color='darkcyan')
        axes[1, 1].set(xlabel='Epoch', ylabel='Sobel Gradient Energy', title='High-Frequency Edge Energy')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        curve_path = out_dir / 'loss_curve.png'
        fig.savefig(curve_path, dpi=160)
        plt.close(fig)
        print(f"Saved loss curve to: {curve_path}")
        
        best_row = min(rows, key=lambda x: x['validation_l1'])
        summary = {
            'stage': '3.7',
            'experiment': 'C_l1_ssim_sobel',
            'scale': SCALE,
            'loss_formula': f'{WEIGHT_L1} * L1 + {WEIGHT_SSIM} * (1 - SSIM) + {WEIGHT_SOBEL} * L_Sobel',
            'weights': {'l1': WEIGHT_L1, 'ssim': WEIGHT_SSIM, 'sobel': WEIGHT_SOBEL},
            'best_checkpoint': str(best_path.relative_to(ROOT)),
            'best_row': best_row,
            'runtime_seconds': time.perf_counter() - started,
            'actual_train_steps': len(rows) * 20,
            'actual_validation_steps': len(rows) * 8,
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"Saved summary to: {out_dir / 'summary.json'}")
        return summary
    else:
        print("Smoke test completed successfully!")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run 1 epoch bounded smoke test')
    parser.add_argument('--output', default='outputs/stage3_7_loss_l1_ssim_sobel')
    args = parser.parse_args()
    
    device = resolve_device('cuda')
    if device.type != 'cuda':
        raise RuntimeError('CUDA required for Stage 3.7 experiment on RTX 3050 GPU')
    
    pairs = {p.image_id: p for p in discover_nh_haze_pairs(ROOT / 'data' / 'NH-HAZE')}
    manifest = load_split(ROOT / 'outputs' / 'v0_pilot_001' / 'split.json')
    validate_split(manifest, list(pairs))
    
    out_dir = ROOT / args.output
    if args.smoke_test:
        smoke_dir = ROOT / 'outputs' / 'stage3_7_smoke_test'
        run_experiment(smoke_dir, pairs, manifest, device, epochs=1, is_smoke=True)
    else:
        run_experiment(out_dir, pairs, manifest, device, epochs=EPOCHS, is_smoke=False)


if __name__ == '__main__':
    main()
