"""Evaluate a selected V0 checkpoint on its immutable held-out split."""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch.utils.data import DataLoader
from dehaze.data.dataset import discover_nh_haze_pairs, PairedDehazeDataset
from dehaze.data.splits import load_split, validate_split
from dehaze.evaluation.evaluator import evaluate_loader, conv2d_macs
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.models.v0_baseline import parameter_count
from dehaze.evaluation.checkpoints import load_checkpoint_model
from dehaze.utils.io import save_tensor_image
from dehaze.utils.reproducibility import resolve_device
def generate_contact_sheet(model, pairs, ids, output_path, device):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from dehaze.utils.io import load_rgb, pil_to_tensor, tensor_to_pil
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
    sheet.save(output_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--data', default='data/NH-HAZE')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing to reuse output directory: {out}')
    out.mkdir(parents=True)
    
    device = resolve_device(a.device)
    model, descriptor, ckpt = load_checkpoint_model(a.checkpoint, device)
    pairs = {x.image_id: x for x in discover_nh_haze_pairs(a.data)}
    split = load_split(a.split)
    validate_split(split, list(pairs))
    
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)
    
    log(f"--- TEST EVALUATION START ---")
    log(f"Checkpoint: {a.checkpoint}")
    log(f"Model Descriptor: {descriptor}")
    log(f"Device: {device}")
    log(f"Split: {a.split} (Test scenes: {split.test})")
    
    dataset = PairedDehazeDataset([pairs[x] for x in split.test])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Warmup GPU if CUDA
    if device.type == 'cuda':
        with torch.inference_mode():
            model(torch.zeros(1, 3, 256, 256, device=device))
        torch.cuda.synchronize()
    
    records = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            image_id = batch['id'][0]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_time = t1 - t0
            
            p_val = psnr(outputs[0], targets[0])
            s_val = ssim(outputs[0], targets[0])
            records.append({
                'id': image_id,
                'psnr': p_val,
                'ssim': s_val,
                'inference_seconds': step_time
            })
            log(f"Scene {image_id}: PSNR = {p_val:.4f} dB, SSIM = {s_val:.4f}, Time = {step_time*1000:.2f} ms")
            save_tensor_image(outputs[0], out / 'predictions' / f"{image_id}_pred.png")
    
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'psnr', 'ssim', 'inference_seconds'])
        w.writeheader()
        w.writerows(records)
    
    # Generate test contact sheet
    contact_sheet_path = out / 'test_contact_sheet.png'
    generate_contact_sheet(model, pairs, split.test, contact_sheet_path, device)
    log(f"Saved test contact sheet to: {contact_sheet_path}")
    
    elapsed_total = sum(r['inference_seconds'] for r in records)
    mean_psnr = sum(r['psnr'] for r in records) / len(records)
    mean_ssim = sum(r['ssim'] for r in records) / len(records)
    mean_time = elapsed_total / len(records)
    param_cnt = parameter_count(model)
    macs = conv2d_macs(model, (1, 3, 256, 256), device)
    
    summary = {
        'checkpoint': str(a.checkpoint),
        'model': descriptor,
        'test_images': len(records),
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'inference_seconds_total': elapsed_total,
        'inference_seconds_per_image': mean_time,
        'parameters': param_cnt,
        'conv2d_macs_256x256': macs,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'evaluation.log').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    log(f"--- TEST EVALUATION COMPLETE ---")
    log(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

