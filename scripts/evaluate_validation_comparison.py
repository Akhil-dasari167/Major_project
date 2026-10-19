"""Validation-only comparison of Experiment B and an isolated candidate."""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from PIL import Image, ImageDraw, ImageOps
from dehaze.data.dataset import ImagePair
from dehaze.data.splits import load_split
from dehaze.evaluation.checkpoints import load_checkpoint_model
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.models.v0_baseline import parameter_count
from dehaze.utils.io import load_rgb, pil_to_tensor, tensor_to_pil
from dehaze.utils.reproducibility import resolve_device

def pairs(root, ids):
    return [ImagePair(i, root / f"{i}_hazy.png", root / f"{i}_GT.png") for i in ids]

def run(label, checkpoint, records, device):
    model, descriptor, _ = load_checkpoint_model(checkpoint, device); model.eval()
    with torch.inference_mode():
        warm = pil_to_tensor(load_rgb(records[0].hazy_path)).unsqueeze(0).to(device); model(warm)
    if device.type == "cuda": torch.cuda.synchronize()
    result, previews = [], []
    for pair in records:
        hazy, target = load_rgb(pair.hazy_path), load_rgb(pair.target_path)
        inputs, targets = pil_to_tensor(hazy).unsqueeze(0).to(device), pil_to_tensor(target).unsqueeze(0).to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        t0=time.perf_counter()
        with torch.inference_mode(): output=model(inputs)[0]
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed=time.perf_counter()-t0
        result.append({"model":label,"id":pair.image_id,"psnr":psnr(output,targets[0]),"ssim":ssim(output,targets[0]),"l1":float(torch.mean(torch.abs(output.cpu()-targets[0].cpu()))),"inference_seconds":elapsed})
        previews.append((pair.image_id,hazy,target,tensor_to_pil(output)))
    summary={"label":label,"checkpoint":str(checkpoint),"parameters":parameter_count(model),"checkpoint_bytes":checkpoint.stat().st_size,"mean_psnr":sum(x["psnr"] for x in result)/len(result),"mean_ssim":sum(x["ssim"] for x in result)/len(result),"mean_l1":sum(x["l1"] for x in result)/len(result),"mean_inference_seconds":sum(x["inference_seconds"] for x in result)/len(result),"descriptor":descriptor}
    return summary,result,previews

def contact_sheet(path, reference, candidate):
    w,h,label=200,150,18; sheet=Image.new("RGB",(w*4,(h+label)*len(reference)),"white"); draw=ImageDraw.Draw(sheet)
    for row, ((image_id,hazy,target,ref),(candidate_id,_,_,cand)) in enumerate(zip(reference,candidate)):
        assert image_id==candidate_id
        for column,(name,image) in enumerate((("Hazy",hazy),("GT",target),("Exp B",ref),("Candidate",cand))):
            draw.text((column*w+3,row*(h+label)+2),f"{image_id} {name}",fill="black")
            sheet.paste(ImageOps.fit(image,(w,h),method=Image.Resampling.LANCZOS),(column*w,row*(h+label)+label))
    sheet.save(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--reference",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--split",type=Path,required=True); p.add_argument("--data",type=Path,default=Path("data/NH-HAZE")); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(f"Refusing to overwrite {a.output}")
    manifest=load_split(a.split); validation=pairs(a.data,manifest.validation); device=resolve_device(a.device); a.output.mkdir(parents=True)
    ref_s,ref_r,ref_p=run("Experiment B reference",a.reference,validation,device); can_s,can_r,can_p=run("Multi-model v1",a.candidate,validation,device)
    with (a.output/"per_image_metrics.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=ref_r[0].keys()); w.writeheader(); w.writerows(ref_r+can_r)
    summary={"protocol":"validation-only; no test images used","device":str(device),"validation_ids":manifest.validation,"models":[ref_s,can_s]}; (a.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    contact_sheet(a.output/"validation_comparison.png",ref_p,can_p)
    recommendation="Experiment B reference" if ref_s["mean_l1"] < can_s["mean_l1"] else "Multi-model v1"
    report=f"# Validation Comparison\n\nProtocol: validation-only; the previously examined test split was not accessed.\n\n| Model | PSNR | SSIM | L1 | Params | Checkpoint bytes | Mean latency |\n|---|---:|---:|---:|---:|---:|---:|\n| Experiment B | {ref_s['mean_psnr']:.4f} | {ref_s['mean_ssim']:.4f} | {ref_s['mean_l1']:.6f} | {ref_s['parameters']:,} | {ref_s['checkpoint_bytes']:,} | {ref_s['mean_inference_seconds']*1000:.2f} ms |\n| Multi-model v1 | {can_s['mean_psnr']:.4f} | {can_s['mean_ssim']:.4f} | {can_s['mean_l1']:.6f} | {can_s['parameters']:,} | {can_s['checkpoint_bytes']:,} | {can_s['mean_inference_seconds']*1000:.2f} ms |\n\nRecommendation: **{recommendation}**. The candidate is preserved as an experimental result; it was not selected if its validation L1 is higher.\n"
    (a.output/"report.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
