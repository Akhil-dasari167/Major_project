"""Pre-registered 20-epoch, train/validation-only residual-scale comparison."""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont, ImageOps
from dehaze.data.dataset import discover_nh_haze_pairs, EpochShuffleSampler, PairedDehazeDataset
from dehaze.data.splits import load_split, save_split, validate_split
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.models.residual_scale_variant import ResidualScaleVariant
from dehaze.training.engine import save_checkpoint, write_json_once
from dehaze.utils.io import load_rgb, pil_to_tensor, tensor_to_pil
from dehaze.utils.reproducibility import resolve_device, seed_everything

ROOT=Path(__file__).resolve().parents[1]; SCALES=(0.1,0.2); EPOCHS=20; SEED=42; PATCH=256; BATCH=2; LR=2e-4

def epoch(model, loader, optimizer, device, training, scale):
    model.train(training); total=0.; samples=batches=0; pvals=[]; svals=[]; absv=[]; signed=[]; near=[]; clipped=[]
    context=torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            x,y=batch['input'].to(device),batch['target'].to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            output=model(x); loss=torch.nn.functional.l1_loss(output,y)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite L1 loss')
            if training:
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise FloatingPointError('Non-finite gradient')
                optimizer.step()
            total += loss.item()*x.size(0); samples += x.size(0); batches += 1
            if not training:
                correction=output-x; absv.append(correction.abs().mean().item()); signed.append(correction.mean().item()); near.append((correction.abs()>=scale*.95).float().mean().item()); clipped.append(((output<=0)|(output>=1)).float().mean().item())
                for i in range(x.size(0)): pvals.append(psnr(output[i],y[i])); svals.append(ssim(output[i],y[i]))
    result={'l1':total/samples,'samples':samples,'batches':batches}
    if not training: result.update(psnr=sum(pvals)/len(pvals),ssim=sum(svals)/len(svals),mean_abs_correction=sum(absv)/len(absv),mean_signed_correction=sum(signed)/len(signed),near_bound_fraction=sum(near)/len(near),clamped_fraction=sum(clipped)/len(clipped))
    return result

def contact(model,pairs,ids,path,device):
    w,h,label=220,165,16; font=ImageFont.load_default(); sheet=Image.new('RGB',(w*3,(h+label)*len(ids)),'white'); model.eval()
    with torch.inference_mode():
        for r,image_id in enumerate(ids):
            pair=pairs[image_id]; hazy=load_rgb(pair.hazy_path); target=load_rgb(pair.target_path); pred=tensor_to_pil(model(pil_to_tensor(hazy).unsqueeze(0).to(device))[0])
            for c,(name,img) in enumerate((('Hazy',hazy),('Ground truth',target),('Prediction',pred))):
                panel=Image.new('RGB',(w,h+label),'white'); ImageDraw.Draw(panel).text((3,2),f'{image_id} {name}',fill='black',font=font); panel.paste(ImageOps.fit(img,(w,h),method=Image.Resampling.LANCZOS),(0,label)); sheet.paste(panel,(c*w,r*(h+label)))
    sheet.save(path)

def run(scale,run_dir,pairs,manifest,device):
    if run_dir.exists(): raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True); seed_everything(SEED); save_split(manifest,run_dir/'split.json')
    config={'study':'residual_scale_continuation','model_type':'residual_scale_variant','residual_scale':scale,'epochs':EPOCHS,'patch_size':PATCH,'batch_size':BATCH,'learning_rate':LR,'optimizer':'Adam','loss':'L1','seed':SEED,'device':str(device),'split_source':'outputs/v0_pilot_001/split.json','expected_train_steps_per_epoch':20,'expected_validation_steps_per_epoch':8}
    write_json_once(run_dir/'config.json',config)
    train=PairedDehazeDataset([pairs[i] for i in manifest.train],PATCH,True,SEED); val=PairedDehazeDataset([pairs[i] for i in manifest.validation],None,False,SEED); sampler=EpochShuffleSampler(train,SEED); tl=DataLoader(train,batch_size=BATCH,sampler=sampler,num_workers=0); vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    model=ResidualScaleVariant(scale).to(device); opt=torch.optim.Adam(model.parameters(),lr=LR); rows=[]; best=float('inf'); best_path=None; started=time.perf_counter()
    for e in range(1,EPOCHS+1):
        train.set_epoch(e); sampler.set_epoch(e); tr=epoch(model,tl,opt,device,True,scale); va=epoch(model,vl,None,device,False,scale)
        if tr['batches']!=20 or tr['samples']!=39 or va['batches']!=8 or va['samples']!=8: raise RuntimeError(f'Unexpected step count at epoch {e}: {tr}, {va}')
        row={'epoch':e,'train_l1':tr['l1'],'validation_l1':va['l1'],'validation_psnr':va['psnr'],'validation_ssim':va['ssim'],'train_batches':tr['batches'],'validation_batches':va['batches'],'skipped_train_batches':0,'skipped_validation_batches':0,**{k:va[k] for k in ['mean_abs_correction','mean_signed_correction','near_bound_fraction','clamped_fraction']}}; rows.append(row)
        save_checkpoint(run_dir/f'epoch_{e:03d}.pt',model,opt,e,config,va['l1'])
        if va['l1']<best: best=va['l1']; best_path=run_dir/f'best_epoch_{e:03d}.pt'; save_checkpoint(best_path,model,opt,e,config,va['l1'])
        print({'scale':scale,**row})
    with (run_dir/'history.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    checkpoint=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(checkpoint['model_state']); contact(model,pairs,manifest.validation,run_dir/'validation_contact_sheet_best.png',device)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot([r['epoch'] for r in rows],[r['train_l1'] for r in rows],label='train L1'); ax.plot([r['epoch'] for r in rows],[r['validation_l1'] for r in rows],label='validation L1'); ax.set(xlabel='epoch',ylabel='L1',title=f'Residual scale {scale}'); ax.legend(); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(run_dir/'loss_curve.png',dpi=160); plt.close(fig)
    summary={'scale':scale,'best_checkpoint':str(best_path.relative_to(ROOT)),'best_row':min(rows,key=lambda x:x['validation_l1']),'runtime_seconds':time.perf_counter()-started,'actual_train_steps':sum(r['train_batches'] for r in rows),'actual_validation_steps':sum(r['validation_batches'] for r in rows),'skipped_train_batches':0,'skipped_validation_batches':0}
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return rows,summary

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--scales',nargs='+',type=float,default=list(SCALES)); parser.add_argument('--suffix',default=''); args=parser.parse_args()
    scales=tuple(args.scales)
    if any(scale not in SCALES for scale in scales) or not scales: raise ValueError(f'scales must be a non-empty subset of {SCALES}')
    suffix=f'_{args.suffix}' if args.suffix else ''
    device=resolve_device('cuda')
    if device.type!='cuda': raise RuntimeError('CUDA required by approved protocol')
    pairs={p.image_id:p for p in discover_nh_haze_pairs(ROOT/'data'/'NH-HAZE')}; manifest=load_split(ROOT/'outputs'/'v0_pilot_001'/'split.json'); validate_split(manifest,list(pairs))
    all_rows={}; summaries={}
    for scale in scales: all_rows[scale],summaries[scale]=run(scale,ROOT/'outputs'/f'residual_scale_continuation_{str(scale).replace(".","p")}{suffix}',pairs,manifest,device)
    analysis=ROOT/'analysis'/f'residual_scale_continuation{suffix}'; analysis.mkdir(parents=True,exist_ok=True); plot=analysis/'validation_comparison.png'; summary=analysis/'continuation_summary.json'
    if plot.exists() or summary.exists(): raise FileExistsError('Refusing to overwrite continuation analysis artifacts')
    fig,ax=plt.subplots(figsize=(7,4));
    for scale,rows in all_rows.items(): ax.plot([r['epoch'] for r in rows],[r['validation_l1'] for r in rows],marker='o',label=f'scale {scale}')
    ax.set(xlabel='epoch',ylabel='validation L1',title='Residual-scale continuation'); ax.legend(); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(plot,dpi=160); plt.close(fig)
    summary.write_text(json.dumps(summaries,indent=2),encoding='utf-8'); print(json.dumps(summaries,indent=2))
if __name__=='__main__': main()
