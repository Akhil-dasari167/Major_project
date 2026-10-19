"""Train V0 only when explicitly invoked."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.utils.data import DataLoader
from dehaze.data.dataset import discover_nh_haze_pairs, PairedDehazeDataset, EpochShuffleSampler
from dehaze.data.splits import make_split, save_split, load_split, validate_split
from dehaze.models.v0_baseline import V0ResidualCNN
from dehaze.training.engine import run_epoch, save_checkpoint, append_history, save_history_plot, write_json_once
from dehaze.utils.reproducibility import seed_everything, resolve_device

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/NH-HAZE'); p.add_argument('--output',default=None); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--patch-size',type=int,default=256); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda'); p.add_argument('--resume',default=None); p.add_argument('--max-train-steps',type=int,default=None); p.add_argument('--max-validation-steps',type=int,default=None); a=p.parse_args()
    run=Path(a.output) if a.output else Path('outputs')/'v0_'+time.strftime('%Y%m%d_%H%M%S')
    if run.exists() != bool(a.resume): raise FileExistsError('An existing output directory requires --resume; a new run must not use --resume.')
    if not run.exists(): run.mkdir(parents=True)
    seed_everything(a.seed); device=resolve_device(a.device); pairs=discover_nh_haze_pairs(a.data); lookup={x.image_id:x for x in pairs}; all_ids=list(lookup)
    config=vars(a)|{'device_resolved':str(device),'model':'V0ResidualCNN'}
    if a.resume:
        checkpoint_path=Path(a.resume)
        if not checkpoint_path.is_file(): raise FileNotFoundError(checkpoint_path)
        manifest=load_split(run/'split.json'); validate_split(manifest,all_ids)
    else:
        manifest=make_split(all_ids,a.seed); validate_split(manifest,all_ids); save_split(manifest,run/'split.json'); write_json_once(run/'config.json',config)
    train=PairedDehazeDataset([lookup[x] for x in manifest.train],a.patch_size,True,a.seed); validation=PairedDehazeDataset([lookup[x] for x in manifest.validation],None,False,a.seed)
    train_sampler=EpochShuffleSampler(train,a.seed); train_loader=DataLoader(train,batch_size=a.batch_size,sampler=train_sampler,num_workers=0); val_loader=DataLoader(validation,batch_size=1,shuffle=False,num_workers=0)
    model=V0ResidualCNN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=a.lr); loss_fn=torch.nn.L1Loss(); history=[]; best=float('inf'); start_epoch=1
    if a.resume:
        checkpoint=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer_state']); start_epoch=int(checkpoint['epoch'])+1
        prior_best=[torch.load(path,map_location='cpu',weights_only=False)['validation_loss'] for path in run.glob('best_epoch_*.pt')]
        best=min(prior_best, default=float(checkpoint['validation_loss']))
    for epoch in range(start_epoch,a.epochs+1):
        train.set_epoch(epoch); train_sampler.set_epoch(epoch); tr=run_epoch(model,train_loader,optimizer,loss_fn,device,True,a.max_train_steps); va=run_epoch(model,val_loader,None,loss_fn,device,False,a.max_validation_steps); row={'epoch':epoch,'train_l1':tr,'validation_l1':va}; history.append(row); save_checkpoint(run/f'epoch_{epoch:03d}.pt',model,optimizer,epoch,config,va)
        if va < best: best=va; save_checkpoint(run/f'best_epoch_{epoch:03d}.pt',model,optimizer,epoch,config,va)
        append_history(run/'history.csv',row); save_history_plot(run/f'loss_epoch_{epoch:03d}.png',history); print(row)
if __name__=='__main__': main()
