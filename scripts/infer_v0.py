"""Dehaze one image or every supported image in a folder."""
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from dehaze.models.v0_baseline import V0ResidualCNN
from dehaze.utils.io import load_rgb, pil_to_tensor, save_tensor_image
from dehaze.utils.reproducibility import resolve_device
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); a=p.parse_args(); source,output=Path(a.input),Path(a.output)
 if output.exists(): raise FileExistsError(f'Refusing to overwrite output: {output}')
 device=resolve_device(a.device); ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False); model=V0ResidualCNN().to(device).eval(); model.load_state_dict(ckpt['model_state']); files=[source] if source.is_file() else [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}]
 if not files: raise ValueError('Input must be an image or a folder containing images.')
 if source.is_dir(): output.mkdir(parents=True)
 with torch.inference_mode():
  for file in files:
   pred=model(pil_to_tensor(load_rgb(file)).unsqueeze(0).to(device))[0]; dest=(output/file.name) if source.is_dir() else output; save_tensor_image(pred,dest)
if __name__=='__main__': main()
