"""Unicode-path-safe RGB image loading and saving."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(Path(path)) as image:
        return image.convert("RGB").copy()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)
