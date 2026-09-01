from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF


ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)
for raw_id, train_id in {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}.items():
    ID_TO_20CLASS[raw_id] = train_id


class JointGeometry:
    """Apply exactly the same geometry to image, label and cached continuous logits."""

    def __init__(self, crop_size: tuple[int, int] | None, scale: tuple[float, float], hflip: float):
        self.crop_size = crop_size
        self.scale = scale
        self.hflip = hflip

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor | None = None):
        factor = random.uniform(*self.scale)
        h, w = image.shape[-2:]
        size = (max(1, round(h * factor)), max(1, round(w * factor)))
        image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]
        mask = F.interpolate(mask[None, None].float(), size=size, mode="nearest")[0, 0].long()
        if logits is not None:
            logits = F.interpolate(logits[None].float(), size=size, mode="bilinear", align_corners=False)[0]
        if self.crop_size:
            ch, cw = self.crop_size
            pad_h, pad_w = max(0, ch - size[0]), max(0, cw - size[1])
            if pad_h or pad_w:
                image = F.pad(image, (0, pad_w, 0, pad_h))
                mask = F.pad(mask, (0, pad_w, 0, pad_h), value=19)
                if logits is not None:
                    logits = F.pad(logits, (0, pad_w, 0, pad_h))
            top = random.randint(0, image.shape[-2] - ch)
            left = random.randint(0, image.shape[-1] - cw)
            image = image[:, top:top + ch, left:left + cw]
            mask = mask[top:top + ch, left:left + cw]
            if logits is not None:
                logits = logits[:, top:top + ch, left:left + cw]
        if random.random() < self.hflip:
            image, mask = image.flip(-1), mask.flip(-1)
            if logits is not None:
                logits = logits.flip(-1)
        return image, mask, logits


class Cityscapes20(Dataset):
    """Cityscapes mapping kept compatible with the reference dfm repository."""

    def __init__(self, root: str, split: str, image_size: tuple[int, int], cache_root: str | None = None,
                 augment: JointGeometry | None = None, return_logits: bool = False):
        self.ds = Cityscapes(root=str(Path(root).expanduser()), split=split, mode="fine", target_type="semantic")
        self.split = split
        self.image_size = tuple(image_size)
        self.cache_root = Path(cache_root).expanduser() if cache_root else None
        self.augment = augment
        self.return_logits = return_logits

    def __len__(self):
        return len(self.ds)

    def sample_id(self, idx: int) -> str:
        image_path = Path(self.ds.images[idx])
        return f"{image_path.parent.name}__{image_path.stem}"

    def __getitem__(self, idx: int):
        image, target = self.ds[idx]
        image = TF.resize(TF.pil_to_tensor(image).float() / 255, self.image_size,
                          interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        mask = torch.from_numpy(ID_TO_20CLASS[np.asarray(target, dtype=np.uint8)]).long()
        mask = TF.resize(mask[None], self.image_size, interpolation=TF.InterpolationMode.NEAREST)[0].long()
        sample_id = self.sample_id(idx)
        logits = None
        if self.return_logits:
            if self.cache_root is None:
                raise ValueError("cache_root is required when return_logits=True")
            path = self.cache_root / self.split / f"{sample_id}.pt"
            if not path.exists():
                raise FileNotFoundError(f"missing cached logits: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            logits = payload["logits"] if isinstance(payload, dict) else payload
            if logits.shape[-2:] != image.shape[-2:]:
                raise ValueError(f"cache/image spatial mismatch for {sample_id}: {logits.shape} vs {image.shape}")
        if self.augment:
            image, mask, logits = self.augment(image, mask, logits)
        result = {"image": image, "mask": mask, "sample_id": sample_id}
        if logits is not None:
            result["source_logits"] = logits
        return result


def build_dataset(cfg: dict, split: str, return_logits: bool = False, augment: bool = False) -> Cityscapes20:
    ds_cfg, cache_cfg = cfg["dataset"], cfg.get("source_cache", {})
    aug = None
    if augment:
        a = ds_cfg.get("augmentation", {})
        aug = JointGeometry(tuple(a["crop_size"]) if a.get("crop_size") else None,
                            tuple(a.get("scale", [1.0, 1.0])), float(a.get("horizontal_flip", 0.0)))
    return Cityscapes20(ds_cfg["root"], split, tuple(ds_cfg["image_size"]), cache_cfg.get("root"), aug,
                        return_logits)

