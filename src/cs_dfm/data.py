from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF

from .cache_spec import LABEL_MAPPING_VERSION, cache_directory, canonical_size_hw


ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)
for raw_id, train_id in {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
                         21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                         28: 15, 31: 16, 32: 17, 33: 18}.items():
    ID_TO_20CLASS[raw_id] = train_id


def _resize(image, mask, logits, size):
    image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]
    mask = F.interpolate(mask[None, None].float(), size=size, mode="nearest")[0, 0].long()
    if logits is not None:
        logits = F.interpolate(logits[None].float(), size=size, mode="bilinear", align_corners=False)[0]
    return image, mask, logits


def _valid_crop(mask: torch.Tensor, top: int, left: int, size: tuple[int, int], ratio: float,
                ignore_class: int | None) -> bool:
    crop = mask[top:top + size[0], left:left + size[1]].flatten()
    if ignore_class is not None:
        crop = crop[crop != ignore_class]
    if crop.numel() == 0:
        return False
    counts = torch.bincount(crop)
    return float(counts.max()) / crop.numel() <= ratio


def _crop_position(mask, crop_size, cat_max_ratio, ignore_class, max_attempts):
    h, w = mask.shape; ch, cw = crop_size
    top = left = 0
    for _ in range(max_attempts):
        top = random.randint(0, h - ch); left = random.randint(0, w - cw)
        if cat_max_ratio >= 1 or _valid_crop(mask, top, left, crop_size, cat_max_ratio, ignore_class):
            break
    return top, left


class SegmentationTransform:
    """Config-driven transform; logits are geometrically transformed only for cache runtime."""

    def __init__(self, cfg: dict, split: str, conditioned: bool):
        self.cfg, self.split, self.conditioned = cfg, split, conditioned

    def __call__(self, image, mask, logits=None):
        d = self.cfg["dataset"]; pipeline = d.get("pipeline", "ccdm_fixed")
        if pipeline == "ccdm_fixed":
            a = d.get("augmentation", {}) if self.split == "train" else {}
            if a.get("random_resize", False):
                ratio = random.uniform(*a.get("ratio_range", [.75, 1.5]))
                h, w = image.shape[-2:]; image, mask, logits = _resize(image, mask, logits, (round(h*ratio), round(w*ratio)))
            if a.get("random_crop", False):
                image, mask, logits = self._crop(image, mask, logits, tuple(a.get("crop_size_hw", canonical_size_hw(self.cfg))), a)
            flip_p = float(a.get("horizontal_flip", a.get("horizontal_flip_probability", 0)))
            photo = bool(a.get("photometric", False))
        elif pipeline == "mmseg":
            p = d.get("train_pipeline", {}) if self.split == "train" else d.get("val_pipeline", {})
            if self.split != "train":
                image, mask, logits = _resize(image, mask, logits, tuple(p.get("resize_size_hw", canonical_size_hw(self.cfg))))
            else:
                rr = p.get("random_resize", {})
                if rr.get("enabled", False):
                    ratio = random.uniform(*rr.get("ratio_range", [.5, 2.]))
                    base = rr.get("base_size_hw", canonical_size_hw(self.cfg))
                    image, mask, logits = _resize(image, mask, logits, (max(1, round(base[0]*ratio)), max(1, round(base[1]*ratio))))
                rc = p.get("random_crop", {})
                if rc.get("enabled", False):
                    image, mask, logits = self._crop(image, mask, logits, tuple(rc["crop_size_hw"]), rc)
            flip_p = float(p.get("horizontal_flip", {}).get("probability", 0))
            photo = bool(p.get("photometric", {}).get("enabled", False))
        else:
            raise ValueError(f"unknown dataset.pipeline: {pipeline}")
        if random.random() < flip_p:
            image, mask = image.flip(-1), mask.flip(-1)
            if logits is not None: logits = logits.flip(-1)
        if photo:
            image = TF.adjust_brightness(image, random.uniform(.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(.8, 1.2))
        return image, mask, logits

    def _crop(self, image, mask, logits, crop_size, c):
        ch, cw = crop_size; pad_h, pad_w = max(0, ch-image.shape[-2]), max(0, cw-image.shape[-1])
        if pad_h or pad_w:
            void = int(self.cfg["dataset"].get("void_class_index", 19))
            image = F.pad(image, (0,pad_w,0,pad_h)); mask = F.pad(mask, (0,pad_w,0,pad_h), value=void)
            if logits is not None: logits = F.pad(logits, (0,pad_w,0,pad_h))
        ignore = c.get("ignore_class_for_ratio", None)
        top, left = _crop_position(mask, crop_size, float(c.get("cat_max_ratio", 1)), ignore, int(c.get("max_attempts", 10)))
        image = image[:,top:top+ch,left:left+cw]; mask = mask[top:top+ch,left:left+cw]
        if logits is not None: logits = logits[:,top:top+ch,left:left+cw]
        return image, mask, logits


class Cityscapes20(Dataset):
    def __init__(self, cfg: dict, split: str, return_logits: bool = False, augment: bool = False):
        d = cfg["dataset"]
        self.ds = Cityscapes(root=str(Path(d["root"]).expanduser()), split=split, mode="fine", target_type="semantic")
        self.cfg, self.split, self.return_logits = cfg, split, return_logits
        self.canonical_size = canonical_size_hw(cfg)
        conditioned = cfg.get("source_distribution", {}).get("type") == "image_conditioned"
        self.transform = SegmentationTransform(cfg, split, conditioned) if augment or split != "train" else None
        self.cache_root = cache_directory(cfg) if return_logits else None

    def __len__(self): return len(self.ds)

    def sample_id(self, idx):
        path = Path(self.ds.images[idx]); return f"{path.parent.name}__{path.stem}"

    def expected_sample_ids(self): return [self.sample_id(i) for i in range(len(self))]

    def __getitem__(self, idx):
        image, target = self.ds[idx]
        image = TF.pil_to_tensor(image).float()/255
        mask = torch.from_numpy(ID_TO_20CLASS[np.asarray(target,dtype=np.uint8)]).long()
        if self.cfg["dataset"].get("pipeline", "ccdm_fixed") == "ccdm_fixed":
            image, mask, _ = _resize(image, mask, None, self.canonical_size)
        sid = self.sample_id(idx); logits = None
        if self.return_logits:
            path = self.cache_root/self.split/f"{sid}.pt"
            if not path.exists(): raise FileNotFoundError(f"missing cached logits: {path}")
            logits = torch.load(path,map_location="cpu",weights_only=True)["logits"]
            if tuple(logits.shape) != (int(self.cfg["dataset"]["num_classes"]), *self.canonical_size):
                raise ValueError(f"cached logits shape mismatch for {sid}: {tuple(logits.shape)}")
        if self.transform: image, mask, logits = self.transform(image, mask, logits)
        out = {"image":image,"mask":mask,"sample_id":sid}
        if logits is not None: out["source_logits"] = logits
        return out


def build_dataset(cfg: dict, split: str, return_logits: bool = False, augment: bool = False) -> Cityscapes20:
    return Cityscapes20(cfg, split, return_logits, augment)
