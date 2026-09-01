from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel


BACKBONE_NAMES = {f"b{i}": f"nvidia/mit-b{i}" for i in range(6)}
DEPTHS = {"b0": [2, 2, 2, 2], "b1": [2, 2, 2, 2], "b2": [3, 4, 6, 3],
          "b3": [3, 4, 18, 3], "b4": [3, 8, 27, 3], "b5": [3, 6, 40, 3]}


def segformer_config(variant: str, num_classes: int) -> SegformerConfig:
    if variant not in BACKBONE_NAMES:
        raise ValueError("variant must be b0..b5")
    large = variant != "b0"
    return SegformerConfig(
        num_labels=num_classes,
        hidden_sizes=[64, 128, 320, 512] if large else [32, 64, 160, 256],
        depths=DEPTHS[variant], num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        decoder_hidden_size=768 if large else 256,
    )


class SourceSegFormer(nn.Module):
    """MiT ImageNet backbone plus a newly initialized Cityscapes segmentation head."""

    def __init__(self, variant: str, num_classes: int, initialization: str = "mit_imagenet"):
        super().__init__()
        if initialization not in {"mit_imagenet", "random"}:
            raise ValueError("initialization must be mit_imagenet or random")
        self.variant, self.initialization = variant, initialization
        self.model = SegformerForSemanticSegmentation(segformer_config(variant, num_classes))
        if initialization == "mit_imagenet":
            backbone = SegformerModel.from_pretrained(BACKBONE_NAMES[variant])
            self.model.segformer.load_state_dict(backbone.state_dict(), strict=True)
            del backbone
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.mean.to(image)) / self.std.to(image)
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


def resolve_initialization(source_cfg: dict) -> str:
    if "initialization" in source_cfg:
        return source_cfg["initialization"]
    return "mit_imagenet" if source_cfg.get("pretrained", True) else "random"


def build_source_model(cfg: dict) -> SourceSegFormer:
    source, dataset = cfg["source"], cfg["dataset"]
    if source.get("architecture", "segformer") != "segformer":
        raise ValueError("only segformer source architecture is supported")
    return SourceSegFormer(source.get("variant", "b2"), int(dataset["num_classes"]), resolve_initialization(source))


def load_source_checkpoint(model: nn.Module, path: str, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    return ckpt

