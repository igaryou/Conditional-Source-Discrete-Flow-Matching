from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


MODEL_NAMES = {f"b{i}": f"nvidia/segformer-b{i}-finetuned-ade-512-512" for i in range(6)}
BACKBONE_NAMES = {f"b{i}": f"nvidia/mit-b{i}" for i in range(6)}
DEPTHS = {"b0": [2, 2, 2, 2], "b1": [2, 2, 2, 2], "b2": [3, 4, 6, 3],
          "b3": [3, 4, 18, 3], "b4": [3, 8, 27, 3], "b5": [3, 6, 40, 3]}


class SourceSegFormer(nn.Module):
    def __init__(self, variant: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if variant not in MODEL_NAMES:
            raise ValueError("variant must be b0..b5")
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                MODEL_NAMES[variant], num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            large = variant != "b0"
            cfg = SegformerConfig(
                num_labels=num_classes,
                hidden_sizes=[64, 128, 320, 512] if large else [32, 64, 160, 256],
                depths=DEPTHS[variant],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                decoder_hidden_size=768 if large else 256,
            )
            self.model = SegformerForSemanticSegmentation(cfg)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.mean.to(image)) / self.std.to(image)
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


def build_source_model(cfg: dict) -> SourceSegFormer:
    source, dataset = cfg["source"], cfg["dataset"]
    if source.get("architecture", "segformer") != "segformer":
        raise ValueError("only segformer source architecture is supported")
    return SourceSegFormer(source.get("variant", "b2"), int(dataset["num_classes"]), source.get("pretrained", True))


def load_source_checkpoint(model: nn.Module, path: str, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    return ckpt
