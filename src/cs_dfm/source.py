from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

from .flow.sampling import sample_source
from .utils import amp_context


BACKBONE_NAMES = {f"b{i}": f"nvidia/mit-b{i}" for i in range(6)}
HIDDEN_SIZES = {
    "b0": [32, 64, 160, 256],
    "b1": [64, 128, 320, 512],
    "b2": [64, 128, 320, 512],
    "b3": [64, 128, 320, 512],
    "b4": [64, 128, 320, 512],
    "b5": [64, 128, 320, 512],
}
DEPTHS = {
    "b0": [2, 2, 2, 2],
    "b1": [2, 2, 2, 2],
    "b2": [3, 4, 6, 3],
    "b3": [3, 4, 18, 3],
    "b4": [3, 8, 27, 3],
    "b5": [3, 6, 40, 3],
}
DECODER_HIDDEN_SIZES = {
    "b0": 256,
    "b1": 256,
    "b2": 768,
    "b3": 768,
    "b4": 768,
    "b5": 768,
}


def segformer_config(variant: str, num_classes: int) -> SegformerConfig:
    if variant not in BACKBONE_NAMES:
        raise ValueError("variant must be b0..b5")
    return SegformerConfig(
        num_labels=num_classes,
        hidden_sizes=HIDDEN_SIZES[variant],
        depths=DEPTHS[variant], num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        decoder_hidden_size=DECODER_HIDDEN_SIZES[variant],
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


class Stage2SourceProvider:
    """Provide samples from uniform, cached, or frozen-online source logits."""

    def __init__(self, cfg: dict, device: torch.device, mode: str, model: nn.Module | None = None):
        self.cfg, self.device, self.mode, self.model = cfg, device, mode, model

    @property
    def needs_cache(self) -> bool:
        return self.mode == "cache"

    def sample(self, batch: dict, image: torch.Tensor, shape: tuple[int, int, int],
               generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.cfg["source_distribution"]
        source_logits = None
        if self.mode == "cache":
            source_logits = batch.get("source_logits")
            if source_logits is None:
                raise ValueError("cached source runtime requires batch['source_logits']")
            source_logits = source_logits.to(self.device)
        elif self.mode == "online":
            if self.model is None:
                raise RuntimeError("online source runtime has no source model")
            runtime = self.cfg.get("runtime", {})
            with torch.inference_mode():
                with amp_context(bool(runtime.get("amp", False)), runtime.get("amp_dtype", "bf16"), self.device):
                    source_logits = self.model(image)
        elif self.mode != "none":
            raise ValueError(f"unknown source runtime mode: {self.mode}")
        result = sample_source(
            d["type"], int(self.cfg["dataset"]["num_classes"]), shape, self.device,
            source_logits, float(d.get("lambda", 0)), float(d.get("temperature", 1)), generator,
        )
        source_logits = None
        return result


def build_stage2_source_provider(cfg: dict, device: torch.device) -> Stage2SourceProvider:
    source_type = cfg["source_distribution"]["type"]
    mode = cfg.get("source_runtime", {}).get("mode")
    if source_type == "uniform":
        if mode != "none":
            raise ValueError("uniform source requires source_runtime.mode=none")
        return Stage2SourceProvider(cfg, device, mode="none")
    if mode == "cache":
        return Stage2SourceProvider(cfg, device, mode="cache")
    if mode != "online":
        raise ValueError("image-conditioned source runtime must be cache or online")
    checkpoint = cfg.get("source", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("MMSeg online conditioned Stage 2 requires source.checkpoint")
    if not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"MMSeg online source checkpoint does not exist: {checkpoint}")
    model = build_source_model(cfg)
    load_source_checkpoint(model, checkpoint, map_location="cpu")
    model.eval().requires_grad_(False).to(device)
    return Stage2SourceProvider(cfg, device, mode="online", model=model)
