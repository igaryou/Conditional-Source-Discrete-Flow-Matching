from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        emb = torch.cat((torch.sin(t[:, None] * freq), torch.cos(t[:, None] * freq)), 1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


def norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.net1 = nn.Sequential(norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.net2 = nn.Sequential(norm(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.time = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.net1(x) + self.time(F.silu(temb))[:, :, None, None]
        return self.net2(h) + self.skip(x)


class SegDiffModel(nn.Module):
    """Image/zt/t conditional predictor with the reference objective's [B,K,H,W] output."""

    def __init__(self, num_classes: int = 20, base_channels: int = 64, channel_mults=(1, 2, 4),
                 num_res_blocks: int = 2, time_dim: int = 128, mask_embed_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.mask_embed = nn.Embedding(num_classes, mask_embed_dim)
        self.image_in = nn.Conv2d(3, mask_embed_dim, 3, padding=1)
        self.fuse = nn.Conv2d(mask_embed_dim * 2, base_channels, 3, padding=1)
        self.time = TimeEmbedding(time_dim)
        self.down, self.downsample, skips = nn.ModuleList(), nn.ModuleList(), []
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out, time_dim)); ch = out; skips.append(ch)
            self.down.append(blocks)
            self.downsample.append(nn.Conv2d(ch, ch, 3, 2, 1) if i < len(channel_mults) - 1 else nn.Identity())
        self.middle = ResBlock(ch, ch, time_dim)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        skip_stack = list(reversed(skips))
        for i, mult in enumerate(reversed(channel_mults)):
            out = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch + skip_stack.pop(0), out, time_dim)); ch = out
            self.up.append(blocks)
            self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1))
                                 if i < len(channel_mults) - 1 else nn.Identity())
        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, num_classes, 3, padding=1))

    def forward(self, image: torch.Tensor, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if zt.ndim == 4:
            zt = zt[:, 0]
        mask = self.mask_embed(zt.long()).permute(0, 3, 1, 2)
        h = self.fuse(torch.cat((self.image_in(image), mask), 1))
        temb, saved = self.time(t), []
        for blocks, down in zip(self.down, self.downsample):
            for block in blocks:
                h = block(h, temb); saved.append(h)
            h = down(h)
        h = self.middle(h, temb)
        for blocks, up in zip(self.up, self.upsample):
            for block in blocks:
                skip = saved.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, skip.shape[-2:], mode="nearest")
                h = block(torch.cat((h, skip), 1), temb)
            h = up(h)
        if h.shape[-2:] != image.shape[-2:]:
            h = F.interpolate(h, image.shape[-2:], mode="nearest")
        return self.out(h)


def build_dfm_model(cfg: dict) -> SegDiffModel:
    m = cfg.get("model", {})
    return SegDiffModel(num_classes=int(cfg["dataset"]["num_classes"]),
                        base_channels=int(m.get("base_channels", 64)),
                        channel_mults=tuple(m.get("channel_mults", [1, 2, 4])),
                        num_res_blocks=int(m.get("num_res_blocks", 2)),
                        time_dim=int(m.get("time_dim", 128)),
                        mask_embed_dim=int(m.get("mask_embed_dim", 64)))

