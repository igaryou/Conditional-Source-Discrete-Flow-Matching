from __future__ import annotations

import torch

from .schedulers import LinearScheduler, PowerScheduler, PowerUniformBumpScheduler


class ProbabilityPath:
    def __init__(self, path_type: str, scheduler, num_classes: int):
        self.path_type, self.scheduler, self.num_classes = path_type, scheduler, num_classes

    @torch.no_grad()
    def sample(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
        if z0.shape != z1.shape or z0.ndim != 3:
            raise ValueError("z0 and z1 must have equal [B,H,W] shapes")
        b, h, w = z0.shape
        if self.path_type == "two_term":
            kappa = self.scheduler(t)[:, None, None]
            u = torch.rand((b, h, w), device=z0.device, generator=generator)
            return torch.where(u < kappa, z1, z0)
        weights = self.scheduler(t)
        choices = torch.stack([weights.kappa1, weights.kappa2, weights.kappa3], dim=1)
        branch = torch.multinomial(choices, h * w, replacement=True, generator=generator).view(b, h, w)
        uniform = torch.randint(self.num_classes, (b, h, w), device=z0.device, generator=generator)
        return torch.where(branch == 0, z1, torch.where(branch == 1, uniform, z0))

    def weights(self, t: torch.Tensor):
        return self.scheduler(t)


def build_path(cfg: dict, num_classes: int) -> ProbabilityPath:
    path_type = cfg.get("type", "two_term")
    name = cfg.get("scheduler", "linear")
    if path_type == "two_term":
        scheduler = LinearScheduler() if name == "linear" else PowerScheduler(float(cfg.get("power", 1)))
    elif path_type == "three_term" and name == "power_uniform_bump":
        scheduler = PowerUniformBumpScheduler(float(cfg.get("power", 1)), float(cfg.get("uniform_strength", .3)))
    else:
        raise ValueError(f"unsupported path/scheduler: {path_type}/{name}")
    return ProbabilityPath(path_type, scheduler, num_classes)

