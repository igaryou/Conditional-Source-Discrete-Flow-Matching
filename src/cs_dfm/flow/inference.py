from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedulers import LinearScheduler, PowerScheduler


@torch.no_grad()
def sample_two_term(model, image: torch.Tensor, z0: torch.Tensor, scheduler, num_classes: int,
                    num_steps: int = 50, argmax_final: bool = False,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Reference-compatible probability-velocity jump sampler for a two-term path."""
    if not isinstance(scheduler, (LinearScheduler, PowerScheduler)):
        raise TypeError("sample_two_term requires a linear or power scheduler")
    zt = z0.clone(); b, h, w = zt.shape; times = torch.linspace(0, 1, num_steps + 1, device=image.device)
    power = scheduler.power if isinstance(scheduler, PowerScheduler) else 1.0
    for i in range(num_steps):
        t_scalar, next_t = times[i], times[i + 1]
        t = t_scalar.expand(b); probs = torch.softmax(model(image, zt, t), dim=1)
        if i == num_steps - 1:
            if argmax_final:
                zt = probs.argmax(1)
            else:
                flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
                zt = torch.multinomial(flat, 1, generator=generator).view(b, h, w)
            continue
        safe_t = t_scalar.clamp_min(1e-6)
        kappa = safe_t.pow(power); derivative = power * safe_t.pow(power - 1)
        rate = derivative / (1 - kappa).clamp_min(1e-8)
        rates = rate * probs.permute(0, 2, 3, 1)
        rates = torch.where(F.one_hot(zt, num_classes).bool(), 0, rates).clamp_min(0)
        intensity = rates.sum(-1); jump_prob = 1 - torch.exp(-(next_t - t_scalar) * intensity)
        jump = torch.rand((b, h, w), device=image.device, generator=generator) < jump_prob
        if jump.any():
            candidates = rates[jump]; candidates /= candidates.sum(-1, keepdim=True).clamp_min(1e-12)
            zt = zt.clone(); zt[jump] = torch.multinomial(candidates, 1, generator=generator)[:, 0]
    return zt

