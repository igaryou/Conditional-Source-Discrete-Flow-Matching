from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedulers import LinearScheduler, PowerScheduler, PowerUniformBumpScheduler


def _jump_step(zt: torch.Tensor, rates: torch.Tensor, h: torch.Tensor,
               generator: torch.Generator | None) -> torch.Tensor:
    intensity = rates.sum(-1)
    jump = torch.rand(zt.shape, device=zt.device, generator=generator) < 1 - torch.exp(-h * intensity)
    if jump.any():
        candidates = rates[jump]
        candidates = candidates / candidates.sum(-1, keepdim=True).clamp_min(1e-12)
        zt = zt.clone()
        zt[jump] = torch.multinomial(candidates, 1, generator=generator)[:, 0]
    return zt


def two_term_rates(p1: torch.Tensor, zt: torch.Tensor, t: torch.Tensor, scheduler) -> torch.Tensor:
    """Off-diagonal CTMC rates q_t(zt -> x) for the two-term x-prediction velocity."""
    kappa, derivative = scheduler.value_derivative(t)
    coef = derivative / (1 - kappa).clamp_min(1e-8)
    rates = coef[:, None, None, None] * p1.permute(0, 2, 3, 1)
    return torch.where(F.one_hot(zt, p1.shape[1]).bool(), 0, rates).clamp_min(0)


def three_term_coefficients(t: torch.Tensor, scheduler: PowerUniformBumpScheduler):
    """DFM Theorem-3 coefficients with the required dynamic ell=argmin(dot(kappa)/kappa)."""
    w = scheduler(t)
    kappas = torch.stack((w.kappa1, w.kappa2, w.kappa3), 1)
    derivatives = torch.stack((w.d_kappa1, w.d_kappa2, w.d_kappa3), 1)
    ratios = torch.where(kappas > 1e-12, derivatives / kappas.clamp_min(1e-12), torch.full_like(kappas, float("inf")))
    base = ratios.min(1).values
    coefficients = derivatives - kappas * base[:, None]
    return coefficients.clamp_min(0), base, w


def source_posterior_from_target_posterior(p1: torch.Tensor, source_probs: torch.Tensor,
                                            zt: torch.Tensor, w) -> torch.Tensor:
    """Analytic p(x0|zt,x) from p(x1|zt,x), known p0, and independent endpoint coupling."""
    k = p1.shape[1]; current = F.one_hot(zt, k).permute(0, 3, 1, 2).to(p1)
    p0_z = (source_probs * current).sum(1, keepdim=True)
    p1_z = (p1 * current).sum(1, keepdim=True)
    k1 = w.kappa1[:,None,None,None]; k2 = w.kappa2[:,None,None,None]; k3 = w.kappa3[:,None,None,None]
    c = k2 / k + k3 * p0_z
    normalizer = (p1 * (1-current)).sum(1,keepdim=True) + p1_z * c / (k1+c).clamp_min(1e-12)
    scale = normalizer.clamp_min(1e-12).reciprocal()  # p_t(z) / c
    q_z = scale * p1_z * c / (k1+c).clamp_min(1e-12)
    d = k2 / k + k1 * q_z
    pt_z = scale * c
    posterior = source_probs * d / pt_z.clamp_min(1e-12)
    posterior = posterior + current * source_probs * k3 / pt_z.clamp_min(1e-12)
    return posterior / posterior.sum(1,keepdim=True).clamp_min(1e-12)


def three_term_rates(p1: torch.Tensor, zt: torch.Tensor, t: torch.Tensor,
                     scheduler: PowerUniformBumpScheduler, source_probs: torch.Tensor) -> torch.Tensor:
    r"""Exact Theorem-3 marginal rates using target, uniform, and analytic source posteriors."""
    if scheduler.uniform_strength == 0:
        return two_term_rates(p1, zt, t, PowerScheduler(scheduler.power))
    coefficients, _, w = three_term_coefficients(t, scheduler)
    k = p1.shape[1]
    p0_post = source_posterior_from_target_posterior(p1, source_probs, zt, w)
    rates_chw = (coefficients[:,0,None,None,None] * p1 + coefficients[:,1,None,None,None] / k
                 + coefficients[:,2,None,None,None] * p0_post)
    rates = rates_chw.permute(0,2,3,1)
    return torch.where(F.one_hot(zt, k).bool(), 0, rates).clamp_min(0)


@torch.no_grad()
def _sample(model, image: torch.Tensor, z0: torch.Tensor, scheduler, num_classes: int,
            num_steps: int, argmax_final: bool, generator: torch.Generator | None,
            three_term: bool, source_probs: torch.Tensor | None = None) -> torch.Tensor:
    zt = z0.clone(); b, h, w = zt.shape
    times = torch.linspace(0, 1, num_steps + 1, device=image.device)
    for i in range(num_steps):
        t0, t1 = times[i], times[i + 1]
        t = t0.expand(b)
        p1 = torch.softmax(model(image, zt, t), dim=1)
        if i == num_steps - 1:
            if argmax_final:
                return p1.argmax(1)
            flat = p1.permute(0, 2, 3, 1).reshape(-1, num_classes)
            return torch.multinomial(flat, 1, generator=generator).view(b, h, w)
        rates = three_term_rates(p1, zt, t, scheduler, source_probs) if three_term else two_term_rates(p1, zt, t, scheduler)
        zt = _jump_step(zt, rates, t1 - t0, generator)
    return zt


def sample_two_term(model, image: torch.Tensor, z0: torch.Tensor, scheduler, num_classes: int,
                    num_steps: int = 50, argmax_final: bool = False,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    if not isinstance(scheduler, (LinearScheduler, PowerScheduler)):
        raise TypeError("sample_two_term requires linear/power scheduler")
    return _sample(model, image, z0, scheduler, num_classes, num_steps, argmax_final, generator, False)


def sample_three_term(model, image: torch.Tensor, z0: torch.Tensor, scheduler, num_classes: int,
                      num_steps: int = 50, argmax_final: bool = False,
                      generator: torch.Generator | None = None, source_probs: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(scheduler, PowerUniformBumpScheduler):
        raise TypeError("sample_three_term requires power_uniform_bump scheduler")
    if source_probs is None: raise ValueError("three-term inference requires known source probabilities")
    return _sample(model, image, z0, scheduler, num_classes, num_steps, argmax_final, generator, True, source_probs)


def sample_discrete_flow(model, image: torch.Tensor, z0: torch.Tensor, path, num_steps: int = 50,
                         argmax_final: bool = False, generator: torch.Generator | None = None,
                         source_probs: torch.Tensor | None = None) -> torch.Tensor:
    if path.path_type == "two_term":
        return sample_two_term(model, image, z0, path.scheduler, path.num_classes, num_steps, argmax_final, generator)
    if path.path_type == "three_term":
        return sample_three_term(model, image, z0, path.scheduler, path.num_classes, num_steps, argmax_final, generator, source_probs)
    raise ValueError(f"unsupported path type: {path.path_type}")
