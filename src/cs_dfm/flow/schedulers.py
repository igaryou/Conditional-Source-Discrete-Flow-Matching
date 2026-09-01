from __future__ import annotations

from dataclasses import dataclass

import torch


class LinearScheduler:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t


class PowerScheduler:
    def __init__(self, power: float):
        if power <= 0:
            raise ValueError("power must be > 0")
        self.power = power

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t.pow(self.power)


@dataclass
class ThreeTermWeights:
    kappa1: torch.Tensor
    kappa2: torch.Tensor
    kappa3: torch.Tensor


class PowerUniformBumpScheduler:
    """One valid realization of DFM Eq.(10), not a uniquely prescribed paper scheduler."""

    def __init__(self, power: float, uniform_strength: float):
        if power <= 0:
            raise ValueError("power must be > 0")
        if not 0 <= uniform_strength <= 1:
            raise ValueError("uniform_strength must be in [0,1]")
        self.power, self.uniform_strength = power, uniform_strength

    def __call__(self, t: torch.Tensor) -> ThreeTermWeights:
        g = t.pow(self.power)
        k2 = self.uniform_strength * 4 * t * (1 - t)
        k1 = (1 - k2) * g
        k3 = (1 - k2) * (1 - g)
        return ThreeTermWeights(k1, k2, k3)

