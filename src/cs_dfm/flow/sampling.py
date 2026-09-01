from __future__ import annotations

import torch


def construct_source_probabilities(logits: torch.Tensor, lambda_: float, temperature: float) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B,K,H,W]")
    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda must be in [0,1]")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    work = logits.float()
    q_mu = torch.softmax(work / temperature, dim=1)
    probs = (1 - lambda_) * q_mu + lambda_ / work.shape[1]
    probs = probs.clamp_min(0)
    return probs / probs.sum(dim=1, keepdim=True)


def sample_categorical(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Vectorized per-pixel Cat sampling from [B,K,H,W], never argmax."""
    if probs.ndim != 4:
        raise ValueError("probs must have shape [B,K,H,W]")
    b, k, h, w = probs.shape
    flat = probs.permute(0, 2, 3, 1).reshape(-1, k)
    return torch.multinomial(flat, 1, replacement=True, generator=generator).view(b, h, w)


def sample_source(source_type: str, num_classes: int, shape: tuple[int, int, int], device: torch.device,
                  logits: torch.Tensor | None = None, lambda_: float = 0, temperature: float = 1,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if source_type == "uniform":
        probs = torch.full((shape[0], num_classes, shape[1], shape[2]), 1 / num_classes, device=device)
    elif source_type == "image_conditioned":
        if logits is None:
            raise ValueError("cached logits are required for image_conditioned source")
        probs = construct_source_probabilities(logits.to(device), lambda_, temperature)
    else:
        raise ValueError(f"unknown source type: {source_type}")
    return sample_categorical(probs, generator), probs

