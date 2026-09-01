from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch: int, best_metric: float, config: dict, extra=None):
    raw = model.module if hasattr(model, "module") else model
    payload = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
               "best_metric": best_metric, "config": config}
    if extra: payload.update(extra)
    if scheduler is not None: payload["scheduler"] = scheduler.state_dict()
    if scaler is not None: payload["scaler"] = scaler.state_dict()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)


def resume_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return ckpt
