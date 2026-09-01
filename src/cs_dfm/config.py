from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    cfg = copy.deepcopy(cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    dataset = cfg.get("dataset", {})
    if int(dataset.get("num_classes", 20)) <= 1:
        raise ValueError("dataset.num_classes must be > 1")
    source = cfg.get("source", {})
    if source.get("variant", "b2") not in {f"b{i}" for i in range(6)}:
        raise ValueError("source.variant must be b0..b5")
    dist = cfg.get("source_distribution", {})
    if dist.get("type", "image_conditioned") not in {"image_conditioned", "uniform"}:
        raise ValueError("source_distribution.type must be image_conditioned or uniform")
    lam = float(dist.get("lambda", 0.0))
    if not 0.0 <= lam <= 1.0:
        raise ValueError("source_distribution.lambda must be in [0,1]")
    if float(dist.get("temperature", 1.0)) <= 0:
        raise ValueError("source_distribution.temperature must be > 0")
    if dist.get("sampling", "categorical") != "categorical":
        raise ValueError("only categorical source sampling is supported")
    path = cfg.get("flow", {}).get("path", {})
    if float(path.get("power", 1.0)) <= 0:
        raise ValueError("flow.path.power must be > 0")
    strength = float(path.get("uniform_strength", 0.0))
    if not 0 <= strength <= 1:
        raise ValueError("flow.path.uniform_strength must be in [0,1]")
    cache_enabled = bool(cfg.get("source_cache", {}).get("enabled", False))
    aug = dataset.get("augmentation", {})
    if cache_enabled and bool(aug.get("photometric", False)):
        raise ValueError("photometric augmentation is incompatible with cached source logits")


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

