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
    if source and source.get("variant", "b2") not in {f"b{i}" for i in range(6)}:
        raise ValueError("source.variant must be b0..b5")
    if source and source.get("initialization", "mit_imagenet") not in {"mit_imagenet", "random"}:
        raise ValueError("source.initialization must be mit_imagenet or random")
    dist = cfg.get("source_distribution", {})
    if dist.get("type", "image_conditioned") not in {"image_conditioned", "uniform"}:
        raise ValueError("source_distribution.type must be image_conditioned or uniform")
    if dist.get("type", "image_conditioned") == "image_conditioned":
        lam = float(dist.get("lambda", 0.0))
        if not 0.0 <= lam <= 1.0: raise ValueError("source_distribution.lambda must be in [0,1]")
        if float(dist.get("temperature", 1.0)) <= 0: raise ValueError("source_distribution.temperature must be > 0")
    if dist.get("sampling", "categorical") != "categorical":
        raise ValueError("only categorical source sampling is supported")
    path = cfg.get("flow", {}).get("path", {})
    if float(path.get("power", 1.0)) <= 0:
        raise ValueError("flow.path.power must be > 0")
    strength = float(path.get("uniform_strength", 0.0))
    if not 0 <= strength <= 1:
        raise ValueError("flow.path.uniform_strength must be in [0,1]")
    pipeline = dataset.get("pipeline", "ccdm_fixed")
    if pipeline not in {"ccdm_fixed", "mmseg"}: raise ValueError("dataset.pipeline must be ccdm_fixed or mmseg")
    conditioned = dist.get("type", "image_conditioned") == "image_conditioned"
    is_stage2 = cfg.get("training", {}).get("stage") == "dfm"
    if is_stage2:
        validate_stage2_source_runtime(cfg)
    if is_stage2 and conditioned and cfg.get("source_runtime", {}).get("mode") == "cache":
        if pipeline == "ccdm_fixed": photo = bool(dataset.get("augmentation", {}).get("photometric", False))
        else: photo = bool(dataset.get("train_pipeline", {}).get("photometric", {}).get("enabled", False))
        if photo: raise ValueError("photometric augmentation is incompatible with cached image-conditioned logits")
    training = cfg.get("training", {})
    if training.get("runner", "epoch") not in {"epoch", "iter"}: raise ValueError("training.runner must be epoch or iter")
    optimizer = cfg.get("optimizer", {})
    if optimizer.get("type", "adamw").lower() != "adamw": raise ValueError("optimizer.type must be adamw")
    paramwise = optimizer.get("paramwise", {})
    for key in ("norm_decay_mult", "positional_decay_mult", "decode_head_lr_mult"):
        if float(paramwise.get(key, 0.0 if key != "decode_head_lr_mult" else 10.0)) < 0:
            raise ValueError(f"optimizer.paramwise.{key} must be non-negative")
    sched = cfg.get("scheduler", {"type": "cosine"})
    if sched.get("type", "cosine") not in {"cosine", "poly"}: raise ValueError("scheduler.type must be cosine or poly")
    if float(sched.get("power", 1)) <= 0: raise ValueError("scheduler.power must be > 0")


def validate_stage2_source_runtime(cfg: dict[str, Any]) -> None:
    """Validate the standard Stage-2 source/runtime protocol.

    Keeping this separate makes it straightforward to add research-only runtime
    policies later without coupling them to the source distribution itself.
    """
    source_type = cfg.get("source_distribution", {}).get("type", "image_conditioned")
    pipeline = cfg.get("dataset", {}).get("pipeline", "ccdm_fixed")
    mode = cfg.get("source_runtime", {}).get("mode")
    if source_type == "uniform":
        if mode != "none":
            raise ValueError("uniform source requires source_runtime.mode=none")
        if cfg.get("source_cache", {}).get("enabled", False):
            raise ValueError("uniform source must not enable source_cache")
        return
    required = {"ccdm_fixed": "cache", "mmseg": "online"}[pipeline]
    if mode != required:
        raise ValueError(
            f"image_conditioned + {pipeline} requires source_runtime.mode={required}"
        )
    cache_enabled = bool(cfg.get("source_cache", {}).get("enabled", False))
    if required == "cache" and not cache_enabled:
        raise ValueError("conditioned cache runtime requires source_cache.enabled=true")
    if required == "online" and cache_enabled:
        raise ValueError("conditioned online runtime must not enable source_cache")


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
