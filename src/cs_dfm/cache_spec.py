from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .utils import file_sha256


LABEL_MAPPING_VERSION = "cityscapes_raw_to_19_plus_void_v1"


def canonical_size_hw(cfg: dict) -> tuple[int, int]:
    d = cfg["dataset"]
    if d.get("pipeline", "ccdm_fixed") == "mmseg":
        return tuple(d.get("canonical_size_hw", d.get("image_size", [128, 256])))
    return tuple(d.get("image_size_hw", d.get("image_size", [128, 256])))


def cache_fingerprint_spec(cfg: dict) -> dict:
    checkpoint = Path(cfg["source"]["checkpoint"]).expanduser()
    size = canonical_size_hw(cfg)
    return {
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_architecture": cfg["source"].get("architecture", "segformer"),
        "source_variant": cfg["source"]["variant"],
        "num_classes": int(cfg["dataset"]["num_classes"]),
        "dataset_name": cfg["dataset"]["name"],
        "label_mapping_version": cfg["dataset"].get("label_mapping_version", LABEL_MAPPING_VERSION),
        "canonical_image_preprocessing": {"range": "[0,1]", "model_normalization": "imagenet_mean_std"},
        "canonical_resize_hw": list(size),
        "source_output_resolution_hw": list(size),
    }


def cache_fingerprint(cfg: dict) -> str:
    payload = json.dumps(cache_fingerprint_spec(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def cache_directory(cfg: dict) -> Path:
    return Path(cfg["source_cache"]["root"]).expanduser() / cache_fingerprint(cfg)
