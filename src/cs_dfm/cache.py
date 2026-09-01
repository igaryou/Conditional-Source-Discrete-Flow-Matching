from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset
from .source import build_source_model, load_source_checkpoint
from .utils import file_sha256


DTYPES = {"float16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}


@torch.no_grad()
def create_source_cache(cfg: dict, splits=("train", "val"), device: torch.device | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_cfg, source_cfg = cfg["source_cache"], cfg["source"]
    root = Path(cache_cfg["root"]).expanduser()
    checkpoint = str(Path(source_cfg["checkpoint"]).expanduser())
    model = build_source_model(cfg)
    load_source_checkpoint(model, checkpoint)
    model.eval().requires_grad_(False).to(device)
    dtype = DTYPES[cache_cfg.get("dtype", "float16")]
    overwrite = bool(cache_cfg.get("overwrite", False))
    shapes = set()
    counts = {}
    for split in splits:
        dataset = build_dataset(cfg, split, return_logits=False, augment=False)
        loader = DataLoader(dataset, batch_size=int(cache_cfg.get("batch_size", 4)), shuffle=False,
                            num_workers=int(cfg.get("runtime", {}).get("num_workers", 4)))
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in tqdm(loader, desc=f"cache:{split}"):
            images = batch["image"].to(device)
            logits = model(images).cpu().to(dtype)
            for sid, value in zip(batch["sample_id"], logits):
                path = split_dir / f"{sid}.pt"
                if overwrite or not path.exists():
                    torch.save({"logits": value.contiguous()}, path)
                shapes.add(tuple(value.shape))
                count += 1
        counts[split] = count
    metadata = {
        "source_checkpoint": checkpoint, "source_checkpoint_sha256": file_sha256(checkpoint),
        "architecture": "segformer", "variant": source_cfg["variant"],
        "dataset": cfg["dataset"]["name"], "splits": counts,
        "image_preprocessing": {"base_resize": cfg["dataset"]["image_size"], "normalization": "ImageNet in model"},
        "num_classes": cfg["dataset"]["num_classes"], "logits_shapes": [list(s) for s in sorted(shapes)],
        "created_at": datetime.now(timezone.utc).isoformat(), "dtype": cache_cfg.get("dtype", "float16"),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def verify_cache(cfg: dict, split: str) -> None:
    root = Path(cfg["source_cache"]["root"]).expanduser()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = Path(cfg["source"]["checkpoint"]).expanduser()
    if metadata["source_checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("source checkpoint hash differs from cache metadata")
    dataset = build_dataset(cfg, split, return_logits=True, augment=False)
    if metadata["splits"].get(split) != len(dataset):
        raise ValueError(f"cache count mismatch for {split}")

