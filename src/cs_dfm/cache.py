from __future__ import annotations

import json
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cache_spec import cache_directory, cache_fingerprint, cache_fingerprint_spec, canonical_size_hw
from .data import build_dataset
from .source import build_source_model, load_source_checkpoint


DTYPES = {"float16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}


@torch.no_grad()
def create_source_cache(cfg: dict, splits=("train", "val"), device: torch.device | None = None) -> dict:
    if cfg.get("source_distribution", {}).get("type") == "uniform":
        raise ValueError("uniform source does not need a source-logits cache")
    if cfg.get("source_runtime", {}).get("mode") != "cache":
        raise ValueError("source-logits cache creation requires source_runtime.mode=cache")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_cfg = cfg["source_cache"]; root = cache_directory(cfg); root.mkdir(parents=True, exist_ok=True)
    model = build_source_model(cfg); load_source_checkpoint(model, cfg["source"]["checkpoint"])
    model.eval().requires_grad_(False).to(device)
    dtype_name = cache_cfg.get("dtype", "float16"); dtype = DTYPES[dtype_name]
    overwrite = bool(cache_cfg.get("overwrite", False)); manifest = {"fingerprint": cache_fingerprint(cfg), "splits": {}}
    shapes, counts = set(), {}
    for split in splits:
        dataset = build_dataset(cfg, split, return_logits=False, augment=False)
        ids = dataset.expected_sample_ids(); manifest["splits"][split] = ids
        loader = DataLoader(dataset,batch_size=int(cache_cfg.get("batch_size",4)),shuffle=False,
                            num_workers=int(cfg.get("runtime",{}).get("num_workers",4)))
        split_dir=root/split; split_dir.mkdir(parents=True,exist_ok=True); count=0
        for batch in tqdm(loader,desc=f"cache:{split}"):
            logits=model(batch["image"].to(device)).cpu().to(dtype)
            for sid,value in zip(batch["sample_id"],logits):
                path=split_dir/f"{sid}.pt"
                if overwrite or not path.exists(): torch.save({"logits":value.contiguous(),"sample_id":sid},path)
                shapes.add(tuple(value.shape)); count += 1
        counts[split]=count
    spec=cache_fingerprint_spec(cfg)
    metadata={"fingerprint":cache_fingerprint(cfg),"fingerprint_spec":spec,"source_checkpoint":cfg["source"]["checkpoint"],
              "splits":counts,"logits_shapes":[list(s) for s in sorted(shapes)],
              "created_at":datetime.now(timezone.utc).isoformat(),"dtype":dtype_name}
    (root/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for split in splits: verify_cache(cfg,split)
    return metadata


def verify_cache(cfg: dict, split: str) -> None:
    root=cache_directory(cfg); metadata_path=root/"metadata.json"; manifest_path=root/"manifest.json"
    if not metadata_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"cache metadata/manifest missing for fingerprint {cache_fingerprint(cfg)}: {root}")
    metadata=json.loads(metadata_path.read_text(encoding="utf-8")); manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fp=cache_fingerprint(cfg); expected_spec=cache_fingerprint_spec(cfg)
    if metadata.get("fingerprint") != expected_fp or manifest.get("fingerprint") != expected_fp:
        raise RuntimeError("cache fingerprint mismatch")
    if metadata.get("fingerprint_spec") != expected_spec:
        raise RuntimeError("cache preprocessing/checkpoint specification mismatch")
    dataset=build_dataset(cfg,split,return_logits=False,augment=False); expected_ids=dataset.expected_sample_ids()
    if manifest.get("splits",{}).get(split) != expected_ids:
        raise RuntimeError(f"cache sample-ID manifest mismatch for {split}")
    if metadata.get("splits",{}).get(split) != len(expected_ids):
        raise RuntimeError(f"cache sample count mismatch for {split}")
    expected_shape=(int(cfg["dataset"]["num_classes"]),*canonical_size_hw(cfg)); expected_dtype=DTYPES[cfg["source_cache"].get("dtype","float16")]
    for sid in expected_ids:
        path=root/split/f"{sid}.pt"
        if not path.exists(): raise RuntimeError(f"missing cached sample: {sid}")
        payload=torch.load(path,map_location="cpu",weights_only=True); value=payload.get("logits")
        if payload.get("sample_id",sid) != sid: raise RuntimeError(f"cache sample ID mismatch: {sid}")
        if value is None or tuple(value.shape) != expected_shape: raise RuntimeError(f"cache logits shape mismatch: {sid}")
        if value.dtype != expected_dtype: raise RuntimeError(f"cache dtype mismatch: {sid}: {value.dtype}")
