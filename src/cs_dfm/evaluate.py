from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import resume_checkpoint
from .data import build_dataset
from .flow.paths import build_path
from .flow.inference import sample_two_term
from .flow.sampling import sample_source
from .metrics import SegmentationMetrics
from .model import build_dfm_model


@torch.no_grad()
def evaluate_dfm(cfg: dict, checkpoint: str, output: str, fixed_t: float | None = None,
                 generative_steps: int | None = None):
    """Evaluate either the conditional objective or two-term probability-velocity generation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditioned = cfg["source_distribution"]["type"] == "image_conditioned"
    dataset = build_dataset(cfg, "val", return_logits=conditioned, augment=False)
    loader = DataLoader(dataset, batch_size=int(cfg["training"].get("batch_size", 4)), shuffle=False,
                        num_workers=int(cfg.get("runtime", {}).get("num_workers", 4)))
    model = build_dfm_model(cfg).to(device); resume_checkpoint(checkpoint, model, map_location=device); model.eval()
    path = build_path(cfg["flow"]["path"], int(cfg["dataset"]["num_classes"]))
    metrics = SegmentationMetrics(int(cfg["dataset"]["num_classes"])); losses = []
    for batch in tqdm(loader, desc="evaluate DFM"):
        image, z1 = batch["image"].to(device), batch["mask"].to(device); d = cfg["source_distribution"]
        cached = batch.get("source_logits"); cached = cached.to(device) if cached is not None else None
        z0, _ = sample_source(d["type"], int(cfg["dataset"]["num_classes"]), tuple(z1.shape), device, cached,
                              float(d.get("lambda", 0)), float(d.get("temperature", 1)))
        if generative_steps is not None:
            if path.path_type != "two_term":
                raise ValueError("generative probability-velocity evaluation currently supports two-term paths")
            pred = sample_two_term(model, image, z0, path.scheduler, int(cfg["dataset"]["num_classes"]), generative_steps)
            metrics.update(pred, z1)
        else:
            t = torch.full((image.shape[0],), fixed_t, device=device) if fixed_t is not None else torch.rand(image.shape[0], device=device)
            zt = path.sample(z0, z1, t); logits = model(image, zt, t)
            losses.append(float(F.cross_entropy(logits, z1))); metrics.update(logits.argmax(1), z1)
    result = metrics.compute(); result.update({"loss": sum(losses)/max(len(losses),1) if losses else None,
                                                "fixed_t": fixed_t, "generative_steps": generative_steps,
                                                "checkpoint": checkpoint})
    out = Path(output); out.mkdir(parents=True, exist_ok=True); (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
