from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .cache import verify_cache
from .checkpoint import resume_checkpoint, save_checkpoint
from .config import save_config
from .data import build_dataset
from .flow.paths import build_path
from .flow.sampling import sample_source
from .metrics import SegmentationMetrics
from .model import build_dfm_model
from .source import build_source_model
from .utils import amp_context, cleanup_distributed, init_distributed, is_main_process, seed_everything, seed_worker


def _device(local_rank: int) -> torch.device:
    return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")


def _loader(dataset, cfg, train, rank, world):
    sampler = DistributedSampler(dataset, world, rank, shuffle=train) if world > 1 else None
    runtime, training = cfg.get("runtime", {}), cfg["training"]
    generator = torch.Generator().manual_seed(int(cfg["experiment"].get("seed", 42)) + rank)
    loader = DataLoader(dataset, batch_size=int(training.get("batch_size", 4)), shuffle=train and sampler is None,
                        sampler=sampler, num_workers=int(runtime.get("num_workers", 4)),
                        pin_memory=bool(runtime.get("pin_memory", True)), worker_init_fn=seed_worker, generator=generator)
    return loader, sampler


def _runtime(cfg, model, device):
    train = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["lr"]),
                                  weight_decay=float(train.get("weight_decay", 1e-3)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(train["epochs"]),
                                                            eta_min=float(train.get("eta_min", 1e-6)))
    amp = cfg.get("runtime", {}).get("amp", False) and device.type == "cuda"
    fp16 = cfg.get("runtime", {}).get("amp_dtype", "bf16") == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and fp16)
    return optimizer, scheduler, scaler


@torch.no_grad()
def _validate_source(model, loader, cfg, device):
    model.eval(); metric = SegmentationMetrics(int(cfg["dataset"]["num_classes"]), int(cfg["dataset"].get("ignore_index", -100)))
    loss_sum = torch.zeros(2, device=device)
    for batch in loader:
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        logits = model(image)
        loss = F.cross_entropy(logits, mask, ignore_index=int(cfg["dataset"].get("ignore_index", -100)))
        loss_sum += torch.tensor([loss.item() * image.shape[0], image.shape[0]], device=device)
        metric.update(logits.argmax(1), mask)
    if dist.is_initialized(): dist.all_reduce(loss_sum)
    metric.synchronize(device); result = metric.compute(); result["loss"] = float(loss_sum[0] / loss_sum[1].clamp_min(1))
    return result


def train_source(cfg: dict) -> None:
    rank, world, local_rank = init_distributed(cfg.get("distributed", {}).get("enabled", "auto"),
                                                cfg.get("distributed", {}).get("backend", "nccl"))
    device = _device(local_rank); seed_everything(int(cfg["experiment"].get("seed", 42)) + rank)
    out = Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    model = build_source_model(cfg).to(device)
    optimizer, scheduler, scaler = _runtime(cfg, model, device)
    start, best = 0, -1.0
    resume = cfg["training"].get("resume")
    if resume:
        ckpt = resume_checkpoint(resume, model, optimizer, scheduler, scaler, device); start = ckpt["epoch"] + 1; best = ckpt.get("best_metric", best)
    if world > 1: model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    train_ds = build_dataset(cfg, "train", augment=True); val_ds = build_dataset(cfg, "val", augment=False)
    train_loader, sampler = _loader(train_ds, cfg, True, rank, world); val_loader, _ = _loader(val_ds, cfg, False, rank, world)
    ignore = int(cfg["dataset"].get("ignore_index", -100)); runtime = cfg.get("runtime", {})
    if is_main_process(): save_config(cfg, out / "config.yaml")
    for epoch in range(start, int(cfg["training"]["epochs"])):
        if sampler: sampler.set_epoch(epoch)
        model.train(); total = 0.0
        iterator = tqdm(train_loader, desc=f"source {epoch+1}", disable=not is_main_process())
        for batch in iterator:
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(bool(runtime.get("amp", False)), runtime.get("amp_dtype", "bf16"), device):
                loss = F.cross_entropy(model(image), mask, ignore_index=ignore)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1)))
            scaler.step(optimizer); scaler.update(); total += loss.item()
        scheduler.step(); metrics = _validate_source(model, val_loader, cfg, device)
        if is_main_process():
            print(json.dumps({"epoch": epoch + 1, "train_loss": total / max(len(train_loader), 1), **metrics}))
            if metrics["mIoU"] > best:
                best = metrics["mIoU"]; save_checkpoint(out / "best.pt", model, optimizer, scheduler, scaler, epoch, best, cfg)
            save_checkpoint(out / "last.pt", model, optimizer, scheduler, scaler, epoch, best, cfg)
    cleanup_distributed()


def _dfm_batch(batch, cfg, path, device):
    image, z1 = batch["image"].to(device), batch["mask"].to(device)
    dist_cfg = cfg["source_distribution"]
    cached = batch.get("source_logits")
    if cached is not None: cached = cached.to(device)
    z0, _ = sample_source(dist_cfg["type"], int(cfg["dataset"]["num_classes"]), tuple(z1.shape), device,
                          cached, float(dist_cfg.get("lambda", 0)), float(dist_cfg.get("temperature", 1)))
    t = torch.rand(image.shape[0], device=device) * (1 - 1e-6)
    zt = path.sample(z0, z1, t)
    return image, z1, z0, zt, t


@torch.no_grad()
def _validate_dfm(model, loader, cfg, path, device):
    model.eval(); metric = SegmentationMetrics(int(cfg["dataset"]["num_classes"])); loss_sum = torch.zeros(2, device=device)
    for batch in loader:
        image, z1, _, zt, t = _dfm_batch(batch, cfg, path, device)
        logits = model(image, zt, t); loss = F.cross_entropy(logits, z1)
        loss_sum += torch.tensor([loss.item() * image.shape[0], image.shape[0]], device=device); metric.update(logits.argmax(1), z1)
    if dist.is_initialized(): dist.all_reduce(loss_sum)
    metric.synchronize(device); result = metric.compute(); result["loss"] = float(loss_sum[0] / loss_sum[1].clamp_min(1)); return result


def train_dfm(cfg: dict) -> None:
    rank, world, local_rank = init_distributed(cfg.get("distributed", {}).get("enabled", "auto"),
                                                cfg.get("distributed", {}).get("backend", "nccl"))
    device = _device(local_rank); seed_everything(int(cfg["experiment"].get("seed", 42)) + rank)
    conditioned = cfg["source_distribution"]["type"] == "image_conditioned"
    if conditioned and not cfg.get("source_cache", {}).get("enabled", False):
        raise ValueError("Stage 2 image_conditioned source requires source_cache.enabled=true")
    if conditioned and cfg.get("source_cache", {}).get("verify", True): verify_cache(cfg, "train"); verify_cache(cfg, "val")
    # Deliberately no source-model construction or forward anywhere in Stage 2.
    model = build_dfm_model(cfg).to(device); optimizer, scheduler, scaler = _runtime(cfg, model, device)
    start, best = 0, -1.0; resume = cfg["training"].get("resume")
    if resume:
        ckpt = resume_checkpoint(resume, model, optimizer, scheduler, scaler, device); start = ckpt["epoch"] + 1; best = ckpt.get("best_metric", best)
    if world > 1: model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    train_ds = build_dataset(cfg, "train", return_logits=conditioned, augment=True)
    val_ds = build_dataset(cfg, "val", return_logits=conditioned, augment=False)
    train_loader, sampler = _loader(train_ds, cfg, True, rank, world); val_loader, _ = _loader(val_ds, cfg, False, rank, world)
    path = build_path(cfg["flow"]["path"], int(cfg["dataset"]["num_classes"])); runtime = cfg.get("runtime", {})
    out = Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    if is_main_process(): save_config(cfg, out / "config.yaml")
    for epoch in range(start, int(cfg["training"]["epochs"])):
        if sampler: sampler.set_epoch(epoch)
        model.train(); total = 0.0
        for batch in tqdm(train_loader, desc=f"dfm {epoch+1}", disable=not is_main_process()):
            image, z1, _, zt, t = _dfm_batch(batch, cfg, path, device); optimizer.zero_grad(set_to_none=True)
            with amp_context(bool(runtime.get("amp", False)), runtime.get("amp_dtype", "bf16"), device):
                # Same objective as the reference dfm: direct pixel-wise CE to z1.
                loss = F.cross_entropy(model(image, zt, t), z1)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1)))
            scaler.step(optimizer); scaler.update(); total += loss.item()
        scheduler.step(); metrics = _validate_dfm(model, val_loader, cfg, path, device)
        if is_main_process():
            print(json.dumps({"epoch": epoch + 1, "train_loss": total / max(len(train_loader), 1), **metrics}))
            if metrics["mIoU"] > best:
                best = metrics["mIoU"]; save_checkpoint(out / "best.pt", model, optimizer, scheduler, scaler, epoch, best, cfg)
            save_checkpoint(out / "last.pt", model, optimizer, scheduler, scaler, epoch, best, cfg)
    cleanup_distributed()
