from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .cache import verify_cache
from .checkpoint import resume_checkpoint, save_checkpoint
from .config import save_config
from .data import build_dataset
from .flow.inference import sample_discrete_flow
from .flow.paths import build_path
from .lr_scheduler import ConfigLRScheduler
from .metrics import SegmentationMetrics
from .model import build_dfm_model
from .source import Stage2SourceProvider, build_source_model, build_stage2_source_provider
from .utils import amp_context, cleanup_distributed, init_distributed, is_main_process, seed_everything, seed_worker


def _device(local_rank): return torch.device("cuda",local_rank) if torch.cuda.is_available() else torch.device("cpu")


def _generator(device, seed):
    return torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(seed)


def _loader(dataset,cfg,train,rank,world):
    sampler=DistributedSampler(dataset,world,rank,shuffle=train) if world>1 else None
    generator=torch.Generator().manual_seed(int(cfg["experiment"].get("seed",42))+rank)
    loader=DataLoader(dataset,batch_size=int(cfg["training"].get("batch_size",4)),shuffle=train and sampler is None,
                      sampler=sampler,num_workers=int(cfg.get("runtime",{}).get("num_workers",4)),
                      pin_memory=bool(cfg.get("runtime",{}).get("pin_memory",True)),worker_init_fn=seed_worker,generator=generator)
    return loader,sampler


_NORM_TYPES = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)


def _normalization_parameter_ids(model: nn.Module) -> set[int]:
    ids = set()
    for module in model.modules():
        if isinstance(module, _NORM_TYPES):
            ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    return ids


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.AdamW:
    """Build AdamW, applying the MMSeg-style recipe only to Stage-1 SegFormer."""
    t = cfg["training"]
    optimizer_cfg = cfg.get("optimizer", {})
    if optimizer_cfg.get("type", "adamw").lower() != "adamw":
        raise ValueError("only optimizer.type=adamw is supported")
    paramwise = optimizer_cfg.get("paramwise", {})
    enabled = (
        bool(paramwise.get("enabled", False))
        and t.get("stage") == "source_pretrain"
        and cfg.get("source", {}).get("architecture", "segformer") == "segformer"
    )
    base_lr = float(t["lr"])
    base_decay = float(t.get("weight_decay", 1e-3))
    if not enabled:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_decay)

    norm_decay = base_decay * float(paramwise.get("norm_decay_mult", 0.0))
    head_lr = base_lr * float(paramwise.get("decode_head_lr_mult", 10.0))
    norm_ids = _normalization_parameter_ids(model)
    buckets: dict[str, list[nn.Parameter]] = {
        "backbone": [],
        "backbone_norm_no_decay": [],
        "decode_head": [],
        "decode_head_norm_no_decay": [],
    }
    names: dict[str, list[str]] = {key: [] for key in buckets}
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    for name, parameter in trainable:
        is_head = name.startswith("model.decode_head.")
        is_norm = id(parameter) in norm_ids
        if is_head and is_norm:
            bucket = "decode_head_norm_no_decay"
        elif is_head:
            bucket = "decode_head"
        elif is_norm:
            bucket = "backbone_norm_no_decay"
        else:
            bucket = "backbone"
        buckets[bucket].append(parameter)
        names[bucket].append(name)

    settings = {
        "backbone": (base_lr, base_decay),
        "backbone_norm_no_decay": (base_lr, norm_decay),
        "decode_head": (head_lr, base_decay),
        "decode_head_norm_no_decay": (head_lr, norm_decay),
    }
    groups = [
        {"params": parameters, "lr": settings[key][0], "weight_decay": settings[key][1],
         "base_lr": settings[key][0], "group_name": key, "parameter_names": names[key]}
        for key, parameters in buckets.items() if parameters
    ]
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [id(parameter) for _, parameter in trainable]
    assert len(grouped_ids) == len(set(grouped_ids)), "optimizer parameter groups overlap"
    assert set(grouped_ids) == set(trainable_ids), "optimizer parameter groups are incomplete"
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=base_decay)


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> list[dict]:
    return [
        {
            "name": group.get("group_name", "all_parameters"),
            "base_lr": group.get("base_lr", group["lr"]),
            "current_lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameter_tensors": len(group["params"]),
            "parameter_elements": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]


def _runtime(cfg,model,device):
    t=cfg["training"]; optimizer=build_optimizer(cfg,model)
    runner=t.get("runner","epoch"); total=int(t.get("epochs",1) if runner=="epoch" else t["max_iters"])
    scheduler=ConfigLRScheduler(optimizer,cfg.get("scheduler",{"type":"cosine","eta_min":t.get("eta_min",1e-6)}),total)
    amp=cfg.get("runtime",{}).get("amp",False) and device.type=="cuda"; fp16=cfg.get("runtime",{}).get("amp_dtype","bf16")=="fp16"
    return optimizer,scheduler,torch.amp.GradScaler("cuda",enabled=amp and fp16)


def _metric(cfg):
    d=cfg["dataset"]
    return SegmentationMetrics(int(d["num_classes"]),int(d.get("loss_ignore_index",-100)),int(d.get("eval_num_classes",d["num_classes"])))


def runner_limit(training_cfg: dict) -> int:
    return int(training_cfg.get("epochs", 1) if training_cfg.get("runner", "epoch") == "epoch" else training_cfg["max_iters"])


def checkpoint_decisions(conditional, generative, best_conditional: float, best_generative: float) -> tuple[bool, bool]:
    return (conditional is not None and conditional["mIoU"] > best_conditional,
            generative is not None and generative["mIoU"] > best_generative)


def dfm_cross_entropy(logits: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    return F.cross_entropy(
        logits, target, ignore_index=int(cfg["dataset"].get("loss_ignore_index", -100))
    )


@torch.no_grad()
def validate_source(model,loader,cfg,device):
    model.eval(); metric=_metric(cfg); sums=torch.zeros(2,device=device)
    for b in loader:
        image,mask=b["image"].to(device),b["mask"].to(device); logits=model(image)
        loss=F.cross_entropy(logits,mask,ignore_index=int(cfg["dataset"].get("loss_ignore_index",-100)))
        sums += torch.tensor([loss.item()*image.shape[0],image.shape[0]],device=device); metric.update(logits.argmax(1),mask)
    if dist.is_initialized(): dist.all_reduce(sums)
    metric.synchronize(device); out=metric.compute(); out["loss"]=float(sums[0]/sums[1].clamp_min(1)); return out


def _source_update(model,b,cfg,device,optimizer,scaler):
    image,mask=b["image"].to(device),b["mask"].to(device); optimizer.zero_grad(set_to_none=True)
    r=cfg.get("runtime",{})
    with amp_context(bool(r.get("amp",False)),r.get("amp_dtype","bf16"),device):
        loss=F.cross_entropy(model(image),mask,ignore_index=int(cfg["dataset"].get("loss_ignore_index",-100)))
    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"].get("grad_clip",1)))
    scaler.step(optimizer); scaler.update(); return float(loss.detach())


def train_source(cfg):
    rank,world,local=init_distributed(cfg.get("distributed",{}).get("enabled","auto"),cfg.get("distributed",{}).get("backend","nccl"))
    device=_device(local); seed_everything(int(cfg["experiment"].get("seed",42))+rank); out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    model=build_source_model(cfg).to(device); optimizer,scheduler,scaler=_runtime(cfg,model,device); start=0; best=-1.
    if is_main_process():
        print(json.dumps({"optimizer_parameter_groups": optimizer_group_summary(optimizer)}))
    if cfg["training"].get("resume"):
        ck=resume_checkpoint(cfg["training"]["resume"],model,optimizer,scheduler,scaler,device); start=ck["epoch"]+1; best=ck.get("best_metric",best)
    if world>1:model=DDP(model,device_ids=[local] if device.type=="cuda" else None)
    tr,sp=_loader(build_dataset(cfg,"train",augment=True),cfg,True,rank,world); va,_=_loader(build_dataset(cfg,"val"),cfg,False,rank,world)
    if is_main_process():save_config(cfg,out/"config.yaml")
    runner=cfg["training"].get("runner","epoch"); limit=runner_limit(cfg["training"])
    iterator=iter(tr); epoch=0
    for progress in range(start,limit):
        model.train(); losses=[]
        batches=tr if runner=="epoch" else [next(iterator,None)]
        if runner=="iter" and batches[0] is None: epoch+=1; sp.set_epoch(epoch) if sp else None; iterator=iter(tr); batches=[next(iterator)]
        for b in tqdm(batches,disable=not is_main_process() or runner=="iter",desc=f"source {progress+1}"): losses.append(_source_update(model,b,cfg,device,optimizer,scaler))
        scheduler.step(); interval=int(cfg["training"].get("val_interval",1))
        if (progress+1)%interval==0 or progress+1==limit:
            metrics=validate_source(model,va,cfg,device)
            if is_main_process():
                print(json.dumps({"progress":progress+1,"train_loss":sum(losses)/len(losses),**metrics}))
                if metrics["mIoU"]>best: best=metrics["mIoU"]; save_checkpoint(out/"best.pt",model,optimizer,scheduler,scaler,progress,best,cfg)
                save_checkpoint(out/"last.pt",model,optimizer,scheduler,scaler,progress,best,cfg)
    cleanup_distributed()


def dfm_batch(batch,cfg,path,device,source_provider: Stage2SourceProvider,generator=None):
    image,z1=batch["image"].to(device),batch["mask"].to(device)
    z0,_=source_provider.sample(batch,image,tuple(z1.shape),generator)
    t=torch.rand(image.shape[0],device=device,generator=generator)*(1-1e-6); zt=path.sample(z0,z1,t,generator)
    return image,z1,z0,zt,t


@torch.no_grad()
def validate_conditional(model,loader,cfg,path,device,source_provider):
    model.eval(); metric=_metric(cfg); sums=torch.zeros(2,device=device)
    for b in loader:
        image,z1,_,zt,t=dfm_batch(b,cfg,path,device,source_provider); logits=model(image,zt,t); loss=dfm_cross_entropy(logits,z1,cfg)
        sums+=torch.tensor([loss.item()*image.shape[0],image.shape[0]],device=device); metric.update(logits.argmax(1),z1)
    if dist.is_initialized():dist.all_reduce(sums)
    metric.synchronize(device); out=metric.compute(); out["loss"]=float(sums[0]/sums[1].clamp_min(1)); return out


@torch.no_grad()
def validate_generative(model,loader,cfg,path,device,source_provider):
    model.eval(); metric=_metric(cfg); v=cfg.get("validation",{}); seed=int(v.get("generative_seed",42)); samples=int(v.get("num_source_samples",1))
    generator=_generator(device,seed+(dist.get_rank() if dist.is_initialized() else 0))
    for b in loader:
        image,z1=b["image"].to(device),b["mask"].to(device)
        for _ in range(samples):
            z0,p0=source_provider.sample(b,image,tuple(z1.shape),generator)
            pred=sample_discrete_flow(model,image,z0,path,int(v.get("generative_steps",20)),bool(v.get("argmax_final",False)),generator,p0)
            metric.update(pred,z1)
    metric.synchronize(device); return metric.compute()


def _dfm_update(model,b,cfg,path,device,optimizer,scaler,source_provider):
    image,z1,_,zt,t=dfm_batch(b,cfg,path,device,source_provider); optimizer.zero_grad(set_to_none=True); r=cfg.get("runtime",{})
    with amp_context(bool(r.get("amp",False)),r.get("amp_dtype","bf16"),device): loss=dfm_cross_entropy(model(image,zt,t),z1,cfg)
    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"].get("grad_clip",1)))
    scaler.step(optimizer); scaler.update(); return float(loss.detach())


def train_dfm(cfg):
    rank,world,local=init_distributed(cfg.get("distributed",{}).get("enabled","auto"),cfg.get("distributed",{}).get("backend","nccl")); device=_device(local)
    seed_everything(int(cfg["experiment"].get("seed",42))+rank)
    source_provider=build_stage2_source_provider(cfg,device)
    if source_provider.needs_cache:
        if cfg["source_cache"].get("verify",True):verify_cache(cfg,"train");verify_cache(cfg,"val")
    model=build_dfm_model(cfg).to(device); optimizer,scheduler,scaler=_runtime(cfg,model,device); start=0; best_c=best_g=-1.
    if cfg["training"].get("resume"):
        ck=resume_checkpoint(cfg["training"]["resume"],model,optimizer,scheduler,scaler,device);start=ck["epoch"]+1;best_c=ck.get("best_conditional",-1.);best_g=ck.get("best_generative",-1.)
    if world>1:model=DDP(model,device_ids=[local] if device.type=="cuda" else None)
    tr,sp=_loader(build_dataset(cfg,"train",source_provider.needs_cache,True),cfg,True,rank,world);va,_=_loader(build_dataset(cfg,"val",source_provider.needs_cache,False),cfg,False,rank,world)
    path=build_path(cfg["flow"]["path"],int(cfg["dataset"]["num_classes"]));out=Path(cfg["experiment"]["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    if is_main_process():save_config(cfg,out/"config.yaml")
    runner=cfg["training"].get("runner","epoch");limit=runner_limit(cfg["training"]);iterator=iter(tr);epoch=0;v=cfg.get("validation",{})
    for progress in range(start,limit):
        model.train();losses=[];batches=tr if runner=="epoch" else [next(iterator,None)]
        if runner=="iter" and batches[0] is None:epoch+=1;sp.set_epoch(epoch) if sp else None;iterator=iter(tr);batches=[next(iterator)]
        for b in tqdm(batches,disable=not is_main_process() or runner=="iter",desc=f"dfm {progress+1}"):losses.append(_dfm_update(model,b,cfg,path,device,optimizer,scaler,source_provider))
        scheduler.step();unit=progress+1;ci=int(v.get("conditional_interval",cfg["training"].get("val_interval",1)));gi=int(v.get("generative_interval",5))
        cond=validate_conditional(model,va,cfg,path,device,source_provider) if unit%ci==0 or unit==limit else None
        gen=validate_generative(model,va,cfg,path,device,source_provider) if unit%gi==0 or unit==limit else None
        if is_main_process() and (cond is not None or gen is not None):
            log={"progress":unit,"train_loss":sum(losses)/len(losses),"conditional":cond,"generative":gen};print(json.dumps(log))
            extra={"best_conditional":best_c,"best_generative":best_g,"validation":log}
            update_c,update_g=checkpoint_decisions(cond,gen,best_c,best_g)
            if update_c:
                best_c=cond["mIoU"];extra["best_conditional"]=best_c;save_checkpoint(out/"best_conditional.pt",model,optimizer,scheduler,scaler,progress,best_c,cfg,extra)
                if v.get("best_metric","generative_mIoU")=="conditional_mIoU":save_checkpoint(out/"best.pt",model,optimizer,scheduler,scaler,progress,best_c,cfg,extra)
            if update_g:
                best_g=gen["mIoU"];extra["best_generative"]=best_g;save_checkpoint(out/"best_generative.pt",model,optimizer,scheduler,scaler,progress,best_g,cfg,extra)
                if v.get("best_metric","generative_mIoU")=="generative_mIoU":save_checkpoint(out/"best.pt",model,optimizer,scheduler,scaler,progress,best_g,cfg,extra)
            save_checkpoint(out/"last.pt",model,optimizer,scheduler,scaler,progress,best_g,cfg,extra)
    cleanup_distributed()
