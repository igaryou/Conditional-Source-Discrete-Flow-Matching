from __future__ import annotations

import json
from itertools import cycle
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
from .flow.inference import sample_discrete_flow
from .flow.paths import build_path
from .flow.sampling import sample_source
from .lr_scheduler import ConfigLRScheduler
from .metrics import SegmentationMetrics
from .model import build_dfm_model
from .source import build_source_model
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


def _runtime(cfg,model,device):
    t=cfg["training"]; optimizer=torch.optim.AdamW(model.parameters(),lr=float(t["lr"]),weight_decay=float(t.get("weight_decay",1e-3)))
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
    scaler.step(optimizer); scaler.update(); return float(loss)


def train_source(cfg):
    rank,world,local=init_distributed(cfg.get("distributed",{}).get("enabled","auto"),cfg.get("distributed",{}).get("backend","nccl"))
    device=_device(local); seed_everything(int(cfg["experiment"].get("seed",42))+rank); out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    model=build_source_model(cfg).to(device); optimizer,scheduler,scaler=_runtime(cfg,model,device); start=0; best=-1.
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


def dfm_batch(batch,cfg,path,device,generator=None):
    image,z1=batch["image"].to(device),batch["mask"].to(device); d=cfg["source_distribution"]
    cached=batch.get("source_logits"); cached=cached.to(device) if cached is not None else None
    z0,_=sample_source(d["type"],int(cfg["dataset"]["num_classes"]),tuple(z1.shape),device,cached,float(d.get("lambda",0)),float(d.get("temperature",1)),generator)
    t=torch.rand(image.shape[0],device=device,generator=generator)*(1-1e-6); zt=path.sample(z0,z1,t,generator)
    return image,z1,z0,zt,t


@torch.no_grad()
def validate_conditional(model,loader,cfg,path,device):
    model.eval(); metric=_metric(cfg); sums=torch.zeros(2,device=device)
    for b in loader:
        image,z1,_,zt,t=dfm_batch(b,cfg,path,device); logits=model(image,zt,t); loss=F.cross_entropy(logits,z1)
        sums+=torch.tensor([loss.item()*image.shape[0],image.shape[0]],device=device); metric.update(logits.argmax(1),z1)
    if dist.is_initialized():dist.all_reduce(sums)
    metric.synchronize(device); out=metric.compute(); out["loss"]=float(sums[0]/sums[1].clamp_min(1)); return out


@torch.no_grad()
def validate_generative(model,loader,cfg,path,device):
    model.eval(); metric=_metric(cfg); v=cfg.get("validation",{}); seed=int(v.get("generative_seed",42)); samples=int(v.get("num_source_samples",1))
    generator=_generator(device,seed+(dist.get_rank() if dist.is_initialized() else 0))
    for b in loader:
        image,z1=b["image"].to(device),b["mask"].to(device); d=cfg["source_distribution"]; cached=b.get("source_logits"); cached=cached.to(device) if cached is not None else None
        for _ in range(samples):
            z0,p0=sample_source(d["type"],int(cfg["dataset"]["num_classes"]),tuple(z1.shape),device,cached,float(d.get("lambda",0)),float(d.get("temperature",1)),generator)
            pred=sample_discrete_flow(model,image,z0,path,int(v.get("generative_steps",20)),bool(v.get("argmax_final",False)),generator,p0)
            metric.update(pred,z1)
    metric.synchronize(device); return metric.compute()


def _dfm_update(model,b,cfg,path,device,optimizer,scaler):
    image,z1,_,zt,t=dfm_batch(b,cfg,path,device); optimizer.zero_grad(set_to_none=True); r=cfg.get("runtime",{})
    with amp_context(bool(r.get("amp",False)),r.get("amp_dtype","bf16"),device): loss=F.cross_entropy(model(image,zt,t),z1)
    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"].get("grad_clip",1)))
    scaler.step(optimizer); scaler.update(); return float(loss)


def train_dfm(cfg):
    rank,world,local=init_distributed(cfg.get("distributed",{}).get("enabled","auto"),cfg.get("distributed",{}).get("backend","nccl")); device=_device(local)
    seed_everything(int(cfg["experiment"].get("seed",42))+rank); conditioned=cfg["source_distribution"]["type"]=="image_conditioned"
    if conditioned:
        if not cfg.get("source_cache",{}).get("enabled",False):raise ValueError("image_conditioned Stage 2 requires cache")
        if cfg["source_cache"].get("verify",True):verify_cache(cfg,"train");verify_cache(cfg,"val")
    model=build_dfm_model(cfg).to(device); optimizer,scheduler,scaler=_runtime(cfg,model,device); start=0; best_c=best_g=-1.
    if cfg["training"].get("resume"):
        ck=resume_checkpoint(cfg["training"]["resume"],model,optimizer,scheduler,scaler,device);start=ck["epoch"]+1;best_c=ck.get("best_conditional",-1.);best_g=ck.get("best_generative",-1.)
    if world>1:model=DDP(model,device_ids=[local] if device.type=="cuda" else None)
    tr,sp=_loader(build_dataset(cfg,"train",conditioned,True),cfg,True,rank,world);va,_=_loader(build_dataset(cfg,"val",conditioned,False),cfg,False,rank,world)
    path=build_path(cfg["flow"]["path"],int(cfg["dataset"]["num_classes"]));out=Path(cfg["experiment"]["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    if is_main_process():save_config(cfg,out/"config.yaml")
    runner=cfg["training"].get("runner","epoch");limit=runner_limit(cfg["training"]);iterator=iter(tr);epoch=0;v=cfg.get("validation",{})
    for progress in range(start,limit):
        model.train();losses=[];batches=tr if runner=="epoch" else [next(iterator,None)]
        if runner=="iter" and batches[0] is None:epoch+=1;sp.set_epoch(epoch) if sp else None;iterator=iter(tr);batches=[next(iterator)]
        for b in tqdm(batches,disable=not is_main_process() or runner=="iter",desc=f"dfm {progress+1}"):losses.append(_dfm_update(model,b,cfg,path,device,optimizer,scaler))
        scheduler.step();unit=progress+1;ci=int(v.get("conditional_interval",cfg["training"].get("val_interval",1)));gi=int(v.get("generative_interval",5))
        cond=validate_conditional(model,va,cfg,path,device) if unit%ci==0 or unit==limit else None
        gen=validate_generative(model,va,cfg,path,device) if unit%gi==0 or unit==limit else None
        if is_main_process() and (cond is not None or gen is not None):
            log={"progress":unit,"train_loss":sum(losses)/len(losses),"conditional":cond,"generative":gen};print(json.dumps(log))
            extra={"best_conditional":best_c,"best_generative":best_g,"validation":log}
            update_c,update_g=checkpoint_decisions(cond,gen,best_c,best_g)
            if update_c:best_c=cond["mIoU"];extra["best_conditional"]=best_c;save_checkpoint(out/"best_conditional.pt",model,optimizer,scheduler,scaler,progress,best_c,cfg,extra)
            if update_g:
                best_g=gen["mIoU"];extra["best_generative"]=best_g;save_checkpoint(out/"best_generative.pt",model,optimizer,scheduler,scaler,progress,best_g,cfg,extra)
                if v.get("best_metric","generative_mIoU")=="generative_mIoU":save_checkpoint(out/"best.pt",model,optimizer,scheduler,scaler,progress,best_g,cfg,extra)
            save_checkpoint(out/"last.pt",model,optimizer,scheduler,scaler,progress,best_g,cfg,extra)
    cleanup_distributed()
