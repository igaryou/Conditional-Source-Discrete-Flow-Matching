from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cache import verify_cache
from .data import build_dataset
from .flow.inference import sample_discrete_flow
from .flow.paths import build_path
from .metrics import SegmentationMetrics
from .model import build_dfm_model
from .source import build_stage2_source_provider


def conditional_cross_entropy(logits: torch.Tensor, target: torch.Tensor, dataset_cfg: dict) -> torch.Tensor:
    return F.cross_entropy(
        logits, target, ignore_index=int(dataset_cfg.get("loss_ignore_index", -100))
    )


@torch.no_grad()
def evaluate_dfm(cfg: dict | None, checkpoint: str, output: str, fixed_t: float | None = None,
                 generative_steps: int | None = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    if cfg is None: cfg=ckpt["config"]
    source_provider=build_stage2_source_provider(cfg,device)
    if source_provider.needs_cache and cfg.get("source_cache",{}).get("verify",True):verify_cache(cfg,"val")
    ds=build_dataset(cfg,"val",source_provider.needs_cache,False);loader=DataLoader(ds,batch_size=int(cfg["training"].get("batch_size",4)),shuffle=False,num_workers=int(cfg.get("runtime",{}).get("num_workers",4)))
    model=build_dfm_model(cfg).to(device);model.load_state_dict(ckpt["model"]);model.eval();path=build_path(cfg["flow"]["path"],int(cfg["dataset"]["num_classes"]))
    dcfg=cfg["dataset"];metric=SegmentationMetrics(int(dcfg["num_classes"]),int(dcfg.get("loss_ignore_index",-100)),int(dcfg.get("eval_num_classes",dcfg["num_classes"])))
    ecfg=cfg.get("evaluation",{});steps=int(generative_steps or ecfg.get("generative_steps",20));seed=int(ecfg.get("seed",42));ns=int(ecfg.get("num_source_samples",1));argmax=bool(ecfg.get("argmax_final",False))
    generator=torch.Generator(device=device.type if device.type=="cuda" else "cpu").manual_seed(seed);losses=[]
    for b in tqdm(loader,desc="evaluate"):
        image,z1=b["image"].to(device),b["mask"].to(device)
        for _ in range(ns):
            z0,p0=source_provider.sample(b,image,tuple(z1.shape),generator)
            if fixed_t is None:
                pred=sample_discrete_flow(model,image,z0,path,steps,argmax,generator,p0);metric.update(pred,z1)
            else:
                t=torch.full((image.shape[0],),fixed_t,device=device);zt=path.sample(z0,z1,t,generator);logits=model(image,zt,t)
                losses.append(float(conditional_cross_entropy(logits,z1,dcfg)));metric.update(logits.argmax(1),z1)
    result=metric.compute();result.update({"mode":"generative" if fixed_t is None else "conditional","loss":sum(losses)/len(losses) if losses else None,
                                           "generative_steps":steps,"seed":seed,"num_source_samples":ns,"checkpoint":checkpoint})
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result
