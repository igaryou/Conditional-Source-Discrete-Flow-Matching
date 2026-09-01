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
from .flow.sampling import sample_source
from .metrics import SegmentationMetrics
from .model import build_dfm_model


@torch.no_grad()
def evaluate_dfm(cfg: dict | None, checkpoint: str, output: str, fixed_t: float | None = None,
                 generative_steps: int | None = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    if cfg is None: cfg=ckpt["config"]
    conditioned=cfg["source_distribution"]["type"]=="image_conditioned"
    if conditioned and cfg.get("source_cache",{}).get("verify",True):verify_cache(cfg,"val")
    ds=build_dataset(cfg,"val",conditioned,False);loader=DataLoader(ds,batch_size=int(cfg["training"].get("batch_size",4)),shuffle=False,num_workers=int(cfg.get("runtime",{}).get("num_workers",4)))
    model=build_dfm_model(cfg).to(device);model.load_state_dict(ckpt["model"]);model.eval();path=build_path(cfg["flow"]["path"],int(cfg["dataset"]["num_classes"]))
    dcfg=cfg["dataset"];metric=SegmentationMetrics(int(dcfg["num_classes"]),int(dcfg.get("loss_ignore_index",-100)),int(dcfg.get("eval_num_classes",dcfg["num_classes"])))
    ecfg=cfg.get("evaluation",{});steps=int(generative_steps or ecfg.get("generative_steps",50));seed=int(ecfg.get("seed",42));ns=int(ecfg.get("num_source_samples",1));argmax=bool(ecfg.get("argmax_final",False))
    generator=torch.Generator(device=device.type if device.type=="cuda" else "cpu").manual_seed(seed);losses=[]
    for b in tqdm(loader,desc="evaluate"):
        image,z1=b["image"].to(device),b["mask"].to(device);dist_cfg=cfg["source_distribution"];cached=b.get("source_logits");cached=cached.to(device) if cached is not None else None
        for _ in range(ns):
            z0,p0=sample_source(dist_cfg["type"],int(dcfg["num_classes"]),tuple(z1.shape),device,cached,float(dist_cfg.get("lambda",0)),float(dist_cfg.get("temperature",1)),generator)
            if fixed_t is None:
                pred=sample_discrete_flow(model,image,z0,path,steps,argmax,generator,p0);metric.update(pred,z1)
            else:
                t=torch.full((image.shape[0],),fixed_t,device=device);zt=path.sample(z0,z1,t,generator);logits=model(image,zt,t)
                losses.append(float(F.cross_entropy(logits,z1)));metric.update(logits.argmax(1),z1)
    result=metric.compute();result.update({"mode":"generative" if fixed_t is None else "conditional","loss":sum(losses)/len(losses) if losses else None,
                                           "generative_steps":steps,"seed":seed,"num_source_samples":ns,"checkpoint":checkpoint})
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result
