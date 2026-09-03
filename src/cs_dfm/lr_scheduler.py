from __future__ import annotations

import math


class ConfigLRScheduler:
    """Update-based cosine/poly schedule with optional linear warmup."""
    def __init__(self, optimizer, cfg: dict, total_updates: int):
        self.optimizer, self.cfg, self.total_updates = optimizer, cfg, max(1, total_updates)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]; self.update = 0
        self._apply()

    def factor(self, update: int) -> float:
        warm = self.cfg.get("warmup", {})
        progress = min(max(update,0)/self.total_updates,1)
        eta_min = float(self.cfg.get("eta_min",0)); base = max(self.base_lrs)
        floor = eta_min/base if base > 0 else 0
        if self.cfg.get("type","cosine") == "cosine": value = .5*(1+math.cos(math.pi*progress))
        else: value = (1-progress)**float(self.cfg.get("power",1))
        decay = floor + (1-floor)*value
        if warm.get("enabled", False) and int(warm.get("begin", 0)) <= update < int(warm.get("end", 0)):
            begin, end = int(warm.get("begin", 0)), int(warm["end"])
            q = (update-begin)/max(end-begin,1)
            warmup = float(warm.get("start_factor",1e-6)) + q*(1-float(warm.get("start_factor",1e-6)))
            return decay * warmup
        return decay

    def _apply(self):
        f=self.factor(self.update)
        for group,base in zip(self.optimizer.param_groups,self.base_lrs): group["lr"]=base*f

    def step(self): self.update += 1; self._apply()
    def state_dict(self): return {"update":self.update,"base_lrs":self.base_lrs}
    def load_state_dict(self,state): self.update=state["update"]; self.base_lrs=state["base_lrs"]; self._apply()
