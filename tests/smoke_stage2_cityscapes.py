"""One-real-sample Stage-2 smoke test for the four standard protocols.

Run explicitly; this is not part of pytest because it requires Cityscapes and a GPU.
"""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import torch

from cs_dfm.data import build_dataset
from cs_dfm.flow.inference import sample_discrete_flow
from cs_dfm.flow.paths import build_path
from cs_dfm.model import build_dfm_model
from cs_dfm.source import build_source_model, build_stage2_source_provider
from cs_dfm.train import _dfm_update


def base_cfg(root: str, checkpoint: str) -> dict:
    return {
        "experiment": {"seed": 42},
        "dataset": {
            "name": "cityscapes", "root": root, "pipeline": "ccdm_fixed",
            "image_size_hw": [32, 64], "canonical_size_hw": [64, 128],
            "num_classes": 20, "void_class_index": 19, "eval_num_classes": 19,
            "loss_ignore_index": -100,
            "augmentation": {"random_resize": False, "random_crop": False,
                             "horizontal_flip": 0.0, "photometric": False},
        },
        "source": {"architecture": "segformer", "variant": "b1",
                   "initialization": "random", "checkpoint": checkpoint},
        "source_distribution": {"type": "image_conditioned", "lambda": 0.2,
                                "temperature": 2.0, "sampling": "categorical"},
        "source_runtime": {"mode": "cache"},
        "source_cache": {"enabled": True, "verify": False, "root": "unused"},
        "flow": {"path": {"type": "two_term", "scheduler": "linear"}},
        "model": {"base_channels": 8, "channel_mults": [1, 2], "num_res_blocks": 1,
                  "time_dim": 8, "mask_embed_dim": 4},
        "training": {"lr": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0},
        "runtime": {"amp": True, "amp_dtype": "bf16", "num_workers": 0},
        "validation": {"generative_steps": 20},
        "evaluation": {"generative_steps": 20},
    }


def run_case(name: str, cfg: dict, batch: dict, device: torch.device) -> dict:
    provider = build_stage2_source_provider(cfg, device)
    source_calls = [0]
    if provider.model is not None:
        provider.model.register_forward_hook(lambda *args: source_calls.__setitem__(0, source_calls[0] + 1))
    model = build_dfm_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = build_path(cfg["flow"]["path"], cfg["dataset"]["num_classes"])
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    loss = _dfm_update(model, batch, cfg, path, device, optimizer, scaler, provider)
    image, z1 = batch["image"].to(device), batch["mask"].to(device)
    z0, p0 = provider.sample(batch, image, tuple(z1.shape), torch.Generator(device=device.type).manual_seed(9))
    pred = sample_discrete_flow(model, image, z0, path, 20, False,
                                torch.Generator(device=device.type).manual_seed(10), p0)
    grads = [] if provider.model is None else [p.grad for p in provider.model.parameters()]
    return {
        "case": name, "loss": loss, "generation_shape": list(pred.shape),
        "generative_steps": 20, "stage2_source_forwards": source_calls[0],
        "source_gradients_none": all(g is None for g in grads),
        "peak_memory_mib": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None),
    }


def main() -> None:
    root = "/home/igarashi_25/datasets/cityscapes"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory(prefix="cs_dfm_smoke_") as tmp:
        checkpoint = str(Path(tmp) / "source.pt")
        cfg = base_cfg(root, checkpoint)
        seed_source = build_source_model(cfg).eval().to(device)
        torch.save({"model": seed_source.state_dict()}, checkpoint)

        ccdm_sample = build_dataset(cfg, "train", return_logits=False, augment=True)[0]
        with torch.inference_mode():
            cached_logits = seed_source(ccdm_sample["image"][None].to(device)).cpu()
        del seed_source
        ccdm_batch = {"image": ccdm_sample["image"][None], "mask": ccdm_sample["mask"][None],
                      "source_logits": cached_logits}
        results = [run_case("ccdm_conditioned_cache", cfg, ccdm_batch, device)]

        uniform = copy.deepcopy(cfg)
        uniform["source_distribution"] = {"type": "uniform", "sampling": "categorical"}
        uniform["source_runtime"] = {"mode": "none"}; uniform["source_cache"] = {"enabled": False}
        uniform_batch = {k: v for k, v in ccdm_batch.items() if k != "source_logits"}
        results.append(run_case("ccdm_uniform", uniform, uniform_batch, device))

        mmseg = copy.deepcopy(cfg)
        mmseg["dataset"]["pipeline"] = "mmseg"
        mmseg["dataset"]["train_pipeline"] = {
            "random_resize": {"enabled": True, "base_size_hw": [64, 128], "ratio_range": [1.0, 1.0]},
            "random_crop": {"enabled": True, "crop_size_hw": [32, 64], "cat_max_ratio": 1.0},
            "horizontal_flip": {"probability": 1.0}, "photometric": {"enabled": False},
        }
        mmseg["source_runtime"] = {"mode": "online"}; mmseg["source_cache"] = {"enabled": False}
        mmseg_sample = build_dataset(mmseg, "train", return_logits=False, augment=True)[0]
        assert "source_logits" not in mmseg_sample
        mmseg_batch = {"image": mmseg_sample["image"][None], "mask": mmseg_sample["mask"][None]}
        results.append(run_case("mmseg_conditioned_online", mmseg, mmseg_batch, device))

        mmseg_uniform = copy.deepcopy(mmseg)
        mmseg_uniform["source_distribution"] = {"type": "uniform", "sampling": "categorical"}
        mmseg_uniform["source_runtime"] = {"mode": "none"}
        results.append(run_case("mmseg_uniform", mmseg_uniform, mmseg_batch, device))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
