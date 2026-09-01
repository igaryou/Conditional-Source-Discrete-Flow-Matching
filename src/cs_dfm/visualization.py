from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset
from .flow.paths import build_path
from .flow.sampling import construct_source_probabilities, sample_categorical


def _entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(0)


def _save_mask(mask, path, title=None):
    plt.imsave(path, mask.cpu().numpy(), vmin=0, vmax=int(mask.max().clamp_min(1)))


@torch.no_grad()
def visualize_source(cfg: dict, index: int, lambdas: list[float], temperatures: list[float], seeds: list[int], out_root: str):
    sample = build_dataset(cfg, "val", return_logits=True, augment=False)[index]
    image, gt, logits, sid = sample["image"], sample["mask"], sample["source_logits"].float(), sample["sample_id"]
    out = Path(out_root) / sid; out.mkdir(parents=True, exist_ok=True)
    plt.imsave(out / "input.png", image.permute(1, 2, 0).numpy().clip(0, 1)); _save_mask(gt, out / "gt.png")
    q_mu = torch.softmax(logits, 0); mu_argmax = q_mu.argmax(0); _save_mask(mu_argmax, out / "mu_argmax.png")
    fig, axes = plt.subplots(len(temperatures), len(lambdas), figsize=(2.5 * len(lambdas), 2.5 * len(temperatures)), squeeze=False)
    stats = []
    for row, temp in enumerate(temperatures):
        for col, lam in enumerate(lambdas):
            p0 = construct_source_probabilities(logits[None], lam, temp)[0]
            samples = []
            for seed in seeds:
                gen = torch.Generator().manual_seed(seed)
                samples.append(sample_categorical(p0[None], gen)[0])
            z0 = samples[0]; axes[row, col].imshow(z0.numpy(), vmin=0, vmax=logits.shape[0] - 1)
            axes[row, col].set_title(f"T={temp:g}, λ={lam:g}"); axes[row, col].axis("off")
            cell = out / f"T{temp:g}_lambda{lam:g}"; cell.mkdir(exist_ok=True)
            for n, z in enumerate(samples): _save_mask(z, cell / f"z0_seed{seeds[n]}.png")
            confidence = p0.max(0).values; entropy = _entropy(p0)
            plt.imsave(cell / "confidence.png", confidence.numpy(), cmap="viridis", vmin=0, vmax=1)
            plt.imsave(cell / "entropy.png", entropy.numpy(), cmap="magma")
            stats.append({"lambda": lam, "temperature": temp, "z0_gt_accuracy": float((z0 == gt).float().mean()),
                          "z0_mu_agreement": float((z0 == mu_argmax).float().mean()),
                          "mean_entropy_p0": float(entropy.mean()), "mean_entropy_q_mu": float(_entropy(q_mu).mean()),
                          "mean_max_confidence": float(confidence.mean()), "sample_seeds": seeds})
    fig.tight_layout(); fig.savefig(out / "grid.png", dpi=150); plt.close(fig)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out


@torch.no_grad()
def source_diagnostics(cfg: dict, lambdas: list[float], temperatures: list[float], out_dir: str, max_samples=None):
    dataset = build_dataset(cfg, "val", return_logits=True, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    k = int(cfg["dataset"]["num_classes"]); accum = {(l, t): {"pixels": 0, "gt": 0, "mu": 0, "changed": 0,
        "entropy": 0., "confidence": 0., "z_hist": torch.zeros(k), "gt_hist": torch.zeros(k), "mu_hist": torch.zeros(k),
        "class_correct": torch.zeros(k), "class_total": torch.zeros(k)} for t in temperatures for l in lambdas}
    for n, batch in enumerate(tqdm(loader, desc="source diagnostics")):
        if max_samples is not None and n >= max_samples: break
        logits, gt = batch["source_logits"].float(), batch["mask"]; mu = logits.argmax(1)
        for (lam, temp), a in accum.items():
            p = construct_source_probabilities(logits, lam, temp); z = sample_categorical(p)
            pixels = z.numel(); a["pixels"] += pixels; a["gt"] += int((z == gt).sum()); a["mu"] += int((z == mu).sum())
            a["changed"] += int((z != mu).sum()); a["entropy"] += float(_entropy(p[0]).sum()); a["confidence"] += float(p.max(1).values.sum())
            a["z_hist"] += torch.bincount(z.flatten(), minlength=k); a["gt_hist"] += torch.bincount(gt.flatten(), minlength=k)
            a["mu_hist"] += torch.bincount(mu.flatten(), minlength=k)
            for c in range(k):
                sel = gt == c; a["class_total"][c] += sel.sum(); a["class_correct"][c] += ((z == c) & sel).sum()
    rows = []
    for (lam, temp), a in accum.items():
        denom = max(a["pixels"], 1)
        rows.append({"lambda": lam, "temperature": temp, "z0_gt_acc": a["gt"] / denom,
                     "z0_mu_agreement": a["mu"] / denom, "changed_ratio": a["changed"] / denom,
                     "mean_entropy": a["entropy"] / denom, "mean_max_confidence": a["confidence"] / denom,
                     "z0_class_histogram": a["z_hist"].tolist(), "gt_class_histogram": a["gt_hist"].tolist(),
                     "mu_class_histogram": a["mu_hist"].tolist(),
                     "per_class_source_accuracy": (a["class_correct"] / a["class_total"].clamp_min(1)).tolist(),
                     "per_class_sampled_frequency": (a["z_hist"] / max(a["z_hist"].sum(), 1)).tolist()})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scalar_keys = [k for k, v in rows[0].items() if not isinstance(v, list)]
    with (out / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys); writer.writeheader(); writer.writerows([{k: r[k] for k in scalar_keys} for r in rows])
    return rows


@torch.no_grad()
def visualize_paths(path_configs: list[dict], num_classes: int, out_dir: str, z0=None, z1=None,
                    times=(0, .1, .25, .5, .75, .9, 1)):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); grid_t = torch.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in path_configs:
        path = build_path(cfg, num_classes); name = cfg.get("name", f"{cfg['type']}-{cfg['scheduler']}")
        weights = path.weights(grid_t)
        if isinstance(weights, torch.Tensor): ax.plot(grid_t, weights, label=name)
        else:
            ax.plot(grid_t, weights.kappa1, label=f"{name}: κ1"); ax.plot(grid_t, weights.kappa2, label=f"{name}: κ2")
            ax.plot(grid_t, weights.kappa3, label=f"{name}: κ3")
        if z0 is not None and z1 is not None:
            samples = [path.sample(z0[None], z1[None], torch.tensor([t]))[0] for t in times]
            pdir = out / name.replace("/", "_"); pdir.mkdir(exist_ok=True)
            f2, axes = plt.subplots(1, len(samples), figsize=(2.2 * len(samples), 2.3))
            for a, value, t in zip(axes, samples, times): a.imshow(value.numpy(), vmin=0, vmax=num_classes-1); a.set_title(f"t={t:g}"); a.axis("off")
            f2.tight_layout(); f2.savefig(pdir / "zt_grid.png", dpi=150); plt.close(f2)
    ax.set(xlabel="t", ylabel="mixture weight", ylim=(-.02, 1.02)); ax.grid(True); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "schedulers.png", dpi=150); plt.close(fig)

