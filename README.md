# CS-DFM: image-conditioned Discrete Flow Matching segmentation

This repository implements two explicitly separated stages for Cityscapes semantic segmentation:

1. pretrain a SegFormer source model `μ(x)` with ordinary pixel-wise cross entropy;
2. freeze it permanently, cache its **logits**, and train the DFM model without loading or forwarding SegFormer.

The reference implementation inspected at `playground_2/DSDFM/dfm` samples a uniform categorical `z0`, uses a linear two-term mixture path, and trains `p(z1 | zt, x, t)` with pixel-wise CE. CS-DFM preserves that objective and changes the source/path construction in modular code.

## Source distribution

For pixel `i`, class count `K`, temperature `T > 0`, and `0 ≤ λ ≤ 1`:

```text
q_mu^i = softmax(mu(x)_i / T)
p0^i   = (1 - lambda) q_mu^i + lambda / K
z0^i  ~ Categorical(p0^i)
```

`λ=0` is fully image-conditioned and `λ=1` exactly recovers the uniform categorical baseline. The actual sample is taken with vectorized `torch.multinomial`; argmax is used only as a diagnostic reference prediction. A separate `source_distribution.type: uniform` baseline is supported.

Implementation: `src/cs_dfm/flow/sampling.py`.

## Dataset and cache/augmentation alignment

The Cityscapes mapping intentionally matches the existing DFM code: the standard 19 semantic classes map to IDs 0–18 and all void IDs map to class 19. Images, masks, and cached logits first use the same deterministic `dataset.image_size` resize. The cache stores one `<city>__<image_stem>.pt` per sample:

```text
cache/source_b2/
├── train/*.pt
├── val/*.pt
└── metadata.json
```

Each `.pt` contains logits `[K,H,W]`, not probabilities. `metadata.json` contains the checkpoint path and SHA-256, SegFormer variant, dataset/split counts, preprocessing, class count, shapes, creation time, and dtype. It does not contain `λ` or `T`.

During Stage 2, `JointGeometry` samples geometry once and applies the same scale, crop, and horizontal flip to image, GT, and logits. Mask interpolation is nearest-neighbor; image/logit interpolation is bilinear. Photometric augmentation is rejected by config validation whenever cache is enabled because cached `μ(x)` cannot represent a color-jittered input. No silent spatial mismatch is allowed.

Implementation: `src/cs_dfm/data.py` and `src/cs_dfm/cache.py`.

## Probability paths

- `two_term + linear`: `κ(t)=t`.
- `two_term + power`: `κ(t)=t^power`, with `power>0`; power 1 is exactly linear.
- `three_term + power_uniform_bump`: one valid implementation of the DFM Eq.(10) form:

```text
g(t)  = t^power
b(t)  = 4 t (1-t)
kappa2 = uniform_strength b(t)
kappa1 = (1-kappa2) g(t)
kappa3 = (1-kappa2) (1-g(t))
```

This scheduler is presented only as a scheduler satisfying the three-term form and endpoint constraints—not as a uniquely specified scheduler used by the paper. The uniform branch samples a categorical class uniformly. All `zt` paths use stochastic categorical/branch sampling, never argmax.

Implementation: `src/cs_dfm/flow/paths.py` and `schedulers.py`.

## Stage 1: SegFormer source pretraining

`b0` through `b5` are configurable. With `pretrained: true`, Hugging Face SegFormer ADE weights initialize the network and the class head is resized to the configured class count. With `false`, the selected MiT-size configuration is initialized locally without downloading a config.

```bash
cd /home/igarashi_25/CS-DFM
uv run python src/train_source.py --config configs/source_pretrain_cityscapes.yaml
```

For multi-GPU DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 src/train_source.py \
  --config configs/source_pretrain_cityscapes.yaml
```

Checkpoints contain model, optimizer, cosine scheduler, AMP scaler, epoch, best metric, and full config. Set `training.resume` to `last.pt` to resume. `best.pt` is selected by validation mIoU. Validation also reports pixel accuracy and class IoU.

## Create the logits cache

Set `source.checkpoint` to the Stage 1 checkpoint, then run:

```bash
uv run python src/cache_source_logits.py \
  --config configs/dfm_cityscapes.yaml --splits train val
```

`source_cache.dtype` supports `float16`, `bf16`, and `float32`; `overwrite` and SHA/count verification are configurable. Float16 cache comparison naturally requires a quantization tolerance.

## Stage 2: DFM training

```bash
uv run python src/train_dfm.py --config configs/dfm_cityscapes.yaml
```

or DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 src/train_dfm.py \
  --config configs/dfm_cityscapes.yaml
```

The iteration is:

```text
(image, z1, cached_logits)
 -> p0 = (1-lambda) softmax(cached_logits/T) + lambda/K
 -> z0 ~ Cat(p0)
 -> t ~ Uniform[0,1)
 -> zt ~ path(. | z0,z1,t)
 -> logits = dfm_model(image,zt,t)
 -> CrossEntropy(logits,z1)
```

`train_dfm` has no construction or invocation of the source model. The source checkpoint is used only by cache verification via its hash. Consequently SegFormer consumes neither Stage 2 VRAM nor per-iteration compute.

Evaluate the conditional DFM objective at random `t`, or pass `--fixed-t`. For a linear/power two-term path, `--generative-steps 50` uses the reference-compatible probability-velocity jump sampler initialized from the image-conditioned cached-logit `z0`:

```bash
uv run python src/evaluate.py --config configs/dfm_cityscapes.yaml \
  --checkpoint outputs/cs_dfm/best.pt --output outputs/cs_dfm/eval
```

## Source diagnostics and λ × T sweep

The sweep uses one image and exactly the same cached logits in every cell. It writes the input, GT, source argmax, confidence/entropy maps, multiple seeded categorical samples, a grid, agreement statistics, and JSON:

```bash
uv run python src/visualize_source.py --config configs/dfm_cityscapes.yaml \
  --index 0 --lambdas 0 .1 .2 .4 .8 1 \
  --temperatures .5 1 2 4 --seeds 42 43 44 45
```

Dataset-wide diagnostics write CSV and JSON with z0/GT accuracy, z0/μ agreement, entropy, confidence, changed ratio, class histograms, per-class source accuracy, and sampled frequency:

```bash
uv run python src/source_diagnostics.py --config configs/dfm_cityscapes.yaml \
  --lambdas 0 .1 .2 .4 .8 1 --temperatures .5 1 2 4
```

## Path visualization

```bash
uv run python src/visualize_path.py --output visualizations/paths
```

This compares linear, powers 0.5/1/2/4, and the three-term scheduler. The reusable visualization function also accepts a `(z0,z1)` pair and saves `zt` at `t={0,.1,.25,.5,.75,.9,1}`.

## Reproducibility and tests

Python, NumPy, PyTorch CPU/CUDA, DistributedSampler epochs, and DataLoader workers are seeded. Visualization accepts explicit seeds so cells are comparable. AMP supports bf16 and fp16 (GradScaler only for fp16); DDP selects NCCL automatically on CUDA.

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python -m compileall -q src
```

The tests cover λ endpoints, probability simplex/non-negativity, non-argmax categorical behavior, temperature sensitivity, scheduler endpoints/equivalence, three-term constraints, cache quantization round-trip, absence of a Stage 2 source-model call, dummy forward/loss, config parsing, and visualization smoke output.
