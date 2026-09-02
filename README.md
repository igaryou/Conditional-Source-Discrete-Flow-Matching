# CS-DFM: image-conditioned Discrete Flow Matching segmentation

This repository implements two explicitly separated stages for Cityscapes semantic segmentation:

1. pretrain a SegFormer source model `μ(x)` with ordinary pixel-wise cross entropy;
2. freeze it permanently and train the DFM from either cached logits (CCDM) or an online frozen-source forward after augmentation (MMSeg).

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

## Source initialization

Stage 1 uses `SegformerModel.from_pretrained("nvidia/mit-b0" ... "nvidia/mit-b5")` for the ImageNet-pretrained MiT backbone only. A fresh `SegformerForSemanticSegmentation` decode/classification head is created for Cityscapes. ADE20K-finetuned segmentation checkpoints are never loaded. Set `source.initialization: random` for fully random initialization; legacy `pretrained` is still interpreted when `initialization` is absent.

## Dataset and Stage 2 source runtime

The Cityscapes mapping intentionally matches the existing DFM code: the standard 19 semantic classes map to IDs 0–18 and all void IDs map to class 19. For CCDM, images, masks, and cached logits use the same deterministic canonical `[H,W]` resize. The cache stores one `<city>__<image_stem>.pt` per sample:

```text
cache/source_b2/<fingerprint>/
├── train/*.pt
├── val/*.pt
├── manifest.json
└── metadata.json
```

The fingerprint hashes checkpoint SHA-256, architecture/variant, class count, dataset, label-mapping version, canonical preprocessing/resize, and source output resolution. A changed checkpoint or preprocessing therefore selects a different directory and cannot inherit stale `.pt` files. Verification checks the fingerprint/spec, every expected sample ID, count, shape, dtype, and preprocessing before Stage 2 starts.

The source distribution and the mechanism used to obtain its logits are configured independently with `source_distribution.type` and `source_runtime.mode`.

- CCDM image-conditioned runs use `source_runtime.mode: cache`. Fixed-resolution images are forwarded through SegFormer in the cache-creation stage; Stage 2 loads the fingerprinted logits and performs no SegFormer construction or forward.
- MMSeg image-conditioned runs use `source_runtime.mode: online`. The Dataset returns only the augmented image and GT after RandomResize, RandomCrop, and RandomFlip. Stage 2 then forwards that augmented image through a checkpoint-loaded source model under `eval()`, `requires_grad_(False)`, AMP, and `torch.inference_mode()`.
- Uniform runs use `source_runtime.mode: none`; neither SegFormer, a source checkpoint, nor a cache is used for either dataset pipeline.

MMSeg deliberately uses `μ(Transform(x))`, because in general `Transform(μ(x)) != μ(Transform(x))`. This is why online source inference, rather than geometrically transforming cached logits, is the standard MMSeg protocol. The frozen source model is not added to the optimizer or DDP; each rank owns one local frozen copy while only the DFM is DDP-wrapped.

Two protocols are explicit in YAML and all sizes use `[H,W]` order:

- `ccdm_fixed`: deterministic fixed-resolution resize by default, optional resize/crop/flip, epoch runner.
- `mmseg`: original image followed by RandomResize, RandomCrop with `cat_max_ratio`, RandomFlip, optional photometric transform, and epoch or iteration runner.

The label policy remains 19 foreground classes plus void class 19. `void_class_index`, `eval_num_classes`, and `loss_ignore_index` have independent meanings: the standard `loss_ignore_index: -100` trains void class 19, while setting it to 19 excludes void only from cross entropy.

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

This scheduler is presented only as a scheduler satisfying the three-term form and endpoint constraints—not as a uniquely specified scheduler used by the paper. Three-term generation uses the general DFM Theorem-3 CTMC velocity, a dynamic base component, and an analytic source posterior from known `p0` and predicted `p1|t`; see [the derivation](docs/three_term_velocity.md). The continuity equation is tested numerically.

Implementation: `src/cs_dfm/flow/paths.py` and `schedulers.py`.

## Stage 1: SegFormer source pretraining

`b0` through `b5` are configurable. `initialization: mit_imagenet` loads only `nvidia/mit-b*`; `initialization: random` initializes backbone and head randomly.

```bash
cd /home/igarashi_25/CS-DFM
uv run python src/train_source.py --config configs/source_pretrain_cityscapes_ccdm.yaml
```

For multi-GPU DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 src/train_source.py \
  --config configs/source_pretrain_cityscapes_ccdm.yaml
```

Checkpoints contain model, optimizer, cosine scheduler, AMP scaler, epoch, best metric, and full config. Set `training.resume` to `last.pt` to resume. `best.pt` is selected by validation mIoU. Validation also reports pixel accuracy and class IoU.

## Create the CCDM logits cache

Set `source.checkpoint` to the Stage 1 checkpoint, then run:

```bash
uv run python src/cache_source_logits.py \
  --config configs/dfm_cityscapes.yaml --splits train val
```

`source_cache.dtype` supports `float16`, `bf16`, and `float32`; `overwrite` and SHA/count verification are configurable. Float16 cache comparison naturally requires a quantization tolerance. MMSeg online and uniform runs do not create or request a cache.

## Stage 2: DFM training

```bash
uv run python src/train_dfm.py --config configs/dfm_cityscapes.yaml
```

or DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 src/train_dfm.py \
  --config configs/dfm_cityscapes.yaml
```

For CCDM conditioned, the iteration is:

```text
(image, z1, cached_logits)
 -> p0 = (1-lambda) softmax(cached_logits/T) + lambda/K
 -> z0 ~ Cat(p0)
 -> t ~ Uniform[0,1)
 -> zt ~ path(. | z0,z1,t)
 -> logits = dfm_model(image,zt,t)
 -> CrossEntropy(logits,z1)
```

For MMSeg conditioned, the iteration is:

```text
(augmented_image, z1)
 -> frozen_source_logits = mu(augmented_image)  [inference mode]
 -> p0 = (1-lambda) softmax(frozen_source_logits/T) + lambda/K
 -> z0 ~ Cat(p0)
 -> zt ~ path(. | z0,z1,t)
 -> DFM forward/backward
```

The temporary source logits are released before the DFM forward/backward. Uniform runs require no `source`, checkpoint, cache, or Stage 1 at all.

Conditional validation reconstructs `z1` from a `zt` that was built using GT and reports conditional loss/mIoU/pixel accuracy. Generative validation starts only from sampled `z0`, never uses GT in generation, and reports generative mIoU/pixel accuracy/class IoU. It runs at its own configurable interval and fixed seed. Checkpoints are `last.pt`, `best_conditional.pt`, and `best_generative.pt`; by default `best.pt` mirrors a new best generative mIoU and is not updated on epochs/iterations without generative validation.

Learning-rate schedules are YAML-controlled cosine or polynomial decay with optional update-based linear warmup. Both epoch and iteration runners are supported.

Standalone evaluation is generative by default for two-term and three-term paths. Generative validation, `best_generative.pt` selection, and final evaluation use 20 steps by default; `--generative-steps` remains available as an override. Pass `--fixed-t` only for conditional reconstruction diagnostics:

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

The tests cover MiT-only initialization, all b0–b5 variants, cache fingerprint/staleness/missing samples, λ and uniform baselines, CCDM/MMSeg geometry, interpolation and crop ratio, epoch/iter runners, cosine/poly/warmup, conditional/generative checkpoint decisions, and the three-term numerical continuity equation.

## Protocol commands

Stage 1 CCDM and MMSeg:

```bash
uv run python src/train_source.py --config configs/source_pretrain_cityscapes_ccdm.yaml
uv run python src/train_source.py --config configs/source_pretrain_cityscapes_mmseg.yaml
```

Create the CCDM conditioned cache:

```bash
uv run python src/cache_source_logits.py --config configs/dfm_cityscapes_ccdm_conditioned.yaml --splits train val
```

Stage 2 four-way ablation:

```bash
uv run python src/train_dfm.py --config configs/dfm_cityscapes_ccdm_conditioned.yaml
uv run python src/train_dfm.py --config configs/dfm_cityscapes_ccdm_uniform.yaml
uv run python src/train_dfm.py --config configs/dfm_cityscapes_mmseg_conditioned.yaml
uv run python src/train_dfm.py --config configs/dfm_cityscapes_mmseg_uniform.yaml
```

Two-term or three-term evaluation uses the path stored in the checkpoint config. Edit `flow.path` before training to choose the path; omit `--config` to restore it from the checkpoint:

```bash
uv run python src/evaluate.py --checkpoint outputs/ccdm_conditioned/best_generative.pt --output outputs/eval_two_term
uv run python src/evaluate.py --checkpoint outputs/three_term/best_generative.pt --output outputs/eval_three_term
```

Source sweep:

```bash
uv run python src/visualize_source.py --config configs/dfm_cityscapes_ccdm_conditioned.yaml --lambdas 0 .1 .2 .4 .8 1 --temperatures .5 1 2 4
```
