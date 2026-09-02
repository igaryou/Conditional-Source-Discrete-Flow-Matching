import pytest
import torch

from cs_dfm.evaluate import conditional_cross_entropy
import cs_dfm.source as source
import cs_dfm.train as train
from cs_dfm.config import validate_stage2_source_runtime
from cs_dfm.flow.inference import sample_discrete_flow
from cs_dfm.flow.paths import build_path


def stage2_cfg(pipeline="ccdm_fixed", source_type="image_conditioned", mode="cache"):
    return {
        "dataset": {
            "name": "cityscapes", "pipeline": pipeline, "num_classes": 3,
            "image_size_hw": [4, 5], "canonical_size_hw": [4, 5],
            "loss_ignore_index": 2, "eval_num_classes": 2,
        },
        "source_distribution": {
            "type": source_type, "lambda": 0.2, "temperature": 2.0,
            "sampling": "categorical",
        },
        "source_runtime": {"mode": mode},
        "source_cache": {"enabled": mode == "cache"},
        "source": {"architecture": "segformer", "variant": "b0", "checkpoint": "source.pt"},
        "runtime": {"amp": False},
    }


def test_stage2_loss_ignores_configured_target_and_conditional_validation_uses_it(monkeypatch):
    cfg = stage2_cfg()
    target = torch.tensor([[[0, 2]]])
    logits_a = torch.tensor([[[[2.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    logits_b = logits_a.clone()
    logits_b[:, :, :, 1] = torch.tensor([1000.0, -1000.0, 500.0])[:, None]
    assert torch.equal(train.dfm_cross_entropy(logits_a, target, cfg),
                       train.dfm_cross_entropy(logits_b, target, cfg))
    assert torch.equal(conditional_cross_entropy(logits_a, target, cfg["dataset"]),
                       conditional_cross_entropy(logits_b, target, cfg["dataset"]))

    seen = []
    original = train.dfm_cross_entropy
    monkeypatch.setattr(train, "dfm_cross_entropy", lambda logits, z1, c: seen.append(c["dataset"]["loss_ignore_index"]) or original(logits, z1, c))

    class Provider:
        def sample(self, batch, image, shape, generator=None):
            probs = torch.full((shape[0], 3, shape[1], shape[2]), 1 / 3)
            return torch.zeros(shape, dtype=torch.long), probs

    class Path:
        def sample(self, z0, z1, t, generator=None): return z0

    class Model(torch.nn.Module):
        def forward(self, image, zt, t): return logits_a.expand(image.shape[0], -1, -1, -1)

    batch = {"image": torch.zeros(1, 3, 1, 2), "mask": target}
    train.validate_conditional(Model(), [batch], cfg, Path(), torch.device("cpu"), Provider())
    assert seen == [2]


@pytest.mark.parametrize("pipeline", ["ccdm_fixed", "mmseg"])
def test_uniform_provider_never_builds_source_or_uses_cache(monkeypatch, pipeline):
    cfg = stage2_cfg(pipeline, "uniform", "none")
    monkeypatch.setattr(source, "build_source_model", lambda cfg: pytest.fail("source model built"))
    provider = source.build_stage2_source_provider(cfg, torch.device("cpu"))
    assert not provider.needs_cache and provider.model is None
    z0, p0 = provider.sample({}, torch.rand(1, 3, 4, 5), (1, 4, 5))
    assert z0.shape == (1, 4, 5)
    assert torch.allclose(p0, torch.full_like(p0, 1 / 3))


def test_ccdm_cache_provider_does_not_build_or_forward_source(monkeypatch):
    cfg = stage2_cfg()
    monkeypatch.setattr(source, "build_source_model", lambda cfg: pytest.fail("source model built"))
    provider = source.build_stage2_source_provider(cfg, torch.device("cpu"))
    assert provider.needs_cache and provider.model is None
    batch = {"source_logits": torch.randn(1, 3, 4, 5)}
    z0, _ = provider.sample(batch, torch.rand(1, 3, 4, 5), (1, 4, 5))
    assert z0.shape == (1, 4, 5)


def test_mmseg_online_loads_frozen_source_and_forwards_augmented_image(monkeypatch, tmp_path):
    cfg = stage2_cfg("mmseg", "image_conditioned", "online")
    cfg["source"]["checkpoint"] = str(tmp_path / "source.pt")
    (tmp_path / "source.pt").write_bytes(b"checkpoint handled by monkeypatch")

    class TrackingSource(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(0.5)); self.seen = None; self.inference = False
        def forward(self, image):
            self.seen = image.detach().clone(); self.inference = torch.is_inference_mode_enabled()
            return self.weight * torch.ones(image.shape[0], 3, *image.shape[-2:], device=image.device)

    mu = TrackingSource().train()
    loaded = []
    monkeypatch.setattr(source, "build_source_model", lambda cfg: mu)
    monkeypatch.setattr(source, "load_source_checkpoint", lambda model, path, map_location=None: loaded.append(path) or {})
    provider = source.build_stage2_source_provider(cfg, torch.device("cpu"))
    assert loaded == [cfg["source"]["checkpoint"]] and not mu.training
    assert all(not p.requires_grad for p in mu.parameters())

    original = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
    augmented = original.flip(-1)
    before = mu.weight.detach().clone()
    z0, p0 = provider.sample({}, augmented, (1, 4, 5))
    assert torch.equal(mu.seen, augmented) and not torch.equal(mu.seen, original)
    assert mu.inference

    dfm = torch.nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.SGD(dfm.parameters(), lr=0.1)
    loss = torch.nn.functional.cross_entropy(dfm(augmented), z0)
    loss.backward(); optimizer.step()
    assert mu.weight.grad is None and torch.equal(mu.weight, before)
    source_ids = {id(p) for p in mu.parameters()}
    assert all(id(p) not in source_ids for group in optimizer.param_groups for p in group["params"])

    class DFM(torch.nn.Module):
        def forward(self, image, zt, t): return torch.zeros(image.shape[0], 3, *image.shape[-2:])

    path = build_path({"type": "two_term", "scheduler": "linear"}, 3)
    pred = sample_discrete_flow(DFM(), augmented, z0, path, 20, False,
                                torch.Generator().manual_seed(1), p0)
    assert pred.shape == z0.shape


def test_standard_runtime_protocol_rejects_crossed_modes():
    with pytest.raises(ValueError, match="requires source_runtime.mode=cache"):
        validate_stage2_source_runtime(stage2_cfg("ccdm_fixed", mode="online"))
    with pytest.raises(ValueError, match="requires source_runtime.mode=online"):
        validate_stage2_source_runtime(stage2_cfg("mmseg", mode="cache"))
    bad = stage2_cfg(source_type="uniform", mode="none")
    bad["source_cache"]["enabled"] = True
    with pytest.raises(ValueError, match="must not enable source_cache"):
        validate_stage2_source_runtime(bad)


def test_online_requires_checkpoint():
    cfg = stage2_cfg("mmseg", "image_conditioned", "online")
    cfg["source"]["checkpoint"] = None
    with pytest.raises(ValueError, match="requires source.checkpoint"):
        source.build_stage2_source_provider(cfg, torch.device("cpu"))


def test_online_requires_existing_checkpoint_file():
    cfg = stage2_cfg("mmseg", "image_conditioned", "online")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        source.build_stage2_source_provider(cfg, torch.device("cpu"))
