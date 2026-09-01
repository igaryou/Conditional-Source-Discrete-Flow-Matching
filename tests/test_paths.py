import torch

from cs_dfm.flow.paths import build_path
from cs_dfm.flow.schedulers import LinearScheduler, PowerScheduler, PowerUniformBumpScheduler
from cs_dfm.flow.inference import sample_two_term


def test_linear_endpoints_and_power_one():
    t = torch.linspace(0, 1, 19)
    linear = LinearScheduler()(t); power = PowerScheduler(1)(t)
    assert linear[0] == 0 and linear[-1] == 1
    assert torch.equal(linear, power)


def test_three_term_simplex_and_endpoints():
    w = PowerUniformBumpScheduler(2, .3)(torch.linspace(0, 1, 51))
    assert torch.all(w.kappa1 >= 0) and torch.all(w.kappa2 >= 0) and torch.all(w.kappa3 >= 0)
    assert torch.allclose(w.kappa1 + w.kappa2 + w.kappa3, torch.ones_like(w.kappa1))
    assert (w.kappa1[0], w.kappa2[0], w.kappa3[0]) == (0, 0, 1)
    assert (w.kappa1[-1], w.kappa2[-1], w.kappa3[-1]) == (1, 0, 0)


def test_path_samples_endpoints():
    z0, z1 = torch.zeros(2,4,5,dtype=torch.long), torch.ones(2,4,5,dtype=torch.long)
    for cfg in ({"type":"two_term","scheduler":"linear"},
                {"type":"three_term","scheduler":"power_uniform_bump","power":2,"uniform_strength":.3}):
        path=build_path(cfg,3)
        assert torch.equal(path.sample(z0,z1,torch.zeros(2)),z0)
        assert torch.equal(path.sample(z0,z1,torch.ones(2)),z1)


def test_two_term_inference_smoke():
    class Model(torch.nn.Module):
        def forward(self, image, zt, t):
            return torch.zeros(image.shape[0], 3, *image.shape[-2:])
    image=torch.randn(1,3,4,5); z0=torch.zeros(1,4,5,dtype=torch.long)
    result=sample_two_term(Model(),image,z0,LinearScheduler(),3,num_steps=3,
                           generator=torch.Generator().manual_seed(1))
    assert result.shape == z0.shape and result.min() >= 0 and result.max() < 3
