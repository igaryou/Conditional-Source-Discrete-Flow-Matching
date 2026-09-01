import torch

from cs_dfm.flow.sampling import construct_source_probabilities, sample_categorical


def test_lambda_endpoints_and_simplex():
    logits = torch.randn(2, 5, 4, 3)
    p_uniform = construct_source_probabilities(logits, 1, 1)
    assert torch.allclose(p_uniform, torch.full_like(p_uniform, .2))
    p_mu = construct_source_probabilities(logits, 0, 2)
    assert torch.allclose(p_mu, torch.softmax(logits / 2, 1))
    assert torch.all(p_mu >= 0)
    assert torch.allclose(p_mu.sum(1), torch.ones_like(p_mu[:, 0]))


def test_temperature_changes_probabilities():
    logits = torch.tensor([[[[4.]], [[1.]], [[0.]]]])
    assert not torch.allclose(construct_source_probabilities(logits, 0, .5),
                              construct_source_probabilities(logits, 0, 2))


def test_sampling_is_not_argmax():
    probs = torch.tensor([.55, .45])[None, :, None, None].expand(1, 2, 1, 1000)
    samples = sample_categorical(probs, torch.Generator().manual_seed(3))
    assert (samples == 1).any() and (samples == 0).any()

