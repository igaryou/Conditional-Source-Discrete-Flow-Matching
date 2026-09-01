import inspect

import torch

from cs_dfm.model import SegDiffModel
from cs_dfm.train import train_dfm


def test_dummy_dfm_forward_and_loss():
    model = SegDiffModel(num_classes=4, base_channels=8, channel_mults=(1,2), num_res_blocks=1,
                         time_dim=8, mask_embed_dim=4)
    image=torch.randn(2,3,16,24); zt=torch.randint(4,(2,16,24)); t=torch.rand(2)
    logits=model(image,zt,t)
    assert logits.shape == (2,4,16,24)
    torch.nn.functional.cross_entropy(logits,torch.randint(4,(2,16,24))).backward()


def test_cache_logits_round_trip_matches_online(tmp_path):
    online=torch.randn(5,8,9); path=tmp_path/"sample.pt"
    torch.save({"logits":online.half()},path)
    cached=torch.load(path,weights_only=True)["logits"].float()
    assert torch.allclose(cached,online,atol=1e-3,rtol=1e-3)


def test_stage2_does_not_construct_or_forward_source_model():
    source=inspect.getsource(train_dfm)
    assert "build_source_model" not in source
    assert "SourceSegFormer" not in source

