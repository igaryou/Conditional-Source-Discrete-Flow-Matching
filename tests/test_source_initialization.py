from pathlib import Path

import torch
from transformers import SegformerModel

import cs_dfm.source as source


def test_all_variants_random_build():
    for variant in source.BACKBONE_NAMES:
        model=source.SourceSegFormer(variant,3,"random")
        assert model.model.decode_head.classifier.out_channels == 3
        del model


def test_random_output_shape():
    model=source.SourceSegFormer("b0",7,"random").eval()
    assert model(torch.rand(1,3,32,64)).shape == (1,7,32,64)


def test_mit_imagenet_loads_only_backbone(monkeypatch):
    pretrained=SegformerModel(source.segformer_config("b0",3))
    for p in pretrained.parameters(): torch.nn.init.constant_(p,.125)
    monkeypatch.setattr(source.SegformerModel,"from_pretrained",lambda name: pretrained)
    model=source.SourceSegFormer("b0",3,"mit_imagenet")
    assert all(torch.allclose(a,b) for a,b in zip(model.model.segformer.parameters(),pretrained.parameters()))
    assert not torch.allclose(model.model.decode_head.classifier.weight,torch.full_like(model.model.decode_head.classifier.weight,.125))
    assert all("segformer-b" not in name or "finetuned" not in name for name in source.BACKBONE_NAMES.values())

