from pathlib import Path

import torch
from transformers import SegformerModel

import cs_dfm.source as source


def test_all_variants_random_build():
    for variant in source.BACKBONE_NAMES:
        model=source.SourceSegFormer(variant,3,"random")
        assert model.model.decode_head.classifier.out_channels == 3
        del model


def test_standard_variant_configs():
    expected_depths = {
        "b0": [2, 2, 2, 2], "b1": [2, 2, 2, 2], "b2": [3, 4, 6, 3],
        "b3": [3, 4, 18, 3], "b4": [3, 8, 27, 3], "b5": [3, 6, 40, 3],
    }
    for variant in source.BACKBONE_NAMES:
        cfg = source.segformer_config(variant, 20)
        assert cfg.depths == expected_depths[variant]
        assert cfg.hidden_sizes == ([32, 64, 160, 256] if variant == "b0" else [64, 128, 320, 512])
        assert cfg.decoder_hidden_size == (256 if variant in {"b0", "b1"} else 768)
        assert cfg.num_attention_heads == [1, 2, 5, 8]
        assert cfg.sr_ratios == [8, 4, 2, 1]
        assert cfg.patch_sizes == [7, 3, 3, 3]
        assert cfg.strides == [4, 2, 2, 2]


def test_b1_random_output_shape():
    model=source.SourceSegFormer("b1",20,"random").eval()
    assert model(torch.rand(1,3,32,64)).shape == (1,20,32,64)


def test_mit_imagenet_loads_only_backbone(monkeypatch):
    pretrained=SegformerModel(source.segformer_config("b0",3))
    for p in pretrained.parameters(): torch.nn.init.constant_(p,.125)
    monkeypatch.setattr(source.SegformerModel,"from_pretrained",lambda name: pretrained)
    model=source.SourceSegFormer("b0",3,"mit_imagenet")
    assert all(torch.allclose(a,b) for a,b in zip(model.model.segformer.parameters(),pretrained.parameters()))
    assert not torch.allclose(model.model.decode_head.classifier.weight,torch.full_like(model.model.decode_head.classifier.weight,.125))
    assert all("segformer-b" not in name or "finetuned" not in name for name in source.BACKBONE_NAMES.values())
