import random
from pathlib import Path

import pytest
import torch

from cs_dfm.config import load_config
from cs_dfm.data import SegmentationTransform, _resize, _valid_crop
from cs_dfm.lr_scheduler import ConfigLRScheduler
from cs_dfm.source import SourceSegFormer
from cs_dfm.train import build_optimizer, checkpoint_decisions, runner_limit


def base_cfg(pipeline="ccdm_fixed"):
    return {"dataset":{"pipeline":pipeline,"image_size_hw":[4,6],"canonical_size_hw":[4,6],"num_classes":3,"void_class_index":2,
                       "augmentation":{"random_resize":False,"random_crop":False,"horizontal_flip":0.,"photometric":False}}}


def test_ccdm_fixed_spatial_size_and_interpolation_modes():
    image=torch.rand(3,4,6);mask=torch.tensor([[0,0,1,1,2,2]]*4);logits=torch.stack([mask.float(),mask.float()*2,mask.float()*3])
    out=SegmentationTransform(base_cfg(),"train",True)(image,mask,logits)
    assert all(x.shape[-2:]==(4,6) for x in out)
    _,m,l=_resize(image,mask,logits,(7,9));assert set(m.unique().tolist())<=set(mask.unique().tolist())
    assert ((l-l.round()).abs()>0).any()


def test_mmseg_shared_geometry_and_cat_ratio():
    c=base_cfg("mmseg");c["dataset"]["train_pipeline"]={"random_resize":{"enabled":False},"random_crop":{"enabled":True,"crop_size_hw":[3,3],"cat_max_ratio":1.,"max_attempts":1},"horizontal_flip":{"probability":1.},"photometric":{"enabled":False}}
    mask=torch.arange(24).reshape(4,6)%3;image=mask.float()[None].repeat(3,1,1);logits=mask.float()[None]
    random.seed(4);i,m,l=SegmentationTransform(c,"train",True)(image,mask,logits)
    assert torch.equal(i[0].long(),m) and torch.equal(l[0].long(),m)
    dominant=torch.tensor([[0,0,0,1],[0,0,0,1],[2,2,2,1],[2,2,2,1]])
    assert not _valid_crop(dominant,0,0,(2,3),.75,None)
    assert _valid_crop(dominant,1,0,(3,3),.75,None)


def test_epoch_iter_scheduler_warmup_and_checkpoint_decisions():
    assert runner_limit({"runner":"epoch","epochs":8})==8 and runner_limit({"runner":"iter","max_iters":17})==17
    p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.SGD([p],lr=1.)
    cosine=ConfigLRScheduler(o,{"type":"cosine","eta_min":0.},10);start=o.param_groups[0]["lr"];cosine.step();assert o.param_groups[0]["lr"]<start
    o=torch.optim.SGD([p],lr=1.);poly=ConfigLRScheduler(o,{"type":"poly","power":1.,"eta_min":0.,"warmup":{"enabled":True,"begin":0,"end":2,"start_factor":.1}},10)
    assert o.param_groups[0]["lr"]==.1;poly.step();assert .1<o.param_groups[0]["lr"]<1
    assert checkpoint_decisions({"mIoU":.6},None,.5,.5)==(True,False)
    assert checkpoint_decisions(None,{"mIoU":.7},.5,.6)==(False,True)


def test_stage1_paramwise_optimizer_is_complete_disjoint_and_scaled():
    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs/source_pretrain_cityscapes_mmseg.yaml")
    model = SourceSegFormer("b1", 20, "random")
    optimizer = build_optimizer(cfg, model)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    assert groups["decode_head"]["lr"] == pytest.approx(6e-4)
    assert groups["decode_head_norm_no_decay"]["lr"] == pytest.approx(6e-4)
    assert groups["backbone_norm_no_decay"]["weight_decay"] == 0
    assert groups["backbone"]["weight_decay"] == pytest.approx(0.01)
    dwconv = next(name for name in groups["backbone"]["parameter_names"] if name.endswith("mlp.dwconv.dwconv.weight"))
    assert dwconv.startswith("model.segformer.")
    assert any("layer_norm" in name for name in groups["backbone_norm_no_decay"]["parameter_names"])
    assert "model.decode_head.classifier.weight" in groups["decode_head"]["parameter_names"]
    assert "model.decode_head.batch_norm.weight" in groups["decode_head_norm_no_decay"]["parameter_names"]
    cfg["optimizer"]["paramwise"]["enabled"] = False
    assert len(build_optimizer(cfg, model).param_groups) == 1
    cfg["optimizer"]["paramwise"]["enabled"] = True
    cfg["training"]["stage"] = "dfm"
    assert len(build_optimizer(cfg, model).param_groups) == 1


def test_stage1_smoke_forward_backward_optimizer_and_scheduler():
    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs/source_pretrain_cityscapes_mmseg.yaml")
    model = SourceSegFormer("b1", 20, "random")
    optimizer = build_optimizer(cfg, model)
    scheduler = ConfigLRScheduler(optimizer, cfg["scheduler"], cfg["training"]["max_iters"])
    logits = model(torch.rand(1, 3, 32, 64))
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 20, (1, 32, 64)))
    loss.backward(); optimizer.step(); scheduler.step()
    assert torch.isfinite(loss)
    assert scheduler.update == 1


def test_stage_budgets_scheduler_points_and_mmseg_parity():
    root = Path(__file__).parents[1]
    stage1 = load_config(root / "configs/source_pretrain_cityscapes_mmseg.yaml")
    conditioned = load_config(root / "configs/dfm_cityscapes_mmseg_conditioned.yaml")
    uniform = load_config(root / "configs/dfm_cityscapes_mmseg_uniform.yaml")
    assert stage1["training"]["max_iters"] == 32000
    assert conditioned["training"]["max_iters"] == 128000
    assert uniform["training"]["max_iters"] == 160000
    assert stage1["training"]["val_interval"] == 4000
    assert stage1["dataset"]["train_pipeline"] == conditioned["dataset"]["train_pipeline"]
    assert conditioned["validation"] == uniform["validation"]
    for key in ("dataset", "flow", "model", "scheduler", "validation", "evaluation", "runtime", "distributed"):
        assert conditioned[key] == uniform[key]
    assert {key: value for key, value in conditioned["training"].items() if key != "max_iters"} == {
        key: value for key, value in uniform["training"].items() if key != "max_iters"
    }
    assert conditioned["validation"]["generative_steps"] == conditioned["evaluation"]["generative_steps"] == 20
    assert stage1["training_budget"] == conditioned["training_budget"] == {
        "stage1_source_pretraining_updates": 32000, "stage2_dfm_updates": 128000,
        "total_updates": 160000, "scope": "proposal_wide",
    }
    assert uniform["training_budget"] == {
        "stage1_source_pretraining_updates": 0, "stage2_dfm_updates": 160000,
        "total_updates": 160000, "scope": "proposal_wide",
    }

    model = SourceSegFormer("b1", 20, "random")
    optimizer = build_optimizer(stage1, model)
    scheduler = ConfigLRScheduler(optimizer, stage1["scheduler"], stage1["training"]["max_iters"])
    backbone_index = next(i for i, group in enumerate(optimizer.param_groups) if group["group_name"] == "backbone")
    head_index = next(i for i, group in enumerate(optimizer.param_groups) if group["group_name"] == "decode_head")
    sampled = {}
    for update in (0, 1, 100, 1499, 1500, 1501, 4000, 16000, 31999, 32000):
        scheduler.update = update; scheduler._apply()
        backbone_lr = optimizer.param_groups[backbone_index]["lr"]
        head_lr = optimizer.param_groups[head_index]["lr"]
        sampled[update] = backbone_lr
        assert head_lr == pytest.approx(backbone_lr * 10)
    assert sampled[32000] == 0
    assert abs(sampled[1500] - sampled[1499]) < stage1["training"]["lr"] * 0.001
    for cfg, total in ((conditioned, 128000), (uniform, 160000)):
        stage2_scheduler = ConfigLRScheduler(
            torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.))], lr=1.),
            cfg["scheduler"], cfg["training"]["max_iters"],
        )
        assert stage2_scheduler.total_updates == total
        stage2_scheduler.update = total; stage2_scheduler._apply()
        assert stage2_scheduler.optimizer.param_groups[0]["lr"] == cfg["scheduler"]["eta_min"]
