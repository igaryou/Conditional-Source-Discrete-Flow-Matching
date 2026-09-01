import random

import torch

from cs_dfm.data import SegmentationTransform, _resize, _valid_crop
from cs_dfm.lr_scheduler import ConfigLRScheduler
from cs_dfm.train import checkpoint_decisions, runner_limit


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

