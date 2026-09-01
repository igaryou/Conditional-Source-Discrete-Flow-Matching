import json

import pytest
import torch

import cs_dfm.cache as cache
from cs_dfm.cache_spec import cache_directory, cache_fingerprint, cache_fingerprint_spec


class FakeDataset:
    def __len__(self): return 2
    def expected_sample_ids(self): return ["a","b"]


def cfg(tmp_path, checkpoint):
    return {"dataset":{"name":"cityscapes","pipeline":"ccdm_fixed","image_size_hw":[4,6],"num_classes":3},
            "source":{"architecture":"segformer","variant":"b0","checkpoint":str(checkpoint)},
            "source_cache":{"root":str(tmp_path/"cache"),"dtype":"float16"},
            "source_distribution":{"type":"image_conditioned"}}


def write_valid(c):
    root=cache_directory(c);(root/"val").mkdir(parents=True)
    fp=cache_fingerprint(c);ids=["a","b"]
    (root/"metadata.json").write_text(json.dumps({"fingerprint":fp,"fingerprint_spec":cache_fingerprint_spec(c),"splits":{"val":2}}))
    (root/"manifest.json").write_text(json.dumps({"fingerprint":fp,"splits":{"val":ids}}))
    for sid in ids:torch.save({"sample_id":sid,"logits":torch.zeros(3,4,6,dtype=torch.float16)},root/"val"/f"{sid}.pt")
    return root


def test_checkpoint_change_cannot_reuse_stale_cache(tmp_path,monkeypatch):
    a=tmp_path/"a.pt";b=tmp_path/"b.pt";a.write_bytes(b"A");b.write_bytes(b"B")
    ca,cb=cfg(tmp_path,a),cfg(tmp_path,b);assert cache_fingerprint(ca)!=cache_fingerprint(cb)
    write_valid(ca);monkeypatch.setattr(cache,"build_dataset",lambda *a,**k:FakeDataset())
    cache.verify_cache(ca,"val")
    with pytest.raises(RuntimeError):cache.verify_cache(cb,"val")


def test_missing_sample_and_preprocessing_mismatch_fail(tmp_path,monkeypatch):
    ck=tmp_path/"a.pt";ck.write_bytes(b"A");c=cfg(tmp_path,ck);root=write_valid(c)
    monkeypatch.setattr(cache,"build_dataset",lambda *a,**k:FakeDataset());(root/"val/b.pt").unlink()
    with pytest.raises(RuntimeError,match="missing"):cache.verify_cache(c,"val")
    c2=cfg(tmp_path,ck);c2["dataset"]["image_size_hw"]=[8,12]
    with pytest.raises(RuntimeError):cache.verify_cache(c2,"val")

