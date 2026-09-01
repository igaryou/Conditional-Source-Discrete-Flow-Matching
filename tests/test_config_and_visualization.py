from pathlib import Path

import torch

from cs_dfm.config import load_config
from cs_dfm.visualization import visualize_paths


def test_configs_parse():
    root=Path(__file__).parents[1]
    assert load_config(root/"configs/source_pretrain_cityscapes.yaml")["training"]["stage"] == "source_pretrain"
    assert load_config(root/"configs/dfm_cityscapes.yaml")["training"]["stage"] == "dfm"


def test_path_visualization_smoke(tmp_path):
    z0=torch.zeros(8,8,dtype=torch.long); z1=torch.ones(8,8,dtype=torch.long)
    visualize_paths([{"name":"linear","type":"two_term","scheduler":"linear"}],2,str(tmp_path),z0,z1)
    assert (tmp_path/"schedulers.png").exists()
    assert (tmp_path/"linear/zt_grid.png").exists()

