from __future__ import annotations

import hashlib
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def amp_context(enabled: bool, dtype: str, device: torch.device):
    if not enabled or device.type != "cuda":
        return nullcontext()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    return torch.autocast("cuda", dtype=amp_dtype)


def init_distributed(enabled: bool | str = "auto", backend: str = "nccl") -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    should_init = world > 1 if enabled == "auto" else bool(enabled)
    if should_init and not dist.is_initialized():
        dist.init_process_group(backend=backend if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

