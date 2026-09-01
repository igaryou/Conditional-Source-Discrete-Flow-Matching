from __future__ import annotations

import torch
import torch.distributed as dist


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = -100, eval_num_classes: int | None = None):
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.eval_num_classes = eval_num_classes or num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = pred.flatten().cpu(), target.flatten().cpu()
        valid = ((target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
                 & (target < self.eval_num_classes))
        ids = target[valid] * self.num_classes + pred[valid]
        self.confusion += torch.bincount(ids, minlength=self.num_classes ** 2).reshape(self.num_classes, -1)

    def synchronize(self, device: torch.device):
        if dist.is_initialized():
            value = self.confusion.to(device)
            dist.all_reduce(value)
            self.confusion = value.cpu()

    def compute(self) -> dict:
        c = self.confusion.float(); tp = c.diag(); union = c.sum(0) + c.sum(1) - tp
        iou = tp / union.clamp_min(1)
        valid = (union > 0) & (torch.arange(self.num_classes) < self.eval_num_classes)
        return {"pixel_accuracy": float(tp.sum() / c.sum().clamp_min(1)),
                "mIoU": float(iou[valid].mean()) if valid.any() else 0.0,
                "class_IoU": [float(x) for x in iou], "confusion_matrix": self.confusion.tolist()}
