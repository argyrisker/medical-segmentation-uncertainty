from __future__ import annotations

from typing import Any

import torch
from monai.losses import DiceCELoss, DiceLoss
from torch import nn
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    """Plain voxel-wise cross-entropy. Labels arrive as B x 1 x H x W x D."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels[:, 0].long())


def build_loss(cfg: dict[str, Any]) -> nn.Module:
    """
    dicece - soft Dice on the foreground classes + cross-entropy (v1)
    dice   - soft Dice only
    ce     - cross-entropy only
    """
    name = cfg["training"].get("loss", "dicece")
    if name == "dicece":
        return DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    if name == "dice":
        return DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    if name == "ce":
        return CrossEntropy()
    raise ValueError(f"Unknown training.loss: {name}")
