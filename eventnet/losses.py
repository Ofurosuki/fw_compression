"""Event-level weighted cross-entropy (valid events only)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_weighted_ce(logits, labels, valid, class_weights):
    """logits [B,H,W,K,C], labels/valid [B,H,W,K]. CE over valid events only."""
    C = logits.shape[-1]
    m = valid.reshape(-1)
    lo = logits.reshape(-1, C)[m]
    la = labels.reshape(-1)[m]
    if lo.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(lo, la, weight=class_weights.to(lo.device))
