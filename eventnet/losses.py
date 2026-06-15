"""Event-level weighted cross-entropy + V-REx domain-generalization loss."""
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


def masked_focal(logits, labels, valid, class_weights, gamma=2.0):
    """Class-weighted focal loss over valid events. Down-weights easy (well-
    classified) events by (1-p_t)^gamma, focusing capacity on the hard minority
    classes (glass/ghost)."""
    C = logits.shape[-1]
    m = valid.reshape(-1)
    lo = logits.reshape(-1, C)[m]
    la = labels.reshape(-1)[m]
    if lo.numel() == 0:
        return logits.sum() * 0.0
    logp = F.log_softmax(lo, dim=1)
    logpt = logp.gather(1, la[:, None]).squeeze(1)
    pt = logpt.exp()
    w = class_weights.to(lo.device)[la]
    return (-w * (1.0 - pt) ** gamma * logpt).mean()


def vrex_loss(logits, labels, valid, scene, class_weights, beta):
    """V-REx (Krueger 2021): ERM + beta * Var over per-scene risks.

    Treats each SCENE in the batch as an environment. Per-frame risk = mean
    weighted CE over that frame's valid events; per-scene risk = mean over its
    frames; the penalty pushes all scenes to similar risk so the model can't rely
    on scene-specific (shortcut) cues. ``scene`` is a (B,) long tensor of scene
    ids. Returns (total_loss, erm_term, penalty) for logging.
    """
    B, C = logits.shape[0], logits.shape[-1]
    cw = class_weights.to(logits.device)
    ce = F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1),
                         weight=cw, reduction="none").reshape(B, -1)        # (B, HWK)
    vf = valid.reshape(B, -1).float()
    frame_risk = (ce * vf).sum(1) / vf.sum(1).clamp_min(1.0)                # (B,) per-frame
    erm = frame_risk.mean()
    scenes = torch.unique(scene)
    if scenes.numel() < 2 or beta <= 0:
        return erm, erm.detach(), torch.zeros((), device=erm.device)
    per_scene = torch.stack([frame_risk[scene == s].mean() for s in scenes])
    penalty = per_scene.var(unbiased=False)
    erm_grpbal = per_scene.mean()                  # group-balanced ERM term
    return erm_grpbal + beta * penalty, erm_grpbal.detach(), penalty.detach()
