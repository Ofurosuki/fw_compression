"""Event-tensor dataset + feature-mode builder.

Reads the cached top-K=8 event frames and produces, per sample, the event
feature tensor for a requested ``K`` and ``feature_mode`` (ablation variants
from ``initial_plan.md``). Top-K selection is by amplitude (greedy-NMS height),
then re-sorted chronologically so ``delta_t`` and the rank embedding are in
time order, exactly as the plan specifies.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from eventnet import paths
from eventnet.cache_events import cache_path

T = float(paths.T_CROPPED)

# columns each mode feeds the shared event MLP (m = valid mask is always last)
# E = behind-energy (full-waveform transmitted-energy-past-peak cue)
FEATURE_COLUMNS = {
    "t_only": ["t", "m"],
    "t_dt":   ["t", "dt", "m"],
    "ta":     ["t", "a", "m"],
    "tdta":   ["t", "dt", "a", "m"],
    "taw":    ["t", "a", "w", "m"],
    "tdtaw":  ["t", "dt", "a", "w", "m"],
    "taE":    ["t", "a", "E", "m"],
    "tdtaE":  ["t", "dt", "a", "E", "m"],
    "tdtaEw": ["t", "dt", "a", "w", "E", "m"],
}


def feature_dim(mode: str) -> int:
    return len(FEATURE_COLUMNS[mode])


def assemble_features(t_bin, a, w, val, mode, e=None):
    """Normalised feature columns for events already top-K & time-sorted.

    t_bin, a, w, (e), val: (..., K). Returns feat (..., K, F). Used at both train
    (after top-K selection) and eval (extractor already returns top-K by time).
    ``e`` (behind-energy, already in [0,1]) is required for modes using "E".
    """
    t_norm = t_bin / T
    w_norm = w / T
    any_valid = val.any(dim=-1, keepdim=True)
    t_first = t_bin[..., :1]                                  # earliest valid (slot 0)
    dt = torch.where(any_valid, (t_bin - t_first) / T, torch.zeros_like(t_bin))
    m = val.float()
    if e is None:
        e = torch.zeros_like(t_bin)
    t_norm, a, w_norm, dt, e = (x * m for x in (t_norm, a * m, w_norm, dt, e))
    cols = {"t": t_norm, "dt": dt, "a": a, "w": w_norm, "E": e, "m": m}
    return torch.stack([cols[c] for c in FEATURE_COLUMNS[mode]], dim=-1)


def select_topk(t_bin, a, w, e, valid, labels, k):
    """Top-k by amplitude (nested under greedy-NMS height ranking), then sorted
    chronologically (invalid pushed last). Returns t_bin,a,w,e,val,lab (...,k)."""
    score = torch.where(valid, a, torch.full_like(a, -1.0))
    idx = score.argsort(dim=-1, descending=True)[..., :k]
    g = lambda x: torch.gather(x, -1, idx)
    t_bin, a, w, e, val, lab = g(t_bin), g(a), g(w), g(e), g(valid), g(labels)
    skey = torch.where(val, t_bin, torch.full_like(t_bin, T + 1.0))
    order = skey.argsort(dim=-1)
    g2 = lambda x: torch.gather(x, -1, order)
    return g2(t_bin), g2(a), g2(w), g2(e), g2(val), g2(lab)


def build_features(events: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor,
                   k: int, mode: str):
    """events (..., 8, 4) [t_bin, a, w, E]; valid/labels (..., 8).

    Returns feat (..., k, F), lab (..., k) long, val (..., k) bool.
    """
    e = events[..., 3] if events.shape[-1] > 3 else torch.zeros_like(events[..., 0])
    t_bin, a, w, e, val, lab = select_topk(
        events[..., 0], events[..., 1], events[..., 2], e, valid, labels, k)
    feat = assemble_features(t_bin, a, w, val, mode, e=e)
    lab = torch.where(val, lab, torch.zeros_like(lab)).long()
    return feat, lab, val


class EventFrameDataset(Dataset):
    def __init__(self, split: str, frame_stride: int, k: int, mode: str,
                 crop=None, augment=False, limit: int = 0):
        self.dir = cache_path(split, frame_stride)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npz")))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"no cached frames in {self.dir}; run cache_events first")
        self.k, self.mode, self.crop, self.augment = k, mode, crop, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        events = torch.from_numpy(z["events"].astype(np.float32))   # (X, Y, 8, 3)
        valid = torch.from_numpy(z["valid"])                        # (X, Y, 8) bool
        labels = torch.from_numpy(z["labels"].astype(np.int64))     # (X, Y, 8)
        X, Y = events.shape[:2]

        if self.crop is not None:
            ch, cw = self.crop
            x0 = np.random.randint(0, X - ch + 1) if self.augment else (X - ch) // 2
            y0 = np.random.randint(0, Y - cw + 1) if self.augment else (Y - cw) // 2
            events = events[x0:x0 + ch, y0:y0 + cw]
            valid = valid[x0:x0 + ch, y0:y0 + cw]
            labels = labels[x0:x0 + ch, y0:y0 + cw]

        if self.augment:
            if np.random.rand() < 0.5:
                events, valid, labels = events.flip(0), valid.flip(0), labels.flip(0)
            if np.random.rand() < 0.5:
                events, valid, labels = events.flip(1), valid.flip(1), labels.flip(1)

        feat, lab, val = build_features(events, valid, labels, self.k, self.mode)
        return {"events": feat, "labels": lab, "valid": val}
