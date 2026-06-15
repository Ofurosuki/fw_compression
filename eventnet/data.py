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

# cached event columns (see eventnet/events.py)
EVENT_COLS = {"t": 0, "a": 1, "w": 2, "E": 3, "D": 4, "I": 5, "L": 6,
              "Tp0": 7, "Tp8": 8, "Tp16": 9, "Tp32": 10, "Tp64": 11}
TP_COLS = ["Tp0", "Tp8", "Tp16", "Tp32", "Tp64"]  # NeRF transmittance decay profile

# columns each mode feeds the shared event MLP (m = valid mask is always last).
# E=raw behind-energy; D=direct mass behind (glass cue); I=indirect/diffuse mass
# behind (ghost cue, the NEW info); L=indirect pedestal around the peak.
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
    # V3: taw + decomposed direct/indirect transport channels
    "tawD":   ["t", "a", "w", "D", "m"],            # + direct-behind (control)
    "tawI":   ["t", "a", "w", "I", "m"],            # + indirect-behind (new info)
    "tawi":   ["t", "a", "w", "D", "I", "L", "m"],  # + full decomposition
    # V4a: taw + NeRF-style peak-anchored transmittance decay profile
    "tawT":   ["t", "a", "w"] + TP_COLS + ["m"],    # + T(peak+δ) survival samples
    "tT":     ["t"] + TP_COLS + ["m"],              # transmittance-only (no a,w) probe
}


def feature_dim(mode: str) -> int:
    return len(FEATURE_COLUMNS[mode])


def assemble_features(ev, val, mode):
    """Normalised feature columns for a full event tensor already top-K &
    time-sorted. ``ev`` (..., K, C) with columns per ``EVENT_COLS``; ``val``
    (..., K) bool. Returns feat (..., K, F).

    t,w are normalised by T; a and the decomposition channels (E,D,I,L) are
    already fractions in [0,1]; dt is computed from the (time-sorted) t. All
    feature columns are zeroed on invalid slots (mask m is kept as-is).
    """
    t_bin = ev[..., EVENT_COLS["t"]]
    any_valid = val.any(dim=-1, keepdim=True)
    dt = torch.where(any_valid, (t_bin - t_bin[..., :1]) / T, torch.zeros_like(t_bin))
    m = val.float()
    raw = {
        "t": t_bin / T, "dt": dt, "a": ev[..., EVENT_COLS["a"]],
        "w": ev[..., EVENT_COLS["w"]] / T, "E": ev[..., EVENT_COLS["E"]],
        "D": ev[..., EVENT_COLS["D"]], "I": ev[..., EVENT_COLS["I"]],
        "L": ev[..., EVENT_COLS["L"]],
    }
    for c in TP_COLS:                       # NeRF transmittance profile (already [0,1])
        raw[c] = ev[..., EVENT_COLS[c]]
    cols = {k: v * m for k, v in raw.items()}
    cols["m"] = m
    return torch.stack([cols[c] for c in FEATURE_COLUMNS[mode]], dim=-1)


def select_topk(ev, valid, labels, k):
    """Top-k events by amplitude (nested under greedy-NMS height ranking), then
    sorted chronologically (invalid pushed last). ev (..., 8, C).
    Returns ev_k (..., k, C), val_k (..., k), lab_k (..., k)."""
    a = ev[..., EVENT_COLS["a"]]
    score = torch.where(valid, a, torch.full_like(a, -1.0))
    idx = score.argsort(dim=-1, descending=True)[..., :k]              # (..., k)
    eidx = idx.unsqueeze(-1).expand(*idx.shape, ev.shape[-1])
    ev_k = torch.gather(ev, -2, eidx)
    val_k = torch.gather(valid, -1, idx)
    lab_k = torch.gather(labels, -1, idx)
    # chronological re-sort
    t = ev_k[..., EVENT_COLS["t"]]
    order = torch.where(val_k, t, torch.full_like(t, T + 1.0)).argsort(dim=-1)
    oidx = order.unsqueeze(-1).expand(*order.shape, ev.shape[-1])
    ev_k = torch.gather(ev_k, -2, oidx)
    val_k = torch.gather(val_k, -1, order)
    lab_k = torch.gather(lab_k, -1, order)
    return ev_k, val_k, lab_k


def build_features(events: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor,
                   k: int, mode: str):
    """events (..., 8, C) per EVENT_COLS; valid/labels (..., 8).

    Returns feat (..., k, F), lab (..., k) long, val (..., k) bool.
    """
    ev_k, val_k, lab_k = select_topk(events, valid, labels, k)
    feat = assemble_features(ev_k, val_k, mode)
    lab = torch.where(val_k, lab_k, torch.zeros_like(lab_k)).long()
    return feat, lab, val_k


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
        # scene id per frame (filename = "{scene}__{hist}__{stem}.npz") for DG/V-REx
        self.scenes = [os.path.basename(f).split("__")[0] for f in self.files]
        self.scene_list = sorted(set(self.scenes))
        self.scene_to_id = {s: i for i, s in enumerate(self.scene_list)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        events = torch.from_numpy(z["events"].astype(np.float32))   # (X, Y, 8, C)
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
        return {"events": feat, "labels": lab, "valid": val,
                "scene": self.scene_to_id[self.scenes[i]]}
