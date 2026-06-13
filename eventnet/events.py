"""Frame-level top-K event extraction + label assignment.

Shared by the cache builder (train/val) and the evaluator (test, on the fly).
Extraction uses the GPU-batch extractor from ``compression.event_extraction``
(height-ranked greedy NMS on per-pixel max-normalised waveforms). Because that
NMS is greedy by height, the top-K=8 set is *nested*: top-K' for any K'<=8 is
exactly the first K' picks, so one K=8 cache serves every K in {1,2,4,8}.

Labels follow the paper's peak scoring (``evaluate_peaks`` in the Ghost-FWL
repo reads ``annotation[d, y, x]`` at the exact peak bin) — each event's label
is the cropped annotation value at its peak bin, so train labels and the test
metric are defined identically. Invalid (padded) events get label 0 (noise).
"""
from __future__ import annotations

import numpy as np
import torch

from compression.event_extraction import extract_topk_events_batch

KMAX = 8  # events stored per pixel in the cache


@torch.no_grad()
def extract_frame_events(
    vox_crop: np.ndarray,
    device,
    k: int = KMAX,
    smooth_sigma: float = 1.5,
    min_height: float = 0.05,
    min_distance: int = 3,
    eps: float = 1e-6,
):
    """vox_crop: raw (X, Y, T) float32 (already y/z cropped).

    Returns:
        events: (X, Y, k, 3) float32 — columns ``[t_bin, a_height, w_fwhm_bin]``.
        valid:  (X, Y, k) bool.
    Background pixels (max<=eps) yield all-invalid events.
    """
    X, Y, T = vox_crop.shape
    w = torch.from_numpy(np.ascontiguousarray(vox_crop)).to(device).reshape(-1, T)
    mx = w.amax(dim=1, keepdim=True)
    valid_px = (mx > eps).squeeze(1)
    wn = torch.where(mx > eps, w / mx, w)
    events, vmask = extract_topk_events_batch(
        wn, k, smooth_sigma=smooth_sigma, min_height=min_height,
        min_distance=min_distance, intensity_mode="height")
    # kill events on background pixels
    vmask = vmask & valid_px[:, None]
    events[~vmask] = 0.0
    events = events.reshape(X, Y, k, 3).cpu().numpy().astype(np.float32)
    valid = vmask.reshape(X, Y, k).cpu().numpy()
    return events, valid


def assign_labels(ann_crop: np.ndarray, events: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Annotation value at each event's exact peak bin. ann_crop: (X, Y, T).

    events: (X, Y, k, 3); valid: (X, Y, k). Returns (X, Y, k) int8."""
    X, Y, k, _ = events.shape
    T = ann_crop.shape[2]
    t = np.clip(events[..., 0].astype(np.int64), 0, T - 1)        # (X, Y, k)
    xi = np.arange(X)[:, None, None]
    yi = np.arange(Y)[None, :, None]
    lab = ann_crop[xi, yi, t].astype(np.int8)                     # (X, Y, k)
    lab[~valid] = 0
    return lab
