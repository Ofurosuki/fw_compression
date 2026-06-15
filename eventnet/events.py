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
        events: (X, Y, k, 12) float32 — columns
            ``[t_bin, a_height, w_fwhm_bin, E_behind, D_after, I_after, I_local,
               Tp0, Tp8, Tp16, Tp32, Tp64]`` where ``Tp{δ}`` is the peak-anchored
            transmittance survival at peak+δ (the NeRF-style decay profile).
        valid:  (X, Y, k) bool.

    Beyond (t, a, w) we attach a **direct/indirect waveform decomposition** (the
    lit-review insight, FW_Event_Net/RESULTS.md "behind-energy"): reconstruct the
    DIRECT model as a sum of Gaussians at the detected peaks
    ``dir(r)=Σ a_k·exp(-(r-t_k)²/2σ_k²)`` (σ=FWHM/2.3548), and the INDIRECT/diffuse
    residual ``resid(r)=max(0, wn-dir)`` — the part the top-K event list throws
    away. Per event (fractions of total energy, ∈[0,1]):
      * ``E_behind`` = total energy at/after the peak (raw transmitted-energy;
        found depth-confounded & scene-non-transferable — kept for ablation).
      * ``D_after``  = DIRECT mass strictly after the peak → "is a real surface
        behind me" (glass cue; largely derivable from the events, control arm).
      * ``I_after``  = INDIRECT/diffuse mass after the peak → broad-transport-behind
        (ghost/multipath cue; the genuinely NEW info not in (t,a,w)).
      * ``I_local``  = diffuse pedestal within ±15 bins of the peak (is this return
        itself sitting on volume/indirect transport).
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
    vmask = vmask & valid_px[:, None]                    # kill background pixels

    N = wn.shape[0]
    rows = torch.arange(N, device=wn.device)
    t_idx = events[..., 0].long().clamp(0, T - 1)        # (N, k)
    cs = torch.cumsum(wn, dim=1)                          # (N, T)
    total = cs[:, -1:].clamp_min(eps)                     # (N, 1)
    e_behind = ((total - cs[rows[:, None], t_idx]) / total
                + wn[rows[:, None], t_idx] / total).clamp(0.0, 1.0)

    # direct/indirect decomposition: Gaussian-reconstruct the detected peaks
    rr = torch.arange(T, device=wn.device, dtype=wn.dtype)            # (T,)
    sig = (events[..., 2] / 2.3548).clamp_min(0.5)                    # (N, k) FWHM->sigma
    direct = torch.zeros_like(wn)                                     # (N, T)
    for j in range(k):
        vj = vmask[:, j]
        g = events[:, j, 1:2] * torch.exp(
            -((rr[None, :] - events[:, j, 0:1]) ** 2) / (2.0 * sig[:, j:j+1] ** 2))
        direct = direct + torch.where(vj[:, None], g, torch.zeros_like(g))
    resid = (wn - direct).clamp_min(0.0)                             # (N, T) indirect/diffuse
    cs_dir = torch.cumsum(direct, dim=1)
    cs_res = torch.cumsum(resid, dim=1)
    g = lambda cs_, idx: cs_[rows[:, None], idx]
    d_after = ((cs_dir[:, -1:] - g(cs_dir, t_idx)) / total).clamp(0.0, 1.0)   # (N,k)
    i_after = ((cs_res[:, -1:] - g(cs_res, t_idx)) / total).clamp(0.0, 1.0)
    Wl = 15
    lo = (t_idx - Wl).clamp(0, T - 1)
    hi = (t_idx + Wl).clamp(0, T - 1)
    i_local = ((g(cs_res, hi) - g(cs_res, lo)) / total).clamp(0.0, 1.0)

    # NeRF-style transmittance survival profile, PEAK-ANCHORED (depth-relative):
    # Tp(δ) = fraction of total ray energy strictly beyond (peak + δ) = transmittance
    # at peak+δ. Sampling the decay shape over δ captures the glass "partial-drop-then-
    # plateau" signature; anchoring at the peak makes it translation-invariant in depth
    # (avoids the absolute-position confound that sank raw behind_energy).
    TP_OFFS = (0, 8, 16, 32, 64)
    tp = []
    for off in TP_OFFS:
        idx = (t_idx + off).clamp(0, T - 1)
        tp.append(((total - g(cs, idx)) / total).clamp(0.0, 1.0))        # (N, k)
    extra = torch.stack([e_behind, d_after, i_after, i_local] + tp, dim=-1)  # (N, k, 4+5)
    events = torch.cat([events, extra], dim=-1)                          # (N, k, 12)
    events[~vmask] = 0.0
    events = events.reshape(X, Y, k, 12).cpu().numpy().astype(np.float32)
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
