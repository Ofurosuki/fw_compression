"""Top-K transport-event extraction from full-waveform LiDAR signals.

The central hypothesis of the event-aware experiment (see
``event_aware_experiment_plan.md``) is that the downstream Ghost-FWL model may
not need the dense waveform ``x[t]`` but only a sparse list of transport events
``{(t_i, a_i, w_i)}``:

* ``t``: peak position / depth bin
* ``a``: peak intensity (height / area / prominence)
* ``w``: FWHM / temporal spread

This module provides two extractors:

* :func:`extract_topk_events` — a faithful, single-waveform reference built on
  ``scipy.signal.find_peaks`` / ``peak_widths`` exactly as the plan specifies.
  Used for tests and as the ground-truth reference. ~70 us/waveform.
* :func:`extract_topk_events_batch` — a GPU-vectorised approximation that
  processes a whole voxel (~200 k waveforms) at once. This is what the
  downstream eval hook uses, because the scipy loop would cost ~2 h/config on
  the split2 test set. It ranks peaks by smoothed height with min-distance NMS
  (waveforms are max-normalised, so height ranking is a close proxy for the
  scipy prominence ranking for the few tallest, well-separated peaks we keep).
"""
from __future__ import annotations

import math

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reference: single-waveform scipy extractor (faithful to the plan spec).
# --------------------------------------------------------------------------- #
def extract_topk_events(
    wave,
    k: int,
    smooth_sigma: float = 1.5,
    min_prominence: float = 0.03,
    min_distance: int = 3,
    rank_by: str = "prominence",
    intensity_mode: str = "area",
):
    """Extract top-K transport events from a 1D waveform (scipy reference).

    Args:
        wave: 1D numpy array or torch tensor of shape ``[T]``. Expected to be
            max-normalised before extraction.
        k: number of events to keep.
        smooth_sigma: Gaussian smoothing sigma before peak detection.
        min_prominence: minimum peak prominence for ``scipy.signal.find_peaks``.
        min_distance: minimum distance (bins) between detected peaks.
        rank_by: ranking criterion — ``"prominence"`` | ``"height"`` | ``"area"``.
        intensity_mode: how to measure ``a_i`` — ``"height"`` | ``"area"`` |
            ``"prominence"``.

    Returns:
        events: numpy array ``[K, 3]`` with columns ``[t, a, w]``; missing
            events padded with zeros and sorted by time.
        valid_mask: boolean array ``[K]``, True for valid events.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths

    if isinstance(wave, torch.Tensor):
        wave = wave.detach().cpu().numpy()
    wave = np.asarray(wave, dtype=np.float64)
    T = wave.shape[0]

    ws = gaussian_filter1d(wave, smooth_sigma) if smooth_sigma > 0 else wave
    peaks, props = find_peaks(ws, prominence=min_prominence, distance=min_distance)

    events = np.zeros((k, 3), dtype=np.float32)
    valid = np.zeros(k, dtype=bool)
    if len(peaks) == 0:
        return events, valid

    prom = props["prominences"]
    height = ws[peaks]
    widths = peak_widths(ws, peaks, rel_height=0.5)[0]  # FWHM in bins

    # local area: sum of the smoothed signal over each peak's FWHM support
    fw = np.clip(widths, 1.0, 80.0)
    area = np.empty(len(peaks))
    for i, p in enumerate(peaks):
        half = int(round(fw[i] / 2))
        lo, hi = max(0, p - half), min(T, p + half + 1)
        area[i] = ws[lo:hi].sum()

    rank_key = {"prominence": prom, "height": height, "area": area}[rank_by]
    order = np.argsort(rank_key)[::-1][:k]            # top-K by rank criterion
    sel = peaks[order]
    a_vals = {"height": height, "area": area, "prominence": prom}[intensity_mode][order]

    keep = np.argsort(sel)                             # sort selected by time
    sel, a_vals, w_vals = sel[keep], a_vals[keep], fw[order][keep]
    n = len(sel)
    events[:n, 0] = sel
    events[:n, 1] = a_vals
    events[:n, 2] = w_vals
    valid[:n] = True
    return events, valid


# --------------------------------------------------------------------------- #
# Fast path: GPU-vectorised batch extractor used by the downstream hook.
# --------------------------------------------------------------------------- #
def _gaussian_kernel(sigma: float, device, dtype):
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).view(1, 1, -1)


def _smooth(wn, sigma):
    """wn: (N, T) -> reflect-padded Gaussian smoothing along T."""
    if sigma <= 0:
        return wn
    kern = _gaussian_kernel(sigma, wn.device, wn.dtype)
    pad = kern.shape[-1] // 2
    x = torch.nn.functional.pad(wn[:, None, :], (pad, pad), mode="reflect")
    return torch.nn.functional.conv1d(x, kern)[:, 0, :]


@torch.no_grad()
def extract_topk_events_batch(
    wn,
    k: int,
    smooth_sigma: float = 1.5,
    min_height: float = 0.03,
    min_distance: int = 3,
    intensity_mode: str = "height",
    max_halfwidth: int = 40,
):
    """Vectorised top-K event extraction for a batch of max-normalised waveforms.

    Ranks local maxima of the smoothed signal by height, greedily keeping the
    top-K with a min-distance non-maximum-suppression window. FWHM is estimated
    from half-max crossings of the smoothed signal.

    Args:
        wn: ``(N, T)`` torch tensor, max-normalised (peak == 1) on the target
            device.
        k: number of events to keep.
        smooth_sigma, min_height, min_distance: peak-detection params.
            ``min_height`` is the minimum smoothed peak height (relative, since
            waveforms are max-normalised) — the batch analogue of
            ``min_prominence``.
        intensity_mode: ``"height"`` (raw value at the peak) or ``"area"`` (sum
            over the FWHM support).
        max_halfwidth: cap (bins) on the half-max crossing search.

    Returns:
        events: ``(N, K, 3)`` float tensor, columns ``[t, a, w]``, sorted by t.
        valid: ``(N, K)`` bool tensor.
    """
    N, T = wn.shape
    dev = wn.device
    ws = _smooth(wn.float(), smooth_sigma)

    # interior local maxima (strict left, ge right) above min_height
    is_max = torch.zeros_like(ws, dtype=torch.bool)
    is_max[:, 1:-1] = (ws[:, 1:-1] > ws[:, :-2]) & (ws[:, 1:-1] >= ws[:, 2:])
    is_max &= ws >= min_height

    cand = torch.where(is_max, ws, torch.full_like(ws, -1.0))
    rows = torch.arange(N, device=dev)
    offsets = torch.arange(-min_distance, min_distance + 1, device=dev)

    t_sel = torch.zeros(N, k, dtype=torch.long, device=dev)
    valid = torch.zeros(N, k, dtype=torch.bool, device=dev)
    for j in range(k):
        val, idx = cand.max(dim=1)                     # (N,)
        valid[:, j] = val > 0.0                        # a real local max remains
        t_sel[:, j] = idx
        cols = (idx[:, None] + offsets[None, :]).clamp_(0, T - 1)   # NMS window
        cand[rows[:, None], cols] = -1.0

    # gather peak heights, FWHM via half-max crossing on the smoothed signal
    win = torch.arange(-max_halfwidth, max_halfwidth + 1, device=dev)   # (2W+1,)
    W = max_halfwidth
    a_out = torch.zeros(N, k, device=dev)
    w_out = torch.zeros(N, k, device=dev)
    for j in range(k):
        t = t_sel[:, j]                                # (N,)
        cols = (t[:, None] + win[None, :]).clamp_(0, T - 1)            # (N,2W+1)
        seg = ws[rows[:, None], cols]                  # (N,2W+1) smoothed window
        peak_h = ws[rows, t]                           # (N,) smoothed height
        half = (peak_h * 0.5)[:, None]
        geq = seg >= half
        right_run = torch.cumprod(geq[:, W + 1:].long(), dim=1).sum(dim=1)
        left_run = torch.cumprod(geq[:, :W].flip(1).long(), dim=1).sum(dim=1)
        fwhm = (left_run + right_run + 1).clamp(1, 80).float()
        if intensity_mode == "area":
            half_w = (fwhm / 2).long()
            mask = win.abs()[None, :] <= half_w[:, None]
            raw_seg = wn[rows[:, None], cols]
            a_out[:, j] = (raw_seg * mask).sum(dim=1)
        else:                                          # "height"
            a_out[:, j] = wn[rows, t]
        w_out[:, j] = fwhm

    t_f = t_sel.float()
    t_f[~valid] = 0.0
    a_out[~valid] = 0.0
    w_out[~valid] = 0.0
    events = torch.stack([t_f, a_out, w_out], dim=-1)  # (N, K, 3)

    # sort each row's events by time (invalid ones have t=0; push them last)
    sort_key = torch.where(valid, t_f, torch.full_like(t_f, float(T) + 1))
    order = sort_key.argsort(dim=1)
    events = torch.gather(events, 1, order[:, :, None].expand(-1, -1, 3))
    valid = torch.gather(valid, 1, order)
    return events, valid
