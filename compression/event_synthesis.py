"""Synthesise a pseudo-waveform from top-K transport events.

Inverse of :mod:`compression.event_extraction`: given events ``[(t_i, a_i, w_i)]``
build a dense Gaussian-pulse waveform

    x_hat[t] = bg + Σ_i a_i exp(-(t - t_i)^2 / (2 σ_i^2)),   σ_i = w_i / (2√(2 ln2))

The ``representation`` argument controls which parameters are honoured vs.
replaced by fixed values — this is the core ablation of the experiment:

* ``"t"``     position only      (a = fixed_amplitude, w = fixed_width)
* ``"ta"``    position+intensity (w = fixed_width)
* ``"tw"``    position+width     (a = fixed_amplitude)
* ``"taw"``   all three
* ``"taw_bg"`` all three + a background floor
"""
from __future__ import annotations

import math

import numpy as np
import torch

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _resolve(representation, fixed_amplitude, fixed_width):
    """Return (use_a, use_w, add_bg) flags for a representation string."""
    use_a = representation in ("ta", "taw", "taw_bg")
    use_w = representation in ("tw", "taw", "taw_bg")
    add_bg = representation == "taw_bg"
    return use_a, use_w, add_bg


def synthesize_waveform_from_events(
    events,
    valid_mask=None,
    T: int = 700,
    representation: str = "taw",
    fixed_amplitude: float = 1.0,
    fixed_width: float = 4.0,
    background: float = 0.0,
    normalize: bool = True,
):
    """Convert top-K event parameters into a pseudo-waveform (single, numpy).

    Args:
        events: array ``[K, 3]``, columns ``[t, a, w]``.
        valid_mask: optional boolean array ``[K]``.
        T: waveform length.
        representation: ``"t" | "ta" | "tw" | "taw" | "taw_bg"``.
        fixed_amplitude: amplitude used when intensity is not in the representation.
        fixed_width: FWHM used when width is not in the representation.
        background: background floor (only added for ``taw_bg``).
        normalize: whether to max-normalise the synthesised waveform.

    Returns:
        wave_hat: array ``[T]``.
    """
    events = np.asarray(events, dtype=np.float64)
    K = events.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(K, dtype=bool)
    use_a, use_w, add_bg = _resolve(representation, fixed_amplitude, fixed_width)

    t_grid = np.arange(T, dtype=np.float64)
    wave = np.zeros(T, dtype=np.float64)
    if add_bg:
        wave += background
    for i in range(K):
        if not valid_mask[i]:
            continue
        t_i = events[i, 0]
        a_i = events[i, 1] if use_a else fixed_amplitude
        w_i = events[i, 2] if use_w else fixed_width
        w_i = min(max(w_i, 1.0), 80.0)
        sigma = w_i * _FWHM_TO_SIGMA
        wave += a_i * np.exp(-((t_grid - t_i) ** 2) / (2.0 * sigma * sigma))

    if normalize:
        mx = wave.max()
        if mx > 1e-12:
            wave = wave / mx
    return wave.astype(np.float32)


@torch.no_grad()
def synthesize_batch(
    events,
    valid,
    T: int = 700,
    representation: str = "taw",
    fixed_amplitude: float = 1.0,
    fixed_width: float = 4.0,
    background: float = 0.0,
    normalize: bool = True,
):
    """Vectorised synthesis for a batch of event sets.

    Args:
        events: ``(N, K, 3)`` torch tensor, columns ``[t, a, w]``.
        valid: ``(N, K)`` bool tensor.
        (other args as in :func:`synthesize_waveform_from_events`).

    Returns:
        ``(N, T)`` torch tensor.
    """
    N, K, _ = events.shape
    dev = events.device
    use_a, use_w, add_bg = _resolve(representation, fixed_amplitude, fixed_width)

    t_i = events[..., 0]                                   # (N,K)
    a_i = events[..., 1] if use_a else torch.full_like(t_i, fixed_amplitude)
    w_i = events[..., 2] if use_w else torch.full_like(t_i, fixed_width)
    w_i = w_i.clamp(1.0, 80.0)
    sigma = w_i * _FWHM_TO_SIGMA
    a_i = torch.where(valid, a_i, torch.zeros_like(a_i))   # drop padded events

    t_grid = torch.arange(T, device=dev, dtype=events.dtype)            # (T,)
    diff = t_grid[None, None, :] - t_i[:, :, None]         # (N,K,T)
    g = a_i[:, :, None] * torch.exp(-(diff ** 2) / (2.0 * (sigma ** 2)[:, :, None]))
    wave = g.sum(dim=1)                                    # (N,T)
    if add_bg:
        wave = wave + background

    if normalize:
        mx = wave.amax(dim=1, keepdim=True)
        wave = torch.where(mx > 1e-12, wave / mx, wave)
    return wave
