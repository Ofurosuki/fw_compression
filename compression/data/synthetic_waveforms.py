"""Synthetic full-waveform LiDAR generator.

Each waveform is a 1D temporal histogram of photon counts of length ``T`` (default
700, matching the Ghost-FWL voxel grid depth ``(400, 512, 700)``). Waveforms contain:

- one primary return (Gaussian, blurred by an IRF-like kernel),
- optional ghost / secondary / multipath returns (the *transport* information we
  care about preserving through compression),
- a background floor + shot (Poisson) noise.

The generator also returns structured peak labels ``(position, intensity, width)``
and a per-waveform ``ghost`` flag, mirroring the real Ghost-FWL ``*_peak.npy``
format ``[peak_position, peak_intensity, peak_width]``.

This is a drop-in stand-in for the real dataset. To swap in real data, implement a
``torch.utils.data.Dataset`` that yields the same ``(waveform, label_dict)`` items
(see ``WaveformDataset`` below) and feed it to the same training / eval code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WaveformConfig:
    T: int = 700
    # primary peak
    primary_amp_range: tuple = (0.6, 1.0)
    primary_width_range: tuple = (3.0, 9.0)  # gaussian sigma in bins (IRF-like)
    margin: int = 20  # keep peaks away from the very edges
    # ghost / multipath returns
    p_ghost: float = 0.55  # probability a waveform carries ghost return(s)
    n_ghost_range: tuple = (1, 2)
    ghost_amp_frac_range: tuple = (0.12, 0.45)  # relative to primary amplitude
    ghost_width_range: tuple = (3.0, 12.0)
    min_peak_sep: int = 25  # minimum separation between peaks (bins)
    # noise / background
    background_range: tuple = (0.0, 0.03)
    poisson_scale: float = 40.0  # photon-count scaling for shot noise
    read_noise_std: float = 0.01


def _gaussian(T: int, center: float, sigma: float, amp: float) -> np.ndarray:
    t = np.arange(T, dtype=np.float64)
    return amp * np.exp(-0.5 * ((t - center) / max(sigma, 1e-3)) ** 2)


def _sample_position(rng, T, margin, existing, min_sep, max_tries=20):
    for _ in range(max_tries):
        pos = rng.uniform(margin, T - margin)
        if all(abs(pos - e) >= min_sep for e in existing):
            return pos
    return None


def generate_waveform(rng: np.random.Generator, cfg: WaveformConfig):
    """Generate a single waveform and its peak labels.

    Returns
    -------
    wave : np.ndarray, shape [T], float32, non-negative.
    peaks : list of (position, intensity, width)  -- ground-truth returns.
    is_ghost : bool                               -- waveform carries a ghost return.
    """
    T = cfg.T
    clean = np.zeros(T, dtype=np.float64)
    peaks: List[tuple] = []

    # --- primary return ---
    p_amp = rng.uniform(*cfg.primary_amp_range)
    p_sig = rng.uniform(*cfg.primary_width_range)
    p_pos = rng.uniform(cfg.margin, T - cfg.margin)
    clean += _gaussian(T, p_pos, p_sig, p_amp)
    peaks.append((p_pos, p_amp, p_sig))
    positions = [p_pos]

    # --- ghost / multipath returns ---
    is_ghost = rng.random() < cfg.p_ghost
    if is_ghost:
        n_ghost = rng.integers(cfg.n_ghost_range[0], cfg.n_ghost_range[1] + 1)
        for _ in range(int(n_ghost)):
            pos = _sample_position(rng, T, cfg.margin, positions, cfg.min_peak_sep)
            if pos is None:
                continue
            g_amp = p_amp * rng.uniform(*cfg.ghost_amp_frac_range)
            g_sig = rng.uniform(*cfg.ghost_width_range)
            clean += _gaussian(T, pos, g_sig, g_amp)
            peaks.append((pos, g_amp, g_sig))
            positions.append(pos)

    # --- background + noise ---
    background = rng.uniform(*cfg.background_range)
    wave = clean + background
    # shot noise: scale to photon counts, sample Poisson, scale back
    lam = np.clip(wave * cfg.poisson_scale, 0, None)
    wave = rng.poisson(lam).astype(np.float64) / cfg.poisson_scale
    wave += rng.normal(0.0, cfg.read_noise_std, size=T)
    wave = np.clip(wave, 0.0, None)

    # sort peaks by position for stable labels
    peaks.sort(key=lambda p: p[0])
    return wave.astype(np.float32), peaks, bool(is_ghost)


def generate_dataset(n: int, cfg: WaveformConfig, seed: int = 0):
    """Generate ``n`` waveforms.

    Returns
    -------
    waves : np.ndarray [n, T] float32
    labels : list of dicts with keys: peaks (list), n_peaks (int), ghost (bool),
             peak_positions (np.ndarray), peak_intensities, peak_widths.
    """
    rng = np.random.default_rng(seed)
    waves = np.zeros((n, cfg.T), dtype=np.float32)
    labels: List[Dict] = []
    for i in range(n):
        w, peaks, ghost = generate_waveform(rng, cfg)
        waves[i] = w
        pos = np.array([p[0] for p in peaks], dtype=np.float32)
        inten = np.array([p[1] for p in peaks], dtype=np.float32)
        wid = np.array([p[2] for p in peaks], dtype=np.float32)
        labels.append(
            dict(
                peaks=peaks,
                n_peaks=len(peaks),
                ghost=ghost,
                peak_positions=pos,
                peak_intensities=inten,
                peak_widths=wid,
            )
        )
    return waves, labels


class WaveformDataset(Dataset):
    """Torch dataset over pre-generated synthetic waveforms.

    Replace this class (keeping the same ``__getitem__`` contract) to plug in real
    Ghost-FWL waveforms: ``__getitem__`` must return ``(wave_tensor[T], label_dict)``.
    """

    def __init__(self, waves: np.ndarray, labels: List[Dict]):
        assert len(waves) == len(labels)
        self.waves = torch.from_numpy(waves).float()
        self.labels = labels

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx], self.labels[idx]


def collate_waveforms(batch):
    """Collate that keeps label dicts as a python list (variable-length peaks)."""
    waves = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return waves, labels


def make_datasets(
    n_train: int = 20000,
    n_val: int = 2000,
    cfg: Optional[WaveformConfig] = None,
    seed: int = 42,
):
    cfg = cfg or WaveformConfig()
    tr_w, tr_l = generate_dataset(n_train, cfg, seed=seed)
    va_w, va_l = generate_dataset(n_val, cfg, seed=seed + 1)
    return (
        WaveformDataset(tr_w, tr_l),
        WaveformDataset(va_w, va_l),
        cfg,
    )
