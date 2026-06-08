"""Physically-motivated full-waveform LiDAR generator (parametric IRF).

This is a more physical replacement for ``synthetic_waveforms.py``. Key physics:

- **Shared system IRF**: an exponentially-modified Gaussian (EMG) -- a Gaussian
  (jitter / laser pulse width) convolved with a one-sided exponential (detector
  tail / SPAD response). All returns share the same *tail* ``tau`` (a system
  property); only the Gaussian width ``sigma`` varies per return (surface tilt,
  roughness, mixed pixels broaden it).
- **Each return** = ``area_k * EMG(t - tof_k; sigma_k, tau)`` where ``EMG`` is
  unit-area, so ``area_k`` is the total returned photons (the physical intensity).
- **Range falloff**: intensity ~ albedo / r^2, with range r mapped from ToF.
- **Multipath / ghost returns**: a secondary, *delayed & attenuated copy* of the
  primary (extra optical path) and/or independent secondary surfaces.
- **Noise**: Poisson (shot) noise on photon counts + ambient/dark-count floor +
  small read noise.

Per-peak ground-truth labels carry ``(position, intensity=area, width=FWHM,
sigma, kind)`` so downstream evaluation can measure how well each pulse's
position / intensity / width survive compression -- split by narrow (high-freq)
vs wide (low-freq) peaks.

The label-dict / dataset contract matches ``synthetic_waveforms.py`` (extra keys
added), so train/eval code is reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import exponnorm
from torch.utils.data import Dataset

from compression.data.synthetic_waveforms import WaveformDataset, collate_waveforms  # re-export


# --------------------------------------------------------------------------- #
# EMG (exponentially-modified Gaussian) unit-area pulse, shared system IRF.
# scipy.stats.exponnorm: pdf(t, K, loc, scale) with K = tau/sigma, loc=mu,
# scale=sigma. Unit area. Numerically robust.
# --------------------------------------------------------------------------- #
def emg_unit_area(t: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    return exponnorm.pdf(t, K=tau / sigma, loc=mu, scale=sigma)


def emg_peak_height(mu: float, sigma: float, tau: float, T: int = 700) -> float:
    """Max of the unit-area EMG (for converting a target peak height to an area)."""
    lo, hi = int(max(0, mu - 6 * sigma - 3 * tau)), int(min(T, mu + 6 * sigma + 6 * tau))
    grid = np.arange(lo, hi + 1, dtype=float)
    return float(emg_unit_area(grid, mu, sigma, tau).max())


def emg_mode(mu: float, sigma: float, tau: float, T: int = 700) -> float:
    grid = np.arange(0, T, dtype=float)
    return float(grid[int(np.argmax(emg_unit_area(grid, mu, sigma, tau)))])


def emg_fwhm(sigma: float, tau: float) -> float:
    """Numeric FWHM of a unit-area EMG with the given (sigma, tau), in bins."""
    span = 8 * sigma + 8 * tau
    grid = np.linspace(-span, span, 4000)
    y = emg_unit_area(grid, 0.0, sigma, tau)
    ymax = y.max()
    above = grid[y >= 0.5 * ymax]
    if above.size < 2:
        return float(2.355 * sigma)
    return float(above[-1] - above[0])


@dataclass
class PhysicalWaveformConfig:
    T: int = 700
    # system IRF
    tau: float = 4.0  # fixed exponential tail (system response), bins
    sigma_irf: float = 2.5  # minimum Gaussian width (jitter/pulse), bins
    # per-return broadening (surface geometry): sigma = sigma_irf * b
    broaden_range: tuple = (1.0, 4.0)  # b; spans narrow (high-freq) -> wide (low-freq)
    # geometry / intensity
    margin: int = 25
    range_min: float = 4.0
    range_max: float = 12.0  # mapped from ToF; controls (gentle) 1/r^2 falloff
    primary_albedo_range: tuple = (0.6, 1.0)
    target_height_range: tuple = (0.6, 1.0)  # peak height before noise (post-falloff ref)
    primary_height_floor: float = 0.35
    # secondary returns
    p_secondary: float = 0.6  # probability of >=1 secondary return
    n_secondary_range: tuple = (1, 2)
    min_peak_sep: int = 45
    # multipath ghost (delayed attenuated copy of primary)
    p_multipath: float = 0.5  # given a secondary exists, prob it is a multipath ghost
    ghost_atten_range: tuple = (0.12, 0.45)  # area fraction of primary
    ghost_delay_range: tuple = (45, 180)  # extra path delay, bins
    ghost_extra_broaden: tuple = (1.0, 1.5)  # ghosts often slightly broader
    # independent secondary surface
    secondary_height_frac: tuple = (0.2, 0.6)  # rel. to primary height
    # noise
    ambient_range: tuple = (0.002, 0.015)  # background count rate (floor)
    poisson_scale: float = 120.0  # photon-count scale for shot noise
    read_noise_std: float = 0.006


def _range_from_tof(tof: float, cfg: PhysicalWaveformConfig) -> float:
    frac = tof / cfg.T
    return cfg.range_min + frac * (cfg.range_max - cfg.range_min)


def _add_return(rate, cfg, mu, sigma, area):
    t = np.arange(cfg.T, dtype=float)
    rate += area * emg_unit_area(t, mu, sigma, cfg.tau)


def _sample_position(rng, cfg, existing, max_tries=30):
    for _ in range(max_tries):
        pos = rng.uniform(cfg.margin, cfg.T - cfg.margin)
        if all(abs(pos - e) >= cfg.min_peak_sep for e in existing):
            return pos
    return None


def generate_waveform(rng: np.random.Generator, cfg: PhysicalWaveformConfig):
    """Generate one physical waveform + per-peak labels.

    Returns (wave[T] float32, peaks: list of dict, is_ghost: bool).
    Each peak dict: position(mode), mu, sigma, fwhm, intensity(area), height, kind.
    """
    T = cfg.T
    rate = np.zeros(T, dtype=float)
    peaks: List[Dict] = []
    positions = []

    def make_peak(mu, sigma, area, kind):
        height = area * emg_peak_height(mu, sigma, cfg.tau, T)
        peaks.append(
            dict(
                position=emg_mode(mu, sigma, cfg.tau, T),
                mu=float(mu),
                sigma=float(sigma),
                fwhm=emg_fwhm(sigma, cfg.tau),
                intensity=float(area),
                height=float(height),
                kind=kind,
            )
        )
        positions.append(mu)

    # --- primary return ---
    p_mu = rng.uniform(cfg.margin, T - cfg.margin)
    p_b = rng.uniform(*cfg.broaden_range)
    p_sigma = cfg.sigma_irf * p_b
    p_range = _range_from_tof(p_mu, cfg)
    albedo = rng.uniform(*cfg.primary_albedo_range)
    target_h = rng.uniform(*cfg.target_height_range)
    falloff = (cfg.range_min / p_range) ** 2  # in [~0.11, 1]
    p_height = max(target_h * albedo * (0.4 + 0.6 * falloff), cfg.primary_height_floor)
    p_area = p_height / emg_peak_height(p_mu, p_sigma, cfg.tau, T)
    _add_return(rate, cfg, p_mu, p_sigma, p_area)
    make_peak(p_mu, p_sigma, p_area, "primary")

    # --- secondary / ghost returns ---
    is_ghost = rng.random() < cfg.p_secondary
    if is_ghost:
        n_sec = int(rng.integers(cfg.n_secondary_range[0], cfg.n_secondary_range[1] + 1))
        for _ in range(n_sec):
            if rng.random() < cfg.p_multipath:
                # multipath: delayed, attenuated copy of primary
                delay = rng.uniform(*cfg.ghost_delay_range)
                mu = p_mu + delay
                if mu >= T - cfg.margin or any(abs(mu - e) < cfg.min_peak_sep for e in positions):
                    mu_alt = _sample_position(rng, cfg, positions)
                    if mu_alt is None:
                        continue
                    mu = mu_alt
                sigma = p_sigma * rng.uniform(*cfg.ghost_extra_broaden)
                area = p_area * rng.uniform(*cfg.ghost_atten_range)
                _add_return(rate, cfg, mu, sigma, area)
                make_peak(mu, sigma, area, "multipath")
            else:
                # independent secondary surface
                mu = _sample_position(rng, cfg, positions)
                if mu is None:
                    continue
                b = rng.uniform(*cfg.broaden_range)
                sigma = cfg.sigma_irf * b
                rng_r = _range_from_tof(mu, cfg)
                h = p_height * rng.uniform(*cfg.secondary_height_frac) * (cfg.range_min / rng_r) ** 2
                h = max(h, 0.03)
                area = h / emg_peak_height(mu, sigma, cfg.tau, T)
                _add_return(rate, cfg, mu, sigma, area)
                make_peak(mu, sigma, area, "secondary")

    # --- ambient floor + shot/read noise ---
    ambient = rng.uniform(*cfg.ambient_range)
    rate = rate + ambient
    lam = np.clip(rate * cfg.poisson_scale, 0, None)
    wave = rng.poisson(lam).astype(np.float64) / cfg.poisson_scale
    wave += rng.normal(0.0, cfg.read_noise_std, size=T)
    wave = np.clip(wave, 0.0, None)

    peaks.sort(key=lambda p: p["position"])
    return wave.astype(np.float32), peaks, bool(is_ghost)


def generate_dataset(n: int, cfg: PhysicalWaveformConfig, seed: int = 0):
    """Generate ``n`` physical waveforms with rich per-peak labels."""
    rng = np.random.default_rng(seed)
    waves = np.zeros((n, cfg.T), dtype=np.float32)
    labels: List[Dict] = []
    for i in range(n):
        w, peaks, ghost = generate_waveform(rng, cfg)
        waves[i] = w
        labels.append(
            dict(
                peaks=peaks,
                n_peaks=len(peaks),
                ghost=ghost,
                peak_positions=np.array([p["position"] for p in peaks], dtype=np.float32),
                peak_mu=np.array([p["mu"] for p in peaks], dtype=np.float32),
                peak_sigma=np.array([p["sigma"] for p in peaks], dtype=np.float32),
                peak_fwhm=np.array([p["fwhm"] for p in peaks], dtype=np.float32),
                peak_intensities=np.array([p["intensity"] for p in peaks], dtype=np.float32),
                peak_heights=np.array([p["height"] for p in peaks], dtype=np.float32),
                peak_kinds=[p["kind"] for p in peaks],
            )
        )
    return waves, labels


def make_datasets(
    n_train: int = 20000,
    n_val: int = 2000,
    cfg: Optional[PhysicalWaveformConfig] = None,
    seed: int = 42,
):
    cfg = cfg or PhysicalWaveformConfig()
    tr_w, tr_l = generate_dataset(n_train, cfg, seed=seed)
    va_w, va_l = generate_dataset(n_val, cfg, seed=seed + 1)
    return WaveformDataset(tr_w, tr_l), WaveformDataset(va_w, va_l), cfg
