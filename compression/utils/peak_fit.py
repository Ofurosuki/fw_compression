"""Per-peak parameter extraction by fitting the known system IRF (EMG).

Given a waveform and a set of candidate peak positions (from the calibrated
detector), we fit an exponentially-modified Gaussian (EMG) -- with the system
tail ``tau`` *fixed* -- in a local window around each candidate, recovering:

    position (mode, bins), intensity (area = total photons), width (FWHM, bins).

This mirrors how one would estimate pulse parameters from a real full-waveform:
the IRF shape is known, only (time-of-flight, amplitude, broadening) are free.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import exponnorm

from compression.data.physical_waveforms import emg_fwhm, emg_mode


def _emg_model_factory(tau: float):
    def model(t, area, mu, sigma):
        sigma = max(sigma, 1e-3)
        return area * exponnorm.pdf(t, K=tau / sigma, loc=mu, scale=sigma)

    return model


def fit_peak(
    wave: np.ndarray,
    pos: float,
    tau: float,
    half_window: int = 45,
    sigma_bounds=(1.0, 40.0),
) -> Dict:
    """Fit a single EMG (tau fixed) in a window around ``pos``.

    Returns dict with position(mode), mu, sigma, fwhm, intensity(area), ok(bool).
    """
    T = len(wave)
    lo, hi = max(0, int(pos) - half_window), min(T, int(pos) + half_window + 1)
    t = np.arange(lo, hi, dtype=float)
    y = wave[lo:hi].astype(float)
    model = _emg_model_factory(tau)

    peak_h = float(y.max())
    # initial guesses: area ~ height * effective width, mu slightly left of mode
    sigma0 = 4.0
    area0 = max(peak_h, 1e-3) * (sigma0 + tau) * 2.5
    p0 = [area0, float(pos) - tau, sigma0]
    bounds = ([0.0, lo - 10, sigma_bounds[0]], [np.inf, hi + 10, sigma_bounds[1]])
    try:
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=4000)
        area, mu, sigma = popt
        return dict(
            position=emg_mode(mu, sigma, tau, T),
            mu=float(mu),
            sigma=float(sigma),
            fwhm=emg_fwhm(sigma, tau),
            intensity=float(area),
            ok=True,
        )
    except Exception:
        # fall back to crude moment estimate
        area = float(y.sum())
        return dict(position=float(pos), mu=float(pos), sigma=float(sigma0),
                    fwhm=float(2.355 * sigma0), intensity=area, ok=False)


def fit_peaks(wave: np.ndarray, positions, tau: float, **kw) -> List[Dict]:
    return [fit_peak(wave, p, tau, **kw) for p in np.atleast_1d(positions)]
