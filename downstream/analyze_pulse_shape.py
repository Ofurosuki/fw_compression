"""Empirical pulse-shape analysis: is the real FW-LiDAR return pulse asymmetric?

Collects clean, isolated single-return waveforms from the split2 test set, aligns
them on their peak (sub-bin, by centroid), max-normalises, averages, and fits a
symmetric Gaussian vs. an asymmetric EMG (exponentially-modified Gaussian). Reports
residual RMSE and the empirical skewness of the mean pulse. This tells us whether
investing in a non-symmetric (Poisson/EMG/gamma) synthesis kernel is worth it.

Run:
  export PATH="$HOME/.local/bin:$PATH"
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  uv run python downstream/analyze_pulse_shape.py
"""
from __future__ import annotations

import glob
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.special import erfc

from hist_lidar.preprocess.custom_blosc2 import load_blosc2  # type: ignore


# ---- waveform sources (a few frames from each test scene) ----------------- #
SCENES = [
    "/data3/user/ikeda/ghost_dataset/36build/data/hist002",
    "/data3/user/ikeda/ghost_dataset/22build/data/hist001",
    "/data3/user/ikeda/ghost_dataset/14build_7floor/data/hist001",
]
FRAMES_PER_SCENE = 2          # keep it quick
WIN = 25                      # +/- bins around the peak for the local pulse window
MIN_HEIGHT = 0.15             # on max-normalised waveform
MAX_PULSES = 40000            # cap collected pulses


def collect_pulses():
    pulses = []
    for sc in SCENES:
        files = sorted(glob.glob(sc + "/*_voxel.b2"))[:FRAMES_PER_SCENE]
        for f in files:
            vox = load_blosc2(f).astype(np.float32)   # (X, Y, T)
            X, Y, T = vox.shape
            flat = vox.reshape(-1, T)
            # keep only pixels with real signal
            mx = flat.max(axis=1)
            cand = np.where(mx > 5.0)[0]
            # subsample for speed
            if len(cand) > 20000:
                cand = cand[np.linspace(0, len(cand) - 1, 20000).astype(int)]
            for idx in cand:
                w = flat[idx]
                m = w.max()
                if m <= 0:
                    continue
                wn = w / m
                ws = gaussian_filter1d(wn, 1.0)
                peaks, props = find_peaks(ws, prominence=0.05, distance=4)
                if len(peaks) != 1:          # require a single, isolated return
                    continue
                p = peaks[0]
                if wn[p] < MIN_HEIGHT or p < WIN or p >= T - WIN:
                    continue
                seg = wn[p - WIN:p + WIN + 1].astype(np.float64)
                # subtract a small local baseline (min of the window edges)
                base = min(seg[0], seg[-1], 0.05)
                seg = np.clip(seg - base, 0, None)
                if seg.max() <= 0:
                    continue
                seg = seg / seg.max()
                pulses.append(seg)
                if len(pulses) >= MAX_PULSES:
                    return np.array(pulses)
    return np.array(pulses)


def gaussian(t, a, mu, sigma):
    return a * np.exp(-((t - mu) ** 2) / (2.0 * sigma ** 2))


def emg(t, a, mu, sigma, lam):
    """Exponentially-modified Gaussian (Gaussian convolved with right exp decay).
    lam = 1/tau decay rate; larger tau (smaller lam) => longer right tail."""
    arg = (sigma * sigma * lam - (t - mu))
    z = arg / (math.sqrt(2.0) * sigma)
    return a * (lam / 2.0) * np.exp((lam / 2.0) * (2.0 * mu + lam * sigma * sigma - 2.0 * t)) \
        * erfc(z)


def main():
    print("[collect] scanning waveforms ...")
    P = collect_pulses()
    print(f"[collect] {len(P)} clean single-return pulses")
    mean = P.mean(axis=0)
    mean = mean / mean.max()
    t = np.arange(len(mean)) - WIN  # peak roughly at 0

    # empirical skewness of the mean pulse treated as a distribution over t
    w = mean / mean.sum()
    mu_e = (t * w).sum()
    var_e = ((t - mu_e) ** 2 * w).sum()
    sd_e = math.sqrt(var_e)
    skew = ((t - mu_e) ** 3 * w).sum() / sd_e ** 3
    print(f"[shape] empirical mean-pulse skewness = {skew:+.3f}  (0 = symmetric)")
    print(f"[shape] centroid offset from peak = {mu_e:+.3f} bins, std = {sd_e:.2f} bins")

    # fit Gaussian
    pg, _ = curve_fit(gaussian, t, mean, p0=[1.0, 0.0, 4.0], maxfev=20000)
    rg = mean - gaussian(t, *pg)
    rmse_g = math.sqrt((rg ** 2).mean())

    # fit EMG
    try:
        pe, _ = curve_fit(emg, t, mean, p0=[1.0, -2.0, 3.0, 0.3], maxfev=40000)
        re = mean - emg(t, *pe)
        rmse_e = math.sqrt((re ** 2).mean())
        tau = 1.0 / pe[3]
    except Exception as ex:
        rmse_e, tau, pe = float("nan"), float("nan"), None
        print(f"[emg] fit failed: {ex}")

    print(f"\n[fit] Gaussian  RMSE = {rmse_g:.5f}  (sigma={pg[2]:.2f})")
    print(f"[fit] EMG       RMSE = {rmse_e:.5f}  (tau={tau:.2f} bins)")
    if rmse_e == rmse_e:  # not nan
        impr = 100.0 * (rmse_g - rmse_e) / rmse_g
        print(f"[fit] EMG reduces residual RMSE by {impr:.1f}% vs Gaussian")

    # dump the mean pulse + fits as a small text table for eyeballing the tail
    print("\n t    mean    gauss    emg")
    for i in range(0, len(mean), 2):
        ev = emg(t[i], *pe) if pe is not None else float("nan")
        print(f"{t[i]:+4d}  {mean[i]:.4f}  {gaussian(t[i], *pg):.4f}  {ev:.4f}")

    np.savez("/home/yoshida/fw_compression/downstream/outputs/pulse_shape.npz",
             mean=mean, t=t, gauss_p=pg, emg_p=(pe if pe is not None else []),
             skew=skew, rmse_g=rmse_g, rmse_e=rmse_e, n=len(P))
    print("\n[saved] downstream/outputs/pulse_shape.npz")


if __name__ == "__main__":
    main()
