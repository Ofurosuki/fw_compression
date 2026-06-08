"""Plotting helpers: original vs reconstructed waveforms, and sweep summaries."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compression.utils.metrics import detect_peaks


def plot_examples(
    x: np.ndarray,
    x_hat: np.ndarray,
    out_path: str,
    labels: Optional[List[Dict]] = None,
    n: int = 6,
    title: str = "",
):
    """Plot ``n`` original-vs-reconstructed waveform pairs in a grid."""
    n = min(n, x.shape[0])
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.4 * rows), squeeze=False)
    for i in range(n):
        ax = axes[i // cols][i % cols]
        ax.plot(x[i], color="black", lw=1.0, label="orig")
        ax.plot(x_hat[i], color="tab:red", lw=1.0, alpha=0.8, label="recon")
        if labels is not None:
            for p in np.asarray(labels[i]["peak_positions"], dtype=float):
                ax.axvline(p, color="tab:blue", ls=":", lw=0.8, alpha=0.6)
        pred_peaks = detect_peaks(x_hat[i])
        ax.scatter(pred_peaks, x_hat[i][pred_peaks], color="tab:red", s=14, zorder=5)
        tag = ""
        if labels is not None:
            tag = f"  (ghost={labels[i].get('ghost')}, npk={labels[i].get('n_peaks')})"
        ax.set_title(f"#{i}{tag}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_sweep(
    results: List[Dict],
    out_path: str,
    metrics=("waveform_mse_mean", "peak_loc_error_mean", "peak_count_acc", "ghost_recall"),
    upper_bound: Optional[Dict] = None,
):
    """Plot metric vs K, one line per encoder.

    ``results`` is a list of dicts each having keys: encoder, K, and the metrics.
    If ``upper_bound`` (the full-waveform / no-compression metrics dict) is given, a
    horizontal reference line is drawn per subplot at the upper-bound value — the
    achievable ceiling (recall-like metrics) or floor (relative-error / MSE metrics).
    """
    encoders = sorted({r["encoder"] for r in results})
    n_m = len(metrics)
    cols = 2
    rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    for mi, metric in enumerate(metrics):
        ax = axes[mi // cols][mi % cols]
        for enc in encoders:
            pts = sorted([(r["K"], r.get(metric)) for r in results if r["encoder"] == enc])
            ks = [p[0] for p in pts]
            vs = [p[1] for p in pts]
            ax.plot(ks, vs, marker="o", label=enc)
        if upper_bound is not None and upper_bound.get(metric) is not None:
            ub = upper_bound[metric]
            ax.axhline(ub, color="black", ls="--", lw=1.2, alpha=0.7,
                       label=f"upper bound ({ub:.3g})")
        ax.set_xlabel("K (latent size)")
        ax.set_ylabel(metric)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(n_m, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
