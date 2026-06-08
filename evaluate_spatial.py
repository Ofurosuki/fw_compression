"""Evaluate spatio-temporal (4x4) compressive autoencoders.

Reconstructs each 4x4 patch, unfolds it back to per-pixel waveforms, and scores with
the SAME real-data metrics as the per-pixel experiment (``real_peak_metrics`` +
``aggregate_metrics``) so results are directly comparable to ``runs/real_{A,B}``.
The block latent K is reported alongside its per-pixel-equivalent K=K/P and the
implied per-pixel compression ratio P*T/K.

Usage:
    uv run python evaluate_spatial.py --cv A --run_name real_A_spatial --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from compression.spatial_coding import build_spatial_autoencoder
from compression.data.spatial_waveforms import (
    SpatialWaveformConfig,
    extract_scene_spatial,
    unfold_patches,
    CV_SPLITS,
)
from compression.utils.metrics import aggregate_metrics, gt_width_threshold, real_peak_metrics
from compression.utils.plot import plot_examples, plot_sweep

DETECT_KW = dict(smooth=2.0, prominence=0.04, distance=12, rel_height=0.06)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_spatial_autoencoder(T=ck["T"], K=ck["K"], P=ck["P"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def reconstruct_patches(model, patches, device, batch=128):
    outs = []
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            xb = torch.from_numpy(np.ascontiguousarray(patches[i : i + batch])).float().to(device)
            xh, _ = model(xb)
            outs.append(xh.cpu().numpy())
    return np.concatenate(outs, axis=0)


def full_metrics(x, x_hat, labels, T, K, width_thr, n_param):
    m = aggregate_metrics(x, x_hat, labels=labels, T=T, K=K, **DETECT_KW)
    m.update(real_peak_metrics(x_hat, labels, width_threshold=width_thr, n_max=n_param, **DETECT_KW))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", choices=["A", "B"], default="A")
    ap.add_argument("--run_name", default="real_A_spatial")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--T", type=int, default=700)
    ap.add_argument("--n_param", type=int, default=3000, help="#pixels for (slow) per-peak param metrics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run dir not found: {run_dir} (train first)")
    P = args.block * args.block

    cfg = SpatialWaveformConfig(T=args.T, block=args.block)
    val_scene = CV_SPLITS[args.cv]["val"]
    patches, plabels = extract_scene_spatial(val_scene, cfg, seed=args.seed + 1)
    x, labels = unfold_patches(patches, plabels)   # per-pixel original waveforms + labels
    print(f"[spatial CV-{args.cv}] eval on {val_scene}: {len(patches)} patches -> {len(x)} labelled pixels")
    width_thr = gt_width_threshold(labels)

    # --- upper bound: full waveform, no compression (x_hat == x) ---
    ub = full_metrics(x, x, labels, args.T, args.T, width_thr, args.n_param)
    ub["encoder"] = "full_waveform_upper_bound"
    ub["K"] = args.T
    with open(os.path.join(run_dir, "upper_bound.json"), "w") as f:
        json.dump(ub, f, indent=2)
    print(f"[upper bound] survival(object={ub['object_recall']:.3f} glass={ub['glass_recall']:.3f} "
          f"ghost={ub['ghost_recall']:.3f})  ghost_int_rel={ub['all_intensity_relerr']:.3f} thr={width_thr:.1f}")

    results = [ub]
    subdirs = sorted((d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and "_K" in d),
                     key=lambda d: int(d.split("_K")[-1]))
    for sd in subdirs:
        ckpt = os.path.join(run_dir, sd, "checkpoint.pt")
        if not os.path.exists(ckpt):
            continue
        model, ck = load_model(ckpt, args.device)
        K = ck["K"]
        x_hat_patches = reconstruct_patches(model, patches, args.device)
        # unfold reconstruction the SAME way (keep pixels that have labels)
        x_hat, _ = unfold_patches(x_hat_patches, plabels)
        m = full_metrics(x, x_hat, labels, args.T, K, width_thr, args.n_param)
        m["encoder"] = "spatial_separable"
        m["K_perpixel"] = K // P
        out = os.path.join(run_dir, sd)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        np.save(os.path.join(out, "x_hat.npy"), x_hat[:4000].astype(np.float32))
        ghost_idx = [i for i in range(len(labels)) if labels[i]["ghost"]][:4]
        plain_idx = [i for i in range(len(labels)) if not labels[i]["ghost"]][:2]
        sel = (ghost_idx + plain_idx) or list(range(6))
        ratio = P * args.T / K
        plot_examples(x[sel], x_hat[sel], os.path.join(out, "examples.png"),
                      labels=[labels[i] for i in sel], n=len(sel),
                      title=f"spatial 4x4 K={K} (per-pixel K={K//P}, CR={ratio:.0f}x)")
        results.append(m)
        print(f"[spatial K={K:4d} (eqK={K//P:3d})] CR={ratio:5.1f}x mse={m['waveform_mse_mean']:.2e} | "
              f"survival obj={m['object_recall']:.2f} glass={m['glass_recall']:.2f} GHOST={m['ghost_recall']:.2f} | "
              f"ghost-fidelity int={m['all_intensity_relerr']:.2f} fwhm={m['all_fwhm_relerr']:.2f}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    sweep = [r for r in results if r["encoder"] != "full_waveform_upper_bound"]
    if sweep:
        plot_sweep(sweep, os.path.join(run_dir, "sweep_survival.png"),
                   metrics=("object_recall", "glass_recall", "ghost_recall", "all_precision"), upper_bound=ub)
        plot_sweep(sweep, os.path.join(run_dir, "sweep.png"),
                   metrics=("waveform_mse_mean", "all_pos_err", "all_intensity_relerr", "all_fwhm_relerr"), upper_bound=ub)
        plot_sweep(sweep, os.path.join(run_dir, "sweep_freq.png"),
                   metrics=("narrow_intensity_relerr", "wide_intensity_relerr", "narrow_fwhm_relerr", "wide_fwhm_relerr"), upper_bound=ub)
    print(f"\nDone. Summary -> {run_dir}/summary.json, plots -> {run_dir}/sweep*.png")


if __name__ == "__main__":
    main()
