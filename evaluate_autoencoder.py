"""Evaluate trained autoencoders.

Reconstruction metrics + (for physical data) IRF-fit per-peak parameter
preservation (position / intensity / FWHM), split into narrow (high-freq) vs
wide (low-freq) peaks. Also saves x_hat / z, example plots, and sweep plots.

Usage:
    uv run python evaluate_autoencoder.py --run_name sweep_phys --data physical
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from compression.autoencoder import build_autoencoder
from compression.downstream.ghost_fwl_hook import proxy_ghost_score
from compression.utils.metrics import (
    aggregate_metrics,
    gt_width_threshold,
    per_peak_param_metrics,
    real_peak_metrics,
)
from compression.utils.plot import plot_examples, plot_sweep

# detector settings calibrated per dataset family (see calibration in dev notes)
DETECT_KW = {
    "physical": dict(smooth=2.0, prominence=0.04, distance=30, rel_height=0.06),
    "synthetic": dict(smooth=2.0, prominence=0.05, distance=20, rel_height=0.1),
    "real": dict(smooth=2.0, prominence=0.04, distance=12, rel_height=0.06),
}


def get_generate(name):
    if name == "physical":
        from compression.data.physical_waveforms import PhysicalWaveformConfig as Cfg, generate_dataset as gd
        return gd, Cfg
    from compression.data.synthetic_waveforms import WaveformConfig as Cfg, generate_dataset as gd
    return gd, Cfg


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_autoencoder(ck["encoder_name"], T=ck["T"], K=ck["K"], decoder_name=ck["decoder_name"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def reconstruct(model, x_t, device, save_n):
    with torch.no_grad():
        sub = x_t[:save_n].to(device)
        x_hat, z = model(sub)
    return x_hat.cpu().numpy(), z.cpu().numpy()


def full_metrics(x, x_hat, labels, T, K, data, tau, width_thr, n_param):
    det = DETECT_KW[data]
    m = aggregate_metrics(x, x_hat, labels=labels, T=T, K=K, **det)
    if data == "real":
        # no synthetic ghost-proxy; use real per-class survival + original-referenced fidelity
        m.update(real_peak_metrics(x_hat, labels, width_threshold=width_thr, n_max=n_param, **det))
    else:
        m.update(proxy_ghost_score(x, x_hat, labels, **det))
        if data == "physical":
            m.update(per_peak_param_metrics(x_hat, labels, tau=tau, width_threshold=width_thr, n_max=n_param, **det))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="sweep_phys")
    ap.add_argument("--data", choices=["physical", "synthetic", "real"], default="physical")
    ap.add_argument("--cv", choices=["A", "B"], default="A",
                    help="real-data cross-validation direction (must match training)")
    ap.add_argument("--T", type=int, default=700)
    ap.add_argument("--n_eval", type=int, default=4000)
    ap.add_argument("--n_param", type=int, default=3000, help="#waveforms for (slow) per-peak param metrics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run dir not found: {run_dir} (train first)")

    if args.data == "real":
        from compression.data.real_waveforms import RealWaveformConfig, extract_scene, CV_SPLITS
        cfg = RealWaveformConfig(T=args.T)
        tau = 0.0
        val_scene = CV_SPLITS[args.cv]["val"]
        # val cache uses train_seed+1 (see make_datasets_cv); match it here
        waves, labels = extract_scene(val_scene, cfg, seed=args.seed + 1)
        waves, labels = waves[: args.n_eval], labels[: args.n_eval]
        print(f"[real CV-{args.cv}] eval on {val_scene}: n={len(waves)} T={args.T}")
        x_t = torch.from_numpy(np.ascontiguousarray(waves)).float()
        width_thr = gt_width_threshold(labels)
    else:
        generate_dataset, Cfg = get_generate(args.data)
        cfg = Cfg(T=args.T)
        tau = getattr(cfg, "tau", 4.0)
        print(f"Generating {args.data} eval set: n={args.n_eval} T={args.T} seed={args.seed}")
        waves, labels = generate_dataset(args.n_eval, cfg, seed=args.seed)
        x_t = torch.from_numpy(waves).float()
        width_thr = gt_width_threshold(labels) if args.data == "physical" else 0.0

    # --- upper bound: full waveform, no compression (x_hat == x) ---
    ub = full_metrics(waves, waves, labels, args.T, args.T, args.data, tau, width_thr, args.n_param)
    ub["encoder"] = "full_waveform_upper_bound"
    ub["K"] = args.T
    with open(os.path.join(run_dir, "upper_bound.json"), "w") as f:
        json.dump(ub, f, indent=2)
    if args.data == "physical":
        print(
            f"[upper bound] mse={ub['waveform_mse_mean']:.2e}  "
            f"narrow(int_rel={ub['narrow_intensity_relerr']:.3f} fwhm_rel={ub['narrow_fwhm_relerr']:.3f} rec={ub['narrow_recall']:.3f})  "
            f"wide(int_rel={ub['wide_intensity_relerr']:.3f} fwhm_rel={ub['wide_fwhm_relerr']:.3f} rec={ub['wide_recall']:.3f})  "
            f"thr={width_thr:.1f}"
        )
    elif args.data == "real":
        print(
            f"[upper bound] mse={ub['waveform_mse_mean']:.2e}  "
            f"survival(object={ub['object_recall']:.3f} glass={ub['glass_recall']:.3f} ghost={ub['ghost_recall']:.3f})  "
            f"ghost_int_rel={ub['all_intensity_relerr']:.3f} thr={width_thr:.1f}"
        )
    else:
        print(f"[upper bound] mse={ub['waveform_mse_mean']:.2e} ghost_f1={ub['ghost_proxy_f1']:.3f}")

    # --- evaluate each trained config ---
    results = [ub]
    subdirs = sorted(d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and "_K" in d)
    for sd in subdirs:
        ckpt = os.path.join(run_dir, sd, "checkpoint.pt")
        if not os.path.exists(ckpt):
            continue
        model, ck = load_model(ckpt, args.device)
        enc, K = ck["encoder_name"], ck["K"]
        x_hat, z = reconstruct(model, x_t, args.device, args.n_eval)
        x = waves[: len(x_hat)]
        lab = labels[: len(x_hat)]
        m = full_metrics(x, x_hat, lab, args.T, K, args.data, tau, width_thr, args.n_param)
        m["encoder"] = enc
        out = os.path.join(run_dir, sd)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        np.save(os.path.join(out, "x_hat.npy"), x_hat.astype(np.float32))
        np.save(os.path.join(out, "z.npy"), z.astype(np.float32))
        ghost_idx = [i for i in range(len(lab)) if lab[i]["ghost"]][:4]
        plain_idx = [i for i in range(len(lab)) if not lab[i]["ghost"]][:2]
        sel = (ghost_idx + plain_idx) or list(range(6))
        plot_examples(x[sel], x_hat[sel], os.path.join(out, "examples.png"),
                      labels=[lab[i] for i in sel], n=len(sel), title=f"{enc} K={K} (CR={args.T/K:.0f}x)")
        results.append(m)
        if args.data == "physical":
            print(
                f"[{enc:17s} K={K:3d}] CR={args.T/K:5.1f}x mse={m['waveform_mse_mean']:.2e} | "
                f"NARROW int={m['narrow_intensity_relerr']:.2f} fwhm={m['narrow_fwhm_relerr']:.2f} rec={m['narrow_recall']:.2f} | "
                f"WIDE int={m['wide_intensity_relerr']:.2f} fwhm={m['wide_fwhm_relerr']:.2f} rec={m['wide_recall']:.2f}"
            )
        elif args.data == "real":
            print(
                f"[{enc:17s} K={K:3d}] CR={args.T/K:5.1f}x mse={m['waveform_mse_mean']:.2e} | "
                f"survival obj={m['object_recall']:.2f} glass={m['glass_recall']:.2f} GHOST={m['ghost_recall']:.2f} | "
                f"ghost-fidelity int={m['all_intensity_relerr']:.2f} fwhm={m['all_fwhm_relerr']:.2f}"
            )
        else:
            print(f"[{enc} K={K}] mse={m['waveform_mse_mean']:.2e} ghost_f1={m['ghost_proxy_f1']:.3f}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    sweep_results = [r for r in results if r["encoder"] != "full_waveform_upper_bound"]
    if sweep_results:
        if args.data == "physical":
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep_narrow.png"),
                       metrics=("narrow_recall", "narrow_pos_err", "narrow_intensity_relerr", "narrow_fwhm_relerr"), upper_bound=ub)
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep_wide.png"),
                       metrics=("wide_recall", "wide_pos_err", "wide_intensity_relerr", "wide_fwhm_relerr"), upper_bound=ub)
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep.png"),
                       metrics=("waveform_mse_mean", "all_pos_err", "all_intensity_relerr", "all_fwhm_relerr"), upper_bound=ub)
        elif args.data == "real":
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep_survival.png"),
                       metrics=("object_recall", "glass_recall", "ghost_recall", "all_precision"), upper_bound=ub)
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep.png"),
                       metrics=("waveform_mse_mean", "all_pos_err", "all_intensity_relerr", "all_fwhm_relerr"), upper_bound=ub)
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep_freq.png"),
                       metrics=("narrow_intensity_relerr", "wide_intensity_relerr", "narrow_fwhm_relerr", "wide_fwhm_relerr"), upper_bound=ub)
        else:
            plot_sweep(sweep_results, os.path.join(run_dir, "sweep.png"),
                       metrics=("waveform_mse_mean", "peak_loc_error_mean", "peak_count_acc", "ghost_proxy_f1"), upper_bound=ub)
    print(f"\nDone. Summary -> {run_dir}/summary.json, plots -> {run_dir}/sweep*.png")


if __name__ == "__main__":
    main()
