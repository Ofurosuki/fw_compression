"""Train spatio-temporal (4x4) compressive autoencoders on real Ghost-FWL patches.

Spatial counterpart of ``train_autoencoder.py``. Compresses a 4x4 neighbourhood of
pixel waveforms jointly (separable coding tensors) and reconstructs the whole block.
K is swept over the *block* latent sizes that match the per-pixel ratios
T/{8,16,32,64,128}, i.e. K_block = 16 * {8,16,32,64,128} (since ratio = P*T/K).

Usage:
    uv run python train_spatial.py --cv A --run_name real_A_spatial --epochs 30 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import envconfig
from compression.spatial_coding import build_spatial_autoencoder
from compression.data.spatial_waveforms import (
    SpatialWaveformConfig,
    make_datasets_spatial_cv,
    make_datasets_spatial_multi,
    collate_patches,
)
from compression.data.real_waveforms import SPLIT2

# block latent sizes matching per-pixel ratios T/{8,16,32,64,128} for a 4x4 (P=16) block
ALL_KS = [128, 256, 512, 1024, 2048]


def build_peak_mask_patch(labels, P, T, width=8):
    """[B, P, T] mask emphasising labelled peak regions across all pixels in the patch."""
    B = len(labels)
    mask = torch.zeros(B, P, T)
    for i, plist in enumerate(labels):
        for p, lab in enumerate(plist):
            if lab is None:
                continue
            for pos in np.asarray(lab["peak_positions"], dtype=int):
                lo, hi = max(0, pos - width), min(T, pos + width + 1)
                mask[i, p, lo:hi] = 1.0
    return mask


def build_bg_mask_patch(labels, P, T, protect=20):
    """[B, P, T] background mask = 1 on non-peak bins, 0 within ``±protect`` of any
    labelled peak in each pixel (anti-hallucination terms). ``protect`` is wider than
    the peak-loss ``width`` so a true peak's shoulders/tail are not penalised."""
    B = len(labels)
    mask = torch.ones(B, P, T)
    for i, plist in enumerate(labels):
        for p, lab in enumerate(plist):
            if lab is None:
                continue
            for pos in np.asarray(lab["peak_positions"], dtype=int):
                lo, hi = max(0, pos - protect), min(T, pos + protect + 1)
                mask[i, p, lo:hi] = 0.0
    return mask


def patch_loss(x_hat, x, peak_mask=None, peak_weight=1.0,
               bg_weight=0.0, fp_weight=0.0, bg_mask=None):
    """MSE + optional peak-aware + anti-hallucination (bg over-shoot / false-peak)
    terms, applied along the T axis of the [B, P, T] patch tensor (see the 1D
    ``reconstruction_loss`` for the rationale)."""
    mse = torch.mean((x_hat - x) ** 2)
    total = mse
    if peak_weight > 0 and peak_mask is not None:
        denom = peak_mask.sum().clamp_min(1.0)
        peak_term = (((x_hat - x) ** 2) * peak_mask).sum() / denom
        total = total + peak_weight * peak_term
    if bg_weight > 0 and bg_mask is not None:
        denom = bg_mask.sum().clamp_min(1.0)
        overshoot = torch.relu(x_hat - x)
        bg_term = ((overshoot ** 2) * bg_mask).sum() / denom
        total = total + bg_weight * bg_term
    if fp_weight > 0 and bg_mask is not None:
        dl = torch.relu(x_hat[:, :, 1:-1] - x_hat[:, :, :-2])
        dr = torch.relu(x_hat[:, :, 1:-1] - x_hat[:, :, 2:])
        peakness = dl * dr
        m = bg_mask[:, :, 1:-1]
        denom = m.sum().clamp_min(1.0)
        total = total + fp_weight * (peakness * m).sum() / denom
    return total


def train_one(K, train_ds, val_ds, T, P, device, epochs, batch_size, lr, peak_weight, out_dir,
              bg_weight=0.0, fp_weight=0.0, protect_width=20, log_every=5):
    model = build_spatial_autoencoder(T=T, K=K, P=P).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_patches, num_workers=4, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_patches, num_workers=2)

    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for x, labels in tl:
            x = x.to(device)
            pm = build_peak_mask_patch(labels, P, T).to(device) if peak_weight > 0 else None
            bm = (build_bg_mask_patch(labels, P, T, protect=protect_width).to(device)
                  if (bg_weight > 0 or fp_weight > 0) else None)
            x_hat, _ = model(x)
            loss = patch_loss(x_hat, x, peak_mask=pm, peak_weight=peak_weight,
                              bg_weight=bg_weight, fp_weight=fp_weight, bg_mask=bm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        train_loss = ep_loss / max(nb, 1)

        model.eval()
        vmse, vnb = 0.0, 0
        with torch.no_grad():
            for x, _ in vl:
                x = x.to(device)
                x_hat, _ = model(x)
                vmse += float(torch.mean((x_hat - x) ** 2))
                vnb += 1
        val_mse = vmse / max(vnb, 1)
        history.append({"epoch": ep, "train_loss": train_loss, "val_mse": val_mse})
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  [spatial K={K}] ep {ep+1}/{epochs} train={train_loss:.5f} val_mse={val_mse:.6f}")

    elapsed = time.time() - t0
    print(f"  [spatial K={K}] done in {elapsed:.1f}s  final val_mse={val_mse:.6f}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "T": T, "K": K, "P": P,
                    "history": history, "elapsed_sec": elapsed}, os.path.join(out_dir, "checkpoint.pt"))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", choices=["A", "B"], default="A")
    ap.add_argument("--split", default=None,
                    help="real-data multi-scene split (e.g. 'split2', 7 train scenes); overrides --cv")
    ap.add_argument("--run_name", default="real_A_spatial")
    ap.add_argument("--ks", nargs="+", type=int, default=ALL_KS)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--T", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--peak_weight", type=float, default=1.0)
    ap.add_argument("--bg_weight", type=float, default=0.0,
                    help="weight on background over-shoot suppression relu(x_hat-x)^2")
    ap.add_argument("--fp_weight", type=float, default=0.0,
                    help="weight on the differentiable false-peak (local-max) penalty")
    ap.add_argument("--protect_width", type=int, default=20,
                    help="bins around each labelled peak excluded from background terms")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    P = args.block * args.block

    cfg = SpatialWaveformConfig(T=args.T, block=args.block)
    if args.split == "split2":
        train_ds, val_ds, cfg = make_datasets_spatial_multi(SPLIT2["train"], cfg=cfg, seed=args.seed)
        print(f"[spatial split2] train={len(train_ds)} val={len(val_ds)} "
              f"({len(SPLIT2['train'])} scenes) patches P={P} T={args.T}")
    else:
        train_ds, val_ds, cfg = make_datasets_spatial_cv(args.cv, cfg=cfg, seed=args.seed)
        print(f"[spatial CV-{args.cv}] train={len(train_ds)} val={len(val_ds)} patches P={P} T={args.T}")

    run_dir = os.path.join(envconfig.output_path("runs"), args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "P": P, "spatial_cfg": cfg.__dict__}, f, indent=2)

    for K in args.ks:
        ratio = P * args.T / K
        print(f"=== training spatial K={K} (P*T/K={ratio:.1f}x, per-pixel-equiv K={K//P}) ===")
        train_one(K, train_ds, val_ds, T=args.T, P=P, device=args.device,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  peak_weight=args.peak_weight, bg_weight=args.bg_weight,
                  fp_weight=args.fp_weight, protect_width=args.protect_width,
                  out_dir=os.path.join(run_dir, f"spatial_K{K}"))
    print(f"All spatial training done. Checkpoints under {run_dir}/")


if __name__ == "__main__":
    main()
