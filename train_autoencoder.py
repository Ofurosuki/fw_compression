"""Train waveform autoencoders across the encoder x K sweep.

Usage:
    uv run python train_autoencoder.py                 # full sweep, default scale
    uv run python train_autoencoder.py --smoke         # tiny/fast sanity run
    uv run python train_autoencoder.py --encoders learnable_linear --ks 32

Outputs (under runs/<run_name>/):
    <encoder>_K<K>/checkpoint.pt        trained autoencoder weights + config
    config.json                          run-level config
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from compression.autoencoder import build_autoencoder, reconstruction_loss
from compression.data.synthetic_waveforms import collate_waveforms


def get_data_api(name):
    """Return (make_datasets, Config) for the chosen dataset family."""
    if name == "physical":
        from compression.data.physical_waveforms import (
            PhysicalWaveformConfig as Cfg,
            make_datasets as md,
        )
        return md, Cfg
    from compression.data.synthetic_waveforms import WaveformConfig as Cfg, make_datasets as md
    return md, Cfg

ALL_ENCODERS = ["coarse_binning", "random_projection", "dct_lowfreq", "learnable_linear"]
ALL_KS = [8, 16, 32, 64, 128]


def build_peak_mask(labels, T, width=8):
    """Build a [B, T] mask emphasising labelled peak regions (for peak-aware loss)."""
    B = len(labels)
    mask = torch.zeros(B, T)
    for i, lab in enumerate(labels):
        for p in np.asarray(lab["peak_positions"], dtype=int):
            lo, hi = max(0, p - width), min(T, p + width + 1)
            mask[i, lo:hi] = 1.0
    return mask


def train_one(
    encoder_name,
    K,
    train_ds,
    val_ds,
    T,
    device,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    energy_weight=0.0,
    peak_weight=1.0,
    out_dir=None,
    log_every=5,
):
    model = build_autoencoder(encoder_name, T=T, K=K, decoder_name="mlp").to(device)
    # only optimize params that require grad (fixed encoders have none in encoder)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_waveforms, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_waveforms, num_workers=2)

    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for x, labels in train_loader:
            x = x.to(device)
            peak_mask = build_peak_mask(labels, T).to(device) if peak_weight > 0 else None
            x_hat, _ = model(x)
            loss, comps = reconstruction_loss(
                x_hat, x, energy_weight=energy_weight, peak_weight=peak_weight, peak_mask=peak_mask
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        train_loss = ep_loss / max(nb, 1)

        # validation MSE
        model.eval()
        vmse = 0.0
        vnb = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                vmse += float(torch.mean((x_hat - x) ** 2))
                vnb += 1
        val_mse = vmse / max(vnb, 1)
        history.append({"epoch": ep, "train_loss": train_loss, "val_mse": val_mse})
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  [{encoder_name} K={K}] ep {ep+1}/{epochs} train={train_loss:.5f} val_mse={val_mse:.6f}")

    elapsed = time.time() - t0
    print(f"  [{encoder_name} K={K}] done in {elapsed:.1f}s  final val_mse={val_mse:.6f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "encoder_name": encoder_name,
                "decoder_name": "mlp",
                "T": T,
                "K": K,
                "history": history,
                "elapsed_sec": elapsed,
            },
            os.path.join(out_dir, "checkpoint.pt"),
        )
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="sweep")
    ap.add_argument("--data", choices=["physical", "synthetic", "real"], default="physical")
    ap.add_argument("--cv", choices=["A", "B"], default="A",
                    help="real-data cross-validation: A=train scene001/val scene002, B=swapped")
    ap.add_argument("--split", default=None,
                    help="real-data multi-scene split (e.g. 'split2', 7 train scenes); overrides --cv")
    ap.add_argument("--encoders", nargs="+", default=ALL_ENCODERS)
    ap.add_argument("--ks", nargs="+", type=int, default=ALL_KS)
    ap.add_argument("--T", type=int, default=700)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    # NOTE: energy_weight defaults OFF. A non-trivial energy term (|sum x_hat - sum x|)
    # dominates the MSE at init (~0.1*27 >> 0.013), so the decoder matches total
    # energy with a broad smear and never localizes peaks. Kept available for ablation.
    ap.add_argument("--energy_weight", type=float, default=0.0)
    ap.add_argument("--peak_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run for sanity checking")
    args = ap.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.epochs = 1000, 200, 3
        args.encoders = ["coarse_binning", "learnable_linear"]
        args.ks = [16, 64]
        args.run_name = "smoke"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data == "real":
        from compression.data.real_waveforms import (
            RealWaveformConfig, make_datasets_cv, make_datasets_multi, SPLIT2,
        )
        cfg = RealWaveformConfig(T=args.T)
        n_tr = None if args.n_train <= 0 else (args.n_train if args.n_train != 20000 else None)
        if args.split == "split2":
            train_ds, val_ds, cfg = make_datasets_multi(
                SPLIT2["train"], cfg=cfg, seed=args.seed, n_train=n_tr, n_val=5000)
            print(f"[real split2] train={len(train_ds)} val={len(val_ds)} "
                  f"({len(SPLIT2['train'])} scenes) T={args.T}")
        else:
            train_ds, val_ds, cfg = make_datasets_cv(args.cv, cfg=cfg, seed=args.seed, n_train=n_tr, n_val=5000)
            print(f"[real CV-{args.cv}] train={len(train_ds)} val={len(val_ds)} T={args.T}")
    else:
        make_datasets, Cfg = get_data_api(args.data)
        cfg = Cfg(T=args.T)
        print(f"Generating {args.data} data: train={args.n_train} val={args.n_val} T={args.T}")
        train_ds, val_ds, cfg = make_datasets(args.n_train, args.n_val, cfg=cfg, seed=args.seed)

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "waveform_cfg": cfg.__dict__}, f, indent=2)

    for enc in args.encoders:
        for K in args.ks:
            out_dir = os.path.join(run_dir, f"{enc}_K{K}")
            print(f"=== training {enc} K={K} (T/K={args.T/K:.1f}x) ===")
            train_one(
                enc,
                K,
                train_ds,
                val_ds,
                T=args.T,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                energy_weight=args.energy_weight,
                peak_weight=args.peak_weight,
                out_dir=out_dir,
            )
    print(f"All training done. Checkpoints under {run_dir}/")


if __name__ == "__main__":
    main()
