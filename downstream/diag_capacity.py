"""Diagnostic: is the val-MSE U-shape (min at K=32 / 22x) a capacity property or an
optimization artifact of the fixed recipe (lr=1e-3, 30 ep, no LR decay)?

A wider bottleneck (K=64/128) can REPRESENT everything K=32 can, so if it cannot
MATCH K=32's MSE it is an optimization failure. Here we retrain K in {32,64,128}
for learnable_linear and random_projection under a stronger recipe (more epochs +
cosine LR decay) on the same SPLIT2 train data, and compare best val MSE.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from compression.autoencoder import build_autoencoder, reconstruction_loss
from compression.data.real_waveforms import RealWaveformConfig, make_datasets_multi, SPLIT2
from compression.data.synthetic_waveforms import collate_waveforms
from train_autoencoder import build_peak_mask

DEVICE = "cuda:0"
T = 700
N_TRAIN = 120000          # subset for speed; same for all configs -> fair comparison
EPOCHS = 80
BATCH = 256


def train(encoder_name, K, train_ds, val_ds, lr=1e-3, cosine=True):
    m = build_autoencoder(encoder_name, T=T, K=K, decoder_name="mlp").to(DEVICE)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=lr * 1e-2) if cosine else None
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_waveforms,
                    num_workers=4, drop_last=True)
    vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_waveforms, num_workers=2)
    best = 1e9
    for ep in range(EPOCHS):
        m.train()
        for x, labels in tl:
            x = x.to(DEVICE)
            pm = build_peak_mask(labels, T).to(DEVICE)
            xh, _ = m(x)
            loss, _ = reconstruction_loss(xh, x, peak_weight=1.0, peak_mask=pm)
            opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
        m.eval(); v = 0.0; n = 0
        with torch.no_grad():
            for x, _ in vl:
                x = x.to(DEVICE)
                xh, _ = m(x)
                v += float(torch.mean((xh - x) ** 2)); n += 1
        best = min(best, v / max(n, 1))
    return best


def main():
    torch.manual_seed(0); np.random.seed(0)
    cfg = RealWaveformConfig(T=T)
    train_ds, val_ds, _ = make_datasets_multi(SPLIT2["train"], cfg=cfg, seed=42, n_train=N_TRAIN, n_val=5000)
    print(f"train={len(train_ds)} val={len(val_ds)}  EPOCHS={EPOCHS} cosine LR\n")
    print(f"{'encoder':18s} {'K':>4} {'ratio':>6} {'best val MSE':>13}")
    print("-" * 46)
    for enc in ["learnable_linear", "random_projection"]:
        for K in [32, 64, 128]:
            t0 = time.time()
            b = train(enc, K, train_ds, val_ds)
            print(f"{enc:18s} {K:>4} {700 // K:>5}x {b:>13.6f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
