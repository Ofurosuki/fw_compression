"""Waveform-level look at EARLY-DEPTH ghosts (d/D <= thr), the ones the event rep
tends to lose (depth was the one separating feature, AUROC 0.71).

Reuses the matched-voxel capture (no-compression vs event taw K=3). Samples true
ghost voxels with d/D <= --thr and plots the FULL model-input waveform (orig black
vs event-synth red, each self-normalised) so the early ghost's relation to the
dominant primary return is visible. Top block = LOST (orig=ghost, event!=ghost),
bottom block = DETECTED, for direct comparison.

Run: PYTHONPATH=<repo>/src uv run python downstream/diag_ghost_early.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from hist_lidar.config import load_config_from_yaml
from hist_lidar.utils import get_model, load_checkpoint, set_seed
from diag_ghost_cases import capture

LABEL = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}
OUT = "downstream/outputs/events/diag"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="downstream/configs/evalA_split2_test.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.3, help="keep ghosts with d/D <= thr")
    ap.add_argument("--ncol", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config_from_yaml(args.config)
    cfg.device = args.device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    model = get_model(cfg).to(device)
    load_checkpoint(cfg.checkpoint_path, model, device)
    model.eval()
    ep = {"k": 3, "representation": "taw", "intensity_mode": "height",
          "smooth_sigma": 1.5, "min_height": 0.03, "min_distance": 3,
          "fixed_width": 4.0, "fixed_amplitude": 1.0}

    ins_o, pred_o, anns = capture(model, cfg, device, "none", ep, args.max_frames)
    ins_e, pred_e, _ = capture(model, cfg, device, "event", ep, args.max_frames)
    D = anns[0].shape[0]
    dmax = args.thr * D

    lost, succ = [], []
    for f in range(len(anns)):
        tg = anns[f] == 3
        for d, h, w in np.argwhere(tg):
            if d > dmax:
                continue
            if pred_o[f][d, h, w] == 3 and pred_e[f][d, h, w] != 3:
                lost.append((f, int(d), int(h), int(w)))
            elif pred_e[f][d, h, w] == 3:
                succ.append((f, int(d), int(h), int(w)))
    print(f"early-depth (d/D<={args.thr}) ghost voxels: LOST={len(lost)} DETECTED={len(succ)}")

    ncol = args.ncol
    blocks = [("LOST  (orig=ghost, event=NOT ghost)", lost, "tab:red"),
              ("DETECTED  (event=ghost)", succ, "tab:green")]
    fig, axes = plt.subplots(4, ncol, figsize=(4.2 * ncol, 12))
    for bi, (label, pool, col) in enumerate(blocks):
        samp = ([pool[i] for i in rng.choice(len(pool), min(2 * ncol, len(pool)), replace=False)]
                if pool else [])
        for k in range(2 * ncol):
            ax = axes[bi * 2 + k // ncol, k % ncol]
            if k >= len(samp):
                ax.axis("off"); continue
            f, d, h, w = samp[k]
            wo, we = ins_o[f][:, h, w], ins_e[f][:, h, w]
            no, ne = wo / max(wo.max(), 1e-6), we / max(we.max(), 1e-6)
            ax.plot(no, "k-", lw=1.1, label="orig")
            ax.plot(ne, "r-", lw=1.1, alpha=0.8, label="event taw K=3")
            gb = np.where(anns[f][:, h, w] == 3)[0]
            if len(gb):
                ax.axvspan(gb.min(), gb.max(), color="red", alpha=0.12)
            ax.axvline(d, color="red", ls="--", lw=1.0)
            ax.set_xlim(0, D); ax.set_ylim(-0.03, 1.08)
            ax.set_title(f"d/D={d/D:.2f}  orig-pred={LABEL[int(pred_o[f][d,h,w])]}  "
                         f"event-pred={LABEL[int(pred_e[f][d,h,w])]}",
                         fontsize=8, color=col)
            if k == 0:
                ax.legend(fontsize=7)
        axes[bi * 2, 0].annotate(label, xy=(0, 1.25), xycoords="axes fraction",
                                 fontsize=11, color=col, fontweight="bold")
    fig.suptitle(f"Early-depth ghosts (d/D ≤ {args.thr}): full model-input waveform, "
                 "orig (black) vs event-synth (red); red band = ghost region", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = f"{OUT}/ghost_early_depth_waveforms.png"
    plt.savefig(p, dpi=130); plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
