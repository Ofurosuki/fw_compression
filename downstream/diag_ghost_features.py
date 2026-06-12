"""Which feature separates DETECTED vs LOST ghosts (event taw K=3)?

Height and clutter were refuted (see ghost_lost_vs_amplitude.png). This tries
richer candidates and QUANTIFIES separability with an AUROC (Mann-Whitney) so we
don't just eyeball overlapping histograms:

  1. recon_err  : |orig - event-synth| at the ghost return (model-input space,
                  each waveform self-normalised) -> did synthesis distort it?
  2. asymmetry  : |right-energy - left-energy| / total around the ghost peak
                  (orig) -> shape the clean Gaussian cannot represent.
  3. separation : bins to the nearest STRONGER return in the pixel (orig)
                  -> is the ghost merged into a dominant neighbour?
  4. spatial_iso: 1 - local ghost-column density in an HxW window (GT)
                  -> is the lost ghost spatially ISOLATED (3D-context hypothesis)?

AUROC ~0.5 = no separation; ->1 or ->0 = the feature distinguishes the two.

Run: PYTHONPATH=<repo>/src uv run python downstream/diag_ghost_features.py --device cuda:0
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
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from hist_lidar.config import load_config_from_yaml
from hist_lidar.utils import get_model, load_checkpoint, set_seed
from diag_ghost_cases import capture

OUT = "downstream/outputs/events/diag"
WIN = 8          # +/- time bins around the ghost return for local shape/error
RAD = 3          # +/- spatial radius (H,W) for ghost-density / isolation


def local_max_positions(c, rel=0.10):
    ismax = (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:]) & (c[1:-1] >= rel)
    return np.where(ismax)[0] + 1


def features_for(coord, ins_o, ins_e, ghost2d):
    f, d, h, w = coord
    col_o = ins_o[f][:, h, w]
    col_e = ins_e[f][:, h, w]
    D = col_o.shape[0]
    mo, me = max(col_o.max(), 1e-6), max(col_e.max(), 1e-6)
    no, ne = col_o / mo, col_e / me
    lo, hi = max(0, d - WIN), min(D, d + WIN + 1)

    recon_err = float(np.abs(no[lo:hi] - ne[lo:hi]).mean())

    seg = no[lo:hi]
    rel = np.arange(lo, hi) - d
    tot = seg.sum() + 1e-9
    left = seg[rel < 0].sum(); right = seg[rel > 0].sum()
    asym = float(abs(right - left) / tot)

    peaks = local_max_positions(no)
    taller = peaks[no[peaks] > no[d] + 1e-6]
    sep = float(np.min(np.abs(taller - d))) if len(taller) else 50.0
    sep = min(sep, 50.0)

    g = ghost2d[f]
    H, W = g.shape
    a, b = max(0, h - RAD), min(H, h + RAD + 1)
    cc, dd = max(0, w - RAD), min(W, w + RAD + 1)
    iso = 1.0 - float(g[a:b, cc:dd].mean())      # small-window isolation
    RAD2 = 12
    a2, b2 = max(0, h - RAD2), min(H, h + RAD2 + 1)
    c2, d2 = max(0, w - RAD2), min(W, w + RAD2 + 1)
    iso_wide = 1.0 - float(g[a2:b2, c2:d2].mean())   # wide-window isolation

    rank = float(min(int((no[peaks] > no[d] + 1e-6).sum()) + 1, 6))  # 1=strongest
    depth = float(d) / D                              # range position 0..1
    return recon_err, asym, sep, iso, iso_wide, rank, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="downstream/configs/evalA_split2_test.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--n_sample", type=int, default=40000)
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
    ghost2d = [(a == 3).any(0) for a in anns]

    succ, lost = [], []
    for f in range(len(anns)):
        tg = anns[f] == 3
        sc = np.argwhere(tg & (pred_e[f] == 3))
        lc = np.argwhere(tg & (pred_o[f] == 3) & (pred_e[f] != 3))
        succ += [(f, *map(int, x)) for x in sc]
        lost += [(f, *map(int, x)) for x in lc]

    NF = 7

    def collect(pool):
        if not pool:
            return np.empty((0, NF))
        idx = rng.choice(len(pool), min(args.n_sample, len(pool)), replace=False)
        return np.array([features_for(pool[i], ins_o, ins_e, ghost2d) for i in idx])

    Fs, Fl = collect(succ), collect(lost)
    names = ["recon_err @ ghost", "asymmetry", "separation to stronger return (bins)",
             "spatial isolation (RAD=3)", "spatial isolation (RAD=12)",
             "ghost peak rank (resized waveform)", "depth / range position"]
    units = ["normalised |orig-synth|", "|R-L|/total", "bins (capped 50)", "fraction",
             "fraction", "rank (1=strongest, capped 6)", "d / D"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for ax in axes.ravel()[NF:]:
        ax.axis("off")
    print(f"DETECTED n={len(Fs)}  LOST n={len(Fl)}")
    for j, (ax, nm, un) in enumerate(zip(axes.ravel(), names, units)):
        s, l = Fs[:, j], Fl[:, j]
        u, _ = mannwhitneyu(s, l, alternative="two-sided")
        auc = u / (len(s) * len(l))                      # P(detected > lost)
        lo = min(s.min(), l.min()); hi = max(s.max(), l.max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(s, bins=bins, density=True, alpha=0.6, color="tab:green",
                label=f"DETECTED (median {np.median(s):.3f})")
        ax.hist(l, bins=bins, density=True, alpha=0.6, color="tab:red",
                label=f"LOST (median {np.median(l):.3f})")
        sep = abs(auc - 0.5) * 2                          # 0=none .. 1=perfect
        ax.set_title(f"{nm}\nAUROC={auc:.3f}  (separability {sep:.2f})",
                     fontsize=10, color="tab:red" if sep > 0.15 else "black")
        ax.set_xlabel(un); ax.set_ylabel("density"); ax.legend(fontsize=8)
        print(f"  {nm:42s} AUROC={auc:.3f}  detected_med={np.median(s):.3f} lost_med={np.median(l):.3f}")
    fig.suptitle("Candidate features separating DETECTED vs LOST ghosts (event taw K=3)\n"
                 "AUROC=0.5 -> no separation;  >0.65 or <0.35 -> the feature distinguishes them",
                 fontsize=12)
    plt.tight_layout()
    p = f"{OUT}/ghost_feature_candidates.png"
    plt.savefig(p, dpi=130); plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
