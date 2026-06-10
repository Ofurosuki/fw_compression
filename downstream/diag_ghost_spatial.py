"""Spatial map of where GHOSTS vanish under the top-K event representation.

Reuses the matched-voxel capture from diag_ghost_cases (frozen FWL-ToPM run twice
on the same frames: no-compression vs event taw K=3) and front-projects the
(D=time, H, W) voxel grid along depth to a 2D (H, W) map of "this pixel column
contains a ghost". For the frames with the most ghost content it shows, per frame:

  GT ghost | no-compression predicted ghost | event taw K=3 predicted ghost | outcome

where `outcome` categorises GT-ghost columns: kept by both (green), LOST by the
event rep (red: orig detected, event missed), missed by both (grey), and event
false-positive ghost columns (blue). This visualises that the event misses are
spatially structured — clusters of ghost, not random voxels — i.e. driven by the
3D context the model integrates, not a per-pixel waveform property.

Run: PYTHONPATH=<repo>/src uv run python downstream/diag_ghost_spatial.py --device cuda:0
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
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))   # for sibling imports
from hist_lidar.config import load_config_from_yaml
from hist_lidar.utils import get_model, load_checkpoint, set_seed
from diag_ghost_cases import capture

OUT = "downstream/outputs/events/diag"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="downstream/configs/evalA_split2_test.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--n_show", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config_from_yaml(args.config)
    cfg.device = args.device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    os.makedirs(OUT, exist_ok=True)

    model = get_model(cfg).to(device)
    load_checkpoint(cfg.checkpoint_path, model, device)
    model.eval()

    ep = {"k": 3, "representation": "taw", "intensity_mode": "height",
          "smooth_sigma": 1.5, "min_height": 0.03, "min_distance": 3,
          "fixed_width": 4.0, "fixed_amplitude": 1.0}

    _, pred_o, anns = capture(model, cfg, device, "none", ep, args.max_frames)
    _, pred_e, _ = capture(model, cfg, device, "event", ep, args.max_frames)

    # project along depth (axis 0): a pixel column "has ghost" if any voxel is ghost
    gt = [(a == 3).any(0) for a in anns]                       # (H,W) bool
    po = [(p == 3).any(0) for p in pred_o]
    pe = [(p == 3).any(0) for p in pred_e]

    order = sorted(range(len(gt)), key=lambda i: -int(gt[i].sum()))[: args.n_show]

    # outcome colourmap: 0 bg, 1 kept(both), 2 LOST(orig-only), 3 missed(neither), 4 event-FP
    cmap = ListedColormap(["#f2f2f2", "#2ca02c", "#d62728", "#bdbdbd", "#1f77b4"])
    titles = ["GT ghost", "no-compression pred", "event taw K=3 pred",
              "outcome (green=kept red=LOST grey=both-miss blue=FP)"]

    fig, axes = plt.subplots(len(order), 4, figsize=(18, 4.0 * len(order)))
    if len(order) == 1:
        axes = axes[None, :]
    for r, f in enumerate(order):
        g, o, e = gt[f], po[f], pe[f]
        outcome = np.zeros(g.shape, dtype=np.int32)
        outcome[g & o & e] = 1                                  # kept by both
        outcome[g & o & ~e] = 2                                 # LOST by event rep
        outcome[g & ~o] = 3                                     # missed by both (baseline too)
        outcome[~g & e] = 4                                     # event false-positive ghost
        n_gt = int(g.sum()); n_lost = int((g & o & ~e).sum()); n_kept = int((g & o & e).sum())
        for c, (img, t) in enumerate(zip([g, o, e, outcome], titles)):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(img, cmap="Greys", vmin=0, vmax=1, aspect="auto")
            else:
                ax.imshow(outcome, cmap=cmap, vmin=0, vmax=4, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=10)
        axes[r, 0].set_ylabel(f"frame {f}\nGT cols={n_gt}\nkept={n_kept} lost={n_lost}",
                              fontsize=9, rotation=0, ha="right", va="center", labelpad=40)
    fig.suptitle("Where do ghosts vanish under top-K events? (front-projected ghost columns, H×W)",
                 fontsize=12)
    plt.tight_layout()
    p = f"{OUT}/ghost_spatial_maps.png"
    plt.savefig(p, dpi=130); plt.close()

    # aggregate recall over shown frames
    tot_gt = sum(int(gt[f].sum()) for f in range(len(gt)))
    tot_o = sum(int((gt[f] & po[f]).sum()) for f in range(len(gt)))
    tot_e = sum(int((gt[f] & pe[f]).sum()) for f in range(len(gt)))
    print(f"column-level ghost recall ({len(gt)} frames): "
          f"no-compression={tot_o/max(1,tot_gt):.3f}  event K3={tot_e/max(1,tot_gt):.3f}")
    print("saved", p)


if __name__ == "__main__":
    main()
