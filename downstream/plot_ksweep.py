"""Plot the taw K-sweep (architecture-controlled ToPM retrain): voxel & peak F1-mean
vs number of top-K transport events, with the full-waveform ceiling as reference.

  uv-py downstream/plot_ksweep.py   # writes downstream/outputs/diag/taw_ksweep.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "outputs", "diag", "taw_ksweep.png")

ap = argparse.ArgumentParser()
ap.add_argument("--peak_only", action="store_true", help="plot peak-level F1 only")
ARGS = ap.parse_args()

# taw, seed42, retrained ToPM (downstream/RETRAIN_RESULTS.md / memory fwc-topm-retrain)
K       = [2,     3,     4,     5,     6]
ratio   = [117,   78,    58,    47,    39]
voxel   = [0.503, 0.532, 0.531, 0.529, 0.540]
peak    = [0.554, 0.581, 0.582, 0.585, 0.595]

FULL_VOXEL = 0.533
FULL_PEAK  = 0.595

out = OUT.replace(".png", "_peak.png") if ARGS.peak_only else OUT
os.makedirs(os.path.dirname(out), exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 4.8))

# full-waveform peak ceiling
ax.axhline(FULL_PEAK, color="tab:orange", ls="--", lw=1, alpha=0.7)
ax.text(6.05, FULL_PEAK, f" full {FULL_PEAK:.3f}", color="tab:orange", va="center", fontsize=8)
ax.plot(K, peak, "o-", color="tab:orange", label="peak-level F1 (paper metric)")
anno = peak

if not ARGS.peak_only:
    ax.axhline(FULL_VOXEL, color="tab:blue", ls="--", lw=1, alpha=0.7)
    ax.text(6.05, FULL_VOXEL, f" full {FULL_VOXEL:.3f}", color="tab:blue", va="center", fontsize=8)
    ax.plot(K, voxel, "s-", color="tab:blue", label="voxel-level F1")
    anno = voxel

# annotate compression ratio under each point
for k, r, y in zip(K, ratio, anno):
    ax.annotate(f"{r}×", (k, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7, color="gray")

# saturation marker
ax.axvspan(2.5, 6.5, color="green", alpha=0.05)
ymid = 0.566 if ARGS.peak_only else 0.515
ax.text(4.5, ymid, "saturated (≈ full)", color="green", ha="center", fontsize=8, alpha=0.8)

ax.set_xlabel("K  (top-K transport events per pixel)")
ax.set_ylabel("F1-mean  (object / glass / ghost)")
title = "taw event K-sweep (peak-level) — ToPM retrained per K" if ARGS.peak_only \
        else "taw event K-sweep — ToPM retrained per K (arch fixed)"
ax.set_title(title)
ax.set_xticks(K)
ax.set_xlim(1.6, 7.0)
ax.set_ylim(0.54, 0.61) if ARGS.peak_only else ax.set_ylim(0.49, 0.61)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"wrote {out}")
