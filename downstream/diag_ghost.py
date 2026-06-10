"""Why does the top-K event representation lose GHOSTS?

The downstream confusion matrices show the ghost gap (taw K=3: 0.45 vs full 0.56)
is a RECALL problem: true ghost voxels get predicted as noise/object as K shrinks
(K=1 loses 90% of ghosts). Hypothesis: ghosts are *secondary / weaker* returns, so
they fall outside the top-K strongest peaks and their pulse is dropped by synthesis.

This script verifies that at the WAVEFORM level (no model inference needed):

1. ghost_waveforms.png — example ghost pixels: original waveform, the ghost-labeled
   return region (shaded), and the top-K event-synthesised waveform for K=1,2,3,8,
   with kept event positions. Shows directly whether the ghost peak survives.
2. ghost_recovery.png — over many ghost pixels:
   (a) ghost-peak RECOVERY rate vs K (fraction of ghost returns with a kept event
       within +/-tol bins);
   (b) histogram of the ghost peak's HEIGHT-RANK among all peaks in its pixel
       (rank 1 = strongest). If ghosts are mostly rank 2-3, you need K>=2-3.

Run: PYTHONPATH=<repo>/src uv run python downstream/diag_ghost.py --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compression.event_extraction import extract_topk_events_batch
from compression.event_synthesis import synthesize_batch

OUT = "downstream/outputs/events/diag"
KS_PLOT = [1, 2, 3, 8]
KS_RECOV = [1, 2, 3, 4, 6, 8]
TOL = 3                      # bins: event counts as recovering a ghost peak if within +/-TOL


def load_frame(vpath, apath, fname):
    import blosc2
    v = blosc2.load_array(os.path.join(vpath, fname)).astype(np.float32)
    a = blosc2.load_array(os.path.join(apath, fname.replace("_voxel.b2", "_annotation_voxel.b2")))
    return v, a


def ghost_peak_t(wn_pix, ann_pix):
    """Bin of the brightest waveform sample inside the ghost-labelled region."""
    gb = np.where(ann_pix == 3)[0]
    if len(gb) == 0:
        return None
    return int(gb[np.argmax(wn_pix[gb])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="downstream/configs/evalA_split2_test.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_frames", type=int, default=6, help="frames to aggregate for stats")
    ap.add_argument("--n_examples", type=int, default=6)
    args = ap.parse_args()
    from hist_lidar.config import load_config_from_yaml
    cfg = load_config_from_yaml(args.config)
    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    vpath, apath = cfg.test_voxel_dirs[0], cfg.test_annotation_dirs[0]
    files = sorted(p for p in os.listdir(vpath) if p.endswith("_voxel.b2"))[: args.n_frames]

    # ---- aggregate ghost-peak recovery + rank over many ghost pixels ----
    recov = {k: [0, 0] for k in KS_RECOV}      # k -> [recovered, total]
    ranks = []                                  # height-rank of the ghost peak
    example_pool = []                           # (wn, ann, gt) for the example figure
    for fname in files:
        v, a = load_frame(vpath, apath, fname)
        T = v.shape[-1]
        flatv, flata = v.reshape(-1, T), a.reshape(-1, T)
        gpix = np.where((flata == 3).any(1) & (flatv.max(1) > 0))[0]
        if len(gpix) == 0:
            continue
        w = torch.from_numpy(flatv[gpix]).to(dev)
        mx = w.amax(1, keepdim=True)
        wn = torch.where(mx > 1e-6, w / mx, w)
        wn_np = wn.cpu().numpy()
        gts = np.array([ghost_peak_t(wn_np[i], flata[gpix[i]]) for i in range(len(gpix))])
        ok = gts != None                                              # noqa: E711
        wn, wn_np, gts, gpix = wn[ok], wn_np[ok], gts[ok].astype(int), gpix[ok]

        for k in KS_RECOV:
            ev, vm = extract_topk_events_batch(wn, k)
            ev_t = ev[..., 0].cpu().numpy()                          # (N,k)
            vmn = vm.cpu().numpy()
            for i in range(len(gts)):
                ts = ev_t[i][vmn[i]]
                hit = len(ts) and np.min(np.abs(ts - gts[i])) <= TOL
                recov[k][0] += int(bool(hit)); recov[k][1] += 1
            if k == 8:        # rank of ghost peak among all kept events (by height)
                ev_a = ev[..., 1].cpu().numpy()
                for i in range(len(gts)):
                    ts, ar = ev_t[i][vmn[i]], ev[..., 1].cpu().numpy()[i][vmn[i]]
                    if not len(ts):
                        continue
                    j = int(np.argmin(np.abs(ts - gts[i])))
                    if abs(ts[j] - gts[i]) <= TOL:
                        ranks.append(int((ar > ar[j]).sum()) + 1)    # 1 = strongest

        for i in range(len(gts)):
            example_pool.append((wn_np[i], flata[gpix[i]].copy(), gts[i]))

    # ---- figure 1: example ghost pixels, orig vs synth for several K ----
    sel = rng.choice(len(example_pool), min(args.n_examples, len(example_pool)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 8: "#d62728"}
    for ax, idx in zip(axes.ravel(), sel):
        wn_np, ann, gt = example_pool[idx]
        T = len(wn_np)
        ax.plot(wn_np, "k-", lw=1.2, label="orig", zorder=5)
        gb = np.where(ann == 3)[0]
        ax.axvspan(gb.min(), gb.max(), color="red", alpha=0.12, label="ghost region")
        ax.axvline(gt, color="red", ls="--", lw=1.0)
        wn_t = torch.from_numpy(wn_np[None]).to(dev)
        for k in KS_PLOT:
            ev, vm = extract_topk_events_batch(wn_t, k)
            rec = synthesize_batch(ev, vm, T=T, representation="taw")[0].cpu().numpy()
            kept = (np.abs(ev[0, :, 0].cpu().numpy()[vm[0].cpu().numpy()] - gt) <= TOL).any()
            ax.plot(rec, "-", color=colors[k], lw=1.0, alpha=0.7,
                    label=f"taw K={k} {'(ghost kept)' if kept else '(ghost LOST)'}")
        lo, hi = max(0, gt - 80), min(T, gt + 80)
        ax.set_xlim(lo, hi)
        ax.set_title(f"ghost peak @t={gt}", fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(fontsize=6.5)
    fig.suptitle("Ghost pixels: does the top-K event synthesis keep the ghost return?")
    plt.tight_layout(); plt.savefig(f"{OUT}/ghost_waveforms.png", dpi=130); plt.close()

    # ---- figure 2: recovery vs K + ghost-peak rank histogram ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ks = KS_RECOV
    rate = [recov[k][0] / max(1, recov[k][1]) for k in ks]
    a1.plot(ks, rate, "o-", color="#d62728", lw=2, ms=8)
    for k, r in zip(ks, rate):
        a1.annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 8), fontsize=9)
    a1.set_xlabel("K (events kept)"); a1.set_ylabel(f"ghost-peak recovery (within +/-{TOL} bins)")
    a1.set_title("Fraction of true ghost returns kept by top-K"); a1.set_ylim(0, 1.02); a1.grid(alpha=0.3)
    if ranks:
        mr = max(ranks)
        a2.hist(ranks, bins=np.arange(0.5, mr + 1.5), color="#9467bd", rwidth=0.85)
        a2.set_xlabel("height-rank of the ghost peak among detected peaks (1=strongest)")
        a2.set_ylabel("# ghost pixels"); a2.set_xticks(range(1, mr + 1))
        a2.set_title(f"Ghost peaks are mostly NOT the strongest return (median rank {int(np.median(ranks))})")
    plt.tight_layout(); plt.savefig(f"{OUT}/ghost_recovery.png", dpi=130); plt.close()

    print("ghost-peak recovery vs K:", {k: round(recov[k][0] / max(1, recov[k][1]), 3) for k in ks})
    if ranks:
        ranks = np.array(ranks)
        print(f"ghost-peak height-rank: median={int(np.median(ranks))}  "
              f"rank1={100*(ranks==1).mean():.0f}%  rank>=2={100*(ranks>=2).mean():.0f}%  "
              f"rank>=3={100*(ranks>=3).mean():.0f}%  (n={len(ranks)})")
    print("saved", f"{OUT}/ghost_waveforms.png", f"{OUT}/ghost_recovery.png")


if __name__ == "__main__":
    main()
