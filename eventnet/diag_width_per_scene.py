"""Per-scene, per-class WIDTH (FWHM) distribution — split train vs test scenes.

Tests the hypothesis behind "width HURTS glass" in the retrain 2-seed result
(downstream/RETRAIN_RESULTS.md): is glass's pulse width a scene-non-transferable cue?
If glass-vs-object width separation SIGN-FLIPS across scenes (glass wider in some, narrower
in others), then a model leaning on w for glass learns a scene-specific shortcut → it helps
on train scenes but hurts on the 3 held-out test scenes. Same failure mode as behind_energy
(FW_Event_Net/SCENE_GEOMETRY.md), now for width.

Per scene we extract top-K events (eventnet.events.extract_frame_events; w = FWHM in bins,
col 2), label each at its peak bin, and report per-class width median [IQR] + the signed
AUC(glass vs object) = P(w_glass > w_object) (>0.5 = glass wider; <0.5 = sign-flip).

Usage:
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  uv-py eventnet/diag_width_per_scene.py --device cuda:2 --dirs_per_scene 2 --frames_per_dir 3
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from sklearn.metrics import roc_auc_score

from hist_lidar.preprocess.custom_blosc2 import load_blosc2
from eventnet import paths
from eventnet.events import extract_frame_events, assign_labels

TEST_SCENES = {"36build", "22build", "14build_7floor"}
SIG = {1: "object", 2: "glass", 3: "ghost"}


def scene_of(path: str) -> str:
    m = re.search(r"ghost_dataset/([^/]+)/", path)
    return m.group(1) if m else "?"


def collect():
    """{scene: list of (voxel_path, ann_path)} across all splits."""
    split = paths.load_split()
    by_scene = {}
    for sp in ("train", "val", "test"):
        d = split[sp]
        for vdir, adir in zip(d["voxel"], d["ann"]):
            by_scene.setdefault(scene_of(vdir), []).append((vdir, adir))
    return by_scene


def signed_auc(w_glass, w_obj):
    """P(w_glass > w_obj). >0.5 glass wider; <0.5 object wider (sign-flip)."""
    if len(w_glass) < 10 or len(w_obj) < 10:
        return float("nan")
    y = np.concatenate([np.ones(len(w_glass)), np.zeros(len(w_obj))])
    s = np.concatenate([w_glass, w_obj])
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--dirs_per_scene", type=int, default=2)
    ap.add_argument("--frames_per_dir", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    by_scene = collect()
    rows = []
    for scene, dirs in sorted(by_scene.items()):
        wcls = {1: [], 2: [], 3: []}
        for vdir, adir in dirs[: args.dirs_per_scene]:
            for vp, ap_ in paths.frame_files(vdir, adir)[: args.frames_per_dir]:
                vox = paths.apply_crop(load_blosc2(vp).astype(np.float32))
                ann = paths.apply_crop(load_blosc2(ap_))
                ev, valid = extract_frame_events(vox, device, k=args.k)
                lab = assign_labels(ann, ev, valid)
                w = ev[..., 2]                              # FWHM in bins
                v = valid.cpu().numpy() if torch.is_tensor(valid) else valid
                w = w.cpu().numpy() if torch.is_tensor(w) else w
                lab = lab.cpu().numpy() if torch.is_tensor(lab) else lab
                for c in (1, 2, 3):
                    wcls[c].append(w[(v) & (lab == c)])
        wcls = {c: (np.concatenate(v) if v else np.array([])) for c, v in wcls.items()}
        def med(c):
            a = wcls[c]; return (np.median(a), np.percentile(a, 25), np.percentile(a, 75), len(a)) if len(a) else (np.nan,)*3 + (0,)
        o, g, gh = med(1), med(2), med(3)
        auc_go = signed_auc(wcls[2], wcls[1])
        auc_gho = signed_auc(wcls[3], wcls[1])
        rows.append((scene, "TEST" if scene in TEST_SCENES else "train",
                     o, g, gh, auc_go, auc_gho))

    # ---- report ----
    print(f"\nWidth (FWHM, bins) per scene  [median (IQR), n]  | AUC = P(w_class > w_object)")
    print(f"{'scene':16} {'set':5} | {'object':14} {'glass':14} {'ghost':14} | "
          f"{'g|o':>6} {'gh|o':>6}")
    print("-" * 96)
    def fmt(m):
        return f"{m[0]:4.1f}({m[1]:.0f}-{m[2]:.0f}) n{m[3]}" if m[3] else "—"
    for scene, st, o, g, gh, ago, agho in rows:
        flag = " <FLIP" if (not np.isnan(ago) and ago < 0.5) else ""
        print(f"{scene:16} {st:5} | {fmt(o):14} {fmt(g):14} {fmt(gh):14} | "
              f"{ago:6.2f} {agho:6.2f}{flag}")

    aucs = [(st, ago) for _, st, *_, ago, _ in
            [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows] if not np.isnan(ago)]
    tr = [a for s, a in aucs if s == "train"]; te = [a for s, a in aucs if s == "TEST"]
    print("-" * 96)
    print(f"glass|object width AUC — train: mean {np.mean(tr):.2f} range [{min(tr):.2f},{max(tr):.2f}] "
          f"flips {sum(a<0.5 for a in tr)}/{len(tr)}")
    print(f"                        TEST : mean {np.mean(te):.2f} range [{min(te):.2f},{max(te):.2f}] "
          f"flips {sum(a<0.5 for a in te)}/{len(te)}")
    print("(>0.5 glass wider than object; <0.5 = sign-flip → width is a non-transferable glass cue)")


if __name__ == "__main__":
    main()


def plot(device, dirs_per_scene=2, frames_per_dir=3, k=8,
         out="downstream/outputs/diag/width_per_scene.png"):
    """Box plot of glass/object/ghost width per scene, train vs test."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_scene = collect()
    scenes = sorted(by_scene, key=lambda s: (s in TEST_SCENES, s))
    fig, axes = plt.subplots(1, len(scenes), figsize=(2.0*len(scenes), 4.2), sharey=True)
    col = {1: "tab:gray", 2: "tab:blue", 3: "tab:red"}
    for ax, scene in zip(axes, scenes):
        wcls = {1: [], 2: [], 3: []}
        for vdir, adir in by_scene[scene][:dirs_per_scene]:
            for vp, ap_ in paths.frame_files(vdir, adir)[:frames_per_dir]:
                vox = paths.apply_crop(load_blosc2(vp).astype(np.float32))
                ann = paths.apply_crop(load_blosc2(ap_))
                ev, valid = extract_frame_events(vox, device, k=k)
                lab = assign_labels(ann, ev, valid)
                w = ev[..., 2]
                w = w.cpu().numpy() if torch.is_tensor(w) else w
                v = valid.cpu().numpy() if torch.is_tensor(valid) else valid
                lab = lab.cpu().numpy() if torch.is_tensor(lab) else lab
                for c in (1, 2, 3):
                    wcls[c].append(w[v & (lab == c)])
        data = [np.concatenate(wcls[c]) if wcls[c] and len(np.concatenate(wcls[c])) else np.array([np.nan]) for c in (1,2,3)]
        # subsample for plotting
        data = [d[np.random.default_rng(0).choice(len(d), min(len(d),20000), replace=False)] if len(d)>1 else d for d in data]
        bp = ax.boxplot(data, labels=["obj","gls","gho"], showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], (1,2,3)): patch.set_facecolor(col[c]); patch.set_alpha(0.6)
        st = "TEST" if scene in TEST_SCENES else "train"
        ax.set_title(f"{scene}\n[{st}]", fontsize=8, color=("crimson" if st=="TEST" else "black"))
        ax.tick_params(labelsize=7); ax.set_ylim(2, 20)
        ax.axhline(np.nanmedian(data[0]), color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("width FWHM (bins)")
    fig.suptitle("Per-event width by class & scene (train vs TEST) — glass width sign-flips/shifts; ghost consistently narrower", fontsize=10)
    plt.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130); print("saved", out)


if __name__ == "__main__" and "--plot" in sys.argv:
    import torch as _t
    plot(_t.device("cuda:2"))
