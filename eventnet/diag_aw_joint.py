"""Why does `ta` (t,a) nearly match `taw` (t,a,w)? — per-class joint (a,w) analysis.

Hypothesis (user): if width `w` is a (near-)deterministic function of amplitude `a`
*within each class*, then a multi-echo (t,a) representation already encodes what `w`
would tell the model -> width is redundant -> ta ~= taw. And if glass breaks that
relationship (different a-w coupling, or a non-transferable width cue), it explains the
per-class width effect (helps ghost/object, hurts glass; net ~= 0; RETRAIN_RESULTS.md).

Key structural fact this script makes precise: `a` is the **per-ray max-normalised** peak
height, so the primary return (object/glass) saturates at a~=1.0. So per-event amplitude
cannot separate object from glass -- only flags the dim secondary (ghost, a~=0.26). We test
whether width is in the *same* boat, and whether (a,w) is redundant for class discrimination.

Outputs (printed tables + figure downstream/outputs/diag/aw_joint.png):
  (A) per-class marginal a, w medians/IQR  (train vs test scenes)
  (B) per-class Spearman corr(a,w) + R^2(w ~ a)  -- is w predictable from a within class?
  (C) E[w | a-decile] per class               -- does w separate classes at MATCHED a?
  (D) per class-PAIR separability: AUC from a alone / w alone / (a,w) logistic,
      in-scene and train->test transfer  -- the direct "does width add over amplitude" test.

Usage:
  export PATH="$HOME/.local/bin:$PATH"
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  uv run python eventnet/diag_aw_joint.py --device cuda:2 --dirs_per_scene 2 --frames_per_dir 4
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hist_lidar.preprocess.custom_blosc2 import load_blosc2
from eventnet import paths
from eventnet.events import extract_frame_events, assign_labels

TEST_SCENES = {"36build", "22build", "14build_7floor"}
SIG = {1: "object", 2: "glass", 3: "ghost"}


def scene_of(path: str) -> str:
    m = re.search(r"ghost_dataset/([^/]+)/", path)
    return m.group(1) if m else "?"


def collect_dirs():
    split = paths.load_split()
    by_scene = {}
    for sp in ("train", "val", "test"):
        d = split[sp]
        for vdir, adir in zip(d["voxel"], d["ann"]):
            by_scene.setdefault(scene_of(vdir), []).append((vdir, adir))
    return by_scene


def gather(device, dirs_per_scene, frames_per_dir, k):
    """Return a flat record array of valid events: t, a, w, label(1/2/3), scene, is_test."""
    by_scene = collect_dirs()
    recs = {"t": [], "a": [], "w": [], "lab": [], "scene": [], "test": []}
    for scene, dirs in sorted(by_scene.items()):
        for vdir, adir in dirs[:dirs_per_scene]:
            for vp, ap_ in paths.frame_files(vdir, adir)[:frames_per_dir]:
                vox = paths.apply_crop(load_blosc2(vp).astype(np.float32))
                ann = paths.apply_crop(load_blosc2(ap_))
                ev, valid = extract_frame_events(vox, device, k=k)
                lab = assign_labels(ann, ev, valid)
                m = valid & np.isin(lab, (1, 2, 3))
                recs["t"].append(ev[..., 0][m])
                recs["a"].append(ev[..., 1][m])
                recs["w"].append(ev[..., 2][m])
                recs["lab"].append(lab[m].astype(np.int8))
                recs["scene"].append(np.full(m.sum(), scene, dtype=object))
                recs["test"].append(np.full(m.sum(), scene in TEST_SCENES))
    out = {key: np.concatenate(v) for key, v in recs.items()}
    return out


def fmt_iqr(x):
    if len(x) < 5:
        return "—"
    return f"{np.median(x):4.1f}[{np.percentile(x,25):.1f}-{np.percentile(x,75):.1f}]"


def report_marginals(R):
    print("\n=== (A) per-class marginal a / w  [median (IQR), n] ===")
    print(f"{'set':6} {'class':7} | {'a (max-norm height)':22} {'w (FWHM bins)':18} {'n':>7}")
    print("-" * 70)
    for st, name in ((False, "train"), (True, "TEST")):
        for c in (1, 2, 3):
            m = (R["test"] == st) & (R["lab"] == c)
            a, w = R["a"][m], R["w"][m]
            print(f"{name:6} {SIG[c]:7} | {fmt_iqr(a):22} {fmt_iqr(w):18} {m.sum():7d}")
    # fraction of object/glass events with a saturated near 1.0
    print("\n  amplitude saturation (a >= 0.98): "
          + ", ".join(f"{SIG[c]} {np.mean(R['a'][R['lab']==c] >= 0.98):.0%}" for c in (1, 2, 3)))


def report_corr(R):
    print("\n=== (B) within-class a-w coupling: Spearman rho(a,w) + R^2(w ~ a, binned) ===")
    print("  (high |rho| / R^2 => w is predictable from a within the class => w redundant given a)")
    print(f"{'class':7} | {'rho(a,w)':>9} {'R^2(w~a)':>9}   reading")
    print("-" * 60)
    for c in (1, 2, 3):
        m = R["lab"] == c
        a, w = R["a"][m], R["w"][m]
        if len(a) < 50:
            print(f"{SIG[c]:7} |   (n<50)"); continue
        rho = spearmanr(a, w).correlation
        # R^2 of E[w|a] via deciles of a (nonparametric, captures monotone+curved coupling)
        order = np.argsort(a)
        nb = 10
        idx = np.array_split(order, nb)
        wmean_tot = w.mean()
        ss_res = sum(((w[i] - w[i].mean()) ** 2).sum() for i in idx if len(i))
        ss_tot = ((w - wmean_tot) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"{SIG[c]:7} | {rho:9.3f} {r2:9.3f}")


def report_cond_width(R):
    """E[w | a-decile] per class -- does width separate classes AT MATCHED amplitude?"""
    print("\n=== (C) E[w | a-bin] per class  (does w separate classes at matched a?) ===")
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.001])
    hdr = "  ".join(f"[{edges[i]:.2f},{edges[i+1]:.2f})" for i in range(len(edges) - 1))
    print(f"{'class':7} | {hdr}")
    print("-" * 78)
    for c in (1, 2, 3):
        m = R["lab"] == c
        a, w = R["a"][m], R["w"][m]
        cells = []
        for i in range(len(edges) - 1):
            sel = (a >= edges[i]) & (a < edges[i + 1])
            cells.append(f"{np.median(w[sel]):5.1f}" if sel.sum() >= 20 else "   . ")
        print(f"{SIG[c]:7} | " + "       ".join(cells))
    print("  (if object & glass rows overlap at every a-bin => width can't split obj/glass even given a)")


def pair_auc(R, c_pos, c_neg):
    """Per-scene + train->test AUC for separating c_pos vs c_neg from a / w / (a,w)."""
    name = f"{SIG[c_pos]} vs {SIG[c_neg]}"
    print(f"\n  -- {name} --   AUC>0.5 => {SIG[c_pos]} has higher score; "
          f"for (a,w) logistic, |AUC-0.5| = separability")
    print(f"  {'scene':16} {'set':5} | {'AUC(a)':>7} {'AUC(w)':>7} {'AUC(a,w)':>9} {'lift_w':>7}")
    scenes = sorted(set(R["scene"]), key=lambda s: (s in TEST_SCENES, s))
    for scene in scenes:
        m = (R["scene"] == scene) & np.isin(R["lab"], (c_pos, c_neg))
        if m.sum() < 40:
            continue
        y = (R["lab"][m] == c_pos).astype(int)
        if y.sum() < 15 or (1 - y).sum() < 15:
            continue
        a, w = R["a"][m], R["w"][m]
        auc_a = roc_auc_score(y, a)
        auc_w = roc_auc_score(y, w)
        X = np.column_stack([a, w])
        lr = LogisticRegression(max_iter=500).fit(X, y)
        auc_aw = roc_auc_score(y, lr.decision_function(X))
        lift = auc_aw - max(auc_a, 1 - auc_a)  # gain of (a,w) over best single-sign a
        st = "TEST" if scene in TEST_SCENES else "train"
        flagw = " <Wflip" if (auc_w < 0.5) else ""
        print(f"  {scene:16} {st:5} | {auc_a:7.2f} {auc_w:7.2f} {auc_aw:9.2f} {lift:+7.2f}{flagw}")
    # transfer: fit on pooled train, eval per test scene (a vs a,w)
    tr = (~R["test"]) & np.isin(R["lab"], (c_pos, c_neg))
    ytr = (R["lab"][tr] == c_pos).astype(int)
    if ytr.sum() >= 20 and (1 - ytr).sum() >= 20:
        lr_a = LogisticRegression(max_iter=500).fit(R["a"][tr][:, None], ytr)
        lr_aw = LogisticRegression(max_iter=500).fit(np.column_stack([R["a"][tr], R["w"][tr]]), ytr)
        print(f"  {'TRANSFER train->':16} {'TEST':5} | (fit on train, eval each test scene)")
        for scene in [s for s in scenes if s in TEST_SCENES]:
            m = (R["scene"] == scene) & np.isin(R["lab"], (c_pos, c_neg))
            if m.sum() < 40:
                continue
            y = (R["lab"][m] == c_pos).astype(int)
            if y.sum() < 15 or (1 - y).sum() < 15:
                continue
            aa = roc_auc_score(y, lr_a.decision_function(R["a"][m][:, None]))
            aaw = roc_auc_score(y, lr_aw.decision_function(np.column_stack([R["a"][m], R["w"][m]])))
            print(f"  {scene:16} {'TEST':5} | {aa:7.2f} {'':7} {aaw:9.2f} {aaw-aa:+7.2f}"
                  f"   (w lift on held-out)")


def report_pairs(R):
    print("\n=== (D) class-PAIR separability from a / w / (a,w) -- the direct ta-vs-taw test ===")
    pair_auc(R, 1, 2)   # object vs glass  (the pair amplitude can't split: both a~=1)
    pair_auc(R, 1, 3)   # object vs ghost  (ghost dim+narrow: is w redundant with a?)
    pair_auc(R, 2, 3)   # glass vs ghost


def make_figure(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {1: "tab:green", 2: "tab:blue", 3: "tab:red"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # marginals a
    for c in (1, 2, 3):
        a = R["a"][R["lab"] == c]
        ax[0, 0].hist(a, bins=50, range=(0, 1.05), density=True, histtype="step",
                      lw=2, color=col[c], label=SIG[c])
    ax[0, 0].set(title="(a) amplitude marginal (per-ray max-norm)", xlabel="a (height)",
                 ylabel="density"); ax[0, 0].legend()

    # marginals w
    for c in (1, 2, 3):
        w = R["w"][R["lab"] == c]
        ax[0, 1].hist(w, bins=40, range=(0, 25), density=True, histtype="step",
                      lw=2, color=col[c], label=SIG[c])
    ax[0, 1].set(title="(b) width marginal", xlabel="w (FWHM bins)", ylabel="density")
    ax[0, 1].legend()

    # joint a-w: per-class median w over a-bins (conditional width curves) + IQR band
    edges = np.linspace(0, 1.0, 11)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for c in (1, 2, 3):
        m = R["lab"] == c
        a, w = R["a"][m], R["w"][m]
        med, lo, hi = [], [], []
        for i in range(len(edges) - 1):
            sel = (a >= edges[i]) & (a < edges[i + 1])
            if sel.sum() >= 20:
                med.append(np.median(w[sel])); lo.append(np.percentile(w[sel], 25))
                hi.append(np.percentile(w[sel], 75))
            else:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        med, lo, hi = map(np.array, (med, lo, hi))
        ax[1, 0].plot(ctr, med, "-o", color=col[c], lw=2, label=SIG[c])
        ax[1, 0].fill_between(ctr, lo, hi, color=col[c], alpha=0.15)
    ax[1, 0].set(title="(c) E[w | a] per class  (overlap => w redundant given a)",
                 xlabel="a (height)", ylabel="median w (FWHM)"); ax[1, 0].legend()

    # 2D hexbin density a vs w, all classes overlaid as cont scatter sample
    rng = np.random.default_rng(0)
    for c in (1, 2, 3):
        m = R["lab"] == c
        a, w = R["a"][m], R["w"][m]
        if len(a) > 4000:
            i = rng.choice(len(a), 4000, replace=False); a, w = a[i], w[i]
        ax[1, 1].scatter(a, w, s=4, alpha=0.15, color=col[c], label=SIG[c])
    ax[1, 1].set(title="(d) a vs w scatter", xlabel="a (height)", ylabel="w (FWHM)",
                 ylim=(0, 25)); ax[1, 1].legend()

    fig.suptitle("Per-class joint (a, w) — why ta ≈ taw", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print("\nsaved figure:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--dirs_per_scene", type=int, default=2)
    ap.add_argument("--frames_per_dir", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="downstream/outputs/diag/aw_joint.png")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"gathering events: dirs/scene={args.dirs_per_scene} frames/dir={args.frames_per_dir} k={args.k}")
    R = gather(device, args.dirs_per_scene, args.frames_per_dir, args.k)
    print(f"total valid signal events: {len(R['lab'])} "
          f"(obj {np.sum(R['lab']==1)}, glass {np.sum(R['lab']==2)}, ghost {np.sum(R['lab']==3)})")

    report_marginals(R)
    report_corr(R)
    report_cond_width(R)
    report_pairs(R)
    make_figure(R, args.out)


if __name__ == "__main__":
    main()
