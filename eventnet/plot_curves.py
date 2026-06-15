"""Learning-curve plots from train_log.json files (val F1 / per-class / loss vs epoch).

Produces a curated set of figures documenting the V1 -> V2 -> v2sa arc and the
feature/loss ablations. Outputs to FW_Event_Net/figs/curves/.

  PYTHONPATH=<repo>/src uv run python -m eventnet.plot_curves
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "downstream/outputs"
FIG = "FW_Event_Net/figs/curves"


def load(run_dir):
    d = json.load(open(os.path.join(OUT, run_dir, "train_log.json")))
    h = d["history"]
    ep = [r["epoch"] for r in h]
    return {
        "epoch": ep,
        "val_f1": [r["val_f1_mean"] for r in h],
        "glass": [r["per"]["glass"] for r in h],
        "ghost": [r["per"]["ghost"] for r in h],
        "object": [r["per"]["object"] for r in h],
        "loss": [r["loss"] for r in h],
        "best": d["best_val_f1_mean"],
    }


def panel(ax, series, key, title, ylabel="val F1-mean"):
    for label, run in series:
        try:
            c = load(run)
        except FileNotFoundError:
            continue
        ax.plot(c["epoch"], c[key], lw=1.4, label=f"{label}")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)


def fig_multi(series, fname, suptitle, keys=(("val_f1", "val F1-mean"),
                                            ("glass", "glass F1"),
                                            ("ghost", "ghost F1"))):
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.2))
    if len(keys) == 1:
        axes = [axes]
    for ax, (k, yl) in zip(axes, keys):
        panel(ax, series, k, yl, ylabel=yl)
    fig.suptitle(suptitle)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, fname), dpi=120)
    plt.close(fig)
    print("saved", os.path.join(FIG, fname))


def main():
    os.makedirs(FIG, exist_ok=True)

    # 1. V1 feature ablation (K=4)
    fig_multi([("t_only", "eventnet_sweep/t_only_K4"), ("t_dt", "eventnet_sweep/t_dt_K4"),
               ("ta", "eventnet_sweep/ta_K4"), ("taw", "eventnet_sweep/taw_K4"),
               ("tdta", "eventnet_sweep/tdta_K4"), ("tdtaw", "eventnet_sweep/tdtaw_K4")],
              "v1_feature_ablation.png", "V1 feature ablation (K=4) — val curves")

    # 2. V1 K-sweep (tdtaw)
    fig_multi([("K=1", "eventnet_sweep/tdtaw_K1"), ("K=2", "eventnet_sweep/tdtaw_K2"),
               ("K=4", "eventnet_sweep/tdtaw_K4"), ("K=8", "eventnet_sweep/tdtaw_K8")],
              "v1_ksweep.png", "V1 tdtaw K-sweep — val curves")

    # 3. V2 feature ablation (seed42)
    fig_multi([("ta", "eventnet_sweep_v2/ta_K4"), ("taw", "eventnet_sweep_v2/taw_K4"),
               ("tdta", "eventnet_sweep_v2/tdta_K4"), ("tdtaw", "eventnet_sweep_v2/tdtaw_K4")],
              "v2_feature_ablation.png", "V2 feature ablation (K=4, seed42) — val curves")

    # 4. Architecture comparison on taw (V1 vs V2 vs v2sa-fair)
    fig_multi([("V1 taw", "eventnet_sweep/taw_K4"),
               ("V2 taw (s42)", "eventnet_sweep_v2/taw_K4"),
               ("V2 base fair (s42)", "eventnet_v2sa2/base_s42"),
               ("v2sa fair (s42)", "eventnet_v2sa2/v2sa_s42")],
              "arch_compare_taw.png", "Architecture comparison on taw — val curves")

    # 5. Spatial-attention fair retest (base vs v2sa, 2 seeds) + train loss
    fig_multi([("base s42", "eventnet_v2sa2/base_s42"), ("base s43", "eventnet_v2sa2/base_s43"),
               ("v2sa s42", "eventnet_v2sa2/v2sa_s42"), ("v2sa s43", "eventnet_v2sa2/v2sa_s43")],
              "v2sa_fair_retest.png", "Spatial-attention fair retest (lr5e-4/warmup5/50ep)",
              keys=(("val_f1", "val F1-mean"), ("ghost", "ghost F1"), ("loss", "train loss")))

    # 6. Feature/training experiments that did not transfer (val F1, seed42)
    fig_multi([("taw (ctrl)", "eventnet_sweep_v3/taw_K4"),
               ("+decomp tawi", "eventnet_sweep_v3/tawi_K4"),
               ("+behind_E taE", "eventnet_sweep_E/taE_K4"),
               ("+NeRF tawT", "eventnet_v4/tawT_s42"),
               ("V-REx b10", "eventnet_vrex/vrex_b10_s42"),
               ("glass-wt x3", "eventnet_loss/glassw3_s42")],
              "negative_experiments.png", "Feature/training experiments (val curves, seed42)",
              keys=(("val_f1", "val F1-mean"), ("glass", "glass F1")))


if __name__ == "__main__":
    main()
