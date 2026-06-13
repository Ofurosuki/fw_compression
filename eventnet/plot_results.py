"""Aggregate sweep eval.json files -> ablation table + plots.

Plots (initial_plan.md): F1-mean vs K, glass F1 vs K, ghost F1 vs K (for tdtaw),
and a bar chart comparing feature modes at K=4. All numbers are the PAPER-
COMPLIANT peak-level F1 from ``evaluate.py``.

  PYTHONPATH=<repo>/src uv run python -m eventnet.plot_results \
      --sweep_root outputs/eventnet/sweep --out_dir outputs/eventnet/sweep/plots
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE_ORDER = ["t_only", "t_dt", "ta", "tdta", "taw", "tdtaw"]


def load(sweep_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(sweep_root, "*", "eval.json"))):
        r = json.load(open(f))
        rows.append(r)
    return rows


def fmt_table(rows):
    rows = sorted(rows, key=lambda r: (MODE_ORDER.index(r["feature_mode"])
                                       if r["feature_mode"] in MODE_ORDER else 99, r["K"]))
    lines = ["| feature_mode | K | object F1 | glass F1 | ghost F1 | F1-mean | event-F1 |",
             "| --- | --: | --: | --: | --: | --: | --: |"]
    for r in rows:
        lines.append(f"| {r['feature_mode']} | {r['K']} | {r['object_f1']:.3f} | "
                     f"{r['glass_f1']:.3f} | {r['ghost_f1']:.3f} | **{r['f1_mean']:.3f}** | "
                     f"{r.get('event_level_f1_mean', float('nan')):.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", default="outputs/eventnet/sweep")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.sweep_root, "plots")
    os.makedirs(out, exist_ok=True)

    rows = load(args.sweep_root)
    if not rows:
        print("no eval.json files found in", args.sweep_root)
        return
    table = fmt_table(rows)
    print(table)
    open(os.path.join(out, "table.md"), "w").write(table + "\n")

    by = {(r["feature_mode"], r["K"]): r for r in rows}

    # K sweep for tdtaw
    ks = sorted(k for (m, k) in by if m == "tdtaw")
    if len(ks) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, title in zip(
                axes, ["f1_mean", "glass_f1", "ghost_f1"],
                ["F1-mean vs K", "glass F1 vs K", "ghost F1 vs K"]):
            ax.plot(ks, [by[("tdtaw", k)][key] for k in ks], "o-")
            ax.set_xlabel("K"); ax.set_ylabel(key); ax.set_title("tdtaw: " + title)
            ax.set_xticks(ks); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "k_sweep.png"), dpi=120)
        plt.close(fig)
        print("saved k_sweep.png")

    # feature-mode bar chart at K=4
    modes = [m for m in MODE_ORDER if (m, 4) in by]
    if modes:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(modes))
        for off, cls, col in [(-0.27, "object_f1", "tab:green"), (0.0, "glass_f1", "tab:blue"),
                              (0.27, "ghost_f1", "tab:red")]:
            ax.bar([i + off for i in x], [by[(m, 4)][cls] for m in modes], 0.25,
                   label=cls.replace("_f1", ""), color=col)
        ax.plot(list(x), [by[(m, 4)]["f1_mean"] for m in modes], "k--o", label="F1-mean")
        ax.set_xticks(list(x)); ax.set_xticklabels(modes); ax.set_ylabel("peak-level F1")
        ax.set_title("Feature-mode ablation at K=4 (paper peak-level F1)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(out, "mode_bar_K4.png"), dpi=120)
        plt.close(fig)
        print("saved mode_bar_K4.png")


if __name__ == "__main__":
    main()
