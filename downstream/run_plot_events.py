"""Aggregate the top-K event sweep -> tables + plots, vs the AE baselines.

Outputs (downstream/outputs/events/):
  summary.txt         full event table (representation, K, dim, ratio, per-class F1)
  f1_vs_ratio.png     event reprs + AE baselines + full-waveform line, F1 vs ratio
  f1_vs_k.png         per-class F1 vs K for each representation
  compare.txt         headline comparison table (events vs best AE vs full)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = "downstream/outputs/events"
SWEEP = "downstream/outputs/sweep"
SWEEP_AH = "downstream/outputs/sweep_ah"
T, P = 700, 16
N_PARAMS = {"t": 1, "ta": 2, "tw": 2, "taw": 3, "taw_bg": 3}


def load(path):
    d = json.load(open(path))
    return d


def ae_ratio(meta):
    if meta is None:
        return None
    return (P * T / meta["K"]) if "P" in meta else (T / meta["K"])


# ---- collect event rows ----
ev_rows = []
base = None
for f in sorted(glob.glob(f"{EV}/*.json")):
    d = load(f)
    if d["compress"] == "none":
        base = d
        continue
    if d["compress"] != "event":
        continue
    m = d["ae_meta"]
    ev_rows.append({
        "repr": m["representation"], "K": m["k"], "dim": m["dim"],
        "ratio": T / m["dim"], "f1": d["macro_f1"],
        **{f"f1_{k}": v for k, v in d["per_class_f1"].items()},
    })

# ---- collect AE baselines (base loss + anti-hallucination) ----
def ae_rows(folder, label):
    out = []
    for f in sorted(glob.glob(f"{folder}/*.json")):
        d = load(f)
        if d["compress"] != "ae":
            continue
        out.append({"label": label, "tag": os.path.basename(f)[:-5],
                    "ratio": ae_ratio(d["ae_meta"]), "f1": d["macro_f1"],
                    **{f"f1_{k}": v for k, v in d["per_class_f1"].items()}})
    return out


ae_base = ae_rows(SWEEP, "base") if os.path.isdir(SWEEP) else []
ae_ah = ae_rows(SWEEP_AH, "ah") if os.path.isdir(SWEEP_AH) else []

# ---- table ----
hdr = f"{'repr':6s} {'K':>2s} {'dim':>4s} {'ratio':>7s} {'object':>7s} {'glass':>7s} {'ghost':>7s} {'F1-mean':>8s}"
lines = [hdr, "-" * len(hdr)]
if base:
    bf = base["per_class_f1"]
    lines.append(f"{'FULL':6s} {'-':>2s} {700:>4d} {'1x':>7s} {bf['object']:7.3f} "
                 f"{bf['glass']:7.3f} {bf['ghost']:7.3f} {base['macro_f1']:8.3f}")
for r in sorted(ev_rows, key=lambda r: (r["repr"], r["K"])):
    lines.append(f"{r['repr']:6s} {r['K']:>2d} {r['dim']:>4d} {r['ratio']:6.0f}x "
                 f"{r['f1_object']:7.3f} {r['f1_glass']:7.3f} {r['f1_ghost']:7.3f} {r['f1']:8.3f}")
print("\n".join(lines))
open(f"{EV}/summary.txt", "w").write("\n".join(lines) + "\n")

# ---- plot 1: F1-mean vs compression ratio (events + AE baselines) ----
fig, ax = plt.subplots(figsize=(9, 6))
colors = {"t": "#1f77b4", "ta": "#ff7f0e", "tw": "#2ca02c", "taw": "#d62728"}
for rp in ["t", "ta", "tw", "taw"]:
    pts = sorted([(r["ratio"], r["f1"]) for r in ev_rows if r["repr"] == rp])
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[rp],
                lw=2, ms=7, label=f"event {rp}")


def ae_series(rows, pred):
    pts = sorted([(r["ratio"], r["f1"]) for r in rows if pred(r) and r["ratio"]])
    return [p[0] for p in pts], [p[1] for p in pts]


for rows, ls, mk in [(ae_base, "--", "d"), (ae_ah, ":", "s")]:
    if not rows:
        continue
    suf = rows[0]["label"]
    x, y = ae_series(rows, lambda r: r["tag"].startswith("1d_ll"))
    if x:
        ax.plot(x, y, mk + ls, color="#7f7f7f", lw=1.3, ms=6, alpha=0.8, label=f"AE 1D ({suf})")
    x, y = ae_series(rows, lambda r: r["tag"].startswith("sp_"))
    if x:
        ax.plot(x, y, mk + ls, color="#9467bd", lw=1.3, ms=6, alpha=0.8, label=f"AE spatial ({suf})")
if base:
    ax.axhline(base["macro_f1"], color="k", ls="-", lw=1.2, label=f"full waveform ({base['macro_f1']:.3f})")
ax.set_xscale("log", base=2)
ax.set_xlabel("compression ratio (T / dim)")
ax.set_ylabel("downstream F1-mean (object/glass/ghost)")
ax.set_title("Top-K transport events vs AE compression (FWL-ToPM, split2 test)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{EV}/f1_vs_ratio.png", dpi=140)
plt.close()

# ---- plot 2: per-class F1 vs K, one panel per class ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, cls in zip(axes, ["object", "glass", "ghost"]):
    for rp in ["t", "ta", "tw", "taw"]:
        pts = sorted([(r["K"], r[f"f1_{cls}"]) for r in ev_rows if r["repr"] == rp])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[rp],
                    lw=2, ms=6, label=f"event {rp}")
    if base:
        ax.axhline(base["per_class_f1"][cls], color="k", ls="-", lw=1.0,
                   label="full waveform")
    ax.set_title(f"{cls} F1 vs K")
    ax.set_xlabel("K (events kept)")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("F1")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{EV}/f1_vs_k.png", dpi=140)
plt.close()

# ---- compare.txt: events at matched ratios vs best AE ----
clines = ["Top-K events vs AE baselines (F1-mean, split2 test, divide=3)", ""]
if base:
    clines.append(f"full waveform (no compression):  {base['macro_f1']:.3f}")
best_ev = max(ev_rows, key=lambda r: r["f1"]) if ev_rows else None
if best_ev:
    clines.append(f"best event config: {best_ev['repr']} K={best_ev['K']} "
                  f"(dim={best_ev['dim']}, {best_ev['ratio']:.0f}x)  ->  {best_ev['f1']:.3f}")
for rows, name in [(ae_base, "AE base"), (ae_ah, "AE anti-halluc")]:
    if rows:
        b = max(rows, key=lambda r: r["f1"])
        clines.append(f"best {name}: {b['tag']} ({b['ratio']:.0f}x)  ->  {b['f1']:.3f}")
open(f"{EV}/compare.txt", "w").write("\n".join(clines) + "\n")
print("\n".join(clines))
print("\nsaved", f"{EV}/summary.txt", f"{EV}/f1_vs_ratio.png", f"{EV}/f1_vs_k.png", f"{EV}/compare.txt")
