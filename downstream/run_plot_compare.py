"""Overlay downstream F1-vs-ratio for the non-AH vs anti-hallucination AEs."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

T, P = 700, 16
DIRS = {"base": "downstream/outputs/sweep", "AH (bg5/fp0.5)": "downstream/outputs/sweep_ah"}


def ratio(meta):
    if meta is None:
        return None
    return (P * T / meta["K"]) if "P" in meta else (T / meta["K"])


def load(d):
    rows = {}
    for f in glob.glob(f"{d}/*.json"):
        x = json.load(open(f))
        rows[os.path.basename(f)[:-5]] = x
    return rows


data = {k: load(v) for k, v in DIRS.items()}

# ---- comparison table ----
hdr = f"{'tag':12s} {'ratio':>6s} {'F1 base':>8s} {'F1 AH':>8s} {'Δ':>7s}"
print(hdr); print("-" * len(hdr))
lines = [hdr, "-" * len(hdr)]
tags = sorted(set(data["base"]) | set(data["AH (bg5/fp0.5)"]),
              key=lambda t: (not t.startswith("1d"), t))
for tag in tags:
    b = data["base"].get(tag)
    a = data["AH (bg5/fp0.5)"].get(tag)
    if not b or not a:
        continue
    rt = ratio(b.get("ae_meta"))
    rts = f"{rt:.0f}x" if rt else "-"
    d = a["macro_f1"] - b["macro_f1"]
    line = f"{tag:12s} {rts:>6s} {b['macro_f1']:8.3f} {a['macro_f1']:8.3f} {d:+7.3f}"
    print(line); lines.append(line)
open("downstream/outputs/sweep_ah/compare.txt", "w").write("\n".join(lines) + "\n")

# ---- overlay plot ----
fig, ax = plt.subplots(figsize=(8.5, 5.5))
styles = {"1d_ll": ("#d62728", "1D learnable"), "sp": ("#9467bd", "spatial 4x4")}
for label, rows in data.items():
    dashed = label.startswith("AH")
    for pref, (col, name) in styles.items():
        pts = sorted([(ratio(r["ae_meta"]), r["macro_f1"]) for k, r in rows.items()
                      if k.startswith(pref) and r.get("ae_meta")], key=lambda x: x[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ("--" if dashed else "-") + ("*" if pref == "sp" else "o"),
                color=col, lw=2, ms=(12 if pref == "sp" else 7), alpha=0.85,
                label=f"{name} [{'AH' if dashed else 'base'}]")
base_none = data["base"].get("none", {}).get("macro_f1")
if base_none:
    ax.axhline(base_none, color="k", ls=":", lw=1.3, label=f"no compression ({base_none:.3f})")
ax.set_xscale("log", base=2)
ax.set_xlabel("compression ratio (T/K, per-pixel-equivalent)")
ax.set_ylabel("downstream F1-mean (object/glass/ghost)")
ax.set_title("Downstream F1: baseline loss vs anti-hallucination loss (FWL-ToPM, split2)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2)
ax.xaxis.set_major_locator(FixedLocator([5.5, 11, 22, 44, 88]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}x"))
plt.tight_layout()
plt.savefig("downstream/outputs/sweep_ah/f1_compare.png", dpi=140)
print("\nsaved downstream/outputs/sweep_ah/compare.txt and f1_compare.png")
