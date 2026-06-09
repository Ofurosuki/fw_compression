"""Aggregate downstream sweep JSONs -> summary table + F1-vs-compression-ratio plot."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "downstream/outputs/sweep"
T, P = 700, 16


def ratio(meta):
    if meta is None:
        return None
    if "P" in meta:                      # spatial: per-pixel-equiv ratio
        return P * T / meta["K"]
    return T / meta["K"]


rows = []
for f in sorted(glob.glob(f"{OUT}/*.json")):
    d = json.load(open(f))
    tag = os.path.basename(f)[:-5]
    rows.append({
        "tag": tag, "compress": d["compress"], "meta": d.get("ae_meta"),
        "ratio": ratio(d.get("ae_meta")), "f1": d["macro_f1"],
        **{f"f1_{k}": v for k, v in d["per_class_f1"].items()},
    })

base = next((r["f1"] for r in rows if r["compress"] == "none"), None)

# ---- table ----
rows_sorted = sorted(rows, key=lambda r: (r["compress"] == "none" and -1 or r["ratio"] or 0))
hdr = f"{'tag':16s} {'ratio':>7s} {'F1-mean':>8s} {'object':>7s} {'glass':>7s} {'ghost':>7s}"
print(hdr); print("-" * len(hdr))
lines = [hdr, "-" * len(hdr)]
for r in rows_sorted:
    rt = f"{r['ratio']:.0f}x" if r["ratio"] else "-"
    line = (f"{r['tag']:16s} {rt:>7s} {r['f1']:8.3f} "
            f"{r.get('f1_object', 0):7.3f} {r.get('f1_glass', 0):7.3f} {r.get('f1_ghost', 0):7.3f}")
    print(line); lines.append(line)
open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")

# ---- plot: F1 vs compression ratio ----
def series(pred):
    pts = sorted([(r["ratio"], r["f1"]) for r in rows if pred(r) and r["ratio"]], key=lambda x: x[0])
    return [p[0] for p in pts], [p[1] for p in pts]

fig, ax = plt.subplots(figsize=(8, 5.5))
x, y = series(lambda r: r["tag"].startswith("1d_ll"))
ax.plot(x, y, "o-", color="#d62728", lw=2, ms=8, label="1D learnable_linear")
x, y = series(lambda r: r["tag"].startswith("sp_"))
ax.plot(x, y, "*-", color="#9467bd", lw=2, ms=13, label="spatial 4x4")
x, y = series(lambda r: r["tag"].startswith("1d_cb"))
ax.plot(x, y, "d--", color="#7f7f7f", lw=1.5, ms=7, label="1D coarse_binning (naive)")
if base:
    ax.axhline(base, color="k", ls=":", lw=1.5, label=f"no compression ({base:.3f})")
ax.set_xscale("log", base=2)
ax.set_xlabel("compression ratio (T/K, per-pixel-equivalent)")
ax.set_ylabel("downstream F1-mean (object/glass/ghost)")
ax.set_title("Ghost-FWL downstream F1 vs waveform compression ratio (FWL-ToPM, split2 test)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
from matplotlib.ticker import FixedLocator, FuncFormatter
xs = [5.5, 11, 22, 44, 88]
ax.xaxis.set_major_locator(FixedLocator(xs))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}x"))
plt.tight_layout()
plt.savefig(f"{OUT}/f1_vs_ratio.png", dpi=140)
print("\nsaved", f"{OUT}/summary.txt", "and", f"{OUT}/f1_vs_ratio.png")
