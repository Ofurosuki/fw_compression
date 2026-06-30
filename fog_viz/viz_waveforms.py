#!/usr/bin/env python
"""Visualize per-pixel full-waveform LiDAR returns from the Keio fog-chamber dataset,
overlaying the dense voxel annotation as colored class bands along the range (t) axis.

Goal: for randomly sampled pixels in each annotated scene, show *where along the waveform*
each class sits (i.e. which part of the return is fog vs. object), as colored axvspan bands.

Outputs: fog_viz/outputs/<scene>_fig<N>.png  (4x3 grid of waveforms per image).

Run:
  export PATH="$HOME/.local/bin:$PATH"
  uv run python fog_viz/viz_waveforms.py
"""
import os, glob, argparse
import numpy as np
import blosc2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/data3/user/ikeda/fog/keio_chamber"
ANN_DIR = f"{ROOT}/annotation_v1"
OUT = os.path.join(os.path.dirname(__file__), "outputs")
SCENES = ["place1", "place2_only_lidar", "scene1", "scene2", "scene3"]

# class -> (color, label). Confirmed by user 2026-06-29: 1=fog, 2=object, 3=others.
# AnnotationLabel (per user, same enum as ghost dataset + FOG):
#   0=UNKNOWN 1=OBJECT 2=GLASS 3=GHOST 4=FOG. FOG only labeled in scene3.
CLASS_STYLE = {
    1: ("tab:blue",   "class 1 = OBJECT"),
    2: ("tab:green",  "class 2 = GLASS"),
    3: ("tab:red",    "class 3 = GHOST"),
    4: ("tab:purple", "class 4 = FOG"),
}
CLASS_NAME = {1: "object", 2: "glass", 3: "ghost", 4: "fog"}

def load(path):
    s = blosc2.open(path)
    shape = s.vlmeta["__pack_tensor__"][1]
    dt = s.vlmeta["__pack_tensor__"][2]
    return np.frombuffer(s[:], dtype=np.dtype(dt)).reshape(shape)

def build_raw_index():
    idx = {}
    for p in glob.glob(f"{ROOT}/b2/**/*.b2", recursive=True):
        idx[os.path.basename(p)] = p
    return idx

def contiguous_runs(ts, cls):
    """Given sorted t indices `ts` with matching class `cls`, yield (t0, t1, class) spans."""
    runs = []
    if len(ts) == 0:
        return runs
    order = np.argsort(ts)
    ts, cls = ts[order], cls[order]
    s = 0
    for i in range(1, len(ts) + 1):
        if i == len(ts) or ts[i] != ts[i-1] + 1 or cls[i] != cls[s]:
            runs.append((int(ts[s]), int(ts[i-1]), int(cls[s])))
            s = i
    return runs

BAND_PAD = 2.0  # min half-width (range bins) so single-voxel labels read as a visible band

def draw_pixel(ax, raw, ann, h, w, focus_class):
    T = raw.shape[2]
    wf = raw[h, w]
    lab = ann[h, w]
    ax.plot(np.arange(T), wf, color="0.25", lw=0.9, zorder=3)
    sig_t = np.where(lab > 0)[0]
    for (t0, t1, c) in contiguous_runs(sig_t, lab[sig_t]):
        col = CLASS_STYLE.get(c, ("0.5", str(c)))[0]
        ax.axvspan(t0 - BAND_PAD, t1 + BAND_PAD, color=col, alpha=0.35, zorder=1)
    active = np.where(wf > 0.02 * max(wf.max(), 1))[0]
    hi = max(active.max() if len(active) else 100, sig_t.max() if len(sig_t) else 100)
    ax.set_xlim(0, min(T, hi + 40))
    present = sorted(set(lab[sig_t].tolist()))
    present_names = ",".join(CLASS_NAME.get(c, f"c{c}") for c in present)
    fcol = CLASS_STYLE[focus_class][0]
    ax.set_title(f"focus={CLASS_NAME[focus_class]} | (h={h},w={w}) [{present_names}]",
                 fontsize=9, color=fcol)
    ax.set_xlabel("range bin t"); ax.set_ylabel("intensity")
    ax.grid(alpha=0.25)

def plot_stratified(raw, ann, pools, title, out_path, ncol=4):
    """pools: dict class -> list of (h,w). One row per class, ncol samples per row."""
    classes = [c for c in sorted(pools) if pools.get(c)]
    nrow = len(classes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.0 * nrow), squeeze=False)
    for r, c in enumerate(classes):
        px = pools[c]
        for j in range(ncol):
            ax = axes[r][j]
            if j < len(px):
                draw_pixel(ax, raw, ann, px[j][0], px[j][1], c)
            else:
                ax.axis("off")
    present_styles = {c: CLASS_STYLE[c] for c in classes if c in CLASS_STYLE}
    handles = [Patch(facecolor=col, alpha=0.35, label=lbl) for col, lbl in present_styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    row_names = " / ".join(CLASS_NAME.get(c, f"c{c}") for c in classes)
    fig.suptitle(title + f"   [rows = {row_names}; pixels CONTAINING that class, all bands overlaid]",
                 y=1.015, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)

def sample_class_pixels(ann, cls, n, rng):
    """Sample up to n pixels that contain >=1 voxel of class `cls`."""
    hs, ws = np.where((ann == cls).any(axis=2))
    if len(hs) == 0:
        return []
    sel = rng.choice(len(hs), size=min(n, len(hs)), replace=False)
    return list(zip(hs[sel].tolist(), ws[sel].tolist()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs_per_scene", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4, help="samples per class row")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    raw_idx = build_raw_index()
    rng = np.random.default_rng(args.seed)

    for scene in SCENES:
        anns = sorted(glob.glob(f"{ANN_DIR}/{scene}/*_annotation.b2"))
        if not anns:
            print(f"[skip] {scene}: no annotations"); continue
        for fi in range(args.figs_per_scene):
            ann_path = anns[rng.integers(len(anns))]
            base = os.path.basename(ann_path).replace("_annotation.b2", ".b2")
            raw_path = raw_idx.get(base)
            if raw_path is None:
                print(f"[warn] no raw for {base}"); continue
            ann = load(ann_path)
            raw = load(raw_path)
            present = [int(c) for c in np.unique(ann) if c > 0]
            pools = {c: sample_class_pixels(ann, c, args.cols, rng) for c in present}
            if not any(pools.values()):
                print(f"[skip] {scene} frame has no labels"); continue
            frame_id = base.split("_")[-1].replace(".b2", "")
            title = f"{scene}  frame {frame_id}"
            out_path = os.path.join(OUT, f"{scene}_fig{fi+1}.png")
            plot_stratified(raw, ann, pools, title, out_path, ncol=args.cols)

if __name__ == "__main__":
    main()
