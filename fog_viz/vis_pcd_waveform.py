#!/usr/bin/env python
"""Standalone (.rrd) fog-chamber viewer: GT point cloud + per-pixel full-waveform plots.

Standalone, no extra install, no notebook callbacks. A BarChartView is REQUIRED to render a
waveform (the selection panel only shows raw component values, not a chart), and a view cannot
follow click-selection without notebook callbacks. So we drive ONE fixed BarChart entity via a
`pixel` timeline:
  - full GT cloud      -> scene/ground_truth     (context, colored by class, static)
  - current pixel      -> inspect/highlight      (white point in 3D, per `pixel` step)
  - current waveform   -> inspect/waveform        (BarChart, per `pixel` step)
  - current pixel info -> inspect/info            (TextDocument: coords + labeled bins)
  - blueprint: [ 3D view | BarChartView + info ] side by side.
Interaction: SCRUB the `pixel` timeline (bottom panel) to step through sampled pixels (ordered
by class). The white 3D point shows WHERE the current pixel is; the bar chart + info update in
sync. Sampled `--per_class` pixels per class.

Coordinate transform is FIXED (user-approved): geom=centered, FOV 90x26 deg, bin_to_m=0.15.

Run:
  export PATH="$HOME/.local/bin:$PATH"
  uv run python fog_viz/vis_pcd_waveform.py --scene scene2 --per_class 300
  # then locally:  rerun fog_viz/outputs/pcd_wf_scene2.rrd
"""
import argparse, glob, os
import numpy as np
import blosc2
import rerun as rr
import rerun.blueprint as rrb

ROOT = "/data3/user/ikeda/fog/keio_chamber"
ANN_DIR = f"{ROOT}/annotation_v1"
OUT = os.path.join(os.path.dirname(__file__), "outputs")
W_RAW, H_RAW = 400, 512

COLOR = {0: (128, 128, 128), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0), 4: (255, 255, 0)}
NAME = {0: "unknown", 1: "object", 2: "glass", 3: "ghost", 4: "fog"}

# FIXED geometry
BIN_TO_M = 0.15
FOV_H, FOV_V = 90.0, 26.0


def load(p):
    return blosc2.load_array(p)


# scene -> clear (no-fog) reference capture dir under b2/20260619
SCENE_GT_DIR = {
    "scene1": f"{ROOT}/b2/20260619/scene1-gt",
    "scene2": f"{ROOT}/b2/20260619/scene2_gt",
    "scene3": f"{ROOT}/b2/20260619/scene3-gt",
    "place1": f"{ROOT}/b2/place1/gt",
    "place2_only_lidar": f"{ROOT}/b2/place2/gt",
}

def map_pairs(scene):
    """List (raw_path, ann_path, raw_dirname) for annotated frames, fog frames first."""
    raw = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/b2/**/*.b2", recursive=True)}
    out = []
    for a in sorted(glob.glob(f"{ANN_DIR}/{scene}/*_annotation.b2")):
        base = os.path.basename(a).replace("_annotation.b2", ".b2")
        if base in raw:
            d = os.path.basename(os.path.dirname(raw[base]))
            out.append((raw[base], a, d))
    out.sort(key=lambda t: ("fog" not in t[2].lower(), t[0]))  # fog dirs first
    return out

def load_clear_mean(scene, k=10):
    """Mean of up to k clear (-gt) frames as the no-fog reference waveform grid, or None."""
    d = SCENE_GT_DIR.get(scene)
    if not d:
        return None
    files = sorted(glob.glob(f"{d}/*.b2"))[:k]
    if not files:
        return None
    acc = None
    for f in files:
        a = load(f).astype(np.float32)
        acc = a if acc is None else acc + a
    return acc / len(files)


def lift(px_x, px_y, bins):
    az = np.deg2rad((np.asarray(px_x, float) / W_RAW - 0.5) * FOV_H)
    alt = np.deg2rad((np.asarray(px_y, float) / H_RAW - 0.5) * FOV_V)
    d = np.asarray(bins, float) * BIN_TO_M
    return np.stack([d * np.cos(az) * np.cos(alt),
                     -d * np.sin(az) * np.cos(alt),
                     -d * np.sin(alt)], axis=-1).astype(np.float32)


def gt_cloud(ann):
    idx = np.argwhere(ann > 0)
    labels = ann[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)
    pts = lift(idx[:, 0], idx[:, 1], idx[:, 2])
    cols = np.array([COLOR[int(l)] for l in labels], dtype=np.uint8)
    return pts, cols, labels


def waveform_strip(wf, scale):
    """(700,) intensity -> (700,2) [bin, -intensity*scale] polyline (y negated so peaks point up)."""
    T = len(wf)
    return np.stack([np.arange(T, dtype=np.float32), -wf.astype(np.float32) * scale], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene1")
    ap.add_argument("--frame", type=int, default=0, help="index into annotated frames (fog first)")
    ap.add_argument("--per_class", type=int, default=300, help="inspect-pixels sampled per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_overlay", action="store_true", help="don't overlay clear(-gt) waveform")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pairs = map_pairs(args.scene)
    if not pairs:
        raise SystemExit(f"no frames for {args.scene}")
    raw_path, ann_path, raw_dir = pairs[args.frame]
    raw = load(raw_path).astype(np.float32)   # fog/main waveform grid (400,512,700)
    ann = load(ann_path)                       # labels (400,512,700) int8
    clear = None if args.no_overlay else load_clear_mean(args.scene)
    overlay = clear is not None and clear.shape == raw.shape
    print(f"[{args.scene}] fog/main frame = {raw_dir}/{os.path.basename(raw_path)}")
    print(f"  clear(-gt) overlay: {'ON (mean of -gt frames)' if overlay else 'OFF'}")

    rr.init(f"fog wf {args.scene}")
    if args.serve:
        uri = rr.serve_grpc(); rr.serve_web_viewer(open_browser=False, connect_to=uri)
        print(f"[rerun] web viewer; grpc={uri}")

    # waveform shown via overlaid LineStrips2D (clear=cyan, fog=red) in a Spatial2DView (BarChart
    # can't overlay two series). A view can't follow click-selection without notebook callbacks,
    # so a `pixel` timeline drives fixed entities; scrub it to browse. 3D highlight + plot sync.
    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["scene/**", "inspect/highlight"],
                              name="point cloud (white = current pixel)"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="inspect/wave",
                                  name="waveform: cyan=clear(gt) gray=fog; dots=annotation "
                                       "(green=obj blue=glass red=ghost yellow=fog)"),
                rrb.TextDocumentView(origin="inspect/info", name="pixel info"),
                row_shares=[3, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
    )
    rr.send_blueprint(bp)

    # context cloud (static)
    pts, cols, labels = gt_cloud(ann)
    rr.log("scene/ground_truth", rr.Points3D(positions=pts, colors=cols,
                                             class_ids=labels, radii=0.05), static=True)
    print(f"  GT points = {len(pts)}")

    # sample pixels per class (coverage), then order by PIXEL position for the timeline
    samples = []   # (class, x, y)
    for cls in (1, 2, 3, 4):
        pix = np.argwhere((ann == cls).any(axis=2))
        if len(pix) == 0:
            continue
        sel = rng.choice(len(pix), size=min(args.per_class, len(pix)), replace=False)
        samples += [(cls, int(x), int(y)) for (x, y) in pix[sel]]
    samples.sort(key=lambda t: (t[1], t[2]))   # pixel-order timeline

    for i, (cls, x, y) in enumerate(samples):
        rr.set_time("pixel", sequence=i)
        wf = raw[x, y]
        lab = ann[x, y]
        cls_bins = np.where(lab == cls)[0]
        bsel = cls_bins[np.argmax(wf[cls_bins])]
        pos = lift([x], [y], [bsel])
        rr.log("inspect/highlight", rr.Points3D(positions=pos, colors=[(255, 255, 255)], radii=0.4))
        # overlaid waveforms (neutral lines so class-colored annotation markers stand out):
        #   fog/main = gray, clear(gt) = cyan
        rr.log("inspect/wave/fog", rr.LineStrips2D([waveform_strip(wf, 1.0)],
                                                   colors=[(170, 170, 170)], radii=0.6))
        if overlay:
            rr.log("inspect/wave/clear", rr.LineStrips2D([waveform_strip(clear[x, y], 1.0)],
                                                         colors=[(0, 200, 255)], radii=0.6))
        # annotation markers ON the fog waveform: a class-colored dot at every labeled bin
        abins = np.where(lab > 0)[0]
        if len(abins):
            apos = np.stack([abins.astype(np.float32), -wf[abins]], axis=1)
            acols = np.array([COLOR[int(lab[b])] for b in abins], dtype=np.uint8)
            rr.log("inspect/wave/annotation", rr.Points2D(apos, colors=acols, radii=2.5))
        else:
            rr.log("inspect/wave/annotation", rr.Clear(recursive=False))
        present = {int(c): np.where(lab == c)[0].tolist() for c in np.unique(lab) if c > 0}
        rr.log("inspect/info", rr.TextDocument(
            f"# pixel {i+1}/{len(samples)}  (focus = {NAME[cls]})\n"
            f"h={x}, w={y}\n3D pos = {pos[0].round(2).tolist()} m\n"
            f"peak: clear={'%.0f'%clear[x,y].max() if overlay else 'NA'}  fog={wf.max():.0f}\n\n"
            "labeled bins:\n" + "\n".join(f"- {NAME[c]}: {present[c]}" for c in present)))
    print(f"  sampled pixels on `pixel` timeline: {len(samples)} (scrub to browse)")

    if not args.serve:
        out = os.path.join(OUT, f"pcd_wf_{args.scene}.rrd")
        rr.save(out)
        print(f"\nsaved {out}\nopen:  rerun {out}")


if __name__ == "__main__":
    main()
