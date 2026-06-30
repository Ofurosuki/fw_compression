#!/usr/bin/env python
"""Standalone rerun point-cloud visualizer for the Keio fog-chamber dataset (GT annotation).

No model / inference needed: we just lift the dense voxel annotation (and optionally the raw
returns) into 3D using the SAME SRL LiDAR geometry as the ghost-fwl repo
(neurips2026/.../visualize/point/pointcloud_wrapper.py: PointcloudWrapper, mode="SRL"):

    distance[m] = bin * 1e-7 * c / 2
    azimuth[deg]  = (Pixel_X * 47 - 4512) / 100
    altitude[deg] = (Pixel_Y * 47 - 1316) / 100
    x =  d cos(az) cos(alt);  y = -d sin(az) cos(alt);  z = -d sin(alt)

NOTE (to debug together): the *47/offset constants are calibrated for the ghost-fwl PROCESSED
grid, not necessarily the raw (400,512,700) fog grid — so the absolute scale / FOV may be off.
This script exposes --bin_to_m, --az_scale/--az_off, --alt_scale/--alt_off to retune live.

Headless: saves an .rrd (open locally with `rerun file.rrd`). Use --serve to host a web viewer.

Run:
  export PATH="$HOME/.local/bin:$PATH"
  uv run python fog_viz/vis_pcd_rerun.py --scene scene2 --frames 1
"""
import argparse, glob, os
import numpy as np
import blosc2
import rerun as rr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299792458.0
ROOT = "/data3/user/ikeda/fog/keio_chamber"
ANN_DIR = f"{ROOT}/annotation_v1"
OUT = os.path.join(os.path.dirname(__file__), "outputs")

# AnnotationLabel: 0=UNKNOWN 1=OBJECT 2=GLASS 3=GHOST 4=FOG  (matches ghost-fwl color_map)
COLOR = {
    0: (128, 128, 128),
    1: (0, 255, 0),     # OBJECT  green
    2: (0, 0, 255),     # GLASS   blue
    3: (255, 0, 0),     # GHOST   red
    4: (255, 255, 0),   # FOG     yellow
}
NAME = {0: "unknown", 1: "object", 2: "glass", 3: "ghost", 4: "fog"}


def load(path):
    return blosc2.load_array(path)  # (400,512,700) float32 / int8


def map_raw_to_paths(scene):
    raw = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/b2/**/*.b2", recursive=True)}
    anns = sorted(glob.glob(f"{ANN_DIR}/{scene}/*_annotation.b2"))
    pairs = []
    for a in anns:
        base = os.path.basename(a).replace("_annotation.b2", ".b2")
        if base in raw:
            pairs.append((raw[base], a))
    return pairs


W_RAW, H_RAW = 400, 512  # native grid (axis0=width, axis1=height)

def pixel_to_angles(px_x, px_y, args):
    """Returns (azimuth_rad, altitude_rad). Two modes:
       - geom=ghost: exact ghost-fwl SRL linear constants (calibrated for their processed grid)
       - geom=centered: center the (400,512) grid, span ±fov_h/2, ±fov_v/2 (debuggable guess)
    """
    if args.geom == "ghost":
        az = (px_x * args.az_scale + args.az_off) / 100.0
        alt = (px_y * args.alt_scale + args.alt_off) / 100.0
    else:  # centered
        az = (px_x / W_RAW - 0.5) * args.fov_h
        alt = (px_y / H_RAW - 0.5) * args.fov_v
    return np.deg2rad(az), np.deg2rad(alt)

def lift_points(idx, dist, args):
    """idx: (N,3) int (px_x, px_y, bin). dist: (N,) m. Returns (N,3) XYZ via SRL geometry."""
    az, alt = pixel_to_angles(idx[:, 0].astype(np.float64), idx[:, 1].astype(np.float64), args)
    x = dist * np.cos(az) * np.cos(alt)
    y = -dist * np.sin(az) * np.cos(alt)
    z = -dist * np.sin(alt)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def save_projection_png(pts, labels, out_path, title):
    """Top view (X fwd vs Y) and side view (X vs Z) scatter, colored by class — so we can
    sanity-check geometry without the interactive viewer."""
    if len(pts) == 0:
        return
    col = np.array([COLOR.get(int(l), (150, 150, 150)) for l in labels]) / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (i, j, xl, yl, ttl) in zip(axes, [(0, 1, "X forward [m]", "Y left [m]", "TOP view (X-Y)"),
                                              (0, 2, "X forward [m]", "Z up [m]", "SIDE view (X-Z)")]):
        ax.scatter(pts[:, i], pts[:, j], c=col, s=1, alpha=0.4)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(color=np.array(COLOR[k]) / 255.0, label=NAME[k]) for k in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def gt_pointcloud(ann, args):
    """Every annotated voxel (label>0, or include 0 if show_unknown) -> one 3D point."""
    mask = ann > 0 if not args.show_unknown else ann != -1
    idx = np.argwhere(mask)                       # (N,3): x,y,bin
    labels = ann[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)
    dist = idx[:, 2].astype(np.float64) * args.bin_to_m
    pts = lift_points(idx, dist, args)
    cols = np.array([COLOR.get(int(l), (200, 200, 200)) for l in labels], dtype=np.uint8)
    return pts, cols, labels


def raw_strongest_pointcloud(raw, args, thresh=3.0):
    """One point per pixel at its strongest return bin (context cloud, intensity-gray)."""
    amax = raw.argmax(axis=2)                      # (X,Y)
    peak = np.take_along_axis(raw, amax[:, :, None], axis=2)[:, :, 0]
    xs, ys = np.where(peak > thresh)
    bins = amax[xs, ys]
    idx = np.stack([xs, ys, bins], axis=1)
    dist = bins.astype(np.float64) * args.bin_to_m
    pts = lift_points(idx, dist, args)
    inten = peak[xs, ys]
    g = (255 * np.clip(inten / max(inten.max(), 1), 0.1, 1)).astype(np.uint8)
    cols = np.stack([g, g, g], axis=1)
    return pts, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene2",
                    help="place1 | place2_only_lidar | scene1 | scene2 | scene3")
    ap.add_argument("--frames", type=int, default=1, help="number of frames (time sequence)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--show_unknown", action="store_true")
    ap.add_argument("--with_raw", action="store_true", help="also log raw strongest-return cloud")
    # geometry knobs
    ap.add_argument("--geom", choices=["centered", "ghost"], default="centered",
                    help="centered: center (400,512) grid w/ fov_h/fov_v; ghost: exact ghost-fwl constants")
    ap.add_argument("--bin_to_m", type=float, default=0.15,
                    help="meters per range bin (ghost-fwl uses 1e-7*c/2≈14.99; chamber likely ~0.1)")
    ap.add_argument("--fov_h", type=float, default=90.0, help="horizontal FOV deg (centered mode)")
    ap.add_argument("--fov_v", type=float, default=26.0, help="vertical FOV deg (centered mode)")
    # exact ghost-fwl SRL constants (used when --geom ghost)
    ap.add_argument("--az_scale", type=float, default=47.0)
    ap.add_argument("--az_off", type=float, default=-4512.0)
    ap.add_argument("--alt_scale", type=float, default=47.0)
    ap.add_argument("--alt_off", type=float, default=-1316.0)
    ap.add_argument("--no_png", action="store_true", help="skip 2D projection PNGs")
    ap.add_argument("--serve", action="store_true", help="host web viewer instead of saving .rrd")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pairs = map_raw_to_paths(args.scene)[args.start:args.start + args.frames]
    if not pairs:
        raise SystemExit(f"no frames for scene {args.scene}")
    print(f"bin_to_m={args.bin_to_m:.4f} m/bin  az=({args.az_scale},{args.az_off}) "
          f"alt=({args.alt_scale},{args.alt_off})")

    rr.init("fog_chamber GT pointcloud")
    if args.serve:
        uri = rr.serve_grpc()
        rr.serve_web_viewer(open_browser=False, connect_to=uri)
        print(f"[rerun] web viewer hosted; grpc={uri}")

    for t, (raw_path, ann_path) in enumerate(pairs):
        ann = load(ann_path)
        pts, cols, labels = gt_pointcloud(ann, args)
        rr.set_time("frame", sequence=t)
        rr.log("scene/ground_truth", rr.Points3D(positions=pts, colors=cols,
                                                  class_ids=labels, radii=0.05))
        # per-class counts + coord ranges (debug)
        u, c = np.unique(labels, return_counts=True)
        cnt = {NAME.get(int(k), k): int(v) for k, v in zip(u, c)}
        rng = (pts.min(0).round(2).tolist(), pts.max(0).round(2).tolist()) if len(pts) else None
        print(f"[frame {t}] {os.path.basename(ann_path)}  N={len(pts)}  classes={cnt}")
        print(f"           XYZ min/max = {rng}  (dist full-scale = {ann.shape[2]*args.bin_to_m:.1f} m)")
        if t == 0 and not args.no_png:
            save_projection_png(pts, labels, os.path.join(OUT, f"pcd_{args.scene}_proj.png"),
                                f"{args.scene} GT  [geom={args.geom} fov=({args.fov_h},{args.fov_v}) "
                                f"bin_to_m={args.bin_to_m}]")
        if args.with_raw:
            rpts, rcols = raw_strongest_pointcloud(load(raw_path), args)
            rr.log("scene/raw_strongest", rr.Points3D(positions=rpts, colors=rcols, radii=0.02))
            print(f"           raw strongest cloud N={len(rpts)}")

    if not args.serve:
        out = os.path.join(OUT, f"pcd_{args.scene}.rrd")
        rr.save(out)
        print(f"\nsaved {out}\nopen locally with:  rerun {out}")


if __name__ == "__main__":
    main()
