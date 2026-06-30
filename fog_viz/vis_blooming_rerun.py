#!/usr/bin/env python
"""Standalone rerun visualizer for the Keio *blooming* dataset (raw only).

Unlike the fog-chamber set, blooming has **no annotation and no clear(gt) reference** — just
raw full-waveform captures. So we lift each pixel's strongest return into 3D using the SAME
approved "centered" SRL geometry as fog_viz/vis_pcd_rerun.py (see USAGE.md), and attach a
per-pixel full-waveform inspector (just the raw line — no class markers, no gt overlay).

Data layout: /data3/user/ikeda/blooming/b2/<timestamp_scene>/*.b2  — each timestamp dir is one
"scene" (a capture session); each .b2 is one frame. We emit ONE .rrd per scene with:
  - scene/raw_strongest   3D strongest-return cloud, one frame per .b2 on the `frame` timeline
  - inspect/highlight     white 3D point = currently inspected pixel (on `pixel` timeline)
  - inspect/wave          that pixel's full waveform (LineStrips2D, on `pixel` timeline)
  - inspect/info          pixel coords / peak (TextDocument)
Layout: [ 3D view | waveform + info ]. SCRUB `frame` to play the sequence; SCRUB `pixel` to
browse sampled pixels' waveforms (sampled on --ref_frame, default 0).

Geometry is FIXED (user-approved): geom=centered, FOV 90x26 deg, bin_to_m=0.15.

Run:
  export PATH="$HOME/.local/bin:$PATH"
  uv run python fog_viz/vis_blooming_rerun.py                       # all scenes, one .rrd each
  uv run python fog_viz/vis_blooming_rerun.py --scene 20260630160055 --per_pixels 400
"""
import argparse, glob, os
import numpy as np
import blosc2
import rerun as rr
import rerun.blueprint as rrb

ROOT = "/data3/user/ikeda/blooming/b2"
OUT = os.path.join(os.path.dirname(__file__), "outputs")

W_RAW, H_RAW = 400, 512  # native grid (axis0=width, axis1=height)
# FIXED geometry (approved, same as fog_viz)
BIN_TO_M = 0.15
FOV_H, FOV_V = 90.0, 26.0


def load(path):
    return blosc2.load_array(path)  # (400,512,700) float32


def list_scenes():
    return sorted(d for d in glob.glob(f"{ROOT}/*") if os.path.isdir(d))


def lift(px_x, px_y, bins):
    az = np.deg2rad((np.asarray(px_x, float) / W_RAW - 0.5) * FOV_H)
    alt = np.deg2rad((np.asarray(px_y, float) / H_RAW - 0.5) * FOV_V)
    d = np.asarray(bins, float) * BIN_TO_M
    return np.stack([d * np.cos(az) * np.cos(alt),
                     -d * np.sin(az) * np.cos(alt),
                     -d * np.sin(alt)], axis=-1).astype(np.float32)


def raw_strongest_pointcloud(raw, thresh):
    """One point per pixel at its strongest return bin, gray-scaled by intensity."""
    amax = raw.argmax(axis=2)                       # (X,Y)
    peak = np.take_along_axis(raw, amax[:, :, None], axis=2)[:, :, 0]
    xs, ys = np.where(peak > thresh)
    bins = amax[xs, ys]
    pts = lift(xs, ys, bins)
    inten = peak[xs, ys]
    g = (255 * np.clip(inten / max(inten.max(), 1), 0.1, 1)).astype(np.uint8)
    cols = np.stack([g, g, g], axis=1)
    return pts, cols, inten


def waveform_strip(wf):
    """(700,) intensity -> (700,2) [bin, -intensity] polyline (y negated so peaks point up)."""
    T = len(wf)
    return np.stack([np.arange(T, dtype=np.float32), -wf.astype(np.float32)], axis=1)


def process_scene(scene_dir, args):
    name = os.path.basename(scene_dir)
    frames = sorted(glob.glob(f"{scene_dir}/*.b2"))
    if not frames:
        print(f"[skip] {name}: no frames")
        return

    rr.init(f"blooming {name}")
    if args.serve:
        uri = rr.serve_grpc()
        rr.serve_web_viewer(open_browser=False, connect_to=uri)
        print(f"[rerun] web viewer hosted; grpc={uri}")

    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["scene/**", "inspect/highlight"],
                              name="raw cloud (white = current pixel)"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="inspect/wave",
                                  name="full waveform (x=range bin, up=intensity)"),
                rrb.TextDocumentView(origin="inspect/info", name="pixel info"),
                row_shares=[3, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
    )
    rr.send_blueprint(bp)

    # ONE sampled frame per scene
    fi = min(args.frame, len(frames) - 1)
    fp = frames[fi]
    raw = load(fp).astype(np.float32)
    print(f"\n=== scene {name}: frame {fi}/{len(frames)-1}  {os.path.basename(fp)} ===")

    # 3D strongest-return cloud (static)
    pts, cols, inten = raw_strongest_pointcloud(raw, args.thresh)
    rr.log("scene/raw_strongest", rr.Points3D(positions=pts, colors=cols, radii=0.03), static=True)
    rng_xyz = (pts.min(0).round(2).tolist(), pts.max(0).round(2).tolist()) if len(pts) else None
    print(f"  cloud N={len(pts)}  peak_max={inten.max() if len(inten) else 0:.1f}  XYZ min/max={rng_xyz}")

    # per-pixel waveform inspector over EVERY pixel with a return, on the `pixel` timeline
    peak = raw.max(axis=2)
    cand = np.argwhere(peak > args.thresh)
    cand = cand[np.lexsort((cand[:, 1], cand[:, 0]))]   # pixel-order timeline
    if args.per_pixels:                                  # optional cap (0 = all)
        cand = cand[: args.per_pixels]
    for i, (x, y) in enumerate(cand):
        x, y = int(x), int(y)
        wf = raw[x, y]
        bsel = int(wf.argmax())
        pos = lift([x], [y], [bsel])
        rr.set_time("pixel", sequence=i)
        rr.log("inspect/highlight", rr.Points3D(positions=pos, colors=[(255, 255, 255)], radii=0.4))
        rr.log("inspect/wave/raw", rr.LineStrips2D([waveform_strip(wf)],
                                                   colors=[(170, 170, 170)], radii=0.6))
        rr.log("inspect/info", rr.TextDocument(
            f"# pixel {i+1}/{len(cand)}\n"
            f"h={x}, w={y}\n3D pos = {pos[0].round(2).tolist()} m\n"
            f"peak intensity = {wf.max():.0f} @ bin {bsel} ({bsel*BIN_TO_M:.1f} m)"))
    print(f"  waveform inspector: {len(cand)} pixels on `pixel` timeline")

    if not args.serve:
        out = os.path.join(OUT, f"blooming_{name}.rrd")
        rr.save(out)
        print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None,
                    help="single timestamp scene dir name (default: all scenes)")
    ap.add_argument("--frame", type=int, default=0, help="which frame to sample per scene")
    ap.add_argument("--per_pixels", type=int, default=0,
                    help="cap pixels on the waveform timeline (0 = ALL pixels with a return)")
    ap.add_argument("--thresh", type=float, default=3.0, help="min peak intensity to keep a pixel")
    ap.add_argument("--serve", action="store_true", help="host web viewer instead of saving .rrd")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    scenes = [os.path.join(ROOT, args.scene)] if args.scene else list_scenes()
    for sc in scenes:
        process_scene(sc, args)


if __name__ == "__main__":
    main()
