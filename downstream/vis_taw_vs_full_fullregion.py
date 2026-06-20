"""Rerun viz (CORRECTED): full-waveform vs taw multi-echo point clouds, coloured by ToPM
segmentation — over the FULL sensor plane (no random crop).

Why this replaces vis_taw_vs_full_rerun.py: that one was built on vis_pointcloud_rerun.py,
which feeds the model a RANDOM 200x168 crop. The SRL point geometry
(`azimuth=(Pixel_X*47-4512)/100`, `altitude=(Pixel_Y*47-1316)/100`) assumes consistent
absolute pixel indices, so a per-frame random crop shifts/tilts the cloud → "looks weird".
The geometrically-correct repo path is the FULL-region one (`FullRegionVoxelDataset` +
`SlidingWindowInference`, no crop, absolute pixels). We reuse that for inference and the
rerun multi-echo point generator (find_peaks per pixel) for the cloud, inserting only the
event transform + the two retrained models.

Usage (remote → save .rrd, scp, open locally):
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  uv-py downstream/vis_taw_vs_full_fullregion.py \
      --full_ckpt .../full_ceiling/.../epoch_50....pth \
      --taw_ckpt  .../taw_k4/.../epoch_50....pth \
      --event_k 4 --device cuda:2 --start 0 --num 5 \
      --rrd downstream/outputs/vis/full_vs_taw_fullregion.rrd
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rerun as rr  # noqa: E402
import run_eval     # noqa: E402  event_voxel

from hist_lidar.config import load_config_from_yaml          # noqa: E402
from hist_lidar.data.dataset_full_region import FullRegionVoxelDataset, SlidingWindowInference  # noqa: E402
from hist_lidar.utils import get_model, load_checkpoint, set_seed  # noqa: E402
from hist_lidar.visualize.point.pointcloud_wrapper import PointcloudWrapper  # noqa: E402
from hist_lidar.visualize.point.vis_pointcloud_rerun import (  # noqa: E402
    generate_pointcloud_from_voxel,   # multi-echo: find_peaks per pixel, one point per peak
    voxel_to_dataframe,
)

LABEL_MAP = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}


def build_model(config, ckpt, device):
    m = get_model(config).to(device)
    load_checkpoint(ckpt, m, device)
    m.eval()
    return m


def points_for(voxel_xyz, label_xyz, show_unknown):
    """Full-plane multi-echo points coloured by label_xyz; offsets 0 (absolute pixels)."""
    pcw = PointcloudWrapper(voxel_to_dataframe(voxel_xyz, 0, 0), mode="SRL")
    return generate_pointcloud_from_voxel(voxel_xyz, label_xyz, pcw, 0, 0, show_unknown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "configs", "evalA_split2_test_best.yaml"))
    ap.add_argument("--full_ckpt", required=True)
    ap.add_argument("--taw_ckpt", required=True)
    ap.add_argument("--event_k", type=int, default=4)
    ap.add_argument("--event_repr", default="taw")
    ap.add_argument("--event_smooth_sigma", type=float, default=1.5)
    ap.add_argument("--event_min_height", type=float, default=0.03)
    ap.add_argument("--event_min_distance", type=int, default=3)
    ap.add_argument("--event_fixed_width", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--limit_dirs", type=int, default=0)
    ap.add_argument("--show_unknown", action="store_true")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--separate", type=float, default=0.0, help="x-shift (m) between clouds; 0=overlay")
    ap.add_argument("--web_port", type=int, default=9090)
    ap.add_argument("--open_browser", action="store_true")
    ap.add_argument("--rrd", default=None, help="save to .rrd instead of serving")
    args = ap.parse_args()

    config = load_config_from_yaml(args.config)
    config.device = args.device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    if args.limit_dirs:
        config.test_voxel_dirs = config.test_voxel_dirs[: args.limit_dirs]
        config.test_annotation_dirs = config.test_annotation_dirs[: args.limit_dirs]

    full_model = build_model(config, args.full_ckpt, device)
    taw_model = build_model(config, args.taw_ckpt, device)
    win = tuple(config.target_size)
    full_sli = SlidingWindowInference(model=full_model, device=device, window_size=win, batch_size=args.batch_size)
    taw_sli = SlidingWindowInference(model=taw_model, device=device, window_size=win, batch_size=args.batch_size)

    ds = FullRegionVoxelDataset(
        voxel_dirs=config.test_voxel_dirs, annotation_dirs=config.test_annotation_dirs,
        downsample_z=config.downsample_z, divide=1,
        y_crop_top=config.y_crop_top, y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
    )
    ep = {"k": args.event_k, "representation": args.event_repr, "intensity_mode": "height",
          "smooth_sigma": args.event_smooth_sigma, "min_height": args.event_min_height,
          "min_distance": args.event_min_distance, "fixed_width": args.event_fixed_width,
          "fixed_amplitude": 1.0, "kernel": "gaussian", "emg_tau": 2.65}

    rr.init("ToPM seg: full vs taw (full region, multi-echo)")
    if args.rrd:
        os.makedirs(os.path.dirname(os.path.abspath(args.rrd)), exist_ok=True)
        rr.save(args.rrd)
        print(f"[Rerun] saving to {args.rrd}", flush=True)
    else:
        uri = rr.serve_grpc()
        rr.serve_web_viewer(web_port=args.web_port, open_browser=args.open_browser, connect_to=uri)
        print(f"[Rerun] open http://localhost:{args.web_port} (tunnel {args.web_port} & 9876 if remote)", flush=True)

    n = len(ds)
    end = min(args.start + args.num, n)
    print(f"frames total={n}, showing {args.start}..{end-1}", flush=True)

    for t, idx in enumerate(range(args.start, end)):
        sample = ds[idx]
        raw = sample["voxel_grid"]                       # (X,Y,Z) uint16, full plane
        ann = sample["annotation"]
        synth = np.rint(run_eval.event_voxel(raw.astype(np.float32), ep, device)).astype(raw.dtype)

        pred_full = full_sli.predict(raw)["prediction"]
        pred_taw = taw_sli.predict(synth)["prediction"]

        f_pts, f_cols, f_cls, f_cnt = points_for(raw, pred_full, args.show_unknown)
        t_pts, t_cols, t_cls, t_cnt = points_for(synth, pred_taw, args.show_unknown)
        if args.separate and len(t_pts):
            t_pts = t_pts.copy(); t_pts[:, 0] += args.separate

        rr.set_time("frame", sequence=t)
        rr.log("scene/full/pred", rr.Points3D(positions=f_pts, colors=f_cols, class_ids=f_cls))
        rr.log("scene/taw/pred", rr.Points3D(positions=t_pts, colors=t_cols, class_ids=t_cls))
        if not args.no_gt:
            g_pts, g_cols, g_cls, _ = points_for(raw, ann, args.show_unknown)
            if args.separate and len(g_pts):
                g_pts = g_pts.copy(); g_pts[:, 0] += 2 * args.separate
            rr.log("scene/ground_truth", rr.Points3D(positions=g_pts, colors=g_cols, class_ids=g_cls))

        fmt = lambda c: ", ".join(f"{LABEL_MAP.get(k, k)}={v}" for k, v in sorted(c.items()))
        print(f"[{t:3d}] frame {idx}: full[{fmt(f_cnt)}]  taw[{fmt(t_cnt)}]", flush=True)

    if args.rrd:
        print(f"done; wrote {args.rrd}", flush=True)
        return
    print("done logging; viewer stays alive (Ctrl-C to stop).", flush=True)
    import time
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
