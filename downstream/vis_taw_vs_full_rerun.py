"""Rerun visualisation: multi-echo point clouds coloured by ToPM segmentation, for the
**full-waveform** vs **taw-event** representation, side by side.

For each frame we load the raw T=700 voxel, build the taw pseudo-waveform with the same
event hook used in training/eval, run the matching retrained ToPM on each, turn the
per-pixel multi-echo peaks into 3D points (repo's PointcloudWrapper, ToF->XYZ) and colour
them by the model's predicted class (OBJECT green / GLASS blue / GHOST red / noise grey).
Both reps share the crop (np.random crop vs torch transform -> seed-fixed identical), so
the point sets are directly comparable. Optional ground-truth cloud too.

Logged rerun entities (toggle in the viewer):
  scene/full/pred , scene/taw/pred , scene/ground_truth   (+ a copy x-shifted if --separate)

Usage (remote box -> open the printed URL over an SSH tunnel):
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  uv-py downstream/vis_taw_vs_full_rerun.py \
      --config downstream/configs/evalA_split2_test_best.yaml \
      --full_ckpt downstream/outputs/retrain/full_ceiling/0618/...epoch_50....pth \
      --taw_ckpt  downstream/outputs/retrain/taw_k4/0618/...epoch_50....pth \
      --event_k 4 --device cuda:2 --start 0 --num 20 --web_port 9090
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
from hist_lidar.data import VoxelDatasetWithToMe             # noqa: E402
from hist_lidar.utils import get_model, load_checkpoint, set_seed  # noqa: E402
from hist_lidar.visualize.point.pointcloud_wrapper import PointcloudWrapper  # noqa: E402
from hist_lidar.visualize.point.vis_pointcloud_rerun import (  # noqa: E402
    generate_pointcloud_from_voxel,
    inference_single_sample,
    voxel_to_dataframe,
)

LABEL_MAP = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}


def build_model(config, ckpt, device):
    m = get_model(config).to(device)
    load_checkpoint(ckpt, m, device)
    m.eval()
    return m


def points_for(voxel_xyz, label_xyz, x_off, y_off, show_unknown):
    """Multi-echo points from voxel_xyz (X,Y,hist), coloured by label_xyz (X,Y,hist)."""
    pcw = PointcloudWrapper(voxel_to_dataframe(voxel_xyz, x_off, y_off), mode="SRL")
    pts, cols, cls, counts = generate_pointcloud_from_voxel(
        voxel_xyz, label_xyz, pcw, x_off, y_off, show_unknown)
    return pts, cols, cls, counts


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
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--limit_dirs", type=int, default=0)
    ap.add_argument("--x_offset", type=int, default=0)
    ap.add_argument("--y_offset", type=int, default=0)
    ap.add_argument("--show_unknown", action="store_true")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--separate", type=float, default=0.0,
                    help="x-shift (metres) between full/taw/gt clouds; 0 = overlay")
    ap.add_argument("--web_port", type=int, default=9090)
    ap.add_argument("--open_browser", action="store_true")
    ap.add_argument("--rrd", default=None,
                    help="save to this .rrd file instead of serving (download + open locally; "
                         "best for a remote box — no port tunnelling)")
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

    ds = VoxelDatasetWithToMe(
        voxel_dirs=config.test_voxel_dirs, annotation_dirs=config.test_annotation_dirs,
        target_size=config.target_size, divide=1,
        y_crop_top=config.y_crop_top, y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front, z_crop_back=config.z_crop_back,
    )
    ep = {"k": args.event_k, "representation": args.event_repr, "intensity_mode": "height",
          "smooth_sigma": args.event_smooth_sigma, "min_height": args.event_min_height,
          "min_distance": args.event_min_distance, "fixed_width": args.event_fixed_width,
          "fixed_amplitude": 1.0, "kernel": "gaussian", "emg_tau": 2.65}

    rr.init("ToPM seg: full vs taw multi-echo")
    if args.rrd:
        os.makedirs(os.path.dirname(os.path.abspath(args.rrd)), exist_ok=True)
        rr.save(args.rrd)
        print(f"[Rerun] saving to {args.rrd} (download + `rerun {os.path.basename(args.rrd)}` locally)", flush=True)
    else:
        uri = rr.serve_grpc()
        rr.serve_web_viewer(web_port=args.web_port, open_browser=args.open_browser, connect_to=uri)
        print(f"[Rerun] gRPC: {uri}", flush=True)
        print(f"[Rerun] open http://localhost:{args.web_port} (tunnel ports {args.web_port} & 9876 if remote)", flush=True)

    n = len(ds.voxel_files)
    end = min(args.start + args.num, n)
    print(f"frames total={n}, showing {args.start}..{end-1}")

    @torch.no_grad()
    def infer(model, vox_xyz):
        return inference_single_sample(model, device, vox_xyz, None)["prediction"]

    for t, idx in enumerate(range(args.start, end)):
        raw = ds._load_voxel_grid(ds.voxel_files[idx]).astype(np.float32)
        ann = ds._load_annotation_voxel(ds.annotation_files[idx])
        synth = run_eval.event_voxel(raw, ep, device)
        # y/z crop, then one shared random crop
        raw, synth, ann = (ds._apply_z_crop(ds._apply_y_crop(a)) for a in (raw, synth, ann))
        start = ds._determine_crop_coordinates(raw.shape)
        # uint16 to match the dataset's native dtype (the taw model was trained on uint16
        # cache, and voxel_to_dataframe stringifies values -> needs int, not "0.0")
        raw_c = ds._preprocess_voxel(raw, start).round().astype(np.uint16)
        synth_c = ds._preprocess_voxel(synth, start).round().astype(np.uint16)
        ann_c = ds._preprocess_annotation(ann, start)

        pred_full = infer(full_model, raw_c)
        pred_taw = infer(taw_model, synth_c)

        full_pts, full_cols, full_cls, full_cnt = points_for(raw_c, pred_full, args.x_offset, args.y_offset, args.show_unknown)
        taw_pts, taw_cols, taw_cls, taw_cnt = points_for(synth_c, pred_taw, args.x_offset, args.y_offset, args.show_unknown)

        if args.separate and len(taw_pts):
            taw_pts = taw_pts.copy(); taw_pts[:, 0] += args.separate
        rr.set_time("frame", sequence=t)
        rr.log("scene/full/pred", rr.Points3D(positions=full_pts, colors=full_cols, class_ids=full_cls))
        rr.log("scene/taw/pred", rr.Points3D(positions=taw_pts, colors=taw_cols, class_ids=taw_cls))
        if not args.no_gt:
            gt_pts, gt_cols, gt_cls, _ = points_for(raw_c, ann_c, args.x_offset, args.y_offset, args.show_unknown)
            if args.separate and len(gt_pts):
                gt_pts = gt_pts.copy(); gt_pts[:, 0] += 2 * args.separate
            rr.log("scene/ground_truth", rr.Points3D(positions=gt_pts, colors=gt_cols, class_ids=gt_cls))

        def fmt(c):
            return ", ".join(f"{LABEL_MAP.get(k, k)}={v}" for k, v in sorted(c.items()))
        print(f"[{t:3d}] frame {idx}: full[{fmt(full_cnt)}]  taw[{fmt(taw_cnt)}]")

    if args.rrd:
        # the rr.save() file sink flushes on normal interpreter exit (atexit); no explicit
        # flush/disconnect needed (disconnect() targets the grpc sink and can hang here).
        print(f"done; wrote {args.rrd}", flush=True)
        return
    print("done logging; viewer stays alive (Ctrl-C to stop).", flush=True)
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
