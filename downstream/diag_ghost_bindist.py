"""Why does a compressed representation lose ghost? Bucket ghost peak-recall by
(a) bin-distance to the nearest other return on the same ray and (b) whether the
ghost is the brightest return on its ray — for a retrained ToPM on a given input
representation. Run on full-waveform vs taw to isolate what the COMPRESSION drops
(architecture/recipe held fixed = ToPM; only the input representation changes).

Mirrors run_eval_peak.py's per-frame pipeline (same random crop, same raw-waveform
find_peaks scoring population), but records per-ghost-peak features instead of only
the confusion matrix.

  PYTHONPATH=.../neurips2026/src python downstream/diag_ghost_bindist.py \
      --config downstream/configs/evalA_split2_test_best.yaml --ckpt <retrained.pth> \
      --compress none --device cuda:3 --divide 3 --max_frames 60        # full waveform
      --compress event --event_repr taw --event_k 4 ...                  # taw
"""
from __future__ import annotations

import argparse
import os
import sys

import pyarrow  # noqa: F401  (precede torch)
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import envconfig  # noqa: E402
import run_eval    # noqa: E402
from run_eval_peak import make_transform, predict_dhw  # noqa: E402

from hist_lidar.config import load_config_from_yaml          # noqa: E402
from hist_lidar.data import VoxelDatasetWithToMe              # noqa: E402
from hist_lidar.training.test_unet3d import detect_peaks_in_voxel  # noqa: E402
from hist_lidar.utils import get_model, load_checkpoint, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--compress", choices=["none", "event", "ae"], default="none")
    ap.add_argument("--event_k", type=int, default=4)
    ap.add_argument("--event_repr", default="taw")
    ap.add_argument("--event_intensity", default="height")
    ap.add_argument("--event_smooth_sigma", type=float, default=1.5)
    ap.add_argument("--event_min_height", type=float, default=0.03)
    ap.add_argument("--event_min_distance", type=int, default=3)
    ap.add_argument("--event_fixed_width", type=float, default=4.0)
    ap.add_argument("--event_fixed_amplitude", type=float, default=1.0)
    ap.add_argument("--event_kernel", default="gaussian")
    ap.add_argument("--event_emg_tau", type=float, default=2.65)
    ap.add_argument("--ae_ckpt", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--divide", type=int, default=3)
    ap.add_argument("--limit_dirs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_config_from_yaml(args.config)
    config.test_voxel_dirs = [envconfig.remap_data_dir(d) for d in config.test_voxel_dirs]
    config.test_annotation_dirs = [envconfig.remap_data_dir(d) for d in config.test_annotation_dirs]
    config.checkpoint_path = envconfig.remap_topm(config.checkpoint_path)
    config.device = args.device
    if args.ckpt:
        config.checkpoint_path = args.ckpt
    config.divide = args.divide
    if args.limit_dirs:
        config.test_voxel_dirs = config.test_voxel_dirs[: args.limit_dirs]
        config.test_annotation_dirs = config.test_annotation_dirs[: args.limit_dirs]
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    model = get_model(config).to(device)
    load_checkpoint(config.checkpoint_path, model, device)
    model.eval()
    ds = VoxelDatasetWithToMe(
        voxel_dirs=config.test_voxel_dirs, annotation_dirs=config.test_annotation_dirs,
        target_size=config.target_size, divide=config.divide,
        y_crop_top=config.y_crop_top, y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front, z_crop_back=config.z_crop_back)
    transform = make_transform(args, device)
    ut, th = config.use_threshold_prediction, config.prediction_threshold
    n = min(args.max_frames, len(ds.voxel_files))
    print(f"[{args.tag}] frames={n} compress={args.compress} repr={args.event_repr}")

    rows = []
    cm_check = np.zeros((4, 4), dtype=np.int64)
    for i in tqdm(range(n), desc=args.tag):
        raw = ds._load_voxel_grid(ds.voxel_files[i]).astype(np.float32)
        ann = ds._load_annotation_voxel(ds.annotation_files[i])
        synth = transform(raw) if args.compress != "none" else raw
        raw = ds._apply_z_crop(ds._apply_y_crop(raw))
        synth = ds._apply_z_crop(ds._apply_y_crop(synth))
        ann = ds._apply_z_crop(ds._apply_y_crop(ann))
        start = ds._determine_crop_coordinates(raw.shape)
        raw_c = ds._preprocess_voxel(raw, start)
        synth_c = ds._preprocess_voxel(synth, start)
        ann_c = ds._preprocess_annotation(ann, start)
        pred = predict_dhw(model, synth_c, device, ut, th)            # (D,H,W)
        raw_dhw = np.transpose(raw_c, (2, 1, 0))
        ann_dhw = np.transpose(ann_c, (2, 1, 0)).astype(np.int64)
        W = raw_dhw.shape[2]
        peaks = detect_peaks_in_voxel(raw_dhw)
        if not peaks:
            continue
        pk = np.asarray(peaks, dtype=np.int64)                       # (M,3) d,h,w
        d, h, w = pk[:, 0], pk[:, 1], pk[:, 2]
        ann_at, pred_at = ann_dhw[d, h, w], pred[d, h, w]
        ok = (ann_at >= 0) & (ann_at < 4) & (pred_at >= 0) & (pred_at < 4)
        np.add.at(cm_check, (ann_at[ok], pred_at[ok]), 1)
        rows.append(pd.DataFrame({
            "ray": (i << 20) + h * W + w, "d": d,                    # frame-unique ray id
            "ht": raw_dhw[d, h, w], "ann": ann_at, "pred": pred_at}))

    df = pd.concat(rows, ignore_index=True)
    df["raymax"] = df.groupby("ray")["ht"].transform("max")
    df["is_max"] = df["ht"] >= df["raymax"] - 1e-6
    df = df.sort_values(["ray", "d"])
    dn = df.groupby("ray")["d"].diff()
    dn2 = df.groupby("ray")["d"].diff(-1).abs()
    df["sep"] = pd.concat([dn, dn2], axis=1).min(axis=1)              # nearest other peak on ray
    print(f"\n[{args.tag}] sanity — CM-based peak recall (cf. full obj0.74/glass0.43/ghost0.72):")
    for c, nm in [(1, "object"), (2, "glass"), (3, "ghost")]:
        rc = cm_check[c, c] / max(1, cm_check[c].sum())
        print(f"    {nm:7}: n={int(cm_check[c].sum()):7d} recall={rc:.3f}")
    print(f"    [df cross-check] ghost recall={(df[df['ann']==3]['pred']==3).mean():.3f}")
    g = df[df["ann"] == 3].copy()
    g["hit"] = (g["pred"] == 3).astype(float)
    g["toobj"] = (g["pred"] == 1).astype(float)
    print(f"[{args.tag}] true-ghost peaks={len(g)}  overall recall={g['hit'].mean():.3f}")
    print("  by bin-dist to nearest other return on the ray:")
    for lo, hi in [(0, 4), (4, 8), (8, 15), (15, 40), (40, 1e9)]:
        s = g[(g["sep"] >= lo) & (g["sep"] < hi)]
        if len(s) > 50:
            print(f"    sep∈[{lo},{hi}): n={len(s):6d} recall={s['hit'].mean():.3f} (->obj {s['toobj'].mean():.3f})")
    iso = g[g["sep"].isna()]
    if len(iso) > 50:
        print(f"    sep=alone (only return): n={len(iso):6d} recall={iso['hit'].mean():.3f} (->obj {iso['toobj'].mean():.3f})")
    print("  ghost is the brightest return on its ray?")
    for v, nm in [(True, "ghost IS brightest"), (False, "ghost is secondary")]:
        s = g[g["is_max"] == v]
        if len(s) > 50:
            print(f"    {nm:20}: n={len(s):6d} recall={s['hit'].mean():.3f} (->obj {s['toobj'].mean():.3f})")


if __name__ == "__main__":
    main()
