"""Peak-level F1 (the paper's metric) for a retrained ToPM on a given representation.

The paper's "F1-mean ~0.592" is PEAK-level: predictions are scored only at the
return-peak positions found by scipy `find_peaks` on the waveform (the repo's
`detect_peaks_in_voxel` + `evaluate_peaks`, reused verbatim here). `run_eval.py` skips
this (slow); this script adds it for the headline retrain configs.

Cross-representation fairness (option B): the scoring population = peaks of the **raw**
waveform, the SAME for every config (paper population), so taw/ta are scored at *all*
true return positions — including returns they dropped — not only where they kept events.
To do that under the downstream's random crop, we load BOTH the raw and the
representation-transformed waveform and crop them with the SAME `start_coords`; the model
sees the transformed input, peaks/scoring use the raw. `_determine_crop_coordinates` uses
np.random while the event transform uses torch, so at a fixed seed the crop (hence the raw
peak set) is identical across full/taw/ta → directly comparable.

Usage:
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  uv-py downstream/run_eval_peak.py --config downstream/configs/evalA_split2_test_best.yaml \
      --ckpt <retrained.pth> --compress event --event_repr taw --event_k 4 \
      --device cuda:0 --divide 3 --out downstream/outputs/retrain/peak_taw_k4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import envconfig  # noqa: E402  (machine-dependent paths; see env.yaml.example)
import run_eval  # noqa: E402  event_voxel / compress_voxel / load_ae

from hist_lidar.config import load_config_from_yaml          # noqa: E402
from hist_lidar.data import VoxelDatasetWithToMe              # noqa: E402
from hist_lidar.training.test_ViT3D import calculate_metrics_from_confusion_matrix  # noqa: E402
from hist_lidar.training.test_unet3d import detect_peaks_in_voxel, evaluate_peaks   # noqa: E402
from hist_lidar.utils import get_model, load_checkpoint, set_seed                   # noqa: E402

LABEL_MAP = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}
SIGNAL = [1, 2, 3]


def make_transform(args, device):
    if args.compress == "event":
        ep = {
            "k": args.event_k, "representation": args.event_repr,
            "intensity_mode": args.event_intensity, "smooth_sigma": args.event_smooth_sigma,
            "min_height": args.event_min_height, "min_distance": args.event_min_distance,
            "fixed_width": args.event_fixed_width, "fixed_amplitude": args.event_fixed_amplitude,
            "kernel": args.event_kernel, "emg_tau": args.event_emg_tau,
        }
        return lambda raw: run_eval.event_voxel(raw, ep, device)
    if args.compress == "ae":
        model, kind, _ = run_eval.load_ae(args.ae_ckpt, device)
        return lambda raw: run_eval.compress_voxel(raw, model, kind, device)
    return lambda raw: raw  # none


@torch.no_grad()
def predict_dhw(model, synth_xyz, device, use_threshold, threshold):
    """synth_xyz: (X,Y,Z) model input -> (D,H,W) predicted labels (D=Z,H=Y,W=X)."""
    x = np.transpose(synth_xyz, (2, 1, 0))                  # (D,H,W)
    t = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(device)
    out = model(t)                                          # (1,C,D,H,W)
    if use_threshold:
        prob = torch.softmax(out, dim=1)
        mp, am = torch.max(prob, dim=1)
        pred = torch.where(mp >= threshold, am, torch.zeros_like(am))
    else:
        pred = torch.argmax(out, dim=1)
    return pred[0].cpu().numpy()                            # (D,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="override config.checkpoint_path")
    ap.add_argument("--compress", choices=["none", "event", "ae"], default="none")
    ap.add_argument("--ae_ckpt", default=None)
    ap.add_argument("--event_k", type=int, default=4)
    ap.add_argument("--event_repr", choices=["t", "ta", "tw", "taw", "taw_bg"], default="taw")
    ap.add_argument("--event_intensity", choices=["height", "area"], default="height")
    ap.add_argument("--event_smooth_sigma", type=float, default=1.5)
    ap.add_argument("--event_min_height", type=float, default=0.03)
    ap.add_argument("--event_min_distance", type=int, default=3)
    ap.add_argument("--event_fixed_width", type=float, default=4.0)
    ap.add_argument("--event_fixed_amplitude", type=float, default=1.0)
    ap.add_argument("--event_kernel", choices=["gaussian", "emg"], default="gaussian")
    ap.add_argument("--event_emg_tau", type=float, default=2.65)
    ap.add_argument("--device", default=None)
    ap.add_argument("--divide", type=int, default=0)
    ap.add_argument("--limit_dirs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = load_config_from_yaml(args.config)
    config.test_voxel_dirs = [envconfig.remap_data_dir(d) for d in config.test_voxel_dirs]
    config.test_annotation_dirs = [envconfig.remap_data_dir(d) for d in config.test_annotation_dirs]
    config.checkpoint_path = envconfig.remap_topm(config.checkpoint_path)
    if args.device:
        config.device = args.device
    if args.ckpt:
        config.checkpoint_path = args.ckpt
    if args.divide:
        config.divide = args.divide
    if args.limit_dirs:
        config.test_voxel_dirs = config.test_voxel_dirs[: args.limit_dirs]
        config.test_annotation_dirs = config.test_annotation_dirs[: args.limit_dirs]
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    model = get_model(config).to(device)
    load_checkpoint(config.checkpoint_path, model, device)
    model.eval()

    # dataset WITHOUT the _load_voxel_grid hook: we load raw and transform manually so we
    # keep both raw (for peaks) and transformed (for the model) under the same crop.
    ds = VoxelDatasetWithToMe(
        voxel_dirs=config.test_voxel_dirs, annotation_dirs=config.test_annotation_dirs,
        target_size=config.target_size, divide=config.divide,
        y_crop_top=config.y_crop_top, y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front, z_crop_back=config.z_crop_back,
    )
    transform = make_transform(args, device)
    nc = config.num_classes
    ut, th = config.use_threshold_prediction, config.prediction_threshold
    print(f"frames={len(ds.voxel_files)} compress={args.compress} repr={args.event_repr} k={args.event_k}")

    peak_cm = np.zeros((nc, nc), dtype=np.int64)
    vox_cm = np.zeros((nc, nc), dtype=np.int64)
    for i in tqdm(range(len(ds.voxel_files)), desc="peak-eval"):
        raw = ds._load_voxel_grid(ds.voxel_files[i]).astype(np.float32)   # (X,Y,700) raw
        ann = ds._load_annotation_voxel(ds.annotation_files[i])
        synth = transform(raw) if args.compress != "none" else raw        # (X,Y,700)
        # y/z crop both with the dataset's own methods
        raw = ds._apply_z_crop(ds._apply_y_crop(raw))
        synth = ds._apply_z_crop(ds._apply_y_crop(synth))
        ann = ds._apply_z_crop(ds._apply_y_crop(ann))
        start = ds._determine_crop_coordinates(raw.shape)                 # np.random (seed-fixed)
        raw_c = ds._preprocess_voxel(raw, start)
        synth_c = ds._preprocess_voxel(synth, start)
        ann_c = ds._preprocess_annotation(ann, start)

        pred = predict_dhw(model, synth_c, device, ut, th)               # (D,H,W)
        raw_dhw = np.transpose(raw_c, (2, 1, 0))                          # (D,H,W)
        ann_dhw = np.transpose(ann_c, (2, 1, 0)).astype(np.int64)

        # voxel-level CM (cross-check vs run_eval)
        p, t = pred.ravel(), ann_dhw.ravel()
        vox_cm += np.bincount(t * nc + p, minlength=nc * nc).reshape(nc, nc)
        # peak-level CM at RAW-waveform peaks (paper population)
        peaks = detect_peaks_in_voxel(raw_dhw)
        pm = evaluate_peaks(pred, ann_dhw, raw_dhw, peaks, ignore_labels=[], num_classes=nc)
        peak_cm += pm["peak_confusion_matrix"]

    def f1_mean(cm):
        m = calculate_metrics_from_confusion_matrix(cm, [])
        return float(np.mean([m["f1"][i] for i in SIGNAL])), {LABEL_MAP[i]: float(m["f1"][i]) for i in SIGNAL}

    pk, pk_pc = f1_mean(peak_cm)
    vx, vx_pc = f1_mean(vox_cm)
    res = {
        "compress": args.compress, "event_repr": args.event_repr, "event_k": args.event_k,
        "checkpoint": config.checkpoint_path,
        "peak_macro_f1": pk, "peak_per_class_f1": pk_pc,
        "voxel_macro_f1": vx, "voxel_per_class_f1": vx_pc,
        "peak_confusion_matrix": peak_cm.tolist(),
    }
    print(json.dumps({"peak_macro_f1": pk, "peak_per_class_f1": pk_pc,
                      "voxel_macro_f1": vx}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
