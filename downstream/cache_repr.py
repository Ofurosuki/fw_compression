"""Pre-compute a compressed input representation as uint16 voxel grids on disk, so
ToPM can be retrained on it with normal multi-worker data loading.

Why cache: the event/AE transform runs ~0.67 s/frame on GPU and forces num_workers=0
(GPU work in __getitem__). On-the-fly that is ~50 h for a 50-epoch run. Cached as
uint16 (the dataset's native dtype; rounding integerises the synthesised Gaussians and
zeroes their far tails) each frame is ~4.4 MB and training reads it like any real voxel
file at ~6 min/epoch.

Layout: for an original dir `<...>/ghost_dataset/<rel>` the cache mirrors it at
`<cache_root>/<rel>`. Voxel dirs get transformed *_voxel.b2 files (strided 1/`stride`
subset = the divide-equivalent training subset, frozen so it is reproducible — unlike the
dataset's random.sample divide); annotation dirs get symlinks to the SAME strided files
(labels are unchanged). Train then runs with the matching voxel+annot cache dirs and
divide=1. ``run_retrain.py --cache_root`` consumes this by the same path rewrite.

Usage (shard over GPUs):
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  for s in 0 1 2; do uv-py downstream/cache_repr.py --rep event --event_repr taw --event_k 4 \
      --cache_root /data3/user/yoshida/fwc_cache/taw_k4 --device cuda:$s \
      --nshard 3 --shard $s & done; wait
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import blosc2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_eval       # noqa: E402  event_voxel / compress_voxel / load_ae
import run_retrain    # noqa: E402  build_config (remapped dir lists)


from run_retrain import cache_rel  # noqa: E402  shared original->cache path mapping


def transform_fn(args, device):
    if args.rep == "event":
        ep = {
            "k": args.event_k, "representation": args.event_repr,
            "intensity_mode": args.event_intensity, "smooth_sigma": args.event_smooth_sigma,
            "min_height": args.event_min_height, "min_distance": args.event_min_distance,
            "fixed_width": args.event_fixed_width, "fixed_amplitude": args.event_fixed_amplitude,
            "kernel": args.event_kernel, "emg_tau": args.event_emg_tau,
        }
        return lambda raw: run_eval.event_voxel(raw, ep, device)
    elif args.rep == "ae":
        assert args.ae_ckpt, "--ae_ckpt required for --rep ae"
        model, kind, _ = run_eval.load_ae(args.ae_ckpt, device)
        return lambda raw: run_eval.compress_voxel(raw, model, kind, device)
    raise ValueError("cache only makes sense for rep in {event, ae}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["event", "ae"], required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame (divide-equiv)")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit_train_dirs", type=int, default=0, help="smoke: first N train dirs")
    ap.add_argument("--limit_val_dirs", type=int, default=0, help="smoke: first N val dirs")
    ap.add_argument("--config_train_src", default=run_retrain.TRAIN_SRC)
    ap.add_argument("--config_eval_src", default=run_retrain.EVAL_SRC)
    # event params (mirror run_eval / run_retrain)
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
    ap.add_argument("--ae_ckpt", default=None)
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # remapped train+val dir lists, paired (voxel, annotation)
    ns = argparse.Namespace(rep="none", run_name="_cache", save_root="/tmp",
        config_train_src=args.config_train_src, config_eval_src=args.config_eval_src,
        device=args.device, epochs=1, divide=0, num_workers=0, seed=42, is_log=False,
        limit_train_dirs=0, limit_val_dirs=0)
    cfg, _ = run_retrain.build_config(ns)
    tv, ta = cfg["train_voxel_dirs"], cfg["train_annotation_dirs"]
    vv, va = cfg["valid_voxel_dirs"], cfg["valid_annotation_dirs"]
    if args.limit_train_dirs:
        tv, ta = tv[: args.limit_train_dirs], ta[: args.limit_train_dirs]
    if args.limit_val_dirs:
        vv, va = vv[: args.limit_val_dirs], va[: args.limit_val_dirs]
    pairs = list(zip(tv, ta)) + list(zip(vv, va))
    pairs = [p for i, p in enumerate(pairs) if i % args.nshard == args.shard]

    fn = transform_fn(args, device)
    n_frames = 0
    for di, (vdir, adir) in enumerate(pairs):
        vfiles = sorted(f for f in os.listdir(vdir) if f.endswith("_voxel.b2"))[:: args.stride]
        afiles = sorted(f for f in os.listdir(adir) if f.endswith("_annotation_voxel.b2"))[:: args.stride]
        cv, ca = cache_rel(vdir, args.cache_root), cache_rel(adir, args.cache_root)
        os.makedirs(cv, exist_ok=True)
        os.makedirs(ca, exist_ok=True)
        for vf, af in zip(vfiles, afiles):
            outp = os.path.join(cv, vf)
            if not os.path.exists(outp):
                raw = blosc2.load_array(os.path.join(vdir, vf)).astype(np.float32)
                syn = fn(raw)
                u16 = np.clip(np.rint(syn), 0, 65535).astype(np.uint16)
                blosc2.save_array(np.ascontiguousarray(u16), outp, mode="w")
            # symlink the (unchanged) annotation
            la = os.path.join(ca, af)
            if not os.path.lexists(la):
                os.symlink(os.path.join(adir, af), la)
            n_frames += 1
        print(f"[shard {args.shard}] {di + 1}/{len(pairs)} {vdir.split('ghost_dataset/')[1]} "
              f"({len(vfiles)} frames, total {n_frames})", flush=True)
    print(f"[shard {args.shard}] DONE {n_frames} frames -> {args.cache_root}", flush=True)


if __name__ == "__main__":
    main()
