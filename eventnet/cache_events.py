"""Pre-extract top-K=8 event tensors + labels for train/val and cache to disk.

Raw voxels are ~287 MB each, so we extract once and store compact per-frame
``.npz`` files (events f16, valid bool, labels int8 — ~8 MB/frame). The test
split is NOT cached: the evaluator re-extracts events on the fly because it also
needs the raw waveform for paper-compliant peak detection.

Usage:
  PYTHONPATH=<repo>/src uv run python -m eventnet.cache_events \
      --split train --frame_stride 7 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from hist_lidar.preprocess.custom_blosc2 import load_blosc2

from eventnet import paths
from eventnet.events import KMAX, assign_labels, extract_frame_events


def cache_path(split: str, frame_stride: int, root: str = paths.CACHE_ROOT) -> str:
    return os.path.join(root, f"{split}_stride{frame_stride}")


def frame_key(vpath: str) -> str:
    parts = vpath.split(os.sep)
    scene = next(p for p in parts if "build" in p)
    hist = next(p for p in parts if p.startswith("hist"))
    stem = os.path.basename(vpath).replace("_voxel.b2", "")
    return f"{scene}__{hist}__{stem}"


def build(split: str, frame_stride: int, device: str, limit: int = 0,
          shard: int = 0, nshard: int = 1):
    frames = paths.list_frames(split, frame_stride=frame_stride)
    if limit:
        frames = frames[:limit]
    if nshard > 1:
        frames = frames[shard::nshard]
    out_dir = cache_path(split, frame_stride)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cache] {split} frames={len(frames)} -> {out_dir}")
    t0 = time.time()
    for i, (vpath, apath) in enumerate(frames):
        out = os.path.join(out_dir, frame_key(vpath) + ".npz")
        if os.path.exists(out):
            continue
        vox = paths.apply_crop(load_blosc2(vpath).astype(np.float32))
        ann = paths.apply_crop(load_blosc2(apath))
        events, valid = extract_frame_events(vox, device, k=KMAX)
        labels = assign_labels(ann, events, valid)
        np.savez_compressed(
            out,
            events=events.astype(np.float16),
            valid=valid,
            labels=labels,
        )
        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            dt = time.time() - t0
            print(f"  {i+1}/{len(frames)}  {dt:.0f}s  ({dt/(i+1):.2f}s/frame)", flush=True)
    print(f"[cache] done {split} in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--frame_stride", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    build(args.split, args.frame_stride, args.device, args.limit, args.shard, args.nshard)


if __name__ == "__main__":
    main()
