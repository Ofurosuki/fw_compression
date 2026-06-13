"""Precompute the PAPER peak set for the test split, once, model-independently.

``detect_peaks_in_voxel`` (find_peaks(height=max*0.1, width=3) per pixel) depends
only on the raw waveform, not on any model — so we run it once per test frame and
cache the peak coordinates + the annotation label at each peak. Every config's
evaluation then just indexes its predicted dense volume at these peaks (fast),
instead of re-running the ~7 s/frame scipy loop 9x.

  PYTHONPATH=<repo>/src uv run python -m eventnet.cache_test_peaks \
      --frame_stride 3 --shard 0 --nshard 4
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from hist_lidar.preprocess.custom_blosc2 import load_blosc2
from hist_lidar.training.test_ViT3D import detect_peaks_in_voxel

from eventnet import paths
from eventnet.cache_events import frame_key


def cache_path(frame_stride: int, root: str = paths.CACHE_ROOT) -> str:
    return os.path.join(root, f"test_peaks_stride{frame_stride}")


def build(frame_stride, shard, nshard):
    frames = paths.list_frames("test", frame_stride=frame_stride)
    if nshard > 1:
        frames = frames[shard::nshard]
    out_dir = cache_path(frame_stride)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[peaks] test frames(shard)={len(frames)} -> {out_dir}", flush=True)
    t0 = time.time()
    for i, (vpath, apath) in enumerate(frames):
        out = os.path.join(out_dir, frame_key(vpath) + ".npz")
        if os.path.exists(out):
            continue
        vox = paths.apply_crop(load_blosc2(vpath).astype(np.float32))
        ann = paths.apply_crop(load_blosc2(apath))
        raw_TXY = np.ascontiguousarray(vox.transpose(2, 0, 1))      # (T,X,Y)
        ann_TXY = np.ascontiguousarray(ann.transpose(2, 0, 1))
        peaks = detect_peaks_in_voxel(raw_TXY)                       # list[(d,y,x)]
        if peaks:
            pk = np.array(peaks, dtype=np.int32)                    # (M,3) cols d,y(=x_idx),x(=y_idx)
            ann_at = ann_TXY[pk[:, 0], pk[:, 1], pk[:, 2]].astype(np.int8)
        else:
            pk = np.zeros((0, 3), np.int32); ann_at = np.zeros((0,), np.int8)
        np.savez_compressed(out, peaks=pk.astype(np.int16), ann_at_peak=ann_at)
        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            dt = time.time() - t0
            print(f"  {i+1}/{len(frames)}  {dt:.0f}s ({dt/(i+1):.1f}s/frame)", flush=True)
    print(f"[peaks] done shard{shard} in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    build(args.frame_stride, args.shard, args.nshard)


if __name__ == "__main__":
    main()
