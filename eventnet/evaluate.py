"""Paper-compliant evaluation of a trained EventTensorNet on the SPLIT2 test set.

For each test frame: load raw waveform + annotation, apply the downstream y/z
crop, extract top-K events, run the net to predict a class per event, paint a
dense (T,X,Y) predicted-label volume from those events, then score with the
Ghost-FWL repo's own peak-level metric (find_peaks(height=max*0.1, width=3) on
the raw wave -> confusion matrix of pred-vs-annotation at peak bins). This is
the paper's "F1-mean" scoring population (see SCORE_DISCREPANCY.md), so the
number IS comparable to the paper's ~0.592 and the frozen-model peak baseline.

An event-level F1 (scored at the net's own extracted events) is also reported.

Example:
  PYTHONPATH=<repo>/src uv run python -m eventnet.evaluate \
      --checkpoint outputs/eventnet/tdtaw_K4/best.pth \
      --frame_stride 3 --device cuda:0 --out outputs/eventnet/tdtaw_K4/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from hist_lidar.preprocess.custom_blosc2 import load_blosc2

from eventnet import paths
from eventnet.cache_events import frame_key
from eventnet.cache_test_peaks import cache_path as peak_cache_path
from eventnet.data import assemble_features, feature_dim
from eventnet.events import assign_labels, extract_frame_events
from eventnet.metrics import (event_confusion, f1_from_cm, paint_pred_dense,
                              peak_cm_from_cache, peak_eval_frame)
from eventnet.model import EventTensorNet
from eventnet.paths import NUM_CLASSES, T_CROPPED


@torch.no_grad()
def predict_frame(model, vox_crop, k, mode, device):
    """Extract top-K events and predict a class per event.

    Returns (t_bin, a, w, valid, pred) each (X, Y, K)."""
    events, valid = extract_frame_events(vox_crop, device, k=k)   # (X,Y,K,3) sorted by t
    ev = torch.from_numpy(events).to(device)
    val = torch.from_numpy(valid).to(device)
    t_bin, a, w = ev[..., 0], ev[..., 1], ev[..., 2]
    feat = assemble_features(t_bin, a, w, val, mode)              # (X,Y,K,F)
    logits = model(feat.unsqueeze(0))[0]                          # (X,Y,K,C)
    pred = logits.argmax(-1)
    pred = torch.where(val, pred, torch.zeros_like(pred))
    return (t_bin.cpu().numpy(), a.cpu().numpy(), w.cpu().numpy(),
            valid, pred.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frame_stride", type=int, default=3, help="test frame subsample")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ca = ck["args"]
    K, mode = ca["K"], ca["feature_mode"]
    model = EventTensorNet(K=K, in_dim=feature_dim(mode), emb_dim=ca["emb_dim"],
                           num_classes=NUM_CLASSES, base_channels=ca["base_channels"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    print(f"[eval] {mode} K={K}  val_f1={ck.get('val_f1_mean'):.4f}")

    frames = paths.list_frames("test", frame_stride=args.frame_stride)
    if args.limit:
        frames = frames[:args.limit]
    pkdir = peak_cache_path(args.frame_stride)
    use_cache = os.path.isdir(pkdir)
    print(f"[eval] test frames={len(frames)}  peak_cache={'HIT '+pkdir if use_cache else 'MISS (slow path)'}")

    peak_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    ev_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    t0 = time.time()
    for i, (vpath, apath) in enumerate(frames):
        vox = paths.apply_crop(load_blosc2(vpath).astype(np.float32))   # (X,Y,T)
        ann = paths.apply_crop(load_blosc2(apath))
        t_bin, a, w, valid, pred = predict_frame(model, vox, K, mode, device)

        # event-level CM (at the net's own events; label = ann at exact bin)
        ev_lab = assign_labels(ann, np.stack([t_bin, a, w], -1), valid)
        ev_cm += event_confusion(pred.ravel(), ev_lab.ravel(), valid.ravel())

        # paper peak-level CM (dense reconstruction -> repo find_peaks scoring)
        pred_dense = paint_pred_dense(t_bin, w, pred, valid, a, T_CROPPED)  # (T,X,Y)
        pkf = os.path.join(pkdir, frame_key(vpath) + ".npz") if use_cache else None
        if pkf and os.path.exists(pkf):
            z = np.load(pkf)
            peak_cm += peak_cm_from_cache(pred_dense, z["peaks"], z["ann_at_peak"])
        else:
            ann_TXY = np.ascontiguousarray(ann.transpose(2, 0, 1))
            raw_TXY = np.ascontiguousarray(vox.transpose(2, 0, 1))
            peak_cm += peak_eval_frame(pred_dense, ann_TXY, raw_TXY)
        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            dt = time.time() - t0
            print(f"  {i+1}/{len(frames)}  {dt:.0f}s ({dt/(i+1):.1f}s/frame)", flush=True)

    peak_f1, peak_per, _ = f1_from_cm(peak_cm)
    ev_f1, ev_per, _ = f1_from_cm(ev_cm)
    res = {
        "feature_mode": mode, "K": K, "checkpoint": args.checkpoint,
        "frame_stride": args.frame_stride, "n_frames": len(frames),
        "f1_mean": peak_f1,                       # PAPER-COMPLIANT peak-level F1-mean
        "object_f1": peak_per["object"], "glass_f1": peak_per["glass"],
        "ghost_f1": peak_per["ghost"],
        "peak_per_class_f1": peak_per,
        "event_level_f1_mean": ev_f1, "event_per_class_f1": ev_per,
        "peak_confusion_matrix": peak_cm.tolist(),
        "event_confusion_matrix": ev_cm.tolist(),
    }
    print(json.dumps({k: res[k] for k in
                      ["feature_mode", "K", "f1_mean", "object_f1", "glass_f1",
                       "ghost_f1", "event_level_f1_mean"]}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
