"""Train EventTensorNet on cached top-K event tensors.

Example:
  PYTHONPATH=<repo>/src uv run python -m eventnet.train \
      --K 4 --feature_mode tdtaw --frame_stride 7 \
      --epochs 40 --device cuda:0 --save_dir outputs/eventnet/tdtaw_K4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eventnet.data import EventFrameDataset, feature_dim
from eventnet.losses import masked_weighted_ce
from eventnet.metrics import event_confusion, f1_from_cm
from eventnet.model import EventTensorNet, count_params
from eventnet.paths import NUM_CLASSES


@torch.no_grad()
def evaluate_event_level(model, loader, device):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for b in loader:
        ev = b["events"].to(device)
        logits = model(ev)
        pred = logits.argmax(-1).cpu().numpy().ravel()
        cm += event_confusion(pred, b["labels"].numpy().ravel(),
                              b["valid"].numpy().ravel().astype(bool))
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--feature_mode", default="tdtaw",
                    choices=["t_only", "t_dt", "ta", "tdta", "taw", "tdtaw"])
    ap.add_argument("--frame_stride", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--crop", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--class_weights", type=float, nargs=4, default=[0.2, 1.0, 2.0, 2.0])
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    tr = EventFrameDataset("train", args.frame_stride, args.K, args.feature_mode,
                           crop=tuple(args.crop), augment=True, limit=args.limit)
    va = EventFrameDataset("val", args.frame_stride, args.K, args.feature_mode,
                           crop=None, augment=False, limit=args.limit)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"train frames={len(tr)} val frames={len(va)}")

    model = EventTensorNet(K=args.K, in_dim=feature_dim(args.feature_mode),
                           emb_dim=args.emb_dim, num_classes=NUM_CLASSES,
                           base_channels=args.base_channels).to(device)
    print(f"model params={count_params(model)/1e6:.2f}M  in_dim={feature_dim(args.feature_mode)}")
    cw = torch.tensor(args.class_weights, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for b in tl:
            ev = b["events"].to(device)
            lab = b["labels"].to(device)
            val = b["valid"].to(device)
            logits = model(ev)
            loss = masked_weighted_ce(logits, lab, val, cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        cm = evaluate_event_level(model, vl, device)
        f1m, per, _ = f1_from_cm(cm)
        history.append({"epoch": ep, "loss": tot / max(1, len(tl)), "val_f1_mean": f1m, "per": per})
        flag = ""
        if f1m > best:
            best = f1m
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "in_dim": feature_dim(args.feature_mode), "val_f1_mean": f1m,
                        "val_per_class": per, "epoch": ep},
                       os.path.join(args.save_dir, "best.pth"))
            flag = " *"
        print(f"ep{ep:02d} loss={tot/max(1,len(tl)):.4f} val_f1={f1m:.4f} "
              f"obj={per['object']:.3f} glass={per['glass']:.3f} ghost={per['ghost']:.3f} "
              f"[{time.time()-t0:.0f}s]{flag}", flush=True)

    json.dump({"best_val_f1_mean": best, "history": history, "args": vars(args)},
              open(os.path.join(args.save_dir, "train_log.json"), "w"), indent=2)
    print(f"best val event-F1-mean={best:.4f}  -> {args.save_dir}/best.pth")


if __name__ == "__main__":
    main()
