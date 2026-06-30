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
from eventnet.losses import masked_dice, masked_focal, masked_weighted_ce, vrex_loss
from eventnet.metrics import event_confusion, f1_from_cm
from eventnet.model import build_model, count_params
from eventnet.paths import NUM_CLASSES


@torch.no_grad()
def evaluate_event_level(model, loader, device, amp=False):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for b in loader:
        ev = b["events"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(ev, b["valid"].to(device))
        pred = logits.argmax(-1).cpu().numpy().ravel()
        cm += event_confusion(pred, b["labels"].numpy().ravel(),
                              b["valid"].numpy().ravel().astype(bool))
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--feature_mode", default="tdtaw",
                    choices=["t_only", "t_dt", "ta", "tdta", "taw", "tdtaw",
                             "taE", "tdtaE", "tdtaEw", "tawD", "tawI", "tawi",
                             "tawT", "tT"])
    ap.add_argument("--frame_stride", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--arch", default="v1",
                    choices=["v1", "v2", "v2sa", "setopm", "setopm2", "setopm3", "setopm2r"])
    ap.add_argument("--method", default="erm", choices=["erm", "vrex"],
                    help="erm=pooled CE; vrex=domain-generalization (scene=environment)")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"],
                    help="erm loss type (vrex always uses weighted CE)")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--dice_weight", type=float, default=0.0,
                    help="add dice_weight * masked_dice(signal classes) to the loss (ToPM uses focal+dice)")
    ap.add_argument("--vrex_beta", type=float, default=10.0, help="V-REx penalty weight")
    ap.add_argument("--vrex_warmup", type=int, default=10, help="epochs at beta=0 before ramp")
    ap.add_argument("--emb_dim", type=int, default=0, help="0 = arch default (v1:32, v2:48)")
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--attn_layers", type=int, default=2)
    ap.add_argument("--unet_levels", type=int, default=3)
    # SparseEventToPM (arch=setopm) hyperparameters
    ap.add_argument("--depth", type=int, default=12, help="setopm: # attention blocks")
    ap.add_argument("--window_size", type=int, default=8, help="setopm: spatial window")
    ap.add_argument("--ffn_mult", type=int, default=2, help="setopm: FFN expansion")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (needed for setopm efficient attn)")
    ap.add_argument("--crop", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--class_weights", type=float, nargs=4, default=[0.2, 1.0, 2.0, 2.0])
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpus", default="", help="comma-list e.g. '3,1' -> DataParallel "
                    "(one model across both GPUs, ~1.7x); overrides --device")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if gpu_ids:
        args.device = f"cuda:{gpu_ids[0]}"
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

    model = build_model(args.arch, K=args.K, in_dim=feature_dim(args.feature_mode),
                        num_classes=NUM_CLASSES, emb_dim=(args.emb_dim or None),
                        base_channels=args.base_channels, attn_heads=args.attn_heads,
                        attn_layers=args.attn_layers, unet_levels=args.unet_levels,
                        depth=args.depth, window_size=args.window_size,
                        ffn_mult=args.ffn_mult).to(device)
    print(f"arch={args.arch} params={count_params(model)/1e6:.2f}M  in_dim={feature_dim(args.feature_mode)}")
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"DataParallel over GPUs {gpu_ids} (primary {gpu_ids[0]})")
    cw = torch.tensor(args.class_weights, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:                          # linear warmup -> cosine (attn-friendly)
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.02, total_iters=args.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - args.warmup_epochs))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warm, cos], milestones=[args.warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        pen = 0.0
        # V-REx: ramp beta from 0 after the warmup (Krueger 2021 anneal)
        beta = 0.0
        if args.method == "vrex" and ep >= args.vrex_warmup:
            beta = args.vrex_beta
        for b in tl:
            ev = b["events"].to(device)
            lab = b["labels"].to(device)
            val = b["valid"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                logits = model(ev, val)
                if args.method == "vrex":
                    loss, _, penalty = vrex_loss(logits, lab, val, b["scene"].to(device), cw, beta)
                    pen += float(penalty)
                elif args.loss == "focal":
                    loss = masked_focal(logits, lab, val, cw, args.focal_gamma)
                else:
                    loss = masked_weighted_ce(logits, lab, val, cw)
                if args.dice_weight > 0 and args.method != "vrex":
                    loss = loss + args.dice_weight * masked_dice(logits, lab, val)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        cm = evaluate_event_level(model, vl, device, amp=args.amp)
        f1m, per, _ = f1_from_cm(cm)
        history.append({"epoch": ep, "loss": tot / max(1, len(tl)), "val_f1_mean": f1m, "per": per})
        flag = ""
        if f1m > best:
            best = f1m
            raw = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({"state_dict": raw.state_dict(), "args": vars(args),
                        "in_dim": feature_dim(args.feature_mode), "val_f1_mean": f1m,
                        "val_per_class": per, "epoch": ep},
                       os.path.join(args.save_dir, "best.pth"))
            flag = " *"
        print(f"ep{ep:02d} loss={tot/max(1,len(tl)):.4f} "
              f"{'pen=%.4f β=%.1f ' % (pen/max(1,len(tl)), beta) if args.method=='vrex' else ''}"
              f"val_f1={f1m:.4f} obj={per['object']:.3f} glass={per['glass']:.3f} "
              f"ghost={per['ghost']:.3f} [{time.time()-t0:.0f}s]{flag}", flush=True)

    json.dump({"best_val_f1_mean": best, "history": history, "args": vars(args)},
              open(os.path.join(args.save_dir, "train_log.json"), "w"), indent=2)
    print(f"best val event-F1-mean={best:.4f}  -> {args.save_dir}/best.pth")


if __name__ == "__main__":
    main()
