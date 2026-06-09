"""Downstream Ghost-FWL evaluation of fw_compression autoencoders.

Loads the frozen, pre-trained Ghost-FWL segmentation model (vit3d / FWL-ToPM)
from the evolved repo, optionally inserts a "compress -> reconstruct" transform on
the RAW (T=700) per-pixel waveforms BEFORE the downstream's own crop pipeline, and
reports the per-voxel F1-mean (object/glass/ghost, Noise excluded) — the same
metric the paper reports.

The compression is applied by monkey-patching ``VoxelDataset._load_voxel_grid`` so
all of the downstream's cropping/normalisation is reused untouched. We skip the
repo's slow per-pixel scipy peak detection (it dominates runtime and is not needed
for the headline F1).

Usage:
  PYTHONPATH=<repo>/src uv run python downstream/run_eval.py \
      --config downstream/configs/evalA_split2_test.yaml \
      --compress none --device cuda:1 --out downstream/outputs/evalA_orig.json
  ... --compress ae --ae_ckpt runs/real_split2_1d/learnable_linear_K32/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hist_lidar.config import load_config_from_yaml
from hist_lidar.data import VoxelDatasetWithToMe, voxel_collate_fn
from hist_lidar.data import dataset_voxel as _dv
from hist_lidar.training.test_ViT3D import calculate_metrics_from_confusion_matrix
from hist_lidar.utils import get_model, load_checkpoint, set_seed

LABEL_MAP = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}


# --------------------------------------------------------------------------- #
# Compression: load a trained fw_compression AE and apply it to a raw voxel.
# --------------------------------------------------------------------------- #
def load_ae(ckpt_path, device):
    """Return (model, kind, meta). kind in {'1d','spatial'}; rebuilt from the ckpt."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "P" in ck:  # spatial 4x4 autoencoder
        from compression.spatial_coding import build_spatial_autoencoder
        m = build_spatial_autoencoder(T=ck["T"], K=ck["K"], P=ck["P"])
        kind = "spatial"
        meta = {"T": ck["T"], "K": ck["K"], "P": ck["P"]}
    else:  # per-pixel 1D autoencoder
        from compression.autoencoder import build_autoencoder
        m = build_autoencoder(ck["encoder_name"], T=ck["T"], K=ck["K"],
                              decoder_name=ck["decoder_name"])
        kind = "1d"
        meta = {"T": ck["T"], "K": ck["K"], "encoder_name": ck["encoder_name"]}
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    return m, kind, meta


@torch.no_grad()
def _ae_recon_rows(model, wn, chunk=32768):
    """wn: (N, T) normalised; return (N, T) reconstruction."""
    out = torch.empty_like(wn)
    for i in range(0, wn.shape[0], chunk):
        out[i:i + chunk] = model(wn[i:i + chunk])[0]
    return out


@torch.no_grad()
def compress_voxel(vox_xyz, model, kind, device, eps=1e-6, block=4):
    """vox_xyz: np.float32 (X, Y, T). Per-pixel max-normalise -> AE -> de-normalise.
    Background (max<=eps) pixels are passed through as-is (left ~zero)."""
    X, Y, T = vox_xyz.shape
    v = torch.from_numpy(np.ascontiguousarray(vox_xyz)).to(device)

    if kind == "1d":
        w = v.reshape(-1, T)                      # (P, T)
        mx = w.amax(dim=1, keepdim=True)
        valid = (mx > eps).squeeze(1)
        wn = torch.where(mx > eps, w / mx, w)
        rec = _ae_recon_rows(model, wn)
        rec = torch.clamp(rec, min=0.0) * mx
        w_out = torch.where(valid.unsqueeze(1), rec, w)
        return w_out.reshape(X, Y, T).cpu().numpy()

    # spatial: tile (X,Y) into block x block groups -> (Nblocks, block*block, T)
    B = block
    nbx, nby = X // B, Y // B
    v = v[: nbx * B, : nby * B, :]
    blocks = (v.reshape(nbx, B, nby, B, T)
               .permute(0, 2, 1, 3, 4)
               .reshape(nbx * nby, B * B, T))      # (Nb, 16, T)
    mx = blocks.amax(dim=2, keepdim=True)          # per-pixel max within block
    bn = torch.where(mx > eps, blocks / mx, blocks)
    rec = torch.empty_like(bn)
    for i in range(0, bn.shape[0], 4096):
        rec[i:i + 4096] = model(bn[i:i + 4096])[0]
    rec = torch.clamp(rec, min=0.0) * mx
    rec = torch.where(mx > eps, rec, blocks)
    out = (rec.reshape(nbx, nby, B, B, T)
              .permute(0, 2, 1, 3, 4)
              .reshape(nbx * B, nby * B, T))
    # if X/Y not divisible by B, pad back the untouched border with originals
    if out.shape[0] != X or out.shape[1] != Y:
        full = torch.from_numpy(np.ascontiguousarray(vox_xyz)).to(device)
        full[: out.shape[0], : out.shape[1], :] = out
        out = full
    return out.cpu().numpy()


def install_compression_hook(model, kind, device):
    orig = _dv.VoxelDataset._load_voxel_grid

    def patched(self, file_path):
        vox = orig(self, file_path).astype(np.float32)
        return compress_voxel(vox, model, kind, device)

    _dv.VoxelDataset._load_voxel_grid = patched
    return orig


# --------------------------------------------------------------------------- #
# Evaluation loop (confusion-matrix F1, no peak detection).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_labels,
             use_threshold, threshold):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    ig = torch.tensor(ignore_labels, device=device)
    for batch in tqdm(loader, desc="eval"):
        vox = batch["voxel_grids"].float().to(device)
        ann = batch["annotations"].long().to(device)
        out = model(vox)                          # (B, C, D, H, W)
        if ig.numel() > 0:
            out[:, ig] = -1e9
        if use_threshold:
            prob = torch.softmax(out, dim=1)
            mp, am = torch.max(prob, dim=1)
            pred = torch.where(mp >= threshold, am, torch.zeros_like(am))
        else:
            pred = torch.argmax(out, dim=1)
        p = pred.cpu().numpy().ravel()
        t = ann.cpu().numpy().ravel()
        m = ~np.isin(t, ignore_labels)
        idx = t[m] * num_classes + p[m]
        cm += np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
        del out, pred, vox, ann
    return cm


# --------------------------------------------------------------------------- #
# Waveform-level visualisation: a FIXED set of 6 pixel waveforms, orig vs recon.
# --------------------------------------------------------------------------- #
def pick_fixed_waveforms(config, n=6, seed=42):
    """Deterministically pick n signal-bearing pixel waveforms from the FIRST test
    voxel file, so the same pixels are shown across every compression config."""
    import blosc2
    vpath = config.test_voxel_dirs[0]
    files = sorted(p for p in os.listdir(vpath) if p.endswith("_voxel.b2"))
    f0 = os.path.join(vpath, files[0])
    apath = config.test_annotation_dirs[0]
    af = os.path.join(apath, files[0].replace("_voxel.b2", "_annotation_voxel.b2"))
    vox = blosc2.load_array(f0).astype(np.float32)          # (X,Y,T)
    ann = blosc2.load_array(af)
    X, Y, T = vox.shape
    flatv = vox.reshape(-1, T)
    flata = ann.reshape(-1, T)
    has_ghost = (flata == 3).any(1)
    has_sig = (flata > 0).any(1) & (flatv.max(1) > 0)
    rng = np.random.default_rng(seed)
    gi = np.where(has_sig & has_ghost)[0]
    oi = np.where(has_sig & ~has_ghost)[0]
    ng = min(n // 2, len(gi)); no = n - ng
    sel = np.concatenate([rng.choice(gi, ng, replace=False),
                          rng.choice(oi, min(no, len(oi)), replace=False)])
    return f0, [(int(i), flatv[i].copy(), flata[i].copy()) for i in sel]


def viz_waveforms(config, ae_model, kind, device, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, picks = pick_fixed_waveforms(config)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (pix, w, a) in zip(axes.ravel(), picks):
        mx = float(w.max()) or 1.0
        wn = w / mx
        with torch.no_grad():
            x = torch.from_numpy(wn[None]).float().to(device)
            if kind == "spatial":
                x = x[None].repeat(1, 16, 1)       # replicate into a block; use pixel 0
                rec = ae_model(x)[0][0, 0].cpu().numpy()
            else:
                rec = ae_model(x)[0][0].cpu().numpy()
        rec = np.clip(rec, 0, None)
        ax.plot(wn, "k-", lw=1.0, label="orig")
        ax.plot(rec, "r-", lw=1.0, alpha=0.8, label="recon")
        for c, col in [(1, "tab:green"), (2, "tab:blue"), (3, "tab:red")]:
            bins = np.where(a == c)[0]
            if len(bins):
                pk = bins[np.argmax(wn[bins])]
                ax.plot(pk, rec[pk], "o", color=col, ms=7)
        npk = int(((a > 0)[:-1] != (a > 0)[1:]).sum() // 1)
        ax.set_title(f"pix#{pix} ghost={(a==3).any()} npk≈{int((np.diff((a>0).astype(int))==1).sum())}",
                     fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close()
    print("saved", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--compress", choices=["none", "ae"], default="none")
    ap.add_argument("--ae_ckpt", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit_dirs", type=int, default=0, help="use only first N dirs (smoke)")
    ap.add_argument("--divide", type=int, default=0, help="subsample 1/divide of frames (0=use config)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--viz_out", default=None)
    args = ap.parse_args()

    config = load_config_from_yaml(args.config)
    if args.device:
        config.device = args.device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    if args.limit_dirs:
        config.test_voxel_dirs = config.test_voxel_dirs[:args.limit_dirs]
        config.test_annotation_dirs = config.test_annotation_dirs[:args.limit_dirs]
    if args.divide:
        config.divide = args.divide

    # frozen downstream model
    model = get_model(config).to(device)
    load_checkpoint(config.checkpoint_path, model, device)
    model.eval()

    # optional compression hook (applied to raw T=700 voxel before crops)
    ae_model = ae_kind = ae_meta = None
    nw = config.num_workers
    if args.compress == "ae":
        assert args.ae_ckpt, "--ae_ckpt required for --compress ae"
        ae_model, ae_kind, ae_meta = load_ae(args.ae_ckpt, device)
        install_compression_hook(ae_model, ae_kind, device)
        nw = 0  # GPU compression in __getitem__ needs the main process
        print(f"[compress] {ae_kind} {ae_meta}")

    ds = VoxelDatasetWithToMe(
        voxel_dirs=config.test_voxel_dirs, annotation_dirs=config.test_annotation_dirs,
        target_size=config.target_size, divide=config.divide,
        y_crop_top=config.y_crop_top, y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front, z_crop_back=config.z_crop_back,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                        num_workers=nw, collate_fn=voxel_collate_fn, pin_memory=False)
    print(f"frames={len(ds)} batches={len(loader)} num_workers={nw}")

    cm = evaluate(model, loader, device, config.num_classes,
                  config.ignore_visualize_labels, config.use_threshold_prediction,
                  config.prediction_threshold)
    met = calculate_metrics_from_confusion_matrix(cm, config.ignore_visualize_labels)
    valid = [i for i in range(config.num_classes) if i not in config.ignore_visualize_labels]

    res = {
        "compress": args.compress, "ae_ckpt": args.ae_ckpt, "ae_meta": ae_meta,
        "macro_f1": float(met["macro_f1"]),
        "per_class_f1": {LABEL_MAP[i]: float(met["f1"][i]) for i in valid},
        "per_class_precision": {LABEL_MAP[i]: float(met["precision"][i]) for i in valid},
        "per_class_recall": {LABEL_MAP[i]: float(met["recall"][i]) for i in valid},
        "macro_iou": float(met["macro_iou"]),
        "confusion_matrix": cm.tolist(),
        "checkpoint": config.checkpoint_path,
    }
    print(json.dumps({k: res[k] for k in ["compress", "macro_f1", "per_class_f1"]}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)

    if args.viz_out and ae_model is not None:
        viz_waveforms(config, ae_model, ae_kind, device, args.viz_out,
                      title=f"{ae_kind} {ae_meta}  F1-mean={res['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
