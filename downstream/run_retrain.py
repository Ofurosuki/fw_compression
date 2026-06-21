"""Retrain the FWL-ToPM downstream model on a *compressed input representation*.

Motivation (PI): to argue about a representation's *effectiveness* we must hold the
architecture fixed (= ToPM) and only change the input, otherwise representation and
architecture are confounded (cf. FW_Event_Net, which changed both). So we lift each
representation back to a T=700 pseudo-waveform (top-K events -> Gaussian synthesis, or
AE reconstruction) and **retrain ToPM from scratch** on it under one fixed recipe.

This drives the read-only Ghost-FWL repo's own ``train_vit3d`` WITHOUT editing it:
  * the input transform is the SAME monkey-patch on ``VoxelDataset._load_voxel_grid``
    used by ``run_eval.py`` (installed here before the dataset is built, so it applies
    to training too);
  * the training config is built by merging the repo's split2 *training* config
    (lr/epochs/loss/optimizer + the 280/60 train/val dirs) with this project's eval
    config (the ToPM architecture + pruning params + crop/target), so the retrained
    model is byte-for-byte the same ToPM the eval harness scores.

Decomposition this enables: frozen-ToPM(rep) conflates (i) representation info-loss with
(ii) train/test domain shift; retrained-ToPM(rep) isolates (i). Ceiling = retrained
ToPM on the FULL waveform under the same recipe (should reproduce neurips_best).

Usage (run from project root, with the repo on PYTHONPATH):
  export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
  # smoke: 2 train dirs, 1 val dir, 1 epoch, heavy frame subsample
  uv run python downstream/run_retrain.py --rep none --run_name smoke_full \
      --device cuda:0 --epochs 1 --divide 10 --limit_train_dirs 2 --limit_val_dirs 1
  # taw-K4 event representation
  uv run python downstream/run_retrain.py --rep event --event_repr taw --event_k 4 \
      --run_name taw_k4 --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

# project root (for `compression`) and downstream/ (for `run_eval`) on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import envconfig  # noqa: E402  (machine-dependent paths; see env.yaml.example)
import run_eval  # noqa: E402  (hook installers + AE loader live here)

# Default source configs.
REPO = envconfig.topm_repo_root()
TRAIN_SRC = f"{REPO}/configs/vit3d_ikeda_vastai_train_split2_no-expand.yaml"
EVAL_SRC = os.path.join(_HERE, "configs", "evalA_split2_test_best.yaml")

# Architecture / pruning / crop keys copied from the eval (ToPM) config so the retrained
# model matches the frozen evaluator exactly. Everything else (lr, epochs, loss,
# optimizer, scheduler, train/val dirs, divide) comes from the repo's training config.
ARCH_KEYS = [
    "model_name", "num_classes", "n_channels",
    "y_crop_top", "y_crop_bottom", "z_crop_front", "z_crop_back",
    "target_size", "patch_size",
    "encoder_embed_dim", "encoder_depth", "encoder_num_heads",
    "merge_attn", "merge_mlp", "low_intensity_ratio", "merge_ratio_high",
    "group_token_order", "dst_ratio_high", "pruning_patch_embed_mode",
    "pruning_patch_embed_op", "low_unknown_logit", "low_other_logit",
    "low_pruning_ignore_loss", "low_pruning_ignore_label",
]


def cache_rel(path, cache_root):
    """Map an original .../ghost_dataset/<rel> dir to <cache_root>/<rel> (used by both
    cache_repr.py when writing and here when pointing training at the cache)."""
    return os.path.join(cache_root, path.split("ghost_dataset/", 1)[1])


def _remap_dirs(dirs):
    """Rebase the train config's dataset dirs onto THIS machine's data_root and apply the
    annotation_v1 -> annotation_v1_expand fix (the test metric uses the *expand*
    annotations). See envconfig.remap_data_dir."""
    return [envconfig.remap_data_dir(d) for d in (dirs or [])]


def build_config(args):
    """Merge repo train config + eval (ToPM) arch into one training config dict."""
    with open(args.config_train_src) as f:
        cfg = yaml.safe_load(f)
    with open(args.config_eval_src) as f:
        ev = yaml.safe_load(f)

    for k in ARCH_KEYS:
        if k in ev:
            cfg[k] = ev[k]

    # remap dataset paths to this box's layout
    for k in ("train_voxel_dirs", "train_annotation_dirs",
              "valid_voxel_dirs", "valid_annotation_dirs"):
        cfg[k] = _remap_dirs(cfg.get(k))

    # train on a pre-computed representation cache (uint16 voxels written by cache_repr.py):
    # point at the mirrored cache dirs and disable the on-the-fly hook + divide subsample
    # (the cache is already the strided subset, so divide=1).
    if getattr(args, "cache_root", None):
        assert args.rep == "none", "--cache_root implies a pre-transformed cache; use --rep none"
        for k in ("train_voxel_dirs", "train_annotation_dirs",
                  "valid_voxel_dirs", "valid_annotation_dirs"):
            cfg[k] = [cache_rel(d, args.cache_root) for d in cfg[k]]
        cfg["divide"] = 1

    # run-specific overrides
    cfg["is_train"] = True
    cfg["is_log"] = bool(args.is_log)
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg["epochs"] = args.epochs if args.epochs else cfg["epochs"]
    if args.divide:
        cfg["divide"] = args.divide
    # GPU work happens in __getitem__ when a hook is active -> must use main process
    cfg["num_workers"] = 0 if args.rep != "none" else args.num_workers
    save_dir = os.path.join(args.save_root, args.run_name)
    cfg["save_model_dir"] = save_dir + "/"

    if args.limit_train_dirs:
        cfg["train_voxel_dirs"] = cfg["train_voxel_dirs"][: args.limit_train_dirs]
        cfg["train_annotation_dirs"] = cfg["train_annotation_dirs"][: args.limit_train_dirs]
    if args.limit_val_dirs:
        cfg["valid_voxel_dirs"] = cfg["valid_voxel_dirs"][: args.limit_val_dirs]
        cfg["valid_annotation_dirs"] = cfg["valid_annotation_dirs"][: args.limit_val_dirs]

    return cfg, save_dir


def install_hook(args, device):
    """Install the same _load_voxel_grid transform run_eval.py uses, for training."""
    if args.rep == "event":
        ep = {
            "k": args.event_k, "representation": args.event_repr,
            "intensity_mode": args.event_intensity, "smooth_sigma": args.event_smooth_sigma,
            "min_height": args.event_min_height, "min_distance": args.event_min_distance,
            "fixed_width": args.event_fixed_width, "fixed_amplitude": args.event_fixed_amplitude,
            "kernel": args.event_kernel, "emg_tau": args.event_emg_tau,
        }
        run_eval.install_event_hook(ep, device)
        n_params = {"t": 1, "ta": 2, "tw": 2, "taw": 3, "taw_bg": 3}[args.event_repr]
        print(f"[hook] event {args.event_repr} K={args.event_k} dim={args.event_k * n_params}")
    elif args.rep == "ae":
        assert args.ae_ckpt, "--ae_ckpt required for --rep ae"
        model, kind, meta = run_eval.load_ae(args.ae_ckpt, device)
        run_eval.install_compression_hook(model, kind, device)
        print(f"[hook] ae {kind} {meta}")
    else:
        print("[hook] none (full waveform)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["none", "event", "ae"], default="none")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--save_root", default=envconfig.output_path("retrain"))
    ap.add_argument("--config_train_src", default=TRAIN_SRC)
    ap.add_argument("--config_eval_src", default=EVAL_SRC)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=0, help="0 = use config (50)")
    ap.add_argument("--divide", type=int, default=0, help="0 = use config (3)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cache_root", default=None,
                    help="train on a uint16 representation cache built by cache_repr.py "
                         "(implies --rep none, divide=1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--is_log", action="store_true", help="enable wandb")
    ap.add_argument("--limit_train_dirs", type=int, default=0, help="smoke: first N train dirs")
    ap.add_argument("--limit_val_dirs", type=int, default=0, help="smoke: first N val dirs")
    # event-mode params (mirror run_eval.py)
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

    cfg, save_dir = build_config(args)
    os.makedirs(save_dir, exist_ok=True)
    cfg_path = os.path.join(save_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[config] wrote {cfg_path}")
    print(f"[config] model={cfg['model_name']} target={cfg['target_size']} "
          f"patch={cfg['patch_size']} embed={cfg['encoder_embed_dim']} "
          f"epochs={cfg['epochs']} divide={cfg['divide']} workers={cfg['num_workers']} "
          f"train_dirs={len(cfg['train_voxel_dirs'])} val_dirs={len(cfg['valid_voxel_dirs'])}")

    install_hook(args, device)

    from hist_lidar.training import train_vit3d
    train_vit3d(cfg_path)


if __name__ == "__main__":
    main()
