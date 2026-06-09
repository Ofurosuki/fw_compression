"""Spatial (4x4 neighbourhood) waveform patches for spatio-temporal compression.

This is the spatial extension of ``real_waveforms.py``, modelled on the ICCV2023
"Learned Compressive Representations" paper: instead of compressing each pixel's
``[T]`` waveform independently, we compress a local ``Mr x Mc`` block of neighbouring
pixels jointly, exploiting *spatial* correlations (neighbouring pixels see correlated
depths / transport structure), not just temporal structure.

Each patch is an ``Mr x Mc x T`` block (default 4x4x700). To keep the comparison to
the per-pixel experiment apples-to-apples (same detector calibration, same
normalization), **each pixel waveform is max-normalized independently** — so the
spatial redundancy the encoder can exploit is the correlation of peak *positions /
shapes* across neighbours (the dominant depth/transport redundancy), with each pixel
still in ``[0,1]``.

Per-pixel labels (object/glass/ghost peak positions + measured intensity/FWHM on the
original waveform) are kept for ALL Mr*Mc pixels, so after reconstruction the patch is
unfolded back to per-pixel waveforms and scored with the *same* metrics as the
per-pixel experiment (``real_peak_metrics``).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from compression.data.real_waveforms import (
    CV_SPLITS,
    SPLIT2,
    LABEL_MAP,
    SIGNAL_CLASSES,
    _annotation_path,
    _extract_pixel,
    _list_frames,
    _sample_frames,
    RealWaveformConfig,
)


@dataclass
class SpatialWaveformConfig:
    T: int = 700
    data_root: str = "ghost_datasets"
    annotation_subdir: str = "annotation_v1_expand"
    block: int = 4                      # Mr = Mc = block (spatial patch side)
    frames_per_scene: int = 30
    patches_per_frame: int = 200        # 4x4 patches kept per frame (× ~30 frames ≈ 6000 patches)
    ghost_fraction: float = 0.7         # fraction of kept patches that contain >=1 ghost pixel
    measure_half_window: int = 25
    merge_gap: int = 12
    min_rel_height: float = 0.05
    cache_dir: str = "runs/real_cache_spatial"

    def as_real_cfg(self) -> RealWaveformConfig:
        return RealWaveformConfig(
            T=self.T, data_root=self.data_root, annotation_subdir=self.annotation_subdir,
            measure_half_window=self.measure_half_window, merge_gap=self.merge_gap,
            min_rel_height=self.min_rel_height,
        )


def _extract_frame_patches(voxel_path, cfg: SpatialWaveformConfig, real_cfg, rng):
    import blosc2

    annot_path = _annotation_path(voxel_path, cfg.annotation_subdir)
    if not annot_path.exists():
        return [], []
    B = cfg.block
    vox = blosc2.load_array(str(voxel_path)).astype(np.float32)   # (X, Y, T)
    ann = blosc2.load_array(str(annot_path))                       # (X, Y, T) uint8
    X, Y, T = vox.shape
    nbx, nby = X // B, Y // B  # number of non-overlapping blocks along each axis

    # classify blocks by whether they contain a ghost-labelled pixel
    ghost_pix = (ann == 3).any(2)        # (X, Y)
    signal_pix = (ann > 0).any(2)
    # block-level flags via reshape into (nbx,B,nby,B)
    gp = ghost_pix[: nbx * B, : nby * B].reshape(nbx, B, nby, B).any(axis=(1, 3))
    sp = signal_pix[: nbx * B, : nby * B].reshape(nbx, B, nby, B).any(axis=(1, 3))
    ghost_blocks = np.argwhere(gp & sp)
    other_blocks = np.argwhere(sp & ~gp)

    n_total = cfg.patches_per_frame
    n_ghost = min(len(ghost_blocks), int(round(n_total * cfg.ghost_fraction)))
    n_other = min(len(other_blocks), n_total - n_ghost)
    sel_g = ghost_blocks[rng.choice(len(ghost_blocks), n_ghost, replace=False)] if n_ghost else np.empty((0, 2), int)
    sel_o = other_blocks[rng.choice(len(other_blocks), n_other, replace=False)] if n_other else np.empty((0, 2), int)
    blocks = np.concatenate([sel_g, sel_o], axis=0)

    patches, labels = [], []
    for bx, by in blocks:
        x0, y0 = bx * B, by * B
        block_vox = vox[x0 : x0 + B, y0 : y0 + B, :].reshape(B * B, T)   # (16, T)
        block_ann = ann[x0 : x0 + B, y0 : y0 + B, :].reshape(B * B, T)
        wn = np.zeros((B * B, T), dtype=np.float32)
        plabels: List[Optional[Dict]] = []
        for p in range(B * B):
            w = block_vox[p]
            mx = float(w.max())
            if mx <= 0:
                plabels.append(None)
                continue
            wn[p] = w / mx
            plabels.append(_extract_pixel(wn[p], block_ann[p], real_cfg))
        patches.append(wn)
        labels.append(plabels)
    return patches, labels


def _cache_key(scene, cfg: SpatialWaveformConfig, seed) -> str:
    payload = {**asdict(cfg), "scene": scene, "seed": seed}
    h = hashlib.md5(repr(sorted(payload.items())).encode()).hexdigest()[:10]
    return f"{scene}_b{cfg.block}_{h}"


def extract_scene_spatial(scene, cfg: SpatialWaveformConfig, seed=0, verbose=True):
    """Extract (and cache) 4x4 spatial patches for one scene.

    Returns (patches[N, B*B, T] float32 in [0,1], labels: list of N lists of B*B
    per-pixel label dicts (or None)).
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache = os.path.join(cfg.cache_dir, _cache_key(scene, cfg, seed) + ".npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"  [spatial] loaded cache {cache}  N={len(d['patches'])}")
        return d["patches"], list(d["labels"])

    real_cfg = cfg.as_real_cfg()
    scene_dir = pathlib.Path(cfg.data_root) / scene
    frames = _sample_frames(_list_frames(scene_dir), cfg.frames_per_scene)
    if verbose:
        print(f"  [spatial] extracting {scene}: {len(frames)} frames x ~{cfg.patches_per_frame} patches ...")
    rng = np.random.default_rng(seed)
    allp, alll = [], []
    for fi, vp in enumerate(frames):
        p, l = _extract_frame_patches(vp, cfg, real_cfg, rng)
        allp.extend(p)
        alll.extend(l)
        if verbose and (fi + 1) % 5 == 0:
            print(f"    frame {fi+1}/{len(frames)}  cumulative patches={len(allp)}")
    patches = np.asarray(allp, dtype=np.float32)
    np.savez_compressed(cache, patches=patches, labels=np.array(alll, dtype=object))
    if verbose:
        ng = sum(1 for pl in alll if any(x is not None and x["ghost"] for x in pl))
        print(f"  [spatial] {scene}: N={len(patches)} patches ({ng} ghost-bearing) -> cached {cache}")
    return patches, alll


def unfold_patches(patches: np.ndarray, labels: List[List]):
    """Flatten [N, P, T] patches + per-pixel labels -> per-pixel (waves[M,T], labels[M])
    keeping only pixels that carry >=1 labelled peak (for metric scoring)."""
    waves, labs, keep_mask = [], [], []
    N, P, T = patches.shape
    for i in range(N):
        for p in range(P):
            lab = labels[i][p]
            if lab is not None and lab["n_peaks"] > 0:
                waves.append(patches[i, p])
                labs.append(lab)
    return np.asarray(waves, dtype=np.float32), labs


class SpatialPatchDataset:
    """Torch-style dataset over spatial patches. __getitem__ -> (patch[P,T], labels_list)."""

    def __init__(self, patches: np.ndarray, labels: List[List]):
        import torch

        assert len(patches) == len(labels)
        self.patches = torch.from_numpy(patches).float()
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


def collate_patches(batch):
    import torch

    patches = torch.stack([b[0] for b in batch], dim=0)   # [B, P, T]
    labels = [b[1] for b in batch]
    return patches, labels


def make_datasets_spatial_cv(cv: str, cfg: Optional[SpatialWaveformConfig] = None, seed: int = 42):
    cfg = cfg or SpatialWaveformConfig()
    split = CV_SPLITS[cv]
    tr_p, tr_l = extract_scene_spatial(split["train"], cfg, seed=seed)
    va_p, va_l = extract_scene_spatial(split["val"], cfg, seed=seed + 1)
    return SpatialPatchDataset(tr_p, tr_l), SpatialPatchDataset(va_p, va_l), cfg


def make_datasets_spatial_multi(train_scenes, cfg: Optional[SpatialWaveformConfig] = None,
                                seed: int = 42, n_val: int = 5000):
    """Pool several train scenes of 4x4 patches, shuffle, carve a val split from the
    SAME pool. Test scenes (SPLIT2['test']) are held out for the downstream eval."""
    cfg = cfg or SpatialWaveformConfig()
    all_p, all_l = [], []
    for i, sc in enumerate(train_scenes):
        p, l = extract_scene_spatial(sc, cfg, seed=seed + i)
        if len(p):
            all_p.append(p)
            all_l.extend(l)
    patches = np.concatenate(all_p, axis=0) if all_p else np.zeros((0,), np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patches))
    patches = patches[perm]
    labels = [all_l[i] for i in perm]
    n_val = min(n_val, len(patches) // 5)
    return (SpatialPatchDataset(patches[n_val:], labels[n_val:]),
            SpatialPatchDataset(patches[:n_val], labels[:n_val]), cfg)
