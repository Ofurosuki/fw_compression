"""Real Ghost-FWL waveform loader (drop-in replacement for the synthetic/physical
generators).

The real dataset stores, per frame, a ``(400, 512, 700)`` voxel grid (uint16
photon counts; the last axis is the 700-bin temporal histogram = one waveform per
pixel) plus a per-voxel semantic annotation with the SAME shape, labelling each
bin as ``{0:noise, 1:object, 2:glass, 3:ghost}`` (see Ghost-FWL ``LABEL_MAP``).

Unlike the synthetic generators there is **no parametric ground truth** for peak
shape. Instead:

- peak *positions* and *classes* come from contiguous runs in the annotation,
- reference peak *intensity/width* are **measured on the (uncompressed) original
  waveform** -- so downstream fidelity is "how well does compression preserve the
  original return", and the natural upper bound is the original waveform itself.

Each pixel waveform is **max-normalized** (counts vary hugely across pixels/frames).

The emitted ``(wave[T], label_dict)`` contract matches ``synthetic_waveforms.py``
(extra keys: ``peak_classes``, measured ``peak_intensities``/``peak_fwhm``), so the
existing train/eval code is reused.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from compression.data.synthetic_waveforms import WaveformDataset, collate_waveforms  # re-export
from compression.utils.metrics import measure_peak

# Ghost-FWL semantic labels (src/config/constants.py)
LABEL_MAP = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}
SIGNAL_CLASSES = (1, 2, 3)


@dataclass
class RealWaveformConfig:
    T: int = 700
    data_root: str = "ghost_datasets"
    annotation_subdir: str = "annotation_v1_expand"  # present in BOTH scenes (v1 is scene001-only)
    frames_per_scene: int = 30          # sampled evenly across the 2500 frames / 50 hist dirs
    pixels_per_frame: int = 2500        # labelled pixels kept per frame
    ghost_fraction: float = 0.5         # target fraction of kept pixels that contain a ghost return
    measure_half_window: int = 25       # window for model-free peak measurement
    merge_gap: int = 12                 # merge same-class annotation runs separated by < this many bins
    min_rel_height: float = 0.05        # drop reference peaks fainter than this fraction of the waveform max
    cache_dir: str = "runs/real_cache"


def _annotation_path(voxel_path: pathlib.Path, annotation_subdir: str) -> pathlib.Path:
    """data/histXXX/<stem>_voxel.b2 -> <annotation_subdir>/histXXX/<stem>_annotation_voxel.b2"""
    parts = list(voxel_path.parts)
    # replace the '/data/' segment with the annotation subdir
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data":
            parts[i] = annotation_subdir
            break
    p = pathlib.Path(*parts)
    return p.with_name(p.name.replace("_voxel.b2", "_annotation_voxel.b2"))


def _list_frames(scene_dir: pathlib.Path) -> List[pathlib.Path]:
    return sorted((scene_dir / "data").glob("hist*/*_voxel.b2"))


def _sample_frames(frames: List[pathlib.Path], n: int) -> List[pathlib.Path]:
    if n >= len(frames):
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    idx = np.unique(idx)
    return [frames[i] for i in idx]


def _class_regions(class_bins: np.ndarray, merge_gap: int) -> List[np.ndarray]:
    """Split a single class's labelled bins into runs, merging runs separated by
    < merge_gap bins (the expand annotation is often fragmented/dilated)."""
    if class_bins.size == 0:
        return []
    runs = np.split(class_bins, np.where(np.diff(class_bins) > merge_gap)[0] + 1)
    return runs


def _extract_pixel(wave_norm: np.ndarray, lab_row: np.ndarray, cfg: RealWaveformConfig) -> Optional[Dict]:
    """One peak per (merged) same-class annotated region; faint reference peaks
    below ``min_rel_height`` (undetectable even on the original) are dropped so the
    full-waveform upper bound is a clean ceiling."""
    positions, classes, inten, heights, fwhm = [], [], [], [], []
    for cls in SIGNAL_CLASSES:
        class_bins = np.where(lab_row == cls)[0]
        for region in _class_regions(class_bins, cfg.merge_gap):
            pos = int(region[int(np.argmax(wave_norm[region]))])
            m = measure_peak(wave_norm, pos, half_window=cfg.measure_half_window)
            if m["height"] < cfg.min_rel_height:
                continue
            positions.append(m["position"])
            classes.append(LABEL_MAP[cls])
            inten.append(m["area"])
            heights.append(m["height"])
            fwhm.append(m["fwhm"])
    if not positions:
        return None
    order = np.argsort(positions)
    positions = np.asarray(positions, dtype=np.float32)[order]
    inten = np.asarray(inten, dtype=np.float32)[order]
    heights = np.asarray(heights, dtype=np.float32)[order]
    fwhm = np.asarray(fwhm, dtype=np.float32)[order]
    classes = [classes[i] for i in order]
    is_ghost = any(c == "ghost" for c in classes)
    return dict(
        peak_positions=positions,
        peak_classes=classes,
        peak_intensities=inten,    # measured area on the ORIGINAL (normalized) waveform
        peak_heights=heights,
        peak_fwhm=fwhm,            # measured half-max width on the original waveform
        n_peaks=len(classes),
        ghost=bool(is_ghost),
    )


def _extract_frame(voxel_path: pathlib.Path, cfg: RealWaveformConfig, rng: np.random.Generator):
    import blosc2

    annot_path = _annotation_path(voxel_path, cfg.annotation_subdir)
    if not annot_path.exists():
        return [], []
    vox = blosc2.load_array(str(voxel_path)).reshape(-1, cfg.T).astype(np.float32)   # (P, T)
    ann = blosc2.load_array(str(annot_path)).reshape(-1, cfg.T)                        # (P, T) uint8

    has_signal = (ann > 0).any(1)
    has_ghost = (ann == 3).any(1)
    ghost_pix = np.where(has_signal & has_ghost)[0]
    other_pix = np.where(has_signal & ~has_ghost)[0]

    n_total = cfg.pixels_per_frame
    n_ghost = min(len(ghost_pix), int(round(n_total * cfg.ghost_fraction)))
    n_other = min(len(other_pix), n_total - n_ghost)
    sel_ghost = rng.choice(ghost_pix, n_ghost, replace=False) if n_ghost else np.array([], int)
    sel_other = rng.choice(other_pix, n_other, replace=False) if n_other else np.array([], int)
    sel = np.concatenate([sel_ghost, sel_other]).astype(int)

    waves, labels = [], []
    for p in sel:
        w = vox[p]
        mx = float(w.max())
        if mx <= 0:
            continue
        wn = (w / mx).astype(np.float32)
        lab = _extract_pixel(wn, ann[p], cfg)
        if lab is None:
            continue
        waves.append(wn)
        labels.append(lab)
    return waves, labels


def _cache_key(scene: str, cfg: RealWaveformConfig, seed: int) -> str:
    payload = {**asdict(cfg), "scene": scene, "seed": seed}
    h = hashlib.md5(repr(sorted(payload.items())).encode()).hexdigest()[:10]
    return f"{scene}_{h}"


def extract_scene(scene: str, cfg: RealWaveformConfig, seed: int = 0, verbose: bool = True):
    """Extract (and cache) per-pixel waveforms + labels for one scene.

    Returns (waves[N, T] float32 in [0,1], labels: list of dict).
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache = os.path.join(cfg.cache_dir, _cache_key(scene, cfg, seed) + ".npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"  [real] loaded cache {cache}  N={len(d['waves'])}")
        return d["waves"], list(d["labels"])

    scene_dir = pathlib.Path(cfg.data_root) / scene
    frames = _sample_frames(_list_frames(scene_dir), cfg.frames_per_scene)
    if verbose:
        print(f"  [real] extracting {scene}: {len(frames)} frames x ~{cfg.pixels_per_frame} px ...")
    rng = np.random.default_rng(seed)
    all_w, all_l = [], []
    for fi, vp in enumerate(frames):
        w, l = _extract_frame(vp, cfg, rng)
        all_w.extend(w)
        all_l.extend(l)
        if verbose and (fi + 1) % 5 == 0:
            print(f"    frame {fi+1}/{len(frames)}  cumulative N={len(all_w)}")
    waves = np.asarray(all_w, dtype=np.float32)
    labels = all_l
    np.savez_compressed(cache, waves=waves, labels=np.array(labels, dtype=object))
    if verbose:
        ng = sum(1 for l in labels if l["ghost"])
        print(f"  [real] {scene}: N={len(waves)} ({ng} ghost) -> cached {cache}")
    return waves, labels


# scene assignment for the two cross-validation directions
CV_SPLITS = {
    "A": dict(train="scene001", val="scene002"),
    "B": dict(train="scene002", val="scene001"),
}

# Multi-scene split aligned with the downstream Ghost-FWL repo (vit3d ..._split2):
# 7 train scenes / 3 held-out test scenes. The AE is trained ONLY on the train
# scenes; the test scenes are kept out entirely so the downstream F1 evaluation on
# them is leak-free (the downstream vit3d was trained on the same 7 train scenes).
SPLIT2 = {
    "train": ["11build", "14build_2floor", "16build", "16buildA_large",
              "16buildA_mid", "34build", "gym_build"],
    "test":  ["14build_7floor", "22build", "36build"],
}


def make_datasets_cv(cv: str, cfg: Optional[RealWaveformConfig] = None, seed: int = 42,
                     n_train: Optional[int] = None, n_val: Optional[int] = None):
    """Build train/val datasets for cross-validation direction ``cv`` in {A, B}."""
    cfg = cfg or RealWaveformConfig()
    split = CV_SPLITS[cv]
    tr_w, tr_l = extract_scene(split["train"], cfg, seed=seed)
    va_w, va_l = extract_scene(split["val"], cfg, seed=seed + 1)
    if n_train:
        tr_w, tr_l = tr_w[:n_train], tr_l[:n_train]
    if n_val:
        va_w, va_l = va_w[:n_val], va_l[:n_val]
    return WaveformDataset(tr_w, tr_l), WaveformDataset(va_w, va_l), cfg


def _extract_scenes(scenes: List[str], cfg: RealWaveformConfig, seed: int):
    """Extract + concatenate per-pixel waveforms over several scenes."""
    all_w, all_l = [], []
    for i, sc in enumerate(scenes):
        w, l = extract_scene(sc, cfg, seed=seed + i)
        if len(w):
            all_w.append(w)
            all_l.extend(l)
    waves = np.concatenate(all_w, axis=0) if all_w else np.zeros((0, cfg.T), np.float32)
    return waves, all_l


def make_datasets_multi(train_scenes: List[str], cfg: Optional[RealWaveformConfig] = None,
                        seed: int = 42, n_train: Optional[int] = None, n_val: int = 5000):
    """Pool several train scenes, shuffle, and carve a val split from the SAME pool.

    Test scenes (see ``SPLIT2['test']``) are NOT passed here — they are reserved for
    the leak-free downstream F1 evaluation."""
    cfg = cfg or RealWaveformConfig()
    waves, labels = _extract_scenes(train_scenes, cfg, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(waves))
    waves = waves[perm]
    labels = [labels[i] for i in perm]
    n_val = min(n_val, len(waves) // 5)
    va_w, va_l = waves[:n_val], labels[:n_val]
    tr_w, tr_l = waves[n_val:], labels[n_val:]
    if n_train and n_train < len(tr_w):
        tr_w, tr_l = tr_w[:n_train], tr_l[:n_train]
    return WaveformDataset(tr_w, tr_l), WaveformDataset(va_w, va_l), cfg
