"""Dataset split + crop constants shared across the eventnet pipeline.

The split is the repo's SPLIT2 (3 held-out test scenes), resolved from
``configs/vit3d_ikeda_vastai_cutmix_train_split2_no-expand.yaml`` with two
remaps so it points at data this box actually has:

* ``/data1/user/ikeda/ghost_dataset`` -> ``/data3/user/ikeda/ghost_dataset``
* ``annotation_v1`` -> ``annotation_v1_expand`` (the TEST metric uses the
  *expand* annotations, so train/val/test all use *expand* for label
  consistency; see CLAUDE.md and ``downstream/configs/evalA_split2_test_best``).

The single missing cutmix-augmentation dir is dropped. The resolved lists live
in ``split2_dirs.json`` next to this file (regenerate via the snippet in
``FW_Event_Net/RESULTS.md`` if the dataset moves).

Cropping matches the downstream exactly so peak positions and labels line up
with the frozen Ghost-FWL evaluation: y_crop (top=88, bottom=88) on the 512 axis
-> 336, z_crop (front=25, back=375) on the 700-bin histogram axis -> ``T=300``.
We do NOT replicate the downstream's *random* spatial crop at eval time; the
event net is scored on the full 400x336 plane (the F1 *method* is identical to
the paper's peak-level scoring, only the spatial coverage is fuller).
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import envconfig  # noqa: E402  (machine-dependent paths; see env.yaml.example)

SPLIT_JSON = os.path.join(_HERE, "split2_dirs.json")

# downstream crop constants (evalA_split2_test_best.yaml)
Y_CROP_TOP = 88
Y_CROP_BOTTOM = 88
Z_CROP_FRONT = 25
Z_CROP_BACK = 375
T_CROPPED = 700 - Z_CROP_FRONT - Z_CROP_BACK  # 300

NUM_CLASSES = 4
LABEL_MAP = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}
SIGNAL_CLASSES = [1, 2, 3]  # object, glass, ghost (Noise excluded from the mean)

# default cache location (events are small; raw voxels are not cached) — on the local
# fast-disk cache_root (env.yaml), not NFS.
CACHE_ROOT = envconfig.cache_path("eventnet")


def load_split():
    """SPLIT2 dir lists, with each dataset path remapped to THIS machine's data_root
    (and annotation_v1 -> annotation_v1_expand). The split2_dirs.json may hold absolute
    paths from whatever machine generated it; remap_data_dir rebases them here."""
    split = json.load(open(SPLIT_JSON))
    for d in split.values():
        for key in ("voxel", "ann"):
            if key in d:
                d[key] = [envconfig.remap_data_dir(p) for p in d[key]]
    return split


def frame_files(voxel_dir: str, ann_dir: str):
    """Return aligned (voxel_path, annotation_path) pairs for one hist dir."""
    vfiles = sorted(f for f in os.listdir(voxel_dir) if f.endswith("_voxel.b2"))
    out = []
    for vf in vfiles:
        af = vf.replace("_voxel.b2", "_annotation_voxel.b2")
        ap = os.path.join(ann_dir, af)
        if os.path.exists(ap):
            out.append((os.path.join(voxel_dir, vf), ap))
    return out


def list_frames(split: str, frame_stride: int = 1):
    """All (voxel_path, ann_path) for a split, optionally keeping every
    ``frame_stride``-th frame *per dir* (deterministic subsample)."""
    d = load_split()[split]
    frames = []
    for vdir, adir in zip(d["voxel"], d["ann"]):
        ff = frame_files(vdir, adir)
        if frame_stride > 1:
            ff = ff[::frame_stride]
        frames.extend(ff)
    return frames


def apply_crop(grid):
    """y_crop + z_crop a raw (X, Y, T) grid -> (X, 336, 300)."""
    y0, y1 = Y_CROP_BOTTOM, grid.shape[1] - Y_CROP_TOP
    z0, z1 = Z_CROP_FRONT, grid.shape[2] - Z_CROP_BACK
    return grid[:, y0:y1, z0:z1]
