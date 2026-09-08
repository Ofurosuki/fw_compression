"""Stage 1: how many annotated returns does candidate generation keep at all?

Before any question about which candidates a selector picks, there is the
question of which ones it could pick. This measures that, and nothing else --
no learned parameters, no downstream model. For every annotated return in the
dataset (object / glass / ghost), does the extractor's candidate set contain a
candidate at that position?

Three numbers per class, in increasing order of what they blame:

  present   a local maximum exists at the annotated bin at all. Anything the
            raw waveform does not show, no ranking can recover; this is the
            ceiling for every criterion below.
  pool      the annotated return is among the top K_POOL candidates.
  kept      it is among the top K the representation actually transmits.

The gap between ``present`` and ``pool`` is a ranking failure -- the return is
visible and was passed over. The gap between ``pool`` and ``kept`` is the cost
of the transmission budget. Splitting them matters because the fog side's
whole gain came from the first gap (its ranker was near-exhausted while 16-19%
of rays held no candidate near the true surface at all), and it is not obvious
the same is true here.

Criteria compared:

  height    the current extractor: rank local maxima of the smoothed,
            per-ray max-normalised trace by height.
  snr       the fog side's candidate generation, ported: rank by
            (y - background) / sqrt(background) on raw photon counts, with a
            wide moving-average background. Scale-free in principle, so a
            ray's accumulation depth should not change the ranking -- except
            through the divisor's floor, which is why both floors are here:
              clamp1    max(background, 1), as written for fog, where the
                        background is tens of counts and the floor never binds
              anscombe  background + 3/8, the variance-stabilising constant,
                        which stays meaningful below one count

Everything is stratified by accumulation band, because this sensor integrates
18/36/90/144/36/18 shots depending on the row and the shallow bands have
2.8x worse SNR than the deep one for the same surface.

Usage:
  uv run python downstream/diag_pool_recall.py --frames_per_dir 2 --limit_dirs 6
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envconfig  # noqa: E402
from compression.sensor import GHOST  # noqa: E402

LABEL_NAMES = {1: "object", 2: "glass", 3: "ghost"}

# Rows 0-123 / 124-187 / 188-251 / 252-315 / 316-379 / 380-511 accumulate
# 18 / 36 / 90 / 144 / 36 / 18 laser shots. Measured from the data: each band
# has a hard ceiling on the counts at exactly that value, identical across
# every building, and its background scales by the same ratios.
ACCUMULATION_BANDS = [(0, 124, 18), (124, 188, 36), (188, 252, 90),
                      (252, 316, 144), (316, 380, 36), (380, 512, 18)]

# The downstream crops Y to [88, 424) (y_crop_top/bottom = 88), so every band
# except the outer thirds is fully inside what the model ever sees.
CROP_Y = (88, 424)

# The test split of the ToPM baseline (split2), from CLAUDE.md.
TEST_DIRS = ([("36build", f"hist{i:03d}") for i in range(2, 12)]
             + [("22build", f"hist{i:03d}") for i in range(1, 11)]
             + [("14build_7floor", f"hist{i:03d}") for i in range(1, 11)])


def moving_average(x: torch.Tensor, win: int) -> torch.Tensor:
    kernel = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win
    padded = torch.nn.functional.pad(x[:, None, :], (win // 2, win // 2), mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel)[:, 0, :]


def smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(t ** 2) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    padded = torch.nn.functional.pad(x[:, None, :], (radius, radius), mode="reflect")
    return torch.nn.functional.conv1d(padded, kernel)[:, 0, :]


def score(raw: torch.Tensor, criterion: str, background_win: int, smooth_sigma: float):
    """Returns (score, trace_for_local_maxima). Candidates are local maxima of
    the score; the raw trace decides what is a real peak."""
    if criterion == "height":
        peak = raw.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return smooth(raw / peak, smooth_sigma), None
    background = moving_average(raw, background_win)
    if criterion == "snr_clamp1":
        divisor = background.clamp_min(1.0).sqrt()
    elif criterion == "snr_anscombe":
        divisor = (background + 0.375).sqrt()
    else:
        raise ValueError(criterion)
    return (raw - background) / divisor, None


def topk_peak_bins(score_trace: torch.Tensor, k: int, nms: int, min_height: float | None):
    """Top-k local maxima of ``score_trace`` under min-distance suppression.
    Returns (bins (N,k) long, valid (N,k) bool)."""
    n_rays, n_bins = score_trace.shape
    is_max = torch.zeros_like(score_trace, dtype=torch.bool)
    mid = score_trace[:, 1:-1]
    is_max[:, 1:-1] = (mid > score_trace[:, :-2]) & (mid >= score_trace[:, 2:])
    if min_height is not None:
        is_max &= score_trace >= min_height

    neg_inf = torch.finfo(score_trace.dtype).min
    pool = torch.where(is_max, score_trace, torch.full_like(score_trace, neg_inf))
    rows = torch.arange(n_rays, device=score_trace.device)
    offsets = torch.arange(-nms, nms + 1, device=score_trace.device)

    bins = torch.zeros(n_rays, k, dtype=torch.long, device=score_trace.device)
    valid = torch.zeros(n_rays, k, dtype=torch.bool, device=score_trace.device)
    for slot in range(k):
        best, idx = pool.max(dim=1)
        valid[:, slot] = best > neg_inf
        bins[:, slot] = idx
        cols = (idx[:, None] + offsets[None, :]).clamp_(0, n_bins - 1)
        pool[rows[:, None], cols] = neg_inf
    return bins, valid


def raw_local_max(raw: torch.Tensor) -> torch.Tensor:
    is_max = torch.zeros_like(raw, dtype=torch.bool)
    mid = raw[:, 1:-1]
    is_max[:, 1:-1] = (mid > raw[:, :-2]) & (mid >= raw[:, 2:])
    return is_max


def hit_within(bins: torch.Tensor, valid: torch.Tensor, target: torch.Tensor, tol: int):
    """For each ray, whether any valid candidate lands within ``tol`` of the
    ray's target bin. ``target`` is (N,) long."""
    delta = (bins - target[:, None]).abs()
    return ((delta <= tol) & valid).any(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_per_dir", type=int, default=2)
    ap.add_argument("--limit_dirs", type=int, default=0, help="0 = all test dirs")
    ap.add_argument("--k_pool", type=int, default=15)
    ap.add_argument("--k_keep", type=int, default=4)
    ap.add_argument("--tol_bins", type=int, default=5,
                    help="0.42 of the pulse FWHM, matching the fog side's 0.5 m")
    ap.add_argument("--row_chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(envconfig.topm_repo_root(), "src"))
    from hist_lidar.preprocess.custom_blosc2 import load_blosc2

    data_root = envconfig.data_root()
    dirs = TEST_DIRS[: args.limit_dirs] if args.limit_dirs else TEST_DIRS
    criteria = ["height", "snr_clamp1", "snr_anscombe"]
    device = torch.device(args.device)

    # counts[(criterion, stage, label, shots)] -> [hits, total]
    counts: dict = defaultdict(lambda: [0, 0])  # (criterion, stage, class, band)
    n_frames = 0

    for build, hist in dirs:
        voxels = sorted(glob.glob(os.path.join(data_root, build, "data", hist, "*_voxel.b2")))
        for voxel_path in voxels[: args.frames_per_dir]:
            ann_path = voxel_path.replace(f"{build}/data/", f"{build}/annotation_v1/").replace(
                "_voxel.b2", "_annotation_voxel.b2")
            if not os.path.exists(ann_path):
                continue
            voxel = load_blosc2(voxel_path)
            annotation = load_blosc2(ann_path)
            n_frames += 1

            for band in ACCUMULATION_BANDS:
                y0, y1, shots = band
                for x0 in range(0, voxel.shape[0], args.row_chunk):
                    x1 = min(x0 + args.row_chunk, voxel.shape[0])
                    ann_block = annotation[x0:x1, y0:y1, :]
                    if not ann_block.any():
                        continue
                    raw = torch.as_tensor(
                        np.ascontiguousarray(voxel[x0:x1, y0:y1, :]), device=device
                    ).float().reshape(-1, voxel.shape[2])
                    ann_flat = ann_block.reshape(-1, annotation.shape[2])

                    present_mask = raw_local_max(raw)
                    cache: dict = {}
                    for criterion in criteria:
                        trace, _ = score(raw, criterion, GHOST.background_win_bins, 1.5)
                        min_height = 0.03 if criterion == "height" else None
                        pool_bins, pool_valid = topk_peak_bins(
                            trace, args.k_pool, GHOST.nms_bins, min_height)
                        cache[criterion] = (pool_bins, pool_valid)

                    for label, name in LABEL_NAMES.items():
                        ray_idx, bin_idx = np.nonzero(ann_flat == label)
                        if ray_idx.size == 0:
                            continue
                        rays = torch.as_tensor(ray_idx, device=device, dtype=torch.long)
                        targets = torch.as_tensor(bin_idx, device=device, dtype=torch.long)
                        total = int(rays.numel())

                        exact = present_mask[rays, targets]
                        counts[("--", "present", name, band)][0] += int(exact.sum())
                        counts[("--", "present", name, band)][1] += total

                        for criterion in criteria:
                            pool_bins, pool_valid = cache[criterion]
                            b, v = pool_bins[rays], pool_valid[rays]
                            pool_hit = hit_within(b, v, targets, args.tol_bins)
                            keep_hit = hit_within(
                                b[:, : args.k_keep], v[:, : args.k_keep], targets, args.tol_bins)
                            counts[(criterion, "pool", name, band)][0] += int(pool_hit.sum())
                            counts[(criterion, "pool", name, band)][1] += total
                            counts[(criterion, "kept", name, band)][0] += int(keep_hit.sum())
                            counts[(criterion, "kept", name, band)][1] += total
                    del raw, present_mask, cache

    def rate(criterion, stage, name, band=None):
        if band is None:
            hits = sum(v[0] for k, v in counts.items()
                       if k[0] == criterion and k[1] == stage and k[2] == name)
            total = sum(v[1] for k, v in counts.items()
                        if k[0] == criterion and k[1] == stage and k[2] == name)
        else:
            hits, total = counts[(criterion, stage, name, band)]
        return (hits / total if total else float("nan")), total

    print(f"\nframes={n_frames}  dirs={len(dirs)}  tol=+/-{args.tol_bins} bins "
          f"({args.tol_bins / GHOST.pulse_fwhm_bins:.2f} FWHM)  "
          f"K_POOL={args.k_pool}  K_KEEP={args.k_keep}  nms={GHOST.nms_bins}  "
          f"background_win={GHOST.background_win_bins}")

    print("\n== overall (all rows) ==")
    header = f"{'class':8s} {'returns':>9s} {'atbin':>8s}"
    for criterion in criteria:
        header += f" | {criterion + ' pool':>17s} {'kept':>6s}"
    print(header)
    for name in LABEL_NAMES.values():
        present, total = rate("--", "present", name)
        line = f"{name:8s} {total:9d} {present:8.3f}"
        for criterion in criteria:
            pool, _ = rate(criterion, "pool", name)
            kept, _ = rate(criterion, "kept", name)
            line += f" | {pool:17.3f} {kept:6.3f}"
        print(line)

    print("\n== by accumulation band (pool recall) ==")
    print(f"{'shots':>6s} {'class':8s} {'returns':>9s} {'atbin':>8s} "
          + " ".join(f"{c:>14s}" for c in criteria))
    for band in ACCUMULATION_BANDS:
        y0, y1, shots = band
        inside = "in" if y0 >= CROP_Y[0] and y1 <= CROP_Y[1] else "edge"
        for name in LABEL_NAMES.values():
            present, total = rate("--", "present", name, band)
            if not total:
                continue
            cells = " ".join(f"{rate(c, 'pool', name, band)[0]:14.3f}" for c in criteria)
            print(f"{shots:6d} {name:8s} {total:9d} {present:8.3f} {cells}   rows {y0}-{y1 - 1} ({inside})")


if __name__ == "__main__":
    main()
