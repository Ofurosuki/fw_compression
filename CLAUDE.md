# CLAUDE.md — fw_compression

Full-waveform (FW) LiDAR **waveform-compression research**. Core question: how much does
compressing the per-pixel `T=700` waveform degrade a **downstream, frozen Ghost-FWL
ghost-detection model** — and can a **sparse top-K transport-event** representation
`{(t_i, a_i, w_i)}` replace the dense waveform? We never retrain the downstream model; it is
a fixed judge.

## ⚠️ Read first: the metric you report is NOT the paper's metric
The repo computes **two** F1s from the same predictions; they differ by ~0.07:
- **voxel-level** — scores all ~10M voxels/frame (incl. background/tails). This is what
  `downstream/run_eval.py` emits (`macro_f1` field). Baseline ≈ **0.532** (3-class).
- **peak-level** (`peak_macro_f1`) — scores only at scipy `find_peaks` return-peak positions.
  Baseline ≈ **0.599** (3-class). **This is the paper's "F1-mean ≈ 0.592".**

We deliberately SKIP peak detection (slow, ~2h/config), so our numbers are **voxel-level and
NOT directly paper-comparable**. They ARE self-consistent for *relative* comparisons across
compression methods. For a paper-comparable number, add peak-level scoring (headline configs
only). Full investigation: **`downstream/SCORE_DISCREPANCY.md`**. Everything else (algorithm,
weights, test set, threshold 0.5, ρ_low=0.7 / merge=0.9) is byte-identical to the repo —
verified by running the repo's own `run_test.py`.

"F1-mean" everywhere = mean of per-class F1 over **3 signal classes {object, glass, ghost}**
(Noise/`unknown` dropped from the *average* but kept as a competing class in the CM;
`ignore_visualize_labels=[]`). The 4-class macro that *includes* noise (F1≈0.99) is ~0.64 —
do not confuse the two.

## Environment & how to run (gotchas)
- Python via **`uv run python ...`**; always `export PATH="$HOME/.local/bin:$PATH"` first.
- Downstream model lives in a **read-only repo**: `/data3/user/yoshida/fwl_mae/neurips2026`.
  Run its code via **`export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src`** using
  *this* project's `.venv` (Blackwell GPUs need a **cu128** torch build).
- Dataset: `ghost_datasets` symlink → `/data3/user/ikeda/ghost_dataset`. 10 **named** scenes
  (e.g. `36build`, `22build`, `14build_7floor`), not `scene001/002`. Dirs are
  `drwxrwxr-x ikeda ikeda` (775) — readable by `yoshida` via the `other` r-x bit (verified:
  all 1427 test files load).
- 4 GPUs `cuda:0..3` (~98 GB each). Sweeps fan out one job/GPU (see `run_sweep_*.py`).
- When a **compression/event hook is active**, run with `num_workers=0` (GPU work happens in
  `__getitem__`, needs the main process). Plain `--compress none` can use workers.
- `run_eval.py` is **deterministic** at fixed seed (verified bit-identical across reruns), so
  relative comparisons are trustworthy.

## Downstream evaluation
- Model: **FWL-ToPM** (`vit3d_ordered_pruning_light`). Headline ckpt = `neurips_best`
  (`…/checkpoints/neurips_best/…0423_221908_0.02523.pth`) = the repo's **"baseline"** weight
  (no augmentation). ρ_low=`low_intensity_ratio`=0.7, merge=`merge_ratio_high`=0.9,
  `prediction_threshold`=0.5.
- Test set (= repo split2 TEST, byte-for-byte set-equal): `36build` h002-011, `22build`
  h001-010, `14build_7floor` h001-010 → **30 dirs, 1427 frames**.
- **`divide=3`** (~475 frames) for sweeps; matches **`divide=1`** (full) within ≤0.003, so
  trust divide=3. Config: `downstream/configs/evalA_split2_test_best.yaml` (neurips_best);
  `evalA_split2_test.yaml` (prior cutmix0.2_0.8 ckpt 0.02485).
- Compression is inserted **before** the downstream's own crop/normalize by monkey-patching
  `VoxelDataset._load_voxel_grid` ("T=700-first": per-pixel max-normalize → transform →
  de-normalize). Note the downstream then takes a **random ~43%-volume crop** (post-crop
  grid 400×336×350 vs target 200×168×300) — aggregate F1 variance from this is tiny
  (seed std ≈0.0024), so it doesn't affect conclusions.

## Key files
- `downstream/run_eval.py` — eval harness (frozen model + optional compress/event hook, CM F1).
- `downstream/run_sweep.py` / `run_sweep_ah.py` / `run_sweep_events.py` — GPU-fanned sweeps.
- `downstream/run_plot*.py` — tables + figures. `downstream/RESULTS.md` — **main results doc**.
- `downstream/SCORE_DISCREPANCY.md` — the voxel-vs-peak F1 investigation.
- `compression/event_extraction.py` / `event_synthesis.py` — top-K transport-event rep
  (scipy reference + GPU-batch extractor; Gaussian-pulse synthesis; `t`/`ta`/`tw`/`taw`).
- `compression/autoencoder.py` / `spatial_coding.py` — 1D and spatial-4×4 AEs.
- `train_autoencoder.py` / `train_spatial.py` — AE training (incl. anti-hallucination loss).

## Findings so far (see RESULTS.md for full tables)
- **Sparse `(t,a,w)` events recover most downstream performance**: `taw` K=2 ≈ 0.426 (81 % of
  full-waveform 0.524) at **117× compression**. Headline of the event experiment.
- **Both intensity AND width matter; width > intensity.** Position-only (`t`) ≈ 0.13–0.16;
  `ta` (multi-echo analogue) ≈ 0.18; `tw` ≈ 0.30; `taw` ≈ 0.41. The `ta`→`taw` jump (+0.23,
  mostly **object**) is the value of full-waveform pulse-shape over a multi-echo sensor.
- **Ghost needs K≥2** (ghosts are *secondary*, ~median-rank-2 returns); ghost loss is
  recall-limited and **depth-dependent** (early/near-range ghosts are dropped). Glass is the
  stress class. Diagnostics: `downstream/outputs/events*/diag/`, scripts `diag_ghost*.py`.
- **spatial 4×4 AE** > 1D AE at high compression; **anti-hallucination loss** improves
  downstream F1 even when reconstruction MSE *worsens* → **MSE is not the right proxy**; this
  frozen-judge harness exists precisely because of that.

## Conventions
- Don't edit the Ghost-FWL repo (read-only). Reuse its dataset/metric via PYTHONPATH.
- Reuse the repo's `calculate_metrics_from_confusion_matrix` (don't reimplement F1).
- Long sweeps run in the background and fan over the 4 GPUs; confirm `rc=0` for all jobs.
- A persistent **memory** lives at `~/.claude/projects/-home-yoshida-fw-compression/memory/`
  (`MEMORY.md` index auto-loads each session) — check it for prior context, and record
  non-obvious facts there.
