# Downstream Ghost-FWL evaluation of waveform compression

How much does compressing the full-waveform LiDAR signal degrade the **downstream
ghost-detection** task, and does an anti-hallucination loss help? We answer this by
running a **frozen, pre-trained Ghost-FWL segmentation model** on
compressed-then-reconstructed waveforms and measuring its F1.

---

## 1. Setup

### Downstream model (frozen evaluator)
- **FWL-ToPM** (`vit3d_ordered_pruning_light`) from the evolved Ghost-FWL repo
  (`/data3/user/yoshida/fwl_mae/neurips2026`), paper *"Towards Real-Time Full-Waveform
  LiDAR Transformers via Intensity-Guided Token Reduction and Physics-Aware
  Augmentation."* Checkpoint: `…/pruning0.7merging0.9-vit3d-neurips-refactered/…aug-only-cutmix0.2_0.8/0512/…epoch_50…0.02485.pth`.
- Per-voxel 4-class segmentation `{noise, object, glass, ghost}`; input `(1, 300, 168, 200)`
  (time × H × W), pruning ρ_low=0.7 / merge 0.9. The model is **never retrained** —
  it is a fixed downstream judge.

### Metric
- **F1-mean = mean of the per-class F1 over {object, glass, ghost}**, computed from the
  per-voxel confusion matrix (softmax + threshold-0.5 prediction), **exactly as the
  Ghost-FWL repo / paper do**: `ignore_visualize_labels = []`, i.e. **Noise stays in the
  confusion matrix as a competing class** (false positives onto/from Noise *do* penalise
  the signal classes); Noise is only dropped from the *average*.
  - ⚠️ *Convention matters by ~0.14.* An earlier version of this harness used
    `ignore_visualize_labels = [0]`, which additionally **masks all true-Noise voxels out
    of the confusion matrix** — that inflates F1 (background→ghost mistakes stop counting):
    the no-compression baseline reads **0.66 masked vs 0.52 un-masked**. All numbers below
    use the repo/paper convention (un-masked). The F1 *implementation* is the repo's own
    `calculate_metrics_from_confusion_matrix` (verified: matches `run_test.py`).

### Test data
- Ghost-FWL `split2` **test** scenes (held out from both the downstream model's and
  the autoencoders' training): `36build`, `22build`, `14build_7floor` — 30 hist dirs,
  1427 frames (`/data3/user/ikeda/ghost_dataset`).
- Sweeps use `--divide 3` (≈475 frames) for speed; the headline baseline is the full
  1427 frames.

### How compression is inserted
Each pixel's raw `T=700` waveform is **compress→reconstructed before** the downstream
model's own crop/normalize pipeline (so all downstream preprocessing is reused
untouched — the "T=700-first" choice):

1. per-pixel max-normalize `w/max(w)`  →
2. autoencoder encode→decode  →
3. de-normalize `×max(w)` (background pixels with `max≤ε` are passed through).

The transform monkey-patches `VoxelDataset._load_voxel_grid`; see `run_eval.py`.

### Harness
- `downstream/run_eval.py` — load frozen model, optional compression hook, confusion-matrix
  F1 (the repo's slow per-pixel `scipy.find_peaks` peak metric is **skipped** — it
  dominated runtime; the headline F1 is unaffected). Emits a fixed 6-waveform
  orig-vs-recon figure per config.
- `downstream/run_sweep.py` / `run_sweep_ah.py` — fan the sweep over GPUs.
- `downstream/run_plot.py` / `run_plot_compare.py` — tables + F1-vs-ratio curves.
- Environment: the Ghost-FWL repo is read-only, so we run its code via
  `PYTHONPATH=<repo>/src` using this project's own cu128 venv (Blackwell GPUs need a
  cu128 torch build).

### Autoencoders under test
Trained on the `split2` **train** scenes (7 scenes, `--split split2`, `T=700`):
- **1D learnable_linear** — per-pixel linear encoder + MLP decoder, K ∈ {8,16,32,64,128}.
- **spatial 4×4** — joint 4×4-block encoder + MLP decoder, K ∈ {128,256,512,1024,2048}
  (per-pixel-equivalent K = K/16, i.e. the same ratios as 1D).
- **1D coarse_binning** — fixed (non-learned) downsampling encoder, as a naive baseline.

Compression ratio = `T/K` (per-pixel-equivalent).

---

## 2. Baseline (no compression)

FWL-ToPM on the split2 test set (divide=3 sweep baseline), original waveforms
(repo/paper convention):

| metric | object | glass | ghost | **F1-mean** |
|---|---|---|---|---|
| F1 | 0.694 | 0.298 | 0.558 | **0.517** |

For reference the paper reports F1-mean ≈ **0.592** for FWL-ToPM; our 0.517 on these 3
split2 test scenes with the `aug-only-cutmix0.2_0.8` checkpoint sits a little below it
(plausibly checkpoint/aug-variant and scene differences — same convention, same code).
Glass is intrinsically the hardest class (transparent, minority) — only ~0.30 even
uncompressed — and is the first to degrade under compression.

---

## 3. Sweep with the standard loss (MSE + peak-aware)

`downstream/outputs/sweep/` — figure `f1_vs_ratio.png`, table `summary.txt`.

| ratio | spatial 4×4 | 1D learnable | 1D coarse (naive) |
|---|---|---|---|
| — (none) | **0.517** | | |
| 6× | **0.456** | 0.395 | 0.391 |
| 11× | 0.452 | 0.411 | — |
| 22× | 0.445 | 0.397 | 0.293 |
| 44× | 0.448 | 0.378 | — |
| 88× | 0.426 | 0.331 | 0.101 |

**Findings**
1. **spatial 4×4 is the most robust** — fairly flat (0.43–0.46) from 6× to 88×. Sharing
   the 4×4 spatial neighbourhood preserves peak shape even at extreme compression; at 88×
   it keeps ~82 % of baseline (0.426 / 0.517).
2. **naive coarse-binning collapses at high ratio** — 0.101 at 88×, with glass essentially
   lost (per-class F1 ≈ 0.006).
3. **1D learnable is intermediate** and degrades at high ratio (0.331 at 88×).
4. **glass is the bottleneck class** throughout (0.30 even uncompressed); object/ghost are
   better preserved. The gap to no-compression (~0.06 at low ratio, up to ~0.19 at 88× for
   1D) is larger than the masked convention suggested — because background→signal false
   positives from compression artifacts now count.

---

## 4. Anti-hallucination loss (bg=5.0, fp=0.5)

Motivation: reconstructions recover true peaks but also **hallucinate spurious peaks**
in the background; a downstream detector then raises false ghost/object voxels. The
anti-hallucination loss (merged from `feature/remove_falsepositive`, ported to the
spatial trainer) adds, on non-peak bins:
- `bg_weight·‖relu(x̂−x)‖²` — background over-shoot suppression (asymmetric: only
  penalises reconstructing *above* the truth);
- `fp_weight·relu(slopeₗ)·relu(slopeᵣ)` — a differentiable local-max (false-peak) penalty.

Same AEs retrained with **bg=5.0, fp=0.5** (`real_split2_1d_ah`, `real_split2_spatial_ah`),
re-evaluated downstream. `downstream/outputs/sweep_ah/` — `f1_compare.png`, `compare.txt`.

| ratio | method | base F1 | **AH F1** | Δ |
|---|---|---|---|---|
| 88× | 1D learnable | 0.331 | **0.398** | **+0.066** |
| 44× | 1D learnable | 0.378 | 0.419 | +0.041 |
| 22× | 1D learnable | 0.397 | 0.416 | +0.018 |
| 11× | 1D learnable | 0.411 | **0.465** | **+0.054** |
| 6×  | 1D learnable | 0.395 | 0.392 | −0.003 |
| 88× | spatial 4×4 | 0.426 | 0.452 | +0.026 |
| 44× | spatial 4×4 | 0.448 | 0.459 | +0.011 |
| 22× | spatial 4×4 | 0.445 | 0.469 | +0.024 |
| 11× | spatial 4×4 | 0.452 | **0.483** | **+0.031** |
| 6×  | spatial 4×4 | 0.456 | 0.464 | +0.007 |

**Findings**
1. **Anti-hallucination loss improves downstream F1 for nearly every config** (Δ +0.01…+0.07);
   the only exception is 1D at the lowest ratio (6×, −0.003 — negligible, and where there is
   little hallucination to fix).
2. **Largest gains at high compression** (1D 88×: +0.066, 11×: +0.054) — where information is
   scarce, hallucinated peaks do the most downstream damage, so suppressing them helps most.
3. Both methods improve; the waveform figures show flatter reconstructed backgrounds.
4. The AH models have *higher* reconstruction val-MSE than the base models yet *better*
   downstream F1 — concrete evidence that **reconstruction MSE is not the right proxy** for
   downstream quality, which is exactly why this frozen-model harness exists.

---

## 5. Aside: why is reconstruction val-MSE lowest at ~22× (K=32)?

For `learnable_linear` *and* `random_projection`, the per-pixel val-MSE bottoms at
K=32 (22×) and *rises* for the wider bottlenecks K=64/128 — counter-intuitive, since a
wider bottleneck can represent everything a narrower one can. Evidence so far:
- The MLP **decoder holds ~95 % of the parameters and is the same size for all K**
  (≈0.5 M); the encoder (which scales with K) is tiny. So wider K adds almost no usable
  capacity — it mostly widens the decoder's input fan-in.
- At K=32 both **train and val** loss are ~2× lower than the neighbours (not overfitting),
  and all configs have **plateaued** by epoch 30 — i.e. K=64/128 are stuck at a worse
  minimum, not merely under-trained.
- `random_projection` has a **fixed (non-trained) encoder**, yet shows the same U — so
  the effect lives in **decoder optimisation**, not encoder training.

Hypothesis: an **optimisation/conditioning artifact of the shared fixed recipe**
(lr=1e-3, 30 ep, no LR decay) — wider decoder fan-in is harder to condition, so the
larger-K models settle at a higher plateau.

**Diagnostic** (`diag_capacity.py`): retrain K∈{32,64,128} with **cosine LR + 80 epochs**
on the same data (120 k-waveform subset, so absolute values are higher than the full-data
numbers above — only the *ordering within this table* is the valid comparison):

| encoder | K=32 (22×) | K=64 (11×) | K=128 (5×) |
|---|---|---|---|
| learnable_linear  | 0.001760 | 0.001318 | **0.001192** |
| random_projection | 0.001773 | **0.001490** | 0.001957 |

vs. the original fixed-recipe shape (full data): learnable_linear 0.00094 / 0.00156 / 0.00184
and random_projection 0.00104 / 0.00127 / 0.00150 — both **U-shaped (min at K=32)**.

**Conclusion**
- **learnable_linear: the U-shape is an optimisation artifact — confirmed.** With cosine
  LR + more epochs the ordering flips to **monotonic in capacity (K=128 best)**, as it
  should be (a wider bottleneck provably subsumes a narrower one). The original "22× is
  best" was K=64/128 getting stuck at a worse plateau under the fixed lr/epoch recipe.
- **random_projection: half artifact.** Better optimisation lets K=64 beat K=32 (so part
  of the U *was* optimisation), but **K=128 stays worse** — a genuine effect of its
  **fixed (non-trained) encoder**: extra random projections aren't adapted to the data, and
  a fixed-capacity decoder cannot exploit them, so the inverse problem gets harder past
  some K. The learned encoder adapts its projections, so it keeps improving to K=128.

Takeaway: pick the AE bottleneck by **downstream F1 under a properly-tuned (LR-decayed)
training recipe**, not by MSE under a fixed recipe — and note (per §4.4) downstream F1
does not track MSE anyway.

---

## 6. Reproduce

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src

# baseline (full test set)
uv run python downstream/run_eval.py --config downstream/configs/evalA_split2_test.yaml \
    --compress none --out downstream/outputs/evalA_orig.json

# sweeps (base-loss AEs / anti-hallucination AEs) + plots
uv run python downstream/run_sweep.py    && uv run python downstream/run_plot.py
uv run python downstream/run_sweep_ah.py && uv run python downstream/run_plot_compare.py
```

Retrain the AEs (SPLIT2, with anti-hallucination loss):
```bash
uv run python train_autoencoder.py --data real --split split2 --encoders learnable_linear \
    --bg_weight 5.0 --fp_weight 0.5 --run_name real_split2_1d_ah --device cuda:0
uv run python train_spatial.py --split split2 \
    --bg_weight 5.0 --fp_weight 0.5 --run_name real_split2_spatial_ah --device cuda:1
```

## 7. Caveats / TODO
- **Evaluator B** (base ViT3D *without* token pruning/merging) not yet run — no matching
  base checkpoint on disk; may be realisable by toggling pruning off on the same ckpt.
- Sweeps use `divide=3`; re-confirm headline configs at full resolution.
- Peak-level F1 (the repo's per-peak metric) and the waveform-level spurious-peak
  metrics (`false_ghost_rate`, `evaluate_autoencoder.py`) not yet tabulated AH-vs-base.
- Single seed; weights bg/fp not swept.
