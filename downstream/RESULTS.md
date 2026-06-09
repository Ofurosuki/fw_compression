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
- **F1-mean = macro-F1 over {object, glass, ghost}, Noise excluded** (`ignore_visualize_labels: [0]`),
  the same headline metric as the paper. Computed from the per-voxel confusion matrix
  (argmax with softmax threshold 0.5).

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

FWL-ToPM on the **full** split2 test set, original waveforms:

| metric | object | glass | ghost | **F1-mean** |
|---|---|---|---|---|
| F1 | 0.797 | 0.370 | 0.800 | **0.656** |

(`divide=3` subset gives 0.662, used as the reference line in the sweep plots.)
Glass is intrinsically the hardest class (transparent, minority) — only 0.37 even
uncompressed — and is the first to degrade under compression.

---

## 3. Sweep with the standard loss (MSE + peak-aware)

`downstream/outputs/sweep/` — figure `f1_vs_ratio.png`, table `summary.txt`.

| ratio | spatial 4×4 | 1D learnable | 1D coarse (naive) |
|---|---|---|---|
| — (none) | **0.662** | | |
| 6× | 0.583 | 0.544 | 0.569 |
| 11× | 0.575 | **0.585** | 0.549 |
| 22× | 0.581 | 0.550 | 0.530 |
| 44× | 0.581 | 0.560 | — |
| 88× | **0.560** | 0.509 | 0.441 |

**Findings**
1. **spatial 4×4 is the most robust** — essentially flat (0.56–0.58) from 6× to 88×.
   Sharing the 4×4 spatial neighbourhood lets it preserve peak shape even at extreme
   compression. At 88× it keeps 85 % of baseline (0.560 / 0.662).
2. **naive coarse-binning collapses at high ratio** — 0.441 at 88× (67 % of baseline),
   with the glass class essentially lost (per-class F1 ≈ 0.02).
3. **1D learnable is intermediate** and degrades at high ratio (0.509 at 88×).
4. **glass is the bottleneck class** throughout; object/ghost are far better preserved.

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
| 88× | 1D learnable | 0.509 | **0.577** | **+0.067** |
| 44× | 1D learnable | 0.560 | 0.575 | +0.015 |
| 22× | 1D learnable | 0.550 | 0.568 | +0.018 |
| 11× | 1D learnable | 0.585 | **0.629** | **+0.044** |
| 6×  | 1D learnable | 0.544 | 0.554 | +0.009 |
| 88× | spatial 4×4 | 0.560 | 0.588 | +0.028 |
| 44× | spatial 4×4 | 0.581 | 0.593 | +0.012 |
| 22× | spatial 4×4 | 0.581 | 0.601 | +0.020 |
| 11× | spatial 4×4 | 0.575 | **0.618** | **+0.043** |
| 6×  | spatial 4×4 | 0.583 | 0.596 | +0.013 |

**Findings**
1. **Anti-hallucination loss improves downstream F1 for every config** (Δ +0.01…+0.07).
2. **Largest gains at high compression** (1D 88×: +0.067) — where information is scarce,
   hallucinated peaks do the most downstream damage, so suppressing them helps most.
3. Both methods improve uniformly; the waveform figures show flatter reconstructed
   backgrounds (fewer spurious bumps).
4. The AH models have *higher* reconstruction val-MSE than the base models yet *better*
   downstream F1 — concrete evidence that **reconstruction MSE is not the right proxy**
   for downstream quality, which is exactly why this frozen-model harness exists.

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

Working hypothesis: an **optimisation/conditioning artifact of the shared fixed recipe**
(lr=1e-3, 30 ep, no LR decay) — wider decoder fan-in is harder to condition, so the
larger-K models settle at a higher plateau. A diagnostic (retrain K∈{32,64,128} with
cosine LR + more epochs) is running to confirm; if K=64/128 then match/beat K=32 it is
optimisation, not capacity. *(results: `downstream/outputs/diag_capacity.log` — to be
appended.)* Note this MSE quirk does **not** carry over to downstream F1 (whose 1D peak
is at 11×), reinforcing §4.4.

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
