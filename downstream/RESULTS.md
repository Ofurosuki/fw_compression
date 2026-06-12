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

## 6. Top-K transport-event representation (does Ghost-FWL need a dense waveform?)

Instead of compress→reconstruct, replace each waveform by a **sparse list of top-K
transport events** `{(t_i, a_i, w_i)}` (peak position, intensity, FWHM), synthesise a
Gaussian-pulse pseudo-waveform from them, and feed *that* to the same frozen FWL-ToPM.
This tests whether the downstream model needs the dense signal `x[t]` or only sparse
peak/event parameters. See `event_aware_experiment_plan.md`.

- **Extraction** `compression/event_extraction.py`: a faithful single-waveform scipy
  reference (`find_peaks`/`peak_widths`) **and** a GPU-vectorised batch extractor (height
  ranking + min-distance NMS, half-max-crossing FWHM) used by the hook — the scipy loop
  would cost ~2 h/config (~70 µs × 205 k px × 475 frames); the batch path is ~200 ms/frame.
- **Synthesis** `compression/event_synthesis.py`: `x̂[t]=Σ a_i·exp(−(t−t_i)²/2σ_i²)`,
  σ=FWHM/(2√(2 ln2)). The `representation` flag is the core ablation — `t` (position only,
  fixed a,w), `ta` (+intensity), `tw` (+width), `taw` (all three).
- **Eval**: `run_eval.py --compress event` (same monkey-patch insertion point and
  T=700-first normalise/de-normalise as the AE hook); sweep `run_sweep_events.py`, plots
  `run_plot_events.py`. **dim = K·n_params**, ratio = `T/dim`. Tests in `tests/`.

> **Evaluator updated to `neurips_best`** (2026-06-12). The event sweep was re-run on a
> stronger FWL-ToPM checkpoint
> (`…/checkpoints/neurips_best/vit3d_ordered_pruning_light_finetune_epoch_50_20260423_221908_0.02523.pth`,
> config `downstream/configs/evalA_split2_test_best.yaml`, outputs
> `downstream/outputs/events_best/`). Its full-waveform F1-mean is **0.524** (vs 0.517 on the
> prior `0.02485` checkpoint; glass 0.329 vs 0.298). **The headline table and findings 1–2
> below are on `neurips_best`.** The AE-baseline rows, the ghost diagnostics (finding 3) and
> the multi-echo table were **not** re-run and remain on the prior checkpoint — flagged inline.

Sweep K∈{1,2,3,4,6,8} × repr∈{t,ta,tw,taw} (divide=3), re-run on `neurips_best`
(`downstream/outputs/events_best/` — `f1_vs_ratio.png`, `f1_vs_k.png`, `summary.txt`).
Headline rows:

| representation | K | dim | ratio | object | glass | ghost | **F1-mean** |
|---|--:|--:|--:|--:|--:|--:|--:|
| **full waveform** | — | 700 | 1× | 0.694 | 0.329 | 0.547 | **0.524** |
| `taw` | 2 | 6 | **117×** | 0.606 | 0.247 | 0.424 | **0.426** |
| `taw` | 3 | 9 | **78×** | 0.582 | 0.255 | 0.407 | **0.414** |
| `taw` | 8 | 24 | 29× | 0.595 | 0.258 | 0.393 | 0.415 |
| `taw` | 1 | 3 | 233× | 0.555 | 0.176 | 0.042 | 0.258 |
| `tw`  | 2 | 4 | 175× | 0.465 | 0.247 | 0.309 | 0.341 |
| `ta`  | 4 | 8 | 88× | 0.086 | 0.145 | 0.298 | 0.176 |
| `t`   | 3 | 3 | 233× | 0.096 | 0.133 | 0.246 | 0.158 |

*(prior `0.02485` checkpoint, for reference: full waveform 0.517; best `taw` K=3 = 0.453;
AE spatial-4×4 base 0.456 / anti-halluc 0.483 — see git history for the full prior table.)*

**Findings** (on `neurips_best`)
1. **Sparse `(t,a,w)` events still explain most of the downstream performance.** `taw` peaks
   at **K=2: 0.426 = 81 % of the full-waveform F1 (0.524) at 117× compression**, and is flat
   (0.41–0.43) out to K=8. On the F1-vs-ratio curve `taw` sits well above the other reps and
   only ~0.1 below the full-waveform line across 30–120×. The headline holds: **Ghost-FWL
   perception is largely governed by sparse transport events, not dense waveform fidelity**
   (the plan's *Case A*). Note: this *stronger* checkpoint relies **more** on dense waveform
   than the prior one — the event↔full gap widened (88 %→81 %) and the optimum shifted from
   K=3 to K=2 — i.e. a better-trained model extracts somewhat more from the dense signal.
2. **Both intensity *and* width are needed — neither alone suffices, and the effect is
   sharper here** (*Case B*). Position only (`t`) collapses to ~0.13–0.16; adding *only*
   intensity (`ta`, ~0.15–0.18) barely helps; adding *only* width (`tw`, ~0.28–0.34) helps
   much more; `taw` jumps to ~0.41–0.43. Width (`tw`) again dominates intensity (`ta`),
   especially on **object** (0.47 vs 0.07 at K=2). Pulse *shape* is the key transport cue —
   this distinguishes full-waveform LiDAR from ordinary multi-echo LiDAR.

   **Multi-echo-LiDAR analogue (K=4).** `ta` K=4 — up to 4 returns each with range+intensity
   but *no pulse width/shape* — is exactly what a conventional multi-echo LiDAR delivers.
   Adding width (`taw` K=4) **more than doubles** the downstream F1 (gap even larger on this
   checkpoint, divide=3):

   | repr (K=4) | dim | object | glass | ghost | **F1-mean** |
   |---|--:|--:|--:|--:|--:|
   | `t`  (position only) | 4 | 0.094 | 0.111 | 0.216 | 0.140 |
   | **`ta` (multi-echo: pos+intensity)** | 8 | 0.086 | 0.145 | 0.298 | **0.176** |
   | `tw` (pos+width) | 8 | 0.429 | 0.151 | 0.327 | 0.303 |
   | **`taw` (full-waveform: pos+int+width)** | 12 | 0.580 | 0.253 | 0.391 | **0.408** |
   | full waveform | 700 | 0.694 | 0.329 | 0.547 | 0.524 |

   The **+0.23** from `ta`→`taw` (driven mostly by **object**, 0.09→0.58) is the quantitative
   value of full-waveform's pulse-shape information *over* a multi-echo sensor — larger than
   on the prior checkpoint (+0.19) and almost as large as the entire remaining gap to the
   uncompressed waveform. `ta` (0.176) trails `tw` (0.303) even more decisively here: for this
   downstream task **pulse width matters far more than return intensity**, so full-waveform
   LiDAR is not merely a higher-return-count multi-echo sensor — it carries qualitatively
   different (shape) information.
3. **`taw` plateaus from K≈2–3** and **ghost needs K≥2** — the first 2–3 events capture
   nearly all downstream-relevant structure. (On `neurips_best`, ghost goes 0.042→0.424 from
   K=1→2; unlike the prior checkpoint, K=1 no longer keeps object fully intact — object 0.555
   at K=1 vs 0.606 at K=2 — the stronger model wants ≥2 events even for the primary return.)

   > ⚠️ **The ghost diagnostics below (finding 3 detail) are on the prior `0.02485`
   > checkpoint** (`downstream/outputs/events/diag/`) and were not re-run on `neurips_best`;
   > the qualitative story (ghosts are secondary returns; recall-limited; depth-dependent
   > losses) is expected to carry over but the exact numbers will differ.

   **Why ghost is the bottleneck — diagnosed** (`diag_ghost.py`,
   `downstream/outputs/events/diag/`). The ghost gap is a **recall** problem (precision
   ~baseline): true ghost voxels get predicted as noise/object as K shrinks
   (recall full=0.78 → K3=0.57 → K2=0.37 → K1=0.02). Two compounding causes:
   - **(a) Ghosts are rarely the strongest return.** Over 383 k ghost pixels, the ghost
     peak's height-rank is **median 2**: only 10 % are the strongest (rank 1), 90 % are
     rank ≥2, 26 % rank ≥3. So top-K (which keeps the *tallest* peaks) drops the ghost
     return at small K — ghost-peak recovery (event kept within ±3 bins) is **K1=9 %,
     K2=72 %, K3=96 %, K≥4≈99 %**, mirroring the downstream recall curve.
   - **(b) Even when the peak is kept, the clean Gaussian under-detects.** At K=3 the ghost
     peak is recovered 96 % of the time, yet downstream ghost recall is only 0.57 — so a
     retained ghost return synthesised as a clean symmetric Gaussian still loses the
     shape/asymmetry/residual cues that distinguish a *ghost* secondary return from an
     *object* return. This is the same dense-residual signal that motivates the
     event-faithful / hybrid direction below (and explains the glass collapse).
   - **Detected-vs-lost ghosts are indistinguishable by their own waveform**
     (`diag_ghost_cases.py`, model predictions on matched voxels; orig recall 0.815 ≈ the
     0.78 baseline, event recall 0.596 — sanity-checked). Among true ghosts the original
     waveform *got right*, the ones the event rep then loses have the **same** ghost-return
     height (median 0.25 vs 0.25) and the **same** waveform clutter (median 3 returns vs 3)
     as the ones it keeps. So at K=3 the miss is *not* a per-pixel property (not dimmer,
     not busier) — it is driven by the model's **3D spatial context** + fine pulse shape
     that the clean-Gaussian synthesis discards uniformly. Misses split 59 % → noise,
     41 % → object. Figures: `ghost_success_vs_fail.png` (example detected vs lost voxels,
     orig-vs-synth model input), `ghost_lost_vs_amplitude.png` (the two refuted hypotheses).
   - **The losses are spatially structured, not random** (`diag_ghost_spatial.py`,
     `ghost_spatial_maps.png`): front-projecting ghost columns to (H,W) and comparing
     GT / no-compression / event-K3 predictions, the dropped ghosts form **coherent
     regions/bands** (column-level recall 0.88 → 0.71), not scattered voxels — the signature
     of a perturbed 3D-context decision rather than independent per-pixel peak losses.
   - **The one feature that separates detected vs lost is DEPTH** (`diag_ghost_features.py`,
     `ghost_feature_candidates.png`; AUROC over 40 k voxels each). Tested 7 candidates —
     reconstruction error at the ghost (AUROC 0.43), pulse asymmetry (0.49), separation to
     the nearest stronger return (0.50), spatial isolation at two window sizes (0.46/0.49),
     ghost-peak rank (0.47) are all near-chance. Only **range/depth position is a real
     discriminator: AUROC 0.71** — lost ghosts cluster at *early* depth (median d/D=0.23)
     vs detected at later depth (0.42). I.e. **close-range ghosts (early returns, near the
     strong primary) are the ones the clean-Gaussian event rep loses**; far/well-separated
     ghosts survive. (recon-error and isolation lean the expected way but only weakly.)
4. **Glass is the stress class** (*Case D*): even `taw` tops out ~0.25 and never reaches the
   already-low full-waveform 0.30. Transparent-object cues are not captured by simple top-K
   peaks — they likely need subtle residuals/tails or spatial context. Glass should be a
   stress test, not the success criterion.

**Interpretation / next direction.** Because `(t,a,w)` is strong, the indicated research
direction is **event-faithful full-waveform compression** — an AE (or codec) regularised to
preserve peak position/intensity/width rather than minimising MSE (consistent with §4: the
anti-hallucination loss, which protects peak/background structure, already beats MSE-optimal
recon downstream). Glass and dense-residual structure are where a *hybrid* (sparse events +
a small dense-residual/tail token) could close the remaining gap to the AH-AE.

## 7. Reproduce

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src

# baseline (full test set)
uv run python downstream/run_eval.py --config downstream/configs/evalA_split2_test.yaml \
    --compress none --out downstream/outputs/evalA_orig.json

# sweeps (base-loss AEs / anti-hallucination AEs) + plots
uv run python downstream/run_sweep.py    && uv run python downstream/run_plot.py
uv run python downstream/run_sweep_ah.py && uv run python downstream/run_plot_compare.py

# top-K transport-event sweep + plots (prior 0.02485 checkpoint, with AE overlay)
uv run python downstream/run_sweep_events.py && uv run python downstream/run_plot_events.py

# top-K transport-event sweep on the neurips_best checkpoint (current headline)
uv run python downstream/run_sweep_events.py \
    --config downstream/configs/evalA_split2_test_best.yaml --out downstream/outputs/events_best
uv run python downstream/run_plot_events.py --ev downstream/outputs/events_best --no_ae

uv run --with pytest python -m pytest tests/   # event extraction/synthesis unit tests
```

Retrain the AEs (SPLIT2, with anti-hallucination loss):
```bash
uv run python train_autoencoder.py --data real --split split2 --encoders learnable_linear \
    --bg_weight 5.0 --fp_weight 0.5 --run_name real_split2_1d_ah --device cuda:0
uv run python train_spatial.py --split split2 \
    --bg_weight 5.0 --fp_weight 0.5 --run_name real_split2_spatial_ah --device cuda:1
```

## 8. Caveats / TODO
- **Evaluator B** (base ViT3D *without* token pruning/merging) not yet run — no matching
  base checkpoint on disk; may be realisable by toggling pruning off on the same ckpt.
- Sweeps use `divide=3`; this matches full resolution for the F1 metric — the divide=3
  no-compression `none.json` (object/glass/ghost = 0.694/0.298/0.558) is **identical** to
  the full-res `evalA_noignore.json` (voxel counts are huge, so the confusion matrix
  converges). **Confirmed for the event configs too** (`downstream/outputs/events/fullres/`,
  full 1427 frames): `taw` K=3 = **0.4514** (vs divide=3 0.4525) and `taw` K=2 = **0.4364**
  (vs 0.4374) — agreement within 0.001, so all divide=3 event numbers above stand.
- The downstream **event hook uses the GPU-vectorised extractor** (height ranking + NMS),
  not the scipy `prominence` reference — the two agree on the few tallest well-separated
  peaks we keep but can differ on overlapping/shoulder peaks; `rank_by=prominence`/`area`
  and `intensity_mode=area` ablations not yet swept.
- Event reps not yet tried: `taw_bg` (background floor) and the dense-residual/tail
  extensions suggested by the glass collapse (§6 finding 4).
- Peak-level F1 (the repo's per-peak metric) and the waveform-level spurious-peak
  metrics (`false_ghost_rate`, `evaluate_autoencoder.py`) not yet tabulated AH-vs-base.
- Single seed; weights bg/fp not swept; event detection params (smooth_sigma, min_height,
  min_distance, fixed_width) at defaults.
