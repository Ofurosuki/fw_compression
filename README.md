# fw-compression

Initial research experiment for **full-waveform LiDAR compression**: can a
*lightweight sensor-side encoder* + *off-sensor DNN decoder* preserve the
transport-related information (multi-peak structure, ghost/multipath returns)
that the Ghost-FWL ghost detection/removal pipeline depends on?

See [`initial_research_plan.md`](initial_research_plan.md) for the full plan.

## Setup (uv)

```bash
uv sync
```

Environment: Python 3.12, `torch==2.5.1+cu121` (CUDA), numpy/scipy/matplotlib.

## Pipeline

```
full waveform x [B,T]
  → lightweight encoder E(x)      (binning / random proj / DCT / learnable linear)
  → compressed latent z [B,K]
  → DNN decoder D(z)              (MLP K→256→512→T)
  → reconstructed pseudo-waveform x_hat [B,T]
  → (deferred) Ghost-FWL downstream  /  proxy ghost-detection score
```

## Code layout

| file | role |
|------|------|
| `compression/data/synthetic_waveforms.py` | synthetic multi-peak + ghost waveform generator (T=700, matches Ghost-FWL voxel depth); swap-in point for real data |
| `compression/encoders.py` | `coarse_binning`, `random_projection`, `dct_lowfreq`, `learnable_linear` |
| `compression/decoders.py` | `MLPDecoder` (+ `DeepMLPDecoder`) |
| `compression/autoencoder.py` | encoder→latent→decoder + `reconstruction_loss` (MSE + optional energy + peak-aware) |
| `compression/utils/metrics.py` | waveform MSE, peak localization error, peak-count preservation, energy error, ghost recall, compression ratio |
| `compression/utils/plot.py` | original-vs-reconstructed plots, metric-vs-K sweep plots |
| `compression/downstream/ghost_fwl_hook.py` | stable interface to Ghost-FWL downstream (deferred) + `proxy_ghost_score` |
| `train_autoencoder.py` | trains the encoder × K sweep, saves checkpoints |
| `evaluate_autoencoder.py` | metrics JSON, saved `x_hat`/`z`, example plots, sweep summary |

## Run

```bash
# train the full sweep (4 encoders × K∈{8,16,32,64,128})
uv run python train_autoencoder.py --run_name sweep --epochs 30

# evaluate: metrics, reconstructed waveforms, latents, plots
uv run python evaluate_autoencoder.py --run_name sweep

# fast sanity check
uv run python train_autoencoder.py --smoke && uv run python evaluate_autoencoder.py --run_name smoke
```

Outputs land under `runs/<run_name>/`:
`<encoder>_K<K>/{checkpoint.pt, metrics.json, x_hat.npy, z.npy, examples.png}`,
plus `summary.json`, `sweep.png`, `upper_bound.json`.

## Synthetic data

Each waveform (length `T=700`) has a primary Gaussian return, optional ghost /
multipath secondary returns (probability 0.55, the *transport* signal we test),
a background floor, and Poisson (shot) + read noise. Ground-truth peak labels
`(position, intensity, width)` mirror the real Ghost-FWL `*_peak.npy` format.

**To use real Ghost-FWL data:** replace `WaveformDataset` with a dataset that
yields the same `(wave[T], label_dict)` contract (extract per-pixel waveforms
from the `(400, 512, 700)` voxel grids; peak labels from the `*_peak.npy` files).

## Downstream evaluation (implemented)

The real downstream is now wired up end-to-end (`downstream/`). The frozen,
pre-trained Ghost-FWL segmentation model **FWL-ToPM** (`vit3d_ordered_pruning_light`
from the evolved repo, paper *"Towards Real-Time FWL Transformers…"*) is used as a
fixed evaluator: compression quality is the **downstream F1-mean over
object/glass/ghost** (Noise excluded) on the split2 test scenes.

`downstream/run_eval.py` drives the (read-only) Ghost-FWL repo via `PYTHONPATH`,
loads the frozen model, and optionally inserts a **compress → reconstruct**
transform on the raw `T=700` per-pixel waveforms (per-pixel max-normalize → AE →
de-normalize) *before* the model's own crop pipeline, then reports
confusion-matrix F1 (the repo's slow per-pixel peak detection is skipped). It also
emits a fixed 6-waveform orig-vs-recon figure per config. `run_sweep.py` fans the
sweep over GPUs; `run_plot.py` aggregates to a table + F1-vs-ratio curve.

```bash
# baseline (no compression) and one compressed config:
PYTHONPATH=<ghost-fwl-repo>/src uv run python downstream/run_eval.py \
    --config downstream/configs/evalA_split2_test.yaml --compress none
PYTHONPATH=... uv run python downstream/run_eval.py \
    --config downstream/configs/evalA_split2_test.yaml \
    --compress ae --ae_ckpt runs/real_split2_spatial/spatial_K512/checkpoint.pt \
    --viz_out downstream/outputs/sp_K512.png
uv run python downstream/run_sweep.py && uv run python downstream/run_plot.py
```

**Results** (FWL-ToPM, split2 test; see `downstream/outputs/sweep/`): no-compression
baseline **F1-mean 0.66**. The **spatial 4×4** autoencoder is the most robust —
~0.56–0.58 from 6× to 88× — while the per-pixel 1D learnable encoder is intermediate
and naive coarse-binning collapses at high ratio (glass class effectively lost). See
`downstream/outputs/sweep/f1_vs_ratio.png` and the per-config waveform overlays.

The earlier `proxy_ghost_score` / `GhostFWLEvaluator` stub remains for quick
self-contained checks, but the headline downstream number now comes from the real
frozen FWL-ToPM model above.

## Physical-data experiment & per-peak findings

The second iteration uses `physical_waveforms.py` (parametric EMG IRF, 1/r²
intensity, physical multipath, Poisson+ambient noise, FWHM 8–36 bins) and
evaluates **per-pulse parameter preservation** by fitting the known IRF to
each reconstructed peak — recovering position / intensity (area) / FWHM — and
reporting errors split by **narrow (high-freq)** vs **wide (low-freq)** peaks.

```bash
uv run python train_autoencoder.py --run_name sweep_phys --data physical --epochs 30
uv run python evaluate_autoencoder.py --run_name sweep_phys --data physical
```

Key results (errors are relative; ↓ better. Clean-fit floor ≈ 0.2):

| K=8 (87×) | narrow int | narrow FWHM | narrow recall | wide int | wide FWHM |
|---|---|---|---|---|---|
| coarse_binning | **2.40** | **4.19** | 0.34 | 1.79 | 2.02 |
| dct_lowfreq | 0.80 | 1.54 | 0.69 | 0.65 | 0.73 |
| **learnable_linear** | **0.25** | **0.36** | **0.75** | 0.27 | 0.27 |
| random_projection | 0.43 | 0.52 | 0.70 | 0.31 | 0.29 |

**Findings (support the research hypothesis):**
1. **Low-pass encoders (coarse binning, DCT low-freq) sacrifice high-frequency
   structure.** They show a large narrow-vs-wide gap (binning at K=16: FWHM error
   1.71 narrow vs 0.58 wide) — exactly the "depth-oriented compression loses
   sharp/multi-peak transport info" signature.
2. **The learned linear encoder preserves transport info best and is
   frequency-agnostic** — narrow ≈ wide errors at all K, and near the clean-fit
   floor even at 87× compression where the fixed encoders collapse.
3. **Graceful degradation**: all encoders degrade smoothly with K; ~K=32 (22×) is
   the knee where fixed encoders still hold up. See `sweep_narrow.png`,
   `sweep_wide.png`, and the `learnable_linear_K16` vs `coarse_binning_K16`
   example plots for the qualitative contrast.

## Real Ghost-FWL data experiment (third iteration)

The third iteration runs the same pipeline on the **real Ghost-FWL dataset**
(`ghost_datasets/scene001`, `scene002`; `(400,512,700)` blosc2 voxels + per-voxel
`annotation_v1_expand` semantic labels `{noise,object,glass,ghost}`).
`compression/data/real_waveforms.py` extracts per-pixel waveforms (max-normalized),
deriving peak labels from contiguous same-class annotation runs and **measuring
reference intensity/width on the original (uncompressed) waveform** — there is no
parametric ground truth, so the natural reference is the original waveform itself.

```bash
# cross-validation: A = train scene001 / val scene002 ; B = swapped
uv run python train_autoencoder.py --data real --cv A --run_name real_A --epochs 30 --device cuda:0
uv run python evaluate_autoencoder.py --data real --cv A --run_name real_A --device cuda:0
# (repeat with --cv B --run_name real_B)
```

**How the metrics change from synthetic/physical** (confirmed with the user before
running): no GT peak params and unknown IRF, so the EMG-fit `intensity/fwhm_relerr`
*vs truth* is replaced by **fidelity vs the original waveform** (model-free
`measure_peak`: height/area/FWHM); narrow/wide is split by the **measured** width on
the original (not GT FWHM); and the headline metric becomes **per-class peak survival
recall** (`object/glass/ghost`) against the real semantic labels — `ghost_recall`
being the real transport signal that replaces the synthetic `proxy_ghost_score`.

The AEs are now retrained on **all scenes** via the `SPLIT2` multi-scene split
(7 train / 3 held-out test, aligned with the downstream repo) — `--split split2` on
both train scripts — instead of the old 2-scene cross-validation, and the real
Ghost-FWL 3D segmentation downstream is now **implemented** (see *Downstream
evaluation* above), no longer deferred.

Each scene → 75,000 waveforms (30 frames × 2,500 labelled pixels, half ghost-bearing),
cached under `runs/real_cache/`.

### Key results (ghost survival recall; upper-bound ceiling ≈ 0.98–0.99)

**CV-A (train scene001 → val scene002)** reproduces the synthetic/physical hypothesis
cleanly and monotonically:

| ghost survival | K=8 (87×) | K=16 | K=32 | K=64 |
|---|---|---|---|---|
| **learnable_linear** | **0.90** | 0.95 | 0.96 | 0.97 |
| random_projection | 0.92 | 0.93 | 0.97 | 0.96 |
| dct_lowfreq | 0.72 | 0.81 | 0.91 | 0.96 |
| coarse_binning | 0.67 | 0.79 | 0.87 | 0.94 |

→ low-pass encoders (binning, DCT) drop ghosts first at high compression; the learned
linear encoder preserves both ghost *detectability* and *intensity* (int_relerr 0.31
at K=8 vs 0.48–0.56 for the fixed encoders) — random projection keeps detectability
but distorts intensity more.

**CV-B (train scene002 → val scene001)** exposes a large, asymmetric **cross-scene
domain gap**, not a bug (upper bound is clean in both directions): ghost survival
collapses to 0.31–0.76 and goes non-monotonic in K. Cause: **the two scenes differ
~3× in ghost brightness** — scene002 ghost area median ≈ 5.0 vs scene001 ≈ 1.5. So
CV-A tests on *brighter* (easier) ghosts than trained, while CV-B tests on *fainter*
(harder) ghosts than trained. The adaptive encoders (learnable/random/DCT) show the
biggest A−B asymmetry (+0.3 … +0.7); crude `coarse_binning` is the most
direction-robust (smallest asymmetry) but lowest-ceiling.

**Honest takeaway:** the method is validated *within distribution* (CV-A reproduces
the hypothesis on real ghosts), but **cross-scene generalization to fainter-than-trained
ghosts is the real bottleneck**. With only two scenes — each a single physical layout
(50 hist positions × 50 near-duplicate temporal frames) and mismatched in ghost
brightness — robust deployment needs brightness-diverse training (more scenes / pooled
training), which the current scene-CV deliberately stresses. Outputs:
`runs/real_{A,B}/summary.json`, `sweep_survival.png`, `sweep_freq.png`, `sweep.png`.

## Spatio-temporal (4×4) compression experiment (fourth iteration)

Following ICCV2023 "Learned Compressive Representations" (Gutierrez-Barragan et al.),
which compresses a *local spatial block* of histograms rather than each pixel
independently, this iteration tests whether **spatial context (neighbouring pixels)
preserves ghost/transport information better than per-pixel compression** — and
whether it helps the cross-scene domain gap.

- `compression/data/spatial_waveforms.py` — extracts 4×4 neighbour patches `[16, 700]`
  (each pixel max-normalized, same as per-pixel, so detector calibration is shared and
  the exploitable spatial redundancy is peak-*position/shape* correlation across
  neighbours). Per-pixel labels kept for all 16 pixels.
- `compression/spatial_coding.py` — **separable** learned coding tensors
  `C_k = c^t_k ⊗ c^s_k` (the paper's parameter-efficient design): encode `[B,16,700]→[B,K]`,
  MLP decode `[B,K]→[B,16,700]`.
- `train_spatial.py` / `evaluate_spatial.py` — block K swept over
  `{128,256,512,1024,2048}` = 16×`{8,16,32,64,128}`, so the implied **per-pixel ratio
  `(16·700)/K` matches** the per-pixel experiment exactly. Reconstructed patches are
  unfolded back to per-pixel waveforms and scored with the *same* `real_peak_metrics`.

```bash
uv run python train_spatial.py    --cv A --run_name real_A_spatial --epochs 30 --device cuda:0
uv run python evaluate_spatial.py --cv A --run_name real_A_spatial --device cuda:0
# (repeat --cv B --run_name real_B_spatial)
```

### Key result: spatial context buys cross-scene robustness at a small in-distribution cost

Ghost survival recall, mean over the K sweep (matched per-pixel ratio), both CV directions:

| method | CV-A (in-dist) | CV-B (cross-domain) | mean | A−B gap |
|---|---|---|---|---|
| per-pixel learnable_linear | **0.95** | 0.45 | 0.70 | 0.50 |
| per-pixel random_projection | 0.95 | 0.47 | 0.71 | 0.49 |
| **spatial 4×4 separable** | 0.92 | **0.55** | **0.73** | **0.36** |

- **In-distribution (CV-A)** spatial does *not* help — per-pixel learnable is already
  near the ceiling (≈0.97 at low compression) and spatial plateaus at ≈0.92 across all
  K (it spreads the K budget over 16 pixels; with per-pixel normalization only position
  correlation — not amplitude — is available to exploit).
- **Cross-domain (CV-B, test on ~3× fainter ghosts than trained)** spatial **helps
  substantially**: +0.10…+0.26 ghost survival at most K (e.g. at 22× compression
  0.45→0.71), because a faint ghost ambiguous in one pixel is corroborated by its 4×4
  neighbourhood. This **shrinks the domain-gap asymmetry from 0.50 to 0.36** — directly
  attacking the generalization bottleneck identified in the per-pixel experiment.
- **Trade-off:** spatial reconstructs ghost *intensity* slightly less precisely
  (int_relerr ≈0.28–0.55 vs per-pixel ≈0.17–0.48) — it preserves ghost *detectability*
  better but *amplitude* a little worse.

**Takeaway:** the ICCV2023 spatial-coding thesis transfers to the ghost/transport
regime, but its payoff here is *robustness*, not peak in-distribution accuracy — it is
worth it precisely when train/test ghost brightness differ. Outputs:
`runs/real_{A,B}_spatial/{summary.json, sweep_survival.png, sweep_freq.png, sweep.png}`.
Caveats: still only 2 scenes; spatial gain may grow with larger/varied blocks, amplitude-
preserving (per-patch) normalization, or a conv decoder — none explored yet.

## Notes / calibration findings

- The **energy-preservation loss** (`|Σx_hat − Σx|`) must stay *off* by default:
  at init it dwarfs the MSE and the decoder learns to match total energy with a
  flat smear, never localizing peaks. Use MSE + peak-aware loss instead.
- The peak detector is Gaussian-smoothed + prominence/relative-height thresholded,
  calibrated so detection on a clean original recovers ground-truth peaks
  (count-accuracy ≈0.98, recall ≈0.996) — making compression-induced degradation
  interpretable rather than swamped by noise over-detection.
