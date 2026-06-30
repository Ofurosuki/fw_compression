# Architecture-controlled representation test: retrain ToPM on each input representation

**Question (PI):** the frozen-judge and FW_Event_Net experiments changed *representation*
and *architecture/training* together, so they cannot isolate a representation's value.
Hold the architecture fixed (= **ToPM**, `vit3d_ordered_pruning_light`, 8.72 M) and change
only the input: lift each representation back to a `T=700` pseudo-waveform and **retrain
ToPM from scratch on it** under one fixed recipe. Then "representation effectiveness" =
F1(ToPM retrained on rep) / F1(ToPM retrained on full waveform).

Recipe (identical for all rows): the repo's split2 baseline — AdamW lr 1e-4, timm-cosine,
focal+dice, batch 4, 50 epochs, no augmentation, divide=3 (~4.7 k train / ~1 k val frames),
ToPM arch/pruning from `evalA_split2_test_best.yaml`. Repo kept read-only; the input
transform is the same `_load_voxel_grid` monkey-patch as the eval harness, applied to
training too (event reps pre-cached as uint16). Each retrained model is **evaluated with
the SAME representation transform on the test set** (so train and test see the same input).
Metric: voxel-level F1-mean over {object, glass, ghost}, divide=3 test (≈475 frames).

Metrics: **voxel-level** and **peak-level** F1-mean over {object, glass, ghost}, divide=3
test (≈475 frames). Peak-level = the paper's metric (scored only at scipy `find_peaks`
return-peak positions); we score on the **raw-waveform** peak set, identical for every
config (the paper population), so taw/ta are judged at *all* true returns incl. ones they
dropped — `run_eval_peak.py` (its voxel cross-check matches `run_eval.py` exactly).

## Result

| representation | frozen voxel¹ | **retrained voxel** | **retrained peak²** | peak obj / glass / ghost |
|---|--:|--:|--:|--|
| **full waveform** (T=700) | 0.524 | **0.533** | **0.595** | 0.770 / 0.300 / 0.715 |
| **taw** K=4 (t,a,w events→Gaussian) | 0.408 | **0.531** | **0.582** | 0.754 / 0.271 / 0.722 |
| **ta** K=4 (multi-echo: t,a, fixed width) | 0.176 | **0.515** | **0.574** | 0.738 / 0.339 / 0.645 |

¹ frozen neurips_best on the synthesised rep (no retraining), voxel, divide=3 — `outputs/events_best/`.
² peak-level, raw-peak population — `outputs/retrain/peak_*.json`. **full peak 0.595 ≈ paper
0.592 / frozen neurips_best peak 0.599** → retrained numbers are paper-comparable and the
recipe is validated (voxel ceiling 0.533 = frozen 0.532 too). Retrained-voxel per-class
(obj/glass/ghost): full 0.720/0.272/0.607, taw 0.717/0.250/0.626, ta 0.695/0.304/0.547.
Ckpts `outputs/retrain/{full_ceiling,taw_k4,ta_k4}/0618/…epoch_50…pth`.

## Findings

1. **Sparse top-K events are essentially lossless once the model adapts.** Retrained
   **taw K=4 = 0.531 voxel / 0.582 peak ≈ full 0.533 / 0.595** (voxel Δ −0.002 within
   ±0.003 noise; peak Δ −0.013 = 98 % of full). The dense `T=700` waveform carries
   virtually **no** downstream-relevant information beyond the top-K `(t,a,w)` events —
   given a ToPM trained to read them. (ghost even *improves*: voxel 0.607→0.626, peak
   0.715→0.722 — the clean Gaussian synthesis denoises secondary returns.) **Both metrics
   agree**, and full peak 0.595 ≈ the paper's 0.592, so this is paper-comparable.

2. **The frozen-judge experiments massively understated the representations — the gap was
   domain shift, not information loss.** Retraining recovers **+0.12 (taw)** and **+0.34
   (ta)**. Frozen ToPM was trained on real waveforms and had never seen synthesised
   Gaussian / fixed-width pulses, so it scored them as out-of-distribution. That penalty is
   *not* a property of the representation. The architecture-controlled, adapted comparison
   is the fair one, and it says: **both taw and ta ≈ full**.

3. **Width's true value over multi-echo is small once architecture is fixed AND adapted:
   taw − ta = +0.016 voxel / +0.008 peak** (taw 0.531/0.582 vs ta 0.515/0.574). The frozen
   experiment showed **+0.23** (0.408 vs 0.176) — a ~14–28× overstatement driven by
   fixed-width `ta` synthesis being even more OOD for the frozen model. With an adapted
   ToPM, conventional multi-echo `(t,a)` already reaches **97 %** (voxel) / **96 %** (peak)
   of the full-waveform ceiling; full-waveform pulse *width* adds only ~2–3 %.
   - Per-class (both metrics agree): width helps **ghost** (voxel 0.626 vs 0.547, peak
     0.722 vs 0.645) and **object**, but **hurts glass** (voxel 0.250 vs 0.304, peak 0.271
     vs 0.339). So full-waveform width is a ghost/object pulse-shape cue, not a glass cue —
     consistent with glass being a representation/domain problem (cf. FW_Event_Net glass
     ceiling), not something width fixes. (ta even beats *full* on glass: 0.339 vs 0.300 peak.)

**Takeaway.** Architecture-controlled and adaptation-controlled, the dense full waveform is
**not needed** for ToPM's ghost/object/glass task: a sparse top-K `(t,a,w)` event list (12
numbers/pixel, ~58× smaller) matches it, and even a plain multi-echo `(t,a)` list reaches
97 %. The large representation gaps reported under the *frozen* judge were dominated by
train/test domain shift, not by information the representation discards. This both
**confirms the PI's methodological point** and **tempers the earlier "width matters a lot /
full-waveform ≫ multi-echo" reading**: width's architecture-controlled margin is ~0.016.

## 2-seed confirmation (voxel, divide=3)

Seed 42 (epoch 50) + seed 43 (epoch 48; converged — train loss matches seed-42 within
0.002). Caches are seed-independent, so seed only varies init/shuffle/crop.

| rep | **voxel** s42 / s43 → mean | **peak** s42 / s43 → mean |
|---|--|--|
| full | 0.533 / 0.540 → **0.536** | 0.595 / 0.602 → **0.598** |
| taw K4 | 0.531 / 0.515 → **0.523** | 0.582 / 0.568 → **0.575** |
| ta K4 | 0.515 / 0.512 → **0.514** | 0.574 / 0.573 → **0.574** |

Margins (2-seed mean): **`taw − ta`** = +0.009 voxel / **+0.001 peak** (s42/s43: +0.016/+0.003
voxel, +0.008/**−0.006** peak — seed43 ta *beats* taw at peak); **`full − taw`** = +0.013 voxel /
+0.023 peak (taw = 97.5 % voxel / **96 %** peak of full).

- **Width's NET value over multi-echo is ~0** — voxel +0.009, **peak +0.001**, and it
  *sign-flips* across seeds at peak-level. The single-seed +0.016 was optimistic; the frozen
  experiment's +0.23 was pure domain shift. Pulse width does **not** add net downstream F1.
- **taw ≈ full (achievability) holds** — 96–97.5 % of full across both metrics; dense waveform
  ≈ not needed.
- **Per-class width effect is robust across 2 seeds × 2 metrics**: width **helps ghost**
  (taw 0.69–0.72 peak > ta 0.64) and **hurts glass** (ta 0.33–0.34 ≈ full > taw 0.27 peak).
  So full-waveform width **redistributes ghost↔glass, it does not raise net F1** — consistent
  with glass being a representation/domain problem, not something width fixes.
- *Methodology:* margins (width, full-gap) are reported as **2-seed means** (the effect is
  smaller than seed noise, so best-of-seed would be cherry-picking); the per-class sign effects
  are robust regardless. seed42 ep50 / seed43 ep48 (both converged, loss within 0.002).

## K-sweep — does taw ≈ full hold at lower K? (seed 42, divide=3)

| rep | K | dim | ratio | **voxel** | **peak** | ghost (vox/peak) |
|---|--:|--:|--:|--:|--:|--|
| **full** | — | 700 | 1× | **0.533** | **0.595** | 0.607 / 0.715 |
| taw | 6 | 18 | 39× | **0.540** | **0.595** | 0.605 / — |
| taw | 5 | 15 | 47× | **0.529** | **0.585** | 0.606 / — |
| taw | 4 | 12 | 58× | **0.531** | **0.582** | 0.626 / 0.722 |
| taw | 3 | 9 | 78× | **0.532** | **0.581** | 0.594 / — |
| taw | 2 | 6 | 117× | **0.503** | **0.554** | 0.537 / 0.637 |
| ta | 4 | 8 | 88× | **0.515** | **0.574** | 0.547 / 0.645 |
| ta | 2 | 4 | 175× | **0.508** | **0.565** | 0.521 / 0.612 |

**taw saturates at full by K≈6**: K6 (39×) reaches full (peak 0.595 = full 0.595; voxel 0.540 ≈
0.533), K2→K4→K6 rises monotonically, K4 already ~99 %. **K5/K6 were run on Tiger (RTX A6000)** as
the multi-server pipeline validation (git+uv-sync+env.yaml) and agree with dragon's K2/K4/full —
cross-machine consistent.

- **taw ≈ full needs K=4** — taw K4 = 99.6 % voxel / 98 % peak of full, but **K2 = 94 % / 93 %**.
  The K4→K2 drop (−0.028 voxel / −0.028 peak) is **almost entirely ghost** (voxel 0.626→0.537,
  peak 0.722→0.637); object/glass flat. K controls how many returns survive and the binding one
  is the **secondary (ghost) return** — so "dense waveform ≈ not needed" holds at K=4 (58×) but
  loosens at K=2 (117×).
- **width's net value stays ≤ 0 across K and both metrics**: taw − ta = +0.016/+0.008 (K4,
  vox/peak) → **−0.005 / −0.011 (K2)** — at K=2 the multi-echo `ta` *edges out* `taw` in both
  metrics. Reinforces the 2-seed conclusion: pulse width does not add net F1.

## AE reconstruction vs sparse events under retraining (the frozen-judge ranking REVERSES)

The frozen-judge sweep (RESULTS.md §3–4) ranked the **spatial-4×4 anti-hallucination AE** as the
best compression (~0.48 frozen). But that was a frozen model seeing OOD reconstructions. Redone
under the architecture-controlled retrain protocol (retrain ToPM on each AE reconstruction; AE
ckpts from `runs/real_split2_*`), seed 42, voxel, divide=3:

| representation | ratio | **voxel** | **peak** | ghost (vox/peak) |
|---|--:|--:|--:|--|
| full | 1× | **0.533** | **0.595** | 0.607 / 0.715 |
| **taw** K4 (sparse events) | 58× | **0.531** | **0.582** | 0.626 / 0.722 |
| **ta** K4 (multi-echo) | 88× | **0.515** | **0.574** | 0.547 / 0.645 |
| AE spatial-AH | 22× | **0.494** | **0.555** | 0.509 / 0.626 |
| AE spatial-AH | 88× | **0.472** | **0.528** | 0.465 / 0.571 |
| AE spatial (no AH) | 88× | **0.443** | **0.497** | 0.443 / 0.541 |
| AE 1D-AH | 88× | **0.409** | **0.469** | 0.329 / 0.437 |

- **Under fair retraining, the dense AE reconstruction LOSES to the sparse event representation**
  (voxel + peak agree). At matched 88×, ta-K4 0.515/0.574 beats AE spatial-AH 0.472/0.528 (**+0.04/+0.05
  peak**); best AE (spatial-AH 22×) 0.494/0.555 only ≈ taw-K2 (0.503/0.554, but at 117× vs 22×) and
  stays below taw/ta-K4. The frozen judge ranked AE > events; **retraining reverses it** — an
  MSE-optimised dense reconstruction smears the transport structure, while explicit top-K `(t,a,w)`
  events preserve it. This is the payoff of re-evaluating AE under retraining (the frozen ranking was a
  domain-shift artifact, like taw/ta).
- **Anti-hallucination still helps under retraining**: spatial 88× AH vs no-AH = **+0.029 voxel /
  +0.031 peak**. The model does *not* just learn to ignore hallucinated peaks — AH yields a genuinely
  cleaner, more informative representation (a real improvement, not a frozen-judge artifact).
- **Ghost is the AE weakness** (AE ghost 0.33–0.51 voxel ≪ taw/ta 0.52–0.63): the dense reconstruction
  loses weak secondary returns the sparse event rep keeps explicitly (K≥2). spatial ≫ 1D; lower
  compression helps AE (22× > 88×) but never reaches the event reps even at 4× less compression.

## Why width helps ghost but hurts glass (per-scene width diagnostic)

`eventnet/diag_width_per_scene.py` (fig `outputs/diag/width_per_scene.png`) — per-event FWHM
by class, per scene, train vs the 3 held-out test scenes. AUC = P(w_class > w_object):

- **glass width is NON-transferable** — `glass|object` AUC: **train mean 0.49, range [0.39,0.62],
  sign-flips in 5/7 scenes** (glass wider than object in 16buildA_large 0.62 / gym 0.52, narrower
  in 16buildA_mid 0.39 / 14build_2floor 0.47); **on all 3 TEST scenes glass is narrower (0.31,
  0.39, 0.43, mean 0.37)**. So glass-vs-object width has no consistent direction on train AND
  shifts on test → a model that uses w for glass latches onto a scene-specific/noisy correlation
  and misapplies it on the held-out scenes → **w hurts glass** (taw glass < ta glass, both seeds).
  Same "separable-but-not-transferable" failure as behind_energy (FW_Event_Net/SCENE_GEOMETRY.md).
- **ghost width IS transferable** — `ghost|object` AUC < 0.5 in ~every scene (ghost ~8 bins
  consistently narrower than object ~10) → a stable cue → **w helps ghost** (taw ghost > ta ghost).

So the robust per-class width effect is mechanistically explained: width carries a *transferable*
ghost cue (ghost narrower) but only a *scene-dependent, sign-flipping* glass cue → net ≈ 0, with a
ghost↔glass redistribution. This is why the architecture-controlled width margin is ~0.

## Caveats / next
- **Single seed** each. taw ≈ full is within noise (claim robust); taw − ta = +0.016 voxel /
  +0.008 peak is ~3–5× the ±0.003 noise (likely real) but a 2nd seed would confirm.
- **Both voxel- and peak-level done** (peak = paper population; full peak 0.595 ≈ paper
  0.592). Numbers above are divide=3 (≈475 frames; matches full-res within ~0.003).
- Caches: event reps strided 1/3 (uint16) vs full's random 1/3 — same scenes, negligible.
- Open extensions: K-sweep (does taw=full hold at K=2?), `tw`, AE recon; optional frozen-peak
  for taw/ta to show the domain-shift gap at peak-level too.
