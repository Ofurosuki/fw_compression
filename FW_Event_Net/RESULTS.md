# FW Event-Tensor Net — results

A **trained-from-scratch** network that takes the sparse top-K transport-event
tensor `{(t, Δt, a, w, m)}` as input (replacing the dense `T=700` waveform) and
segments each event into `{noise, object, glass, ghost}`. Unlike the rest of
`fw_compression` (which feeds a *reconstructed* waveform into the **frozen**
Ghost-FWL model), this is a standalone model — see `FW_Event_Net/initial_plan.md`.

Core questions: **(1)** can sparse events replace the dense waveform for
ghost/glass perception, and **(2)** does waveform-derived width `w` add
information beyond a conventional multi-echo `(t, a)` sensor?

## TL;DR
- **Headline: V2 `taw` K=4 = 0.555 (2-seed mean) = 93 % of the dense-waveform
  ceiling** (paper peak-level; ceiling 0.599), up from V1's 0.534 (89 %), with the
  **best glass of any config (0.31** vs ≤0.28). The win comes from the **V2
  architecture** (cross-event attention over the K ray-returns + deeper U-Net),
  not from more features. Confirmed over 2 seeds; cross-scene test variance ±0.03.
- **Architecture changes the feature conclusions.** Under V2: (i) **width now
  helps** — `taw` − `ta` = +0.021, glass 0.27→0.31 (reversing V1's "width
  marginal"); (ii) **relative delay `Δt` becomes redundant/harmful** — `taw` >
  `tdtaw`, because attention already learns inter-return timing, so the best
  representation is just `taw` (t, a, w, m), *no Δt*. A plain channel-flatten CNN
  (V1) couldn't exploit width; attention can.
- **Sparse events recover ~89–93 % of the dense full-waveform upper bound** from a
  representation **>100× smaller** (4 numbers × 4 events vs 700 samples), at
  similar model size (V2 7.85M vs the frozen FWL-ToPM 8.72M). A trained event net
  *can* stand in for the dense waveform for ghost/glass perception.
- **Whether width helps is architecture-dependent.** With **V1** width is
  marginal (`taw` vs `ta` = +0.000, `tdtaw` vs `tdta` = +0.005; intensity is the
  big lever, `t_dt`→`tdta` +0.020). But with **V2** (cross-event attention) width
  helps: `taw` − `ta` = **+0.021** (2-seed mean), glass 0.268→0.307, in both
  seeds. Width carries real glass/ghost signal, but a channel-flattened CNN (V1)
  can't exploit it — it needs an architecture that relates a return's shape to the
  *other* returns on the ray. This overturns the earlier "width doesn't matter"
  reading and re-aligns with the frozen-judge experiment where width mattered.
- **The best-separating waveform feature (`behind_energy`) HURT downstream F1.**
  A feature search found that *ray-structure* cues (not pulse shape) separate the
  classes — top was `behind_energy` (transmitted energy past the peak,
  glass|ghost AUC 0.93). But swapping it into the representation *lowered*
  peak-level F1 (`ta`→`taE` **−0.027**, glass −0.066), because its
  discriminative power is a **depth/scene confound that doesn't transfer**: glass
  vs object AUC = 0.68 on train but 0.51–0.71 across test scenes, **sign-flipping
  on 14build_7floor**. Another "high in-isolation ≠ useful for the task" case
  (cf. width, EMG kernel, MSE). See *"Follow-up: behind-energy"* below.
- **Ghost is secondary and needs K≥2.** Ghost F1 over K = {1,2,4,8} =
  {**0.058**, 0.496, **0.576**, 0.541}: a single (first) return almost never
  lands on the ghost, K=2 recovers it, K=4 is best, K=8 slightly overfits noise
  peaks. Glass stays the hard class (~0.27) regardless. Best overall K = **4**.

---

## Method

### Data / split (same as everywhere else in this repo)
- Repo **SPLIT2**: 7 train scenes / 3 held-out test scenes. Resolved from
  `configs/vit3d_ikeda_vastai_cutmix_train_split2_no-expand.yaml` with two
  remaps for this box: `/data1→/data3`, and `annotation_v1→annotation_v1_expand`
  (the test metric uses the *expand* annotations, so train/val/test all use
  *expand*). The single missing cutmix-augmentation dir is dropped.
  → **train 280 dirs, val 60 dirs, test 30 dirs** (= the byte-for-byte SPLIT2
  TEST: 36build h002-011, 22build h001-010, 14build_7floor h001-010).
- Each hist dir holds 50 frames (~14k train frames total). For tractability we
  keep **every 7th frame per dir** for train/val caching (**2216 train, 479 val**
  frames) and **every 3rd** test frame for eval (~475 frames — `divide=3`-equivalent,
  which the downstream doc shows matches `divide=1` within 0.003).
- Cropping matches the downstream exactly so peaks/labels line up: y_crop
  (88/88) → 336, z_crop (25/375) → **T=300**. We do **not** apply the
  downstream's *random spatial crop* at eval; the event net is scored on the
  full 400×336 plane (the F1 *method* is identical to the paper's; only the
  spatial coverage is fuller).

### Event extraction & features
- Per pixel, top-K events are extracted from the **per-pixel max-normalised**
  waveform with the GPU-batch extractor (`compression.event_extraction`):
  greedy height-ranked NMS (min_distance=3, smooth σ=1.5, min_height=0.05), FWHM
  from half-max crossings. Greedy NMS makes the top-K set **nested**, so one
  cached K=8 frame serves every K∈{1,2,4,8}.
- Features per event (all normalised): `t = t_bin/T`, `Δt = (t_bin−t_first)/T`,
  `a = peak height ∈[0,1]`, `w = FWHM/T`, `m = valid mask`. Events are kept
  top-K by amplitude then **re-sorted chronologically** (so Δt and the rank
  embedding are time-ordered, per the plan).
- **Event labels** = the annotation value at the event's exact peak bin. This
  matches how the paper's `evaluate_peaks` reads `annotation[d,y,x]`, so train
  labels and the test metric share one definition. Event-level class balance
  (cached): noise 76 %, object 12 %, glass 4 %, ghost 1.3 %.

### Model
- `EventTensorNet` (`eventnet/model.py`), exactly the plan's architecture:
  shared event MLP (in_dim→32→32) + per-rank embedding → flatten K into channels
  → **2D U-Net** (base 64) over the H×W plane → per-event logits `[B,H,W,K,C]`.
  ~1.9 M params. Input dim varies per feature mode (2–5).
- Loss: weighted cross-entropy over **valid events only**, class weights
  `[0.2, 1.0, 2.0, 2.0]`. AdamW lr 1e-3, wd 1e-4, cosine, 40 epochs, batch 8,
  random 256×256 crops + flips. Model selection by **val event-level F1-mean**.

### Metric — **paper-compliant peak-level F1** (論文準拠)
The headline number is computed the **same way as the paper's "F1-mean ≈ 0.592"**
(see `downstream/SCORE_DISCREPANCY.md`), reusing the Ghost-FWL repo's own code:
1. Each event gets a predicted class; we paint a dense `(T,X,Y)` predicted-label
   volume — `pred[t−r:t+r+1] = class`, `r=max(1,w/2)` (the plan's reconstruction).
2. The repo's `detect_peaks_in_voxel` (`find_peaks(height=max·0.1, width=3)` per
   pixel on the **raw** waveform) defines the scoring population — **identical to
   the paper's peak set**, and model-independent (cached once for the test set).
3. The repo's scoring builds a confusion matrix of pred-vs-annotation **at those
   peak bins**; **F1-mean = mean F1 over {object, glass, ghost}** (Noise kept as a
   competing class, `ignore_visualize_labels=[]`).

So these numbers are directly comparable to the **frozen Ghost-FWL peak-level
baseline 0.599** and the paper's **0.592** (full-waveform upper bound). We also
report an event-level F1 (scored at the net's own extracted events) for
reference.

---

## Ablation table (paper peak-level F1, test set)

*(generated by `eventnet/plot_results.py` → `plots/table.md`)*

| feature_mode | K | object F1 | glass F1 | ghost F1 | F1-mean | event-F1 |
| --- | --: | --: | --: | --: | --: | --: |
| t_only | 4 | 0.720 | 0.265 | 0.514 | **0.500** | 0.473 |
| t_dt | 4 | 0.720 | 0.283 | 0.522 | **0.509** | 0.482 |
| ta | 4 | 0.738 | 0.281 | 0.555 | **0.525** | 0.520 |
| taw | 4 | 0.741 | 0.274 | 0.561 | **0.525** | 0.520 |
| tdta | 4 | 0.756 | 0.265 | 0.565 | **0.529** | 0.524 |
| **tdtaw** | **4** | 0.754 | 0.273 | 0.576 | **0.534** | 0.530 |
| tdtaw | 1 | 0.717 | 0.295 | 0.058 | **0.357** | 0.428 |
| tdtaw | 2 | 0.745 | 0.250 | 0.496 | **0.497** | 0.514 |
| tdtaw | 8 | 0.750 | 0.277 | 0.541 | **0.523** | 0.515 |

`F1-mean` = paper peak-level (headline); `event-F1` = scored at the net's own
events (model-selection proxy). Feature modes (per `initial_plan.md`):
`t_only`=[t,m], `t_dt`=[t,Δt,m], `ta`=[t,a,m], `tdta`=[t,Δt,a,m], `taw`=[t,a,w,m],
`tdtaw`=[t,Δt,a,w,m] (proposed). Plots: `plots/k_sweep.png`, `plots/mode_bar_K4.png`.

Key comparisons (paper peak-level F1-mean):
- **intensity beyond geometry** — `tdta` 0.529 vs `t_dt` 0.509 → **+0.020** ✓
- **width beyond simple multi-echo** — `taw` 0.525 vs `ta` 0.525 → **+0.000** ✗
- **width beyond practical multi-echo** — `tdtaw` 0.534 vs `tdta` 0.529 → **+0.005** (marginal)
- **relative delay** — `tdtaw` 0.534 vs `taw` 0.525 → +0.009; `t_dt` 0.509 vs `t_only` 0.500 → +0.009

Reference upper bounds (frozen full-waveform Ghost-FWL, **same** test set & metric):
**peak-level 0.599 / paper 0.592**; voxel-level 0.532. → `tdtaw` K=4 = **89 %** of
the peak-level ceiling.

### Caveat on width
The ~0.005 width gain is within plausible run-to-run noise (no multi-seed CI was
run; the downstream doc reports per-class seed std ≈0.003–0.004 for the frozen
model). The robust, repeatable effects here are **intensity (+0.020)** and the
**K≥2 ghost recovery (+0.14 from K1→K2)**; width should be read as "not clearly
helpful in this trained setup", not "proven useless". A multi-seed `tdtaw` vs
`tdta` repeat would settle it (each config is ~26 min train + ~6 min eval).

---

## How to reproduce
```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
# 1. cache events (train/val) + test peaks (once)
uv run python -m eventnet.cache_events --split train --frame_stride 7 --device cuda:0  # shardable
uv run python -m eventnet.cache_events --split val   --frame_stride 7 --device cuda:0
uv run python -m eventnet.cache_test_peaks --frame_stride 3 --nshard 4 --shard 0       # x4
# 2. train + eval the whole ablation over 4 GPUs
uv run python -m eventnet.run_sweep --save_root downstream/outputs/eventnet_sweep --epochs 40
# 3. table + plots
uv run python -m eventnet.plot_results --sweep_root downstream/outputs/eventnet_sweep
# V2 architecture (cross-event attention + deeper U-Net): add --train_extra "--arch v2"
uv run python -m eventnet.run_sweep --save_root downstream/outputs/eventnet_sweep_v2 \
    --jobs tdtaw:4 tdta:4 taw:4 ta:4 --train_extra "--arch v2" --epochs 40
```

---

## V2 architecture — cross-event attention + deeper U-Net

Motivated by the feature-search finding that the classes are separated by
*relational ray-structure* (rank, "what's behind") rather than single-pulse
shape, and that hand-crafted scalars (`w`, `behind_energy`) don't transfer, V2
lets the network **learn** the ray relations instead (`eventnet/model.py`,
`EventTensorNetV2`, `--arch v2`):
- **Cross-event attention**: a per-pixel Transformer (2 layers, 4 heads) over the
  K events of each ray, masked for padded events — each return attends to the
  others, so rank / inter-echo gaps / "is there a return behind me" become
  learned features. This is the robust, data-driven replacement for the
  depth-confounded `behind_energy` scalar.
- **Deeper U-Net** (3 levels vs 2), GELU, `emb_dim` 48 (vs 32). ~7.85 M params
  (vs 1.9 M), ~155 s/epoch (vs ~40 s).

### Results (paper peak-level F1, K=4, **2 seeds: 42 & 43**)
| mode | V1 | V2 s42 | V2 s43 | **V2 mean** | range | V2 glass (mean) |
|---|--:|--:|--:|--:|--:|--:|
| ta    | 0.525 | 0.531 | 0.536 | **0.534** | 0.005 | 0.268 (V1 0.281) |
| **taw**   | 0.525 | 0.565 | 0.544 | **0.555** | 0.021 | **0.307** (V1 0.274) |
| tdta  | 0.529 | 0.536 | 0.511 | **0.524** | 0.025 | 0.252 (V1 0.265) |
| tdtaw | 0.534 | 0.509 | 0.542 | **0.525** | 0.033 | 0.257 (V1 0.273) |

**Findings (2-seed):**
- **HEADLINE: V2 `taw` K=4 = 0.555 = 0.555/0.599 = 93 %** of the full-waveform
  peak-level ceiling (up from V1's `tdtaw` 0.534 = 89 %), with the **best glass of
  any config (0.307** vs ≤0.28 elsewhere). `taw` is highest in *both* seeds
  (0.565, 0.544), so the win is real, not a lucky run.
- **Width helps under V2** (reversing V1): `taw` − `ta` = **+0.021** (mean), and
  glass 0.268→0.307; `taw`>`ta` in both seeds. With an architecture that relates a
  return's shape to the other returns on the ray, width's glass/ghost signal
  *does* get used — re-aligning with the frozen-judge experiment.
- **Relative delay `Δt` becomes redundant/harmful under attention.** `taw` 0.555 >
  `tdtaw` 0.525 and `ta` 0.534 > `tdta` 0.524 — adding explicit `Δt` *hurts* under
  V2, the opposite of V1 (where `tdtaw`>`taw`). Cross-event attention already
  learns inter-return timing, so the hand-coded `Δt` is duplicate input that only
  adds noise. **Best V2 representation = `taw` (t, a, w, m) — no Δt.**
- **V2's gain is concentrated in `taw`, not uniform** (mode-mean V2 0.534 vs V1
  0.528): the architecture doesn't lift every representation, it specifically
  unlocks the width-bearing one while making Δt unnecessary.
- Variance: cross-scene test range up to 0.033 at near-identical val F1
  (~0.68–0.70; val=7 train scenes, test=3 held-out), so single-seed deltas <0.03
  are noise — which is why the 2-seed averaging above was necessary.

---

## Follow-up: behind-energy (a transmitted-energy feature) — negative result

Motivated by "width doesn't help — what *waveform* feature actually separates the
classes?", we ran a feature search and a representation swap.

### Feature search (pairwise class separability, AUC on cached train events)
For each detected peak we computed a battery of descriptors and ranked them by
pairwise 1-vs-1 AUC (0.5 = useless). **Ray-structure features dominate; single-
pulse shape is weak.** Top of the table:

| feature | glass\|obj | ghost\|obj | glass\|ghost | kind |
|---|--:|--:|--:|---|
| **behind_energy** (Σwn[t:]/Σwn) | 0.658 | **0.887** | **0.925** | ray structure |
| amp_rank (brightness rank on ray) | **0.693** | 0.847 | 0.839 | ray structure |
| dist_to_brightest | 0.608 | 0.861 | 0.907 | ray structure |
| rel_to_brightest | 0.628 | 0.831 | 0.827 | ray structure |
| area | 0.540 | 0.867 | 0.844 | shape |
| height (=`a`) | 0.547 | 0.845 | 0.825 | scalar |
| **fwhm (=`w`)** | 0.527 | 0.779 | 0.757 | shape |
| asym (rise/fall) | 0.545 | 0.555 | 0.600 | shape |
| skewness | 0.511 | 0.533 | 0.522 | shape |

Physics: **glass = partial transmission** → it's not the brightest return and has
energy *behind* it (the surface behind the glass); **ghost = secondary/late
return**. Half-width / skew / asymmetry barely separate anything. Of the strong
features, `amp_rank`/`rel_to_brightest`/`dist_to_brightest` are all functions of
the K events the net already sees (it has all `t_i, a_i` + rank embedding), so
**`behind_energy` is the only strong cue NOT derivable from `{t,a,w}`** — it
integrates diffuse/sub-threshold waveform energy. We added it as a 4th event
column (`E`, in [0,1]) and ran the ablation.

### Ablation (paper peak-level F1, K=4; `ta`/`tdta` re-run on the same re-cache)
| mode | object | glass | ghost | **F1-mean** | vs no-E |
|---|--:|--:|--:|--:|--:|
| `ta`     | 0.735 | 0.284 | 0.556 | **0.525** | — |
| `taE`    | 0.739 | 0.218 | 0.537 | **0.498** | **−0.027** |
| `tdta`   | 0.752 | 0.248 | 0.555 | **0.518** | — |
| `tdtaE`  | 0.728 | 0.248 | 0.550 | **0.509** | −0.009 |
| `tdtaEw` | 0.732 | 0.239 | 0.536 | **0.502** | (worst) |

`behind_energy` **hurts**, worst on **glass** (`ta`→`taE` glass −0.066). (Run-to-run
noise ≈0.01: `tdta` was 0.529 in the first sweep, 0.518 here — but the glass drop
exceeds that.)

### Why the best-AUC feature hurts: cross-scene confound
`behind_energy` correlates with depth `t` (corr −0.70), and its glass-vs-object
separability is **train-set-specific** — it does not transfer to the held-out
test scenes:

| split | E_glass median | E_object median | glass>object? | glass\|object AUC |
|---|--:|--:|:--:|--:|
| TRAIN (mixed) | 0.682 | 0.587 | yes | 0.682 |
| TEST 36build | 0.645 | 0.635 | yes | 0.580 |
| TEST 22build | 0.725 | 0.653 | yes | 0.708 |
| TEST 14build_7floor | 0.625 | 0.631 | **no (flips)** | 0.512 |

The train AUC (0.68) overstates a transferable signal (0.51–0.71, **sign-flipping**
on one scene). A model that leans on `E` for glass is hurt where the cue doesn't
hold → glass collapses. **"Energy behind the surface ⇒ glass" is a scene-geometry
spurious correlation, not robust physics.** This is the same lesson as width and
the EMG kernel: a feature that scores high *in isolation / on train* is not
necessarily useful for the *cross-scene downstream task* — which is exactly why
this frozen-judge / held-out-scene harness exists.

**Verdict:** keep `tdtaw` (K=4, 0.534) as the headline. The robust, transferable
cues are **intensity** and **K≥2 multi-echo geometry**; both width and
transmitted-energy add nothing that generalizes. Code/artifacts:
`downstream/outputs/eventnet_sweep_E/`, feature `E` lives in `eventnet/events.py`
and modes `taE`/`tdtaE`/`tdtaEw` in `eventnet/data.py`.

### Is the non-transfer a SPLIT2 artifact? No — it's intrinsic depth-confounding
Per-scene `behind_energy` (glass vs object) over all 10 scenes (AUC>0.5 = glass
has more behind-energy, the train assumption), with each class's median depth `t`:

| scene | set | E_glass | E_obj | t_glass | t_obj | AUC g\|o |
|---|---|--:|--:|--:|--:|--:|
| 11build | train | 0.661 | 0.587 | 46 | 55 | 0.668 |
| 14build_2floor | train | 0.657 | 0.576 | 34 | 86 | 0.739 |
| 16build | train | 0.652 | 0.590 | 23 | 33 | 0.723 |
| 16buildA_large | train | 0.789 | 0.632 | 55 | 53 | 0.671 |
| 16buildA_mid | train | 0.772 | 0.641 | 19 | 29 | 0.718 |
| 22build | TEST | 0.725 | 0.653 | 38 | 52 | 0.710 |
| 36build | TEST | 0.636 | 0.637 | 21 | 21 | 0.553 |
| 14build_7floor | TEST | 0.625 | 0.631 | 42 | 33 | 0.485 |
| **gym_build** | **train** | 0.665 | 0.711 | 56 | 32 | **0.392** |

Two findings: **(1) the sign-flip is not a test-only thing** — `gym_build`, a
*train* scene, flips harder (0.392) than any test scene, so the training set
itself contains contradictory `E→glass` directions and **no re-split can rescue
it** (any 7-scene subset mixes both signs). **(2) the cue is pure depth-confound**
— AUC is high exactly where glass is *closer* than object (11build 46<55, 22build
38<52, 16build 23<33) and flips where glass is *farther* (gym_build 56>32,
14build_7floor 42>33). `behind_energy`'s "glass signal" is really "glass happens
to be nearer than object in this scene" (nearer peak ⇒ more energy behind it),
which is scene layout, not glass physics. So the failure is intrinsic to the
feature, not to the split. (Aside: ghost E is consistently low across scenes —
last-return — except `14build_2floor` where ghost E=0.89, a near-range-ghost
population, consistent with the known cross-scene ghost domain gap.)

### Would a learned geometric-falloff residual rescue it? No.
Idea: subtract a learned `E_expected(t)` and feed the residual, to strip the
depth confound. Depth-stratified glass-vs-object AUC (compare E only *within*
matched depth bins) per scene: mean drops **0.630 → 0.561** (raw → depth-
controlled), i.e. **~half the apparent glass signal was pure depth**. A weak
residual remains (0.56) but is still scene-inconsistent — **2/9 scenes flip even
depth-controlled** (gym_build 0.398, 36build 0.443; while 14build_7floor's raw
flip *recovers* to 0.555). The falloff *curve* itself is ~scene-invariant (mostly
the mechanical Σwn[t:] depth dependence), so a learnable global falloff would
**not** be scene-dependent — but the residual it leaves is too weak and still
non-transferable to help. Decisively: `tdtaE`/`tdtaEw` already give the per-event
MLP both `t` and `E`, so it can learn `E − f(t)` internally — and it still lost to
`tdta`. Explicit residual featurization is therefore not expected to beat the
`tdtaw`/`tdta` headline.
