# Per-scene geometry of object / glass / ghost (Ghost-FWL dataset)

Consolidated record of the **per-scene geometric/transport analyses** run during the
FW_Event_Net investigation. It explains *why* glass is the stress class and why every
per-ray "what's behind" cue (behind_energy, decomposition, transmittance, radiometric
correction) failed to transfer across scenes: **the depth ordering of the classes is
itself scene-dependent**, so cues derived from it sign-flip on the held-out scenes.

> TL;DR — Glass sits **nearer than object in most scenes but FARTHER in `gym_build`
> (train) and `14build_7floor` (test)**. Because "transmitted/behind energy" mechanically
> tracks depth, its glass-vs-object discriminative *direction flips* exactly on those
> scenes. The flip is in a **train** scene too, so it's intrinsic to the feature, not a
> train/test split artifact. See FW_Event_Net/RESULTS.md for the downstream consequences.

---

## 1. Dataset, split, and what was measured

- **10 scenes** (buildings). Repo **SPLIT2**: 3 held-out **TEST** scenes
  (`36build`, `22build`, `14build_7floor`); the rest **TRAIN**
  (`11build`, `14build_2floor`, `16build`, `16buildA_large`, `16buildA_mid`,
  `gym_build`, `34build`).
- All per-scene tables below were computed by extracting top-K events
  (`eventnet.events.extract_frame_events`) on a **sample per scene = first 3 `hist`
  dirs × first 4 frames**, after the downstream y/z crop (T=300). Labels = annotation at
  each event's exact peak bin. `34build` is absent from the tables (its sampled frames had
  <50 glass events under the threshold) — a sampling gap, not a finding.
- **Signed AUC convention:** for a feature `f`, AUC(glass vs object) = P(f_glass >
  f_object). **>0.5 = glass has the higher value; <0.5 = sign-flip** (object higher).
  0.5 = no separation.

---

## 2. Class balance

- **Event-level** (cached top-K events, all valid events): noise **76 %**, object **12 %**,
  glass **4 %**, ghost **1.3 %**. Glass+ghost together are ~5 % of events → minority,
  high-variance classes.
- **Voxel-level** (one representative `36build` frame, raw (400,512,700)): background
  ~140.5 M, object 2.30 M, glass 0.17 M, ghost 0.36 M voxels. Glass is the rarest signal.

---

## 3. Global per-class feature medians (mixed-scene sample)

Where each class's returns sit and what their pulses look like (medians over valid events):

| feature | object | glass | ghost | reading |
|---|--:|--:|--:|---|
| amplitude `a` (peak height, max-norm) | 1.00 | 0.93 | 0.26 | ghost = dim/secondary return |
| width `w` (FWHM, bins) | 10 | 10 | 8 | nearly identical → width barely separates |
| `behind_energy` (Σwn[t:]/Σ) | 0.57 | 0.77 | 0.09 | glass has energy behind; ghost is last |
| `D_after` (direct mass behind) | 0.50 | **0.82** | 0.07 | glass = real object behind it |
| `I_after` (indirect/diffuse behind) | 0.04 | 0.07 | 0.01 | small (ghosts here are discrete, not diffuse) |
| transmittance `Tp(δ)` δ=0/8/16/32/64 | .49/.08/.04/.02/.004 | **.73/.56/.25/.10/.03** | .07/.01/.005/0/0 | glass = slow decay (partial transmission) |

In isolation these look strongly discriminative for glass (esp. `D_after`, the `Tp` decay
shape). The per-scene tables below show why that separation **does not generalize**.

---

## 4. Per-scene: behind-energy and depth (the core geometry table)

`E_*` = median behind_energy per class; `t_*` = median peak depth bin (0–299, larger =
farther). `AUC g|o` = glass-vs-object behind_energy (>0.5 = glass higher).

| scene | set | E_glass | E_obj | E_ghost | t_glass | t_obj | glass vs obj depth | AUC g\|o | AUC gh\|o |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| 11build | train | 0.661 | 0.587 | 0.093 | 46 | 55 | glass nearer | 0.668 | 0.109 |
| 14build_2floor | train | 0.657 | 0.576 | **0.891** | 34 | 86 | glass nearer (big) | 0.739 | 0.814 |
| 16build | train | 0.652 | 0.590 | 0.142 | 23 | 33 | glass nearer | 0.723 | 0.105 |
| 16buildA_large | train | 0.789 | 0.632 | 0.380 | 55 | 53 | ~equal | 0.671 | 0.212 |
| 16buildA_mid | train | 0.772 | 0.641 | 0.493 | 19 | 29 | glass nearer | 0.718 | 0.303 |
| gym_build | train | 0.665 | 0.711 | 0.138 | 56 | 32 | **glass FARTHER** | **0.392** | 0.127 |
| 22build | TEST | 0.725 | 0.653 | 0.094 | 38 | 52 | glass nearer | 0.710 | 0.022 |
| 36build | TEST | 0.636 | 0.637 | 0.128 | 21 | 21 | ~equal | 0.553 | 0.040 |
| 14build_7floor | TEST | 0.625 | 0.631 | 0.423 | 42 | 33 | **glass FARTHER** | 0.485 | 0.191 |

Summary: behind_energy glass\|object AUC **mean 0.63, range [0.39, 0.74]**, **flips (<0.5)
in 2/9 scenes** — `gym_build` (train) and `14build_7floor` (test).

**Geometric reading:** AUC>0.5 (glass higher behind-energy) holds **exactly when glass is
nearer than object** (more waveform sits behind a nearer peak). Where glass is farther
(`gym_build`, `14build_7floor`) the cue **reverses**. So "glass has energy behind it" is a
proxy for "glass happens to be nearer than object in this scene" — scene layout, not glass
physics.

### Ghost depth note
Ghost behind_energy is **low in almost every scene** (0.09–0.49) → ghosts are late/last
returns with little behind them. The exception is **`14build_2floor` (E_ghost 0.891,
AUC gh|o 0.814)** — a near-range-ghost population (ghosts early on the ray with lots
behind), an outlier consistent with the known cross-scene ghost-brightness/geometry domain
gap (see memory `fwc-real-data-domain-gap`).

---

## 5. Per-scene: depth-stratified behind-energy (is the cue *just* depth?)

`raw` = behind_energy glass\|object AUC; `depth-strat` = same AUC computed **within matched
depth bins** (controls for depth). If the cue were pure geometry, stratifying would push it
to 0.5.

| scene | set | raw AUC | depth-strat AUC |
|---|---|--:|--:|
| 11build | train | 0.667 | 0.543 |
| 14build_2floor | train | 0.736 | 0.574 |
| 16build | train | 0.724 | 0.603 |
| 16buildA_large | train | 0.674 | 0.661 |
| 16buildA_mid | train | 0.722 | 0.606 |
| gym_build | train | 0.394 | 0.398 |
| 22build | TEST | 0.709 | 0.669 |
| 36build | TEST | 0.553 | 0.443 |
| 14build_7floor | TEST | 0.488 | 0.555 |
| **mean** | | **0.63** | **0.56** |

**~Half the apparent glass signal is pure depth** (mean 0.63 → 0.56). A weak residual
remains (0.56) but is **still scene-inconsistent** (flips in 2/9: gym_build 0.398,
36build 0.443; 14build_7floor's raw flip *recovers* to 0.555). → a learned geometric-falloff
residual can't rescue it.

---

## 6. Per-scene: range-corrected amplitude (radiometric)

Test of whether converting amplitude to a material backscatter proxy `ρ = a·R(t)²` (range
correction; R∝t+c0) makes the glass cue scene-consistent. Physics hypothesis: glass
(transmissive) → consistently *lower* material backscatter (AUC <0.5 every scene).
`a` = raw amplitude glass\|object AUC; `ρ` at three range zero-points c0; last col =
depth-stratified ρ.

| scene | set | raw `a` | ρ(c0=25) | ρ(c0=125) | ρ(c0=325) | ρ depth-strat |
|---|---|--:|--:|--:|--:|--:|
| 11build | train | 0.473 | 0.370 | 0.379 | 0.393 | 0.479 |
| 14build_2floor | train | 0.565 | 0.221 | 0.244 | 0.263 | 0.511 |
| 16build | train | 0.640 | 0.290 | 0.308 | 0.319 | 0.580 |
| 16buildA_large | train | 0.569 | 0.534 | 0.529 | 0.542 | 0.470 |
| 16buildA_mid | train | 0.564 | 0.246 | 0.266 | 0.324 | 0.390 |
| gym_build | train | 0.676 | 0.672 | 0.684 | 0.687 | 0.728 |
| 22build | TEST | 0.549 | 0.283 | 0.293 | 0.302 | 0.349 |
| 36build | TEST | 0.664 | 0.505 | 0.524 | 0.523 | 0.744 |
| 14build_7floor | TEST | 0.639 | 0.687 | 0.659 | 0.640 | 0.543 |

Range correction does **not** make the cue consistent: ρ mean ~0.43 but **range [0.22, 0.69],
sign still flips (5/9 <0.5, 4/9 >0.5)**, robust to c0; spread *widens* vs raw. The flip
scenes (`gym_build`, `14build_7floor`) are again the glass-farther ones — ×R² over-boosts far
glass and *amplifies* the flip. Confound is relative near/far **geometry**, not radiometric
range alone.

---

## 7. Per-scene: transmittance decay profile (NeRF-style)

`Tp0` = transmittance survival at the peak (≈ behind_energy); `Tp8` = at peak+8;
`ratio8/0` = Tp8/Tp0 (the "partial-transmission plateau" indicator). All glass\|object AUC.

| scene | set | Tp0 | Tp8 | ratio8/0 |
|---|---|--:|--:|--:|
| 11build | train | 0.666 | 0.618 | 0.601 |
| 14build_2floor | train | 0.734 | 0.663 | 0.632 |
| 16build | train | 0.731 | 0.696 | 0.681 |
| 16buildA_large | train | 0.683 | 0.654 | 0.632 |
| 16buildA_mid | train | 0.711 | 0.657 | 0.594 |
| gym_build | train | 0.385 | 0.337 | 0.313 |
| 22build | TEST | 0.706 | 0.684 | 0.656 |
| 36build | TEST | 0.529 | 0.430 | 0.397 |
| 14build_7floor | TEST | 0.448 | 0.461 | 0.478 |
| **mean** | | **0.62** | **0.58** | **0.55** |

The decay shape's leading indicators flip on the same scenes (gym_build, 14build_7floor,
and 36build for the ratio) → the transmittance profile is no more transferable than
behind_energy.

---

## 8. Cross-cutting geometric conclusions

1. **Depth ordering of classes is scene-specific.** Glass nearer than object in 6–7/9
   scenes, farther in `gym_build` and `14build_7floor`. This single geometric fact drives
   the sign-flips of *every* "behind/transmittance/range" cue.
2. **The glass cue's failure is geometric, not radiometric.** Depth-stratification removes
   ~half the signal; range correction can't fix it (flip persists/worsens); the residual is
   scene-inconsistent.
3. **Ghost is geometrically simple but has one outlier scene.** Ghost = low behind-energy
   (late return) everywhere except `14build_2floor` (near-range ghosts), echoing the
   cross-scene ghost domain gap.
4. **The flip occurs in a TRAIN scene (`gym_build`)** → the training set itself contains
   contradictory glass↔geometry relationships, so no re-split and no domain-generalization
   loss (V-REx ≈ ERM) can manufacture an invariant glass cue. It is a **representation**
   limitation: per-ray scalar/shape summaries of "what's behind" are not class-invariant
   across scene geometry.

---

## 9. Provenance & caveats

- Probes (logic preserved in transcript; outputs transcribed here): per-scene behind-energy
  & depth; depth-stratified AUC; radiometric `ρ=a·R²`; transmittance profile `Tp(δ)`. Global
  per-class medians from the `eventnet.events` 12-col cache unit tests.
- Sample = first 3 hist dirs × first 4 frames per scene (not the full test set); absolute
  AUC values carry ~±0.02 sampling noise, but the **sign/flip pattern is the robust result**
  and reproduces across the four independent probes.
- `34build` (train) not represented (sampling threshold). Depths are post-crop bins (0–299).
- Downstream impact of these geometry facts (which features/architectures were tried and why
  they failed) is in **FW_Event_Net/RESULTS.md**; related memory:
  `fwc-eventnet`, `fwc-behind-energy-litreview`, `fwc-real-data-domain-gap`.
