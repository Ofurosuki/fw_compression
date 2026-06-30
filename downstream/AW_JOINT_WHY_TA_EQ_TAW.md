# Why does multi-echo `ta` ≈ full-waveform `taw`? — per-class (a, w) joint distribution

Distribution-level mechanism behind `RETRAIN_RESULTS.md`'s finding that full-waveform
**width `w` has net downstream value ≈ 0** (helps ghost, hurts glass, net flat). Answers the
question "is `ta` close to `taw` because width is redundant given amplitude?"

- **Script:** `eventnet/diag_aw_joint.py`
- **Figure:** `downstream/outputs/diag/aw_joint.png`
- **Date:** 2026-06-23
- **Sample:** top-K events over all 10 scenes (2 dirs × 4 frames each) → **12.97 M valid
  signal events** (object 9.13 M, glass 1.66 M, ghost 2.18 M). `a` = per-ray max-normalised
  peak height ∈ [0,1]; `w` = FWHM in bins.

---

## Hypothesis tested

> "`w` is a class-specific function of `a` (a ∝ w), and glass differs" — so a model with
> `(t, a)` could already reconstruct `w`, making width redundant ⇒ `ta ≈ taw`.

**Refuted in that literal form, and replaced by a cleaner mechanism (below).**

---

## Results

### (A) Per-class marginals — `[median (IQR), n]`

| set | class | a (max-norm height) | w (FWHM bins) | n |
|---|---|---|---|--:|
| train | object | 1.0 [0.9–1.0] | 10 [9–11] | 6.12 M |
| train | glass  | 1.0 [0.6–1.0] | 10 [9–11] | 1.27 M |
| train | ghost  | **0.8** [0.3–1.0] | 9 [8–11] | 1.60 M |
| TEST | object | 1.0 [0.9–1.0] | 10 [9–11] | 3.02 M |
| TEST | glass  | 1.0 [0.8–1.0] | 9 [8–11] | 0.38 M |
| TEST | ghost  | **0.3** [0.2–0.6] | 8 [7–9] | 0.58 M |

Amplitude saturation (a ≥ 0.98): **object 65 %, glass 56 %, ghost 33 %**.

### (B) Within-class `a–w` coupling — the hypothesis test

| class | Spearman ρ(a,w) | R²(w ~ a, decile-binned) |
|---|--:|--:|
| object | 0.296 | **0.006** |
| glass  | 0.500 | **0.081** |
| ghost  | 0.466 | **0.085** |

**`a` and `w` are essentially uncoupled within class (R² ≈ 0).** Width is *not* derivable
from amplitude → the literal "a ∝ w" mechanism is **refuted**.

### (C) `E[w | a-bin]` per class — the real mechanism

| class | a∈[0,0.2) | [0.4,0.6) | [0.8,0.95) | [0.95,1.0) |
|---|--:|--:|--:|--:|
| object | 8 | 9 | 10 | 10 |
| glass  | 8 | 8 | 9 | 11 |
| ghost  | 8 | 8 | 9 | 10 |

At **matched amplitude the three classes have near-identical width**. Width's only
systematic structure is a **class-independent "taller peak → wider" SNR/shape trend**. So
once the model has `a`, `w` adds almost no class-discriminative information ⇒ **`ta ≈ taw`**.

### (D) Class-pair separability — AUC from `a` / `w` / `(a,w)` logistic

- **object vs glass:** AUC(a) ≈ 0.5 in every scene — per-event amplitude **cannot split
  object from glass** (both primary returns pile at a ≈ 1.0). Width's `(a,w)` lift is small
  and **sign-flips across scenes** (`w`-AUC: 16buildA_large 0.38, gym 0.48 ⇄ 14build_7floor
  0.69) → glass width is **non-transferable** ⇒ "width hurts glass."
- **object/glass vs ghost:** `a` alone separates well (AUC 0.76–0.94) in dim-ghost scenes;
  the `w` lift there is ≈ +0.00–0.02. But where amplitude fails it is **+0.09–0.16**
  (16build; TEST 14build_7floor +0.15). **Width backstops ghost only when `a` can't flag
  it** — the transferable "ghost is narrower" cue.

---

## Takeaway

`ta ≈ taw` is explained by **(C) + amplitude saturation**, not by redundancy of the form
"w derivable from a":

1. At matched amplitude, width carries **~no class signal** (E[w|a] overlaps across classes).
2. Amplitude itself **can't even split object/glass** (saturates at 1.0); that separation
   comes from multi-echo geometry / spatial context — consistent with glass being a
   *representation* problem, not a per-event-feature problem.
3. Width's **only** real contribution is rescuing **bright ghosts** where amplitude fails —
   transferable but small and scene-gated.
4. That residual ghost value is **cancelled** by width's **non-transferable glass cue**
   (sign-flips across scenes) ⇒ **net ≈ 0**.

This grounds the retrain conclusion ("width redistributes ghost↔glass, doesn't raise net F1")
at the data-distribution level.

**New nuance:** ghost amplitude is **scene-dependent** (train median a = 0.8 vs TEST a = 0.3,
the known ghost-brightness domain gap). So "ghost = dim (a ≈ 0.26)" is a **TEST-only** fact;
amplitude's ghost-flagging power varies by scene, and width is its backstop.

---

## Reproduce

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
uv run python eventnet/diag_aw_joint.py --device cuda:3 --dirs_per_scene 2 --frames_per_dir 4
```

Related: `downstream/RETRAIN_RESULTS.md` (architecture-controlled width margin ≈ 0),
`eventnet/diag_width_per_scene.py` (per-scene width sign-flip), `FW_Event_Net/SCENE_GEOMETRY.md`.
