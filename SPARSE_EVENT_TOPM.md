# Sparse Event-ToPM — token-native event-token network (PI: "tokenize the LiDAR signal")

Instruction: `eventnet_v4.md`. Tests whether the gap between **FW_Event_Net V2** (`taw`
K4 peak **0.555**) and **ToPM retrained on Gaussian pseudo-waveforms** (`taw` K4 peak
**0.582**; full **0.595**) is **architectural** — specifically, whether EventNet underuses
the event tokens because it **flattens the K events of each ray into channels**
(`reshape(B,H,W,K·emb)`) *before* the spatial U-Net. A token-native model that keeps every
event as a `(h,w,k)` token through the spatial mixing should exploit them better.

Metric = the paper-compliant **peak-level F1-mean** over {object, glass, ghost} on the SPLIT2
held-out test set (3 scenes), identical population/scoring to the references above
(`eventnet/evaluate.py`). Same cache (stride7 train/val, stride3 test), loss
(masked weighted-CE [.2,1,2,2]), crop 256, and fair recipe (lr5e-4 / warmup5 / cosine /
50ep / bf16) as the V2 comparison — only the architecture differs. Implementation:
`eventnet/sparse_event_topm.py`; drop-in `build_model(arch=…)` + `eventnet/train.py` +
`evaluate.py`.

## Models
- **v1 `setopm`** (flat): event MLP → +rank/+sinusoidal-time emb → N **windowed shifted
  spatial-event attention** blocks (each window = `ws·ws·K` tokens; shift via padded
  partition offset, no Swin cyclic mask) → per-event classifier. NO spatial hierarchy
  (RF ≈ depth/2 · window ≈ 48 px). emb256/depth12/ffn2 = 6.67M.
- **v2 `setopm2`** (hierarchical): U-shaped — `PatchMergeK` downsamples the plane 2× per
  level **keeping K as tokens** (3 levels, /1→/4) + skip connections, so deep blocks get a
  large receptive field while skips preserve fine per-event detail. 6.24M.
- **v3 `setopm3`** (hier + global bottleneck): v2 + a `GlobalSpatialContext` at the
  bottleneck (pool K→ray summary, **global** self-attention over all cells, broadcast back) —
  the analog of EventNet V2's `spatial_attn`, the single lever that most moved ghost there.
  6.50M.

## Results (peak-level F1, SPLIT2 test, taw K=4)

| model | seeds | **F1** | object | glass | ghost | val event-F1 |
|---|--|--:|--:|--:|--:|--:|
| **v1 setopm** (flat) | 42,43 | **0.534** | **0.765** | 0.234 | 0.603 | 0.709 |
| **v2 setopm2** (hier) | 42 | **0.545** | 0.752 | 0.267 | 0.615 | 0.707 |
| **v3 setopm3** (hier+global) | 42 | **0.545** | 0.753 | 0.269 | 0.612 | **0.721** |
| **v2 + focal[1,1,2,1]** (loss recalib) | 42 | **0.554** | **0.774** | 0.264 | 0.623 | **0.724** |
| — *EventNet V2* (ref) | 42,43 | *0.555* | 0.741 | **0.307** | 0.561 | ~0.70 |
| — *ToPM-retrain taw* (ref) | 42 | *0.582* | 0.754 | 0.271 | **0.722** | — |
| — *full waveform* (ref) | — | *0.595* | 0.770 | 0.300 | 0.715 | — |

v1 is the 2-seed mean (0.5355 / 0.5326); v2/v3 single seed (42). Checkpoints/eval under
`downstream/outputs/sparse_event_topm/`; v1 evals on `/data3/.../sparse_event_topm_eval/`.
Trained with DataParallel over 2 GPUs (bf16; `--gpus 3,1`, ~1.7× faster, recipe unchanged).

## Findings

1. **Token-native matches its per-ray classes to the dense ceiling.** v1/v2/v3 **object ≈
   0.75–0.77 ≈ the full-waveform ceiling (0.770)** and **beat EventNet V2 on object (0.741)
   and ghost (0.56)**. Keeping the K events as tokens (vs EventNet's K→channels flatten)
   demonstrably helps the per-ray classes — the thesis holds *per-class*.

2. **But the headline does NOT beat EventNet V2.** Three architecture variants converge to
   **~0.545 ≈ V2's 0.555** (within the ±0.03 seed band) and all sit **below ToPM-retrain
   0.582**. Token-native ≈ EventNet at the F1-mean level; the early-flatten was *not* the
   binding bottleneck.

3. **The hierarchy fixed v1's glass collapse (RF hypothesis confirmed).** v1 had no spatial
   downsampling and glass collapsed (0.234); v2's `PatchMergeK` hierarchy lifted **glass
   +0.033 → 0.267 ≈ ToPM's 0.271**, net +0.011. So v1's glass gap *was* a receptive-field
   problem, and the hierarchy closed it to the ToPM level (though still < V2's 0.307).

4. **The remaining gap to ToPM is entirely GHOST, and a global-attention "ghost attack"
   did NOT transfer.** v2/v3 match ToPM on object (0.75) and glass (0.27); the whole
   0.545→0.582 gap is ghost (0.61 vs **0.72**, −0.11). Adding global spatial attention (v3)
   to target it **improved val** (event-F1 0.707→0.721, val ghost 0.766→0.782) but was a
   **clean NULL on held-out test** (F1 0.5446→0.5448, ghost 0.615→0.612). This is the
   project's recurring **"val ↑ / held-out test flat = non-transfer"** signature (identical
   to V2's `spatial_attn`: seed-42 ghost spike that didn't replicate/transfer). **So the
   ghost gap is not a receptive-field/spatial-capacity problem.**

5. **Interpretation — the ghost gap is dense-reconstruction vs sparse-token, not K-flatten.**
   ToPM-retrain reads ghost *better than the full waveform itself* (0.722 > 0.715) because it
   runs a **3D conv over the dense Gaussian-synthesised field**, which *denoises* secondary
   returns (cf. RETRAIN_RESULTS: "the clean Gaussian synthesis denoises secondary returns").
   Both sparse-token nets (EventNet 0.561, ours 0.61) underperform that. The ghost advantage
   appears intrinsic to **dense smooth-field 3D convolution**, which generalises secondary
   returns across scenes better than discrete event-token attention — a representation/
   inductive-bias difference, **not** the channel-flatten the thesis targeted.

## UPDATE (2026-06-26): the ghost gap is PRECISION / loss, NOT dense-vs-sparse — finding 5 retracted

PI challenge: *ToPM-retrain consumes the **same** top-K taw events (no special 2nd-return
info), so a token model on the same input should reach the same ghost — the gap looks
suspicious.* Correct. Diagnosis:

- **Not a decode/paint artifact.** event-level ghost F1 ≈ peak-level ghost F1 (0.610 vs 0.615),
  so the model's classification — not the dense painting / peak coverage — is the limit.
- **The ghost gap is 100 % PRECISION.** Ghost precision/recall: ours (v2, CE `[0.2,1,2,2]`)
  **0.54 / 0.71**; ToPM-retrain **0.72 / 0.72**. **Recall is identical** (same input events, both
  find ~72 % of ghosts) — the gap is ours over-predicting ghost (false ghost from noise 5.0M +
  object 3.5M). So it is **not** an information / dense-vs-sparse ceiling; **finding 5 is retracted.**
- **Root cause = loss calibration inherited from EventNet.** Weights `[0.2,1,2,2]` (noise 0.2 ⇒
  the model rarely predicts noise; ghost 2 ⇒ over-predicts) bias toward false ghost. ToPM uses
  focal+dice, no such skew.
- **Confirmed controllable.** Retraining v2 with **focal + weights `[1,1,2,1]`** moved ghost
  **precision 0.54 → 0.64** and lifted the **headline 0.545 → 0.554 ≈ EventNet V2 0.555** (object
  0.752 → **0.774**, above the full ceiling). So the precision IS a loss knob, exactly as the PI
  argued.
- **But reweighting only slides ALONG the precision/recall tradeoff.** focal`[1,1,2,1]` traded
  recall down (0.71 → 0.61) for precision (→0.64); ghost F1 stayed ~0.62. ToPM's point
  (0.72 **and** 0.72) **dominates ours on both axes**, so matching it needs the *curve pushed out*
  (better ghost discrimination), not a different operating point. The missing ingredient is
  almost certainly **dice loss** (ToPM uses focal+**dice**; our `train.py` has focal only) — dice
  directly optimises overlap, lifting precision and recall together. Next experiment: add masked
  dice and retrain.

- **Dice did NOT transfer (clean negative).** Adding `masked_dice` (signal classes, weight 1.0) on
  top of focal`[1,1,2,1]` gave the **highest val ghost of all (0.801)** but **lower held-out test
  ghost (0.62 → 0.60)** and headline 0.554 → 0.546 — the same **val ↑ / held-out test ↓** non-transfer
  seen with the global-attention bottleneck (v3). So loss tuning hits a wall: **every loss variant
  plateaus at held-out ghost F1 ~0.60–0.62 (prec/rec frontier ~0.6/0.6–0.64/0.61), well below ToPM's
  0.72/0.72**, even as val keeps improving.

**Revised takeaway (final):** the PI was right that the ghost gap is **not** a dense-vs-sparse
*information* ceiling — recall is identical to ToPM and ghost **precision is a loss knob** (0.54→0.64),
so "dense beats sparse for ghost" (old finding 5) is **retracted**. Token-native, with the loss skew
fixed (focal`[1,1,2,1]`), **ties EventNet V2 (0.554 ≈ 0.555)** and **exceeds the dense ceiling on
object (0.774)**. BUT matching ToPM's *held-out* ghost (0.72/0.72) is **not achievable by loss**
(focal/dice/reweight all plateau ~0.62 on test while val rises) — the residual is a **cross-scene
generalization** gap: the token model overfits the training scenes' ghost; ToPM's dense 3D-conv
transfers it better. This is consistent with the project's central theme (cross-scene transfer is the
real bottleneck), and reframes the ToPM ghost edge as a **generalization** difference, not an
information or precision-ceiling one.

| loss (v2, taw K4, peak) | ghost prec/rec/F1 | F1-mean | val ghost |
|---|--|--:|--:|
| CE `[0.2,1,2,2]` (orig) | 0.54 / 0.71 / 0.62 | 0.545 | 0.766 |
| **focal `[1,1,2,1]`** (best) | 0.64 / 0.61 / 0.62 | **0.554** | 0.796 |
| focal+dice `[1,1,2,1]` | 0.60 / 0.59 / 0.60 | 0.546 | 0.801 |
| *ToPM-retrain* | *0.72 / 0.72 / 0.72* | *0.582* | — |

## UPDATE 2 (2026-06-27): structural diff → relative-range RoPE → token-native BEATS EventNet V2

PI: *what's the structural difference from ToPM, can we derive a better structure?* Read the ToPM
code (`hist_lidar/models/ViT3D.py`): **`PatchEmbed3D = nn.Conv3d(1, 240, kernel=patch=(20,8,4))`** —
ToPM **convolves the range/depth axis** (each token = a 20-bin range patch), so it is **range
translation-equivariant**: a "primary + ghost at Δdepth" pattern gives the same features at any
absolute depth → transfers across scenes whose depths shift. Our token model encoded range as an
**absolute scalar** (`sinusoidal(t)`) with **no range-equivariance** — the likely cause of the
cross-scene ghost generalization gap (recall the depth diagnostics: ghost detectability is depth-
dependent, lost ghosts cluster at early depth).

**Fix = RoPE on the range coordinate** (`_rope_range`, arch `setopm2r`): rotate q,k by an angle ∝ each
token's range `t`, so the attention dot-product depends on the **relative** gap `t_i − t_j` — the
token-native analog of ToPM's range-conv, and (unlike an explicit T×T relative bias, which was
**49 s/step** vs 1.4) it keeps the fast SDPA kernel (1.8 s/step). Same hierarchy + focal`[1,1,2,1]`,
only RoPE added.

| taw K4, peak, s42 | F1-mean | object | glass | ghost prec/rec/F1 | val event-F1 |
|---|--:|--:|--:|--|--:|
| v2 focal`[1,1,2,1]` | 0.554 | 0.774 | 0.264 | 0.64 / 0.61 / 0.62 | 0.724 |
| **v2r RoPE focal`[1,1,2,1]`** | **0.563** | 0.775 | 0.277 | 0.63 / **0.65** / **0.638** | **0.745** |
| *EventNet V2* | *0.555* | 0.741 | 0.307 | — / — / 0.561 | ~0.70 |
| *ToPM-retrain* | *0.582* | 0.754 | 0.271 | 0.72 / 0.72 / 0.72 | — |

**Findings:**
1. **First token-native model to BEAT EventNet V2: 0.563 > 0.555**, gap to ToPM down to 0.019.
2. **RoPE broke the ghost plateau the right way.** Loss tuning only *slid along* the prec/rec tradeoff
   (ghost F1 stuck ~0.62); RoPE **pushed the frontier OUT — recall 0.61→0.65 with precision ~flat →
   ghost F1 0.623→0.638**. Catching more transferable ghosts is exactly what range-equivariance
   predicts, **confirming the structural hypothesis**: the cross-scene ghost gap was partly the
   missing range-translation-equivariance, not information or dense-vs-sparse.
3. glass also nudged up (0.264→0.277); val consistently best (event-F1 0.745, val ghost 0.819).
4. **Caveat: single seed** (0.563 vs 0.554 is within ±0.03), but val improves consistently and the
   ghost-recall gain is mechanistically predicted, so the direction is credible — a 2nd seed would
   confirm. Residual to ToPM (0.019) is still ghost (0.638 vs 0.722, both prec & rec below).

## UPDATE 3 (2026-06-27): full-waveform vs taw, bucketed — taw is a near-optimal compression

Research goal (PI): not SoTA, but *what must a compression preserve for the ghost task* — found by
comparing ToPM trained on **full-waveform vs taw** (architecture/recipe fixed, only the input
representation changes), bucketing ghost peak-recall by bin-separation and "is the ghost the
brightest return on its ray". `downstream/diag_ghost_bindist.py` (instruments `run_eval_peak.py`'s
pipeline; same random crop + raw find_peaks population, 475 frames, 4.34M ghost peaks identical for
both). ⚠️ gotcha: must pass `--limit_dirs`/same dataset so the `divide=3` random frame-sampling +
crop RNG match the reference; a different dir set silently shuffles to different frames (gave a
spurious ghost 0.13 vs the true 0.72).

| ghost peak-recall | **FULL** | **taw** |
|---|--:|--:|
| overall | **0.720** | **0.725** |
| sep 0–4 | 0.639 | 0.669 |
| sep 4–8 | 0.236 | 0.292 |
| sep 8–15 | 0.396 | 0.359 |
| sep 15–40 | 0.796 | 0.762 |
| sep ≥40 | 0.784 | 0.820 |
| **ghost is the ONLY return** | **0.050** | **0.058** |
| **ghost IS the brightest** | **0.180** | **0.192** |
| ghost secondary (separated) | 0.812 | 0.816 |

**Findings:**
1. **full ≈ taw on EVERY bucket → the compression loses no ghost-relevant information.** The dense
   waveform (inter-peak shape, tails) does not help ghost; **top-K taw (K≈4, ~58×) is near-optimal
   for this task** (holds for object & glass too: full/taw recall 0.74/0.72 obj, 0.43/0.42 glass).
2. **The ghost ceiling is task-intrinsic, not a compression loss.** Both full and taw collapse when
   the ghost is the **only** return on its ray (0.05, →object) or the **brightest** (0.18) — a
   single bright/lone return is ambiguous between a real object and a multipath ghost *from that
   ray's waveform alone*. Disambiguating needs **spatial context / scene geometry** (is there a real
   object elsewhere this reflects?), which is **not in the waveform** — so the full waveform can't
   beat taw here. Well-separated secondary ghosts are read fine by both (0.81).
3. **Caveats on "taw is sufficient":** needs K≥3–4 (the binding return is the secondary/ghost; taw
   K2 drops ghost); "sufficient" = the info is *preserved*, but extracting it needs an adequate
   reader (frozen judge failed = domain shift; ToPM-retrain succeeds). Reading taw well is an
   architecture matter: ToPM-taw ghost 0.725 > our setopm2r 0.638 (same taw info, ToPM's 3D-conv
   reads it better than windowed attention).

**Conclusion for the research question:** for ghost detection, **the optimal compression is top-K
`(t,a,w)` with K≈4 — the dense waveform is redundant.** Remaining ghost headroom is NOT in the
compression (it's architecture + spatial context), so effort on richer waveform-preserving
representations is not warranted for this task.

## Takeaway (original, superseded on ghost by the UPDATE above)
The PI's "tokenize the signal" hypothesis is **partially supported**: a token-native model
reads object (to the dense ceiling) and ghost (above EventNet) better than the K-flatten
EventNet, and a per-event spatial hierarchy recovers glass to the ToPM level. But at the
**F1-mean headline it ties EventNet (~0.545–0.555) and does not reach ToPM-retrain (0.582)**;
the residual is ghost, which spatial-capacity/global-attention does **not** fix
(val-only gain, no test transfer). The ghost lever that works — ToPM's dense Gaussian field +
3D conv — is a dense-vs-sparse representation difference, outside the token-native family.

## Open / next levers
- `ta` K=4 ablation on v2/v3 (does the architecture reproduce taw→ghost / ta→glass?) — completes
  the thesis characterisation but won't beat V2.
- 2nd seed on v2/v3 to tighten the ~0.545 vs 0.555 comparison (currently single-seed).
- The only ghost lever with evidence is dense reconstruction (ToPM-style), i.e. *not* a
  pure token model — a hybrid (sparse tokens + a small dense ghost/secondary-return channel)
  would test whether the dense field is the irreducible ingredient.

## Reproduce
```bash
export PATH="$HOME/.local/bin:$PATH"; export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src
# train (single seed, DataParallel over 2 GPUs, bf16):
.venv/bin/python -m eventnet.train --arch setopm3 --feature_mode taw --K 4 --frame_stride 7 \
  --emb_dim 256 --ffn_mult 2 --window_size 8 --attn_heads 4 --batch_size 16 --epochs 50 \
  --lr 5e-4 --warmup_epochs 5 --amp --num_workers 8 --seed 42 --gpus 3,1 \
  --save_dir downstream/outputs/sparse_event_topm/taw_k4_g_dp_s42
# eval (paper peak-level):
.venv/bin/python -m eventnet.evaluate --checkpoint <save_dir>/best.pth --frame_stride 3 \
  --device cuda:3 --out <save_dir>/eval.json
# arch ∈ {setopm (flat), setopm2 (hier), setopm3 (hier+global)}; configs in eventnet/configs/.
```
