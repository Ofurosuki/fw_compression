# FW_Event_Net — consolidated experiment log (V1 → present)

Single-page ledger of every experiment in the event-net investigation. All scores are the
**paper-compliant peak-level F1** on the SPLIT2 test set (3 held-out scenes), the same metric &
population as the paper's "F1-mean". Detailed write-ups: `RESULTS.md`; per-scene geometry:
`SCENE_GEOMETRY.md`.

## Reference upper bounds (frozen full-waveform Ghost-FWL / FWL-ToPM, same test & metric)
- **peak-level F1-mean 0.599** (= paper 0.592), voxel-level 0.532. Per-class peak: object
  0.742, glass 0.385, ghost 0.669. Model size **8.72 M** params.

## Headline
- **V2 `taw` K=4 = 0.555 (2-seed) = 93 % of the 0.599 ceiling**, from a **>100× smaller**
  input (≤4 numbers × 4 events vs 700 samples) at **7.85 M** params (≈ the 8.72 M judge).
- object already **beats** the full-waveform model (0.75 vs 0.742); glass is the gap (≈0.31
  vs 0.385) and is a **representation/domain-gap** problem (see "Lessons").

## Inference speed (cuda:2 ~Blackwell, B=1, fp32; warmup + timed iters + CUDA-sync)
| config | input | params | ms/fwd | FPS |
|---|---|--:|--:|--:|
| FWL-ToPM (full-waveform) | crop 300×168×200 (ToMe-pruned) | 8.72 M | 9.8 | 102 |
| V2 `taw` forward | full 400×336×K4×F4 | 7.85 M | 15.6 | 64 |
| V2 event extraction (taw, bare) | raw → events | — | 10 | 100 |
| **V2 `taw` end-to-end** (extract+fwd) | raw → logits | 7.85 M | **25.6** | **39** |

- Per-forward ToPM is ~1.6× faster, but on **43 % the coverage** (a 200×168 crop vs V2's full
  400×336 plane); ToPM is fast because ToMe token-merge + intensity-pruning shrink its input.
- **At equal full-frame coverage they're comparable** (~24 ms each: ToPM ~2.4 crops; V2 25.6 ms
  full plane). V2 end-to-end ≈ **39 FPS**, real-time-capable.
- **The compression win is bandwidth/data-size (>100×), NOT a speed win** — inference speeds
  are the same order. (V3/V4 12-col decomposition extraction is 394 ms; `taw` needs only the
  10 ms bare extraction.)
- **V2 forward breakdown** (params ≠ latency): cross-event attention **11.6 ms / 75 %** (≈0
  params, but runs on B·H·W = 134k per-pixel sequences of length K=4 → memory/launch-bound,
  low GPU utilization), U-Net 3.2 ms / 21 % (holds most params but compute-dense → fast),
  event-MLP 0.7 ms. → V2 is "slow for its param count" because the bottleneck is the
  near-param-free per-pixel attention; optimizing it (lighter relational op / fp16 /
  fewer heads) could cut forward toward ~5 ms. ToPM is param-heavy but fast because ToMe
  merge+prune shrink its effective token count.

---

## Master results table

### A. V1 — `EventTensorNet` (event MLP + rank-emb → channel-flatten → 2-level U-Net, 1.9 M)
Feature ablation at K=4 (single seed):

| feature | F1-mean | object | glass | ghost |
|---|--:|--:|--:|--:|
| t_only | 0.500 | 0.720 | 0.265 | 0.514 |
| t_dt | 0.509 | 0.720 | 0.283 | 0.522 |
| ta | 0.525 | 0.738 | 0.281 | 0.555 |
| taw | 0.525 | 0.741 | 0.274 | 0.561 |
| tdta | 0.529 | 0.756 | 0.265 | 0.565 |
| **tdtaw** | **0.534** | 0.754 | 0.273 | 0.576 |

K-sweep (tdtaw): K1 **0.357** (ghost 0.058!), K2 0.497, **K4 0.534**, K8 0.523.
→ intensity is the big lever (t_dt→tdta +0.020); **width marginal** under V1 (taw−ta +0.00);
ghost needs **K≥2**; K=4 best.

### B. V2 — `EventTensorNetV2` (+ cross-event attention + 3-level U-Net, GELU, 7.85 M)
Feature ablation at K=4 (2-seed mean; seed42/seed43):

| feature | F1-mean | glass | note |
|---|--:|--:|---|
| ta | 0.534 | 0.268 | 0.531 / 0.536 |
| **taw** | **0.555** | **0.307** | 0.565 / 0.544 — **headline** |
| tdta | 0.524 | 0.252 | 0.536 / 0.511 |
| tdtaw | 0.525 | 0.257 | 0.509 / 0.542 |

→ V2 lifts taw +0.03 over V1; **width now helps** (taw−ta +0.021, both seeds); **Δt becomes
redundant/harmful** (taw > tdtaw) — attention learns inter-return timing. Best rep = `taw`.

### C. Feature experiments on V2 `taw` (close the glass gap) — all NEGATIVE
| experiment | config | result vs taw control | verdict |
|---|---|---|---|
| **behind_energy** (raw transmitted-E) | taE / tdtaE / tdtaEw (s42) | ta 0.525→taE **0.498** (glass −0.066) | hurts; depth/scene confound, sign-flips |
| **direct/indirect decomposition** | tawD / tawI / tawi (s42, in-run base 0.536) | tawD 0.500, tawI 0.521, tawi 0.535 | ≈/worse; glass unmoved |
| **radiometric range correction** | diagnostic only (ρ=a·R²) | per-scene AUC still flips | won't transfer (no training run) |
| **NeRF transmittance profile** | tawT (2-seed) | taw 0.538 → tawT **0.536** (glass +0.007) | ≈ taw, within ±0.03; shape doesn't transfer |

Common root: glass's "what's behind" cue is **geometry/scene-dependent** (sign-flips across
scenes incl. a train scene) → no per-ray scalar/shape summary transfers. (`SCENE_GEOMETRY.md`.)

### D. Training-side levers on V2 `taw`
| experiment | config | result | verdict |
|---|---|---|---|
| **V-REx (domain generalization)** | β=1/10/30, scene=env | 2-seed erm 0.531 vs β1 **0.539**; β10/30 hurt | ≈ ERM (Gulrajani-LopezPaz); no real gain |
| **loss: glass class-weight** | ×1.5 / ×3 | 2-seed glass +0.018 (not seed-robust), **F1 flat** | precision/recall trade, no lift |
| **loss: focal (γ=2)** | seed42 | F1 0.535, glass 0.296 | same trade as class-weight |

### E. Architecture: spatial attention (`v2sa`, +1.06 M)
| run | recipe | F1 (2-seed) | ghost | verdict |
|---|---|--:|--:|---|
| v2sa initial | bolt-on (lr1e-3, no warmup, 40ep) | 0.523 | 0.201 | artifact — *under-trained* (val also ↓, not overfit) |
| base (fair) | lr5e-4, warmup5, 50ep | 0.536 | 0.583 | control |
| **v2sa (fair)** | lr5e-4, warmup5, 50ep | **0.541** | 0.612 | **neutral** — ΔF1 +0.005, sign-flips per-seed (+0.029/−0.019); seed-42 ghost +0.08 did NOT replicate (s43 −0.022) |

→ fair recipe resolves the under-training (v2sa val ≈ base), but v2sa **≈ base on test** (seed-
dependent, within ±0.03). Spatial attention is **neutral**, not a headline. F1-mean 0.541 < ToPM 0.599.

---

## Chronological arc (one-line verdicts)
1. **V1 ablation** → sparse events work; intensity≫width; ghost needs K≥2; tdtaw K4 = 0.534 (89 %).
2. **V1 K-sweep** → K=4 optimal; ghost collapses at K=1.
3. **V2 (attention)** → +0.03 → **taw 0.555 (93 %)**; width now helps; Δt redundant. **Headline.**
4. **behind_energy** → strongest single feature in isolation, but **hurts** (depth/scene confound).
5. **per-scene + depth-stratified + radiometric diagnostics** → the confound is **geometry**; not a split artifact (flips in a train scene too).
6. **V3 decomposition (direct/indirect)** → no transfer; glass unmoved.
7. **V4 NeRF transmittance profile** → no transfer (shape flips on same scenes).
8. **V-REx DG** → ≈ ERM.
9. **loss sweep (glass-weight / focal)** → moves glass but precision/recall trade; F1 flat, not seed-robust.
10. **spatial attention** → first run was under-training artifact; **fair-recipe 2-seed = neutral** (ΔF1 +0.005, seed-flipping; seed-42 ghost +0.08 didn't replicate). Not a headline.

---

## Cross-cutting lessons
- **Architecture > features.** The only thing that robustly lifted the headline was the V2
  architecture (V1→V2 +0.03). No bolt-on feature or DG loss did.
- **"Separable but not transferable."** 4 diffuse features (width, behind_energy, decomposition,
  NeRF-T) and radiometric correction look discriminative in-isolation/on-train but don't transfer
  to held-out scenes — because the glass cue is geometry/scene-dependent.
- **Width's value is architecture-dependent** (marginal in V1, real in V2).
- **Δt is redundant once attention is present.**
- **The glass ceiling is a representation/domain-gap problem**, confirmed from 6 angles
  (4 features + DG loss + loss tuning), reinforced by capacity (spatial attn) hurting when
  bolted on. Only lever with headroom left: a different **transport/NeRF end-to-end**
  representation (Option B) — large build.
- **±0.03 run/cache/seed variance is real** (same taw/V2/seed42 = 0.565 on 4-col vs 0.536 on
  7-col cache) → always within-run control + multi-seed; single-run deltas <0.03 are noise.

## Settled vs open
- **Settled:** sparse events reach ~93 % of the dense ceiling at ~judge model size; object
  matches/beats full-waveform; glass is the binding gap and is not closable by per-ray features,
  DG loss, or loss tuning.
- **Open / in-flight:** fair spatial-attention retest (improved recipe, GPU-2); untried arch/
  training knobs (EMA, stronger aug, alternate backbone); **Option B** transport/NeRF
  representation (the main remaining lever).

*(Detailed methodology and tables: `RESULTS.md`. Per-scene geometry: `SCENE_GEOMETRY.md` /
`SCENE_GEOMETRY.ja.md`. Memory: `fwc-eventnet`, `fwc-behind-energy-litreview`.)*
