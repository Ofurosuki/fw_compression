# Session progress — ToPM architecture-controlled representation test (+ viz)

Date: 2026-06-17 → 06-19. Branch `feature/remove_falsepositive` (working copy
`/home/yoshida/ai-task-folder`; canonical project `/home/yoshida/fw_compression`).

## Goal (PI's pivot)
FW_Event_Net and the frozen-judge experiments changed *representation* AND
*architecture/training* together, so they can't isolate a representation's value. Hold the
architecture fixed (**ToPM**, `vit3d_ordered_pruning_light`, 8.72 M) and change only the
input: lift each representation to a T=700 pseudo-waveform and **retrain ToPM from scratch**
on it under one fixed recipe. Effectiveness = F1(ToPM on rep) / F1(ToPM on full waveform).

## Infrastructure built (repo kept read-only)
- `downstream/run_retrain.py` — drives the repo's `train_vit3d` with the `_load_voxel_grid`
  monkey-patch (same hook as eval) so the input transform applies to training too. Config =
  repo split2 train recipe (lr1e-4 / focal+dice / cosine / 50 ep / no-aug) + eval ToPM arch
  + 280/60 train/val dirs. `--cache_root` trains on a pre-built cache; `--seed` for seeds.
- `downstream/cache_repr.py` — pre-computes event/AE reps as **uint16** voxels (~4–9 MB/frame
  vs 71 MB float32) so retraining reads them with normal workers (~6 min/epoch vs ~50 h
  on-the-fly). Strided 1/3 subset, reproducible.
- `downstream/run_eval_peak.py` — paper-metric **peak-level** F1 (raw-waveform peak
  population, identical across reps), reusing the repo's `detect_peaks_in_voxel`/`evaluate_peaks`.
- `downstream/vis_taw_vs_full_rerun.py` — rerun visualisation: multi-echo point clouds from
  full vs taw, coloured by ToPM segmentation (object=green/glass=blue/ghost=red). Live
  (`--web_port`) or `--rrd` file. (added rerun-sdk 0.33 + pandas to the fw_compression venv.)
- `downstream/RETRAIN_RESULTS.md` — full write-up.

## Headline result (seed 42, divide=3 test ≈475 frames)

| representation | frozen voxel | retrained voxel | retrained **peak** |
|---|--:|--:|--:|
| full waveform (T=700) | 0.524 | **0.533** | **0.595** |
| taw K=4 (t,a,w events→Gaussian) | 0.408 | **0.531** | **0.582** |
| ta K=4 (multi-echo: t,a, fixed width) | 0.176 | **0.515** | **0.574** |

Validation: retrained full voxel 0.533 = frozen neurips_best 0.532; full peak 0.595 ≈ paper 0.592.

### Findings
1. **Sparse top-K events are essentially lossless once the model adapts** — taw-K4 0.531/0.582
   ≈ full 0.533/0.595 (voxel Δ within ±0.003; peak = 98 % of full). The dense T=700 waveform
   carries ~no downstream info beyond top-K (t,a,w).
2. **The frozen-judge gaps were DOMAIN SHIFT, not info loss** — retraining recovers +0.12 (taw),
   +0.34 (ta); the frozen model had never seen synthesised pulses.
3. **Width's architecture-controlled value is small**: taw−ta = +0.016 voxel / +0.008 peak
   (frozen showed +0.23 — a ~14–28× overstatement). Multi-echo (t,a) alone reaches 96–97 % of
   full. Width helps ghost/object, slightly HURTS glass (ta even beats full on glass).
4. **Tempers the earlier "width matters a lot / FW ≫ multi-echo" reading** and vindicates the
   PI's demand for an architecture-controlled test.

## In flight (this session)
- **2nd seed (43)** for full/taw/ta-K4 — confirms taw−ta margin. ~epoch 30/50 each (cuda:0/1/3),
  slow (~25 min/epoch) due to shared GPUs; ETA ~10:00. losses track seed 42.
- **K-sweep caches** taw/ta at K=2,3 (cuda:2): K2 done, K3 in progress. K=2,3 retrains queued
  for when seed-43 frees the GPUs.

## Next
2-seed table → K=2,3 retrain+eval (does taw=full hold at lower K?) → optional tw / AE recon /
fuller rerun .rrd over test scenes.

## Env notes
Real env = `/home/yoshida/fw_compression/.venv` (cu128 torch + repo deps); ai-task-folder/.venv
is minimal — run repo code with the fw_compression venv python directly, NOT `uv run`.
PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src. Caches in /home/yoshida/fwc_cache.
