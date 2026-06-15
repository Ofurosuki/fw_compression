# Architecture diagrams (torchview)

Generated with `torchview` (`uv run`, CPU trace, taw input `[1,32,32,K=4,F=4]`).
Architecture evolution V1 → V2 → V2sa. See EXPERIMENT_LOG.md for results.

## Module-level (depth=1, readable at a glance)
- `V1_modules.png` — **V1** `EventTensorNet` (1.94M): event-MLP + rank-emb → flatten K
  into channels → **SmallUNet2D** (2-level). *No attention.*
- `V2_modules.png` — **V2** `EventTensorNetV2` (7.85M): event-MLP + rank-emb →
  **CrossEventAttention** (per-pixel Transformer over the K ray-events) → UNet2D (3-level).
  *V1→V2 change = the CrossEventAttention block.*
- `V2sa_modules.png` — same top-level as V2 (the spatial-attention difference is *inside*
  UNet2D; see below).

## UNet2D internals (depth=2) — shows the V2 → V2sa change
- `UNet2D_v2_noSA.png` — 3-level U-Net, plain conv bottleneck.
- `UNet2D_v2sa_bottleneckSA.png` — **V2sa**: at the 4×4 bottleneck adds
  `flatten → LayerNorm → multi_head_attention_forward → residual` (global spatial
  self-attention) before the decoder. *V2→V2sa change = this bottleneck MHA.*

## Full detail (depth=4, expand_nested; long graphs)
- `EventTensorNet_V1.png`, `EventTensorNetV2.png`, `EventTensorNetV2_spatialAttn.png`.

## Architecture summary
```
V1 :  events[B,H,W,K,F] → eventMLP+rankEmb → [flatten K] → 2D U-Net(2lvl) → logits[B,H,W,K,C]
V2 :  events            → eventMLP+rankEmb → CrossEventAttention(K) → [flatten K] → U-Net(3lvl) → logits
V2sa: V2 with global spatial self-attention at the U-Net bottleneck (4×4 grid)
```
Params: V1 1.94M · V2 7.85M · V2sa 8.91M  (frozen full-waveform judge FWL-ToPM = 8.72M).
