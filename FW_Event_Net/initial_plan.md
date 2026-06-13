# Event Tensor Network for Full-Waveform LiDAR Ghost/Glass Segmentation

## Goal

Build an event-based neural network that replaces dense full-waveform LiDAR input with sparse transport-event tensors.

Instead of feeding a full waveform

```text
x(u, v, t),  t = 1...700
```

to Ghost-FWL, extract the top-K return events from each waveform and represent each event by a compact 5-dimensional feature:

```text
e_i = [t_i, delta_t_i, a_i, w_i, m_i]
```

where:

* `t_i`: normalized peak position / time-of-flight
* `delta_t_i`: normalized delay from the first valid event
* `a_i`: normalized peak amplitude or peak area
* `w_i`: normalized FWHM / peak width
* `m_i`: valid mask, 1 if event exists, 0 if padded

The resulting input tensor is:

```text
events: [B, H, W, K, 5]
```

The model should perform event-level segmentation into:

```text
0: noise
1: object
2: glass
3: ghost
```

The core research question is:

> Can sparse transport events replace dense full-waveform LiDAR for ghost/glass-aware perception?

A second key question is:

> Does waveform-derived width `w_i` provide useful information beyond conventional multi-echo LiDAR, which is roughly represented by `(t_i, a_i)`?

---

# Main Experimental Setup

## Input

Original full-waveform voxel grid:

```text
waveform: [T, H, W]
T = 700
```

For each ray / pixel `(u, v)`, extract up to `K` events:

```text
E[u, v] = {e_1, ..., e_K}
```

Each event:

```text
e_i = [t_i, delta_t_i, a_i, w_i, m_i]
```

Thus the final event tensor is:

```text
events: [H, W, K, 5]
```

For batching:

```text
events: [B, H, W, K, 5]
```

---

# Event Feature Definition

For each detected peak/event `i`:

## 1. `t_i`

Absolute peak time index normalized by waveform length.

```python
t_i_norm = t_i / T
```

This corresponds to range/depth.

## 2. `delta_t_i`

Relative delay from the first valid event.

```python
delta_t_i = (t_i - t_1) / T
```

For the first event:

```python
delta_t_1 = 0.0
```

This is important for ghost, glass, and multipath because their structure often appears as delayed secondary events along the same ray.

## 3. `a_i`

Peak amplitude or local peak area.

Recommended first implementation:

```python
a_i = peak_height / max(waveform)
```

If `max(waveform) <= eps`, mark all events as invalid.

Alternative later ablation:

```python
a_i = local_area_around_peak / total_waveform_energy
```

## 4. `w_i`

Peak width / FWHM normalized by waveform length.

```python
w_i_norm = fwhm_i / T
```

This is the key waveform-derived cue that distinguishes this representation from ordinary multi-echo LiDAR.

## 5. `m_i`

Valid mask.

```python
m_i = 1.0 if event exists else 0.0
```

For padded events:

```python
[t_i, delta_t_i, a_i, w_i, m_i] = [0, 0, 0, 0, 0]
```

---

# Event Ordering

Events should be sorted by time-of-flight, not by amplitude.

```text
event 1 = earliest valid return
event 2 = second earliest valid return
...
event K = K-th earliest valid return
```

This preserves ray-level transport structure.

Do not rank only by amplitude in the final tensor because the meaning of `delta_t_i` depends on chronological ordering.

Peak detection may first collect candidate peaks using prominence/height, but after selecting top-K candidates, sort them by time.

---

# Event Extraction

Create:

```text
compression/event_extraction.py
```

Required function:

```python
def extract_topk_events(
    waveform: np.ndarray,
    K: int,
    T: int = 700,
    smoothing_sigma: float = 1.0,
    min_prominence: float = 0.03,
    min_height: float = 0.03,
    min_distance: int = 2,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Args:
        waveform: [T] raw waveform.
        K: maximum number of events.
        T: waveform length.

    Returns:
        events: [K, 5] array.
            columns = [t_norm, delta_t_norm, a_norm, w_norm, mask]
    """
```

Implementation details:

1. If waveform max is almost zero, return all-zero events.
2. Normalize waveform by its maximum value for event extraction.
3. Optionally smooth waveform with a small Gaussian filter.
4. Detect peaks using `scipy.signal.find_peaks`.
5. Estimate FWHM using `scipy.signal.peak_widths`.
6. Select up to top-K peaks by prominence or height.
7. Sort selected peaks by time index.
8. Compute normalized features.
9. Pad with invalid events if fewer than K peaks exist.

Recommended initial parameters:

```python
smoothing_sigma = 1.0
min_prominence = 0.03
min_height = 0.03
min_distance = 2
```

These should be configurable from CLI.

---

# Dataset

Create:

```text
eventnet/data.py
```

The dataset should load:

1. Full waveform voxel grid
2. Dense label grid
3. Extract top-K events per pixel
4. Assign event-level labels

Expected raw data:

```text
waveform: [T, H, W]
label:    [T, H, W]
```

Output:

```python
{
    "events": event_tensor,       # [H, W, K, 5]
    "labels": event_labels,       # [H, W, K]
    "valid": valid_mask,          # [H, W, K]
}
```

---

# Event Label Generation

For each event position `t_i`, assign a label from the dense voxel label grid.

Naive version:

```python
label_i = dense_label[t_i, u, v]
```

Recommended version:

Use majority label in a local window around the peak:

```python
r = max(1, int(width_i / 2))
label_i = mode(dense_label[t_i-r : t_i+r+1, u, v])
```

If `m_i = 0`, set:

```python
label_i = 0  # noise
```

Also ensure invalid padded events can be ignored or treated as noise depending on the loss setting.

Recommended first implementation:

* Assign invalid events to class 0.
* Use valid mask to optionally ignore padded events in the loss.
* Report metrics both with and without padded invalid events.

---

# Model

Create:

```text
eventnet/model.py
```

## Architecture

Use a shared event MLP, rank embedding, and 2D spatial U-Net.

Input:

```text
events: [B, H, W, K, 5]
```

Pipeline:

```text
[B, H, W, K, 5]
→ shared event MLP
→ [B, H, W, K, D]
→ add rank embedding
→ flatten K dimension
→ [B, K*D, H, W]
→ 2D U-Net / small CNN
→ [B, K*C, H, W]
→ reshape
→ [B, H, W, K, C]
```

Where:

```text
D = event embedding dimension, e.g. 32
C = number of classes = 4
```

## Minimal PyTorch Skeleton

```python
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class EventTensorNet(nn.Module):
    def __init__(
        self,
        K: int,
        in_dim: int = 5,
        emb_dim: int = 32,
        num_classes: int = 4,
        base_channels: int = 64,
    ):
        super().__init__()
        self.K = K
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.event_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

        self.rank_embedding = nn.Embedding(K, emb_dim)

        self.spatial_net = SmallUNet2D(
            in_channels=K * emb_dim,
            out_channels=K * num_classes,
            base_channels=base_channels,
        )

    def forward(self, events):
        """
        Args:
            events: [B, H, W, K, 5]

        Returns:
            logits: [B, H, W, K, C]
        """
        B, H, W, K, F = events.shape
        assert K == self.K

        # Shared event MLP
        feat = self.event_mlp(events)  # [B, H, W, K, D]

        # Add rank embedding
        rank_ids = torch.arange(K, device=events.device)
        rank_feat = self.rank_embedding(rank_ids)  # [K, D]
        feat = feat + rank_feat.view(1, 1, 1, K, self.emb_dim)

        # Flatten event dimension into channels
        feat = feat.reshape(B, H, W, K * self.emb_dim)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, K*D, H, W]

        logits = self.spatial_net(feat)  # [B, K*C, H, W]

        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(B, H, W, K, self.num_classes)

        return logits
```

---

# Loss

Use event-level classification loss.

Initial loss:

```text
weighted cross entropy
```

Input logits:

```text
logits: [B, H, W, K, C]
labels: [B, H, W, K]
valid:  [B, H, W, K]
```

Recommended first option:

* Include valid events only in the loss.
* Ignore padded invalid events.

```python
loss = CE(logits[valid == 1], labels[valid == 1])
```

Class imbalance is expected. Use class weights.

Example:

```python
class_weights = torch.tensor([0.2, 1.0, 2.0, 2.0])
```

Tune later based on label distribution.

Alternative later:

```text
focal loss
```

---

# Metrics

Report per-class F1 and macro-F1 excluding noise.

Classes:

```text
0: noise
1: object
2: glass
3: ghost
```

Primary metric:

```text
F1-mean = macro-F1 over object, glass, ghost
```

That is:

```text
F1-mean = mean(F1_object, F1_glass, F1_ghost)
```

Do not include noise in the mean.

Report:

```text
object F1
glass F1
ghost F1
F1-mean
```

Also report:

```text
valid-event accuracy
valid-event macro-F1
```

Optional additional metric:

Convert event predictions back into a sparse voxel label grid and compute voxel-level F1 to compare more directly with Ghost-FWL.

---

# Dense Voxel Reconstruction for Evaluation

To compare with Ghost-FWL voxel segmentation, optionally map event predictions back to a dense voxel label volume.

For each predicted event:

```text
event: (u, v, t_i, w_i)
predicted class: c_i
```

Fill the label around the event location:

```python
r = max(1, int(width_i / 2))
pred_dense[t_i-r : t_i+r+1, u, v] = c_i
```

This produces:

```text
pred_dense: [T, H, W]
```

Then compute voxel-level F1 using the same metric as Ghost-FWL.

This is optional for the first version, but useful for final comparison.

---

# Baselines

The experiment must include the following baselines.


## Baseline 1: First-return only

Input per pixel:

```text
[t_1, a_1]
```

or optionally:

```text
[t_1, a_1, m_1]
```

Purpose:

```text
Ordinary single-return LiDAR-like representation.
```

This baseline tests whether a single surface hit is enough.

Expected behavior:

* poor ghost performance
* weak glass performance
* decent object performance if first return is object

---

## Baseline 2: Depth-only multi-event

Input:

```text
{t_i}_{i=1}^K
```

or practical version:

```text
[t_i, delta_t_i, m_i]
```

Purpose:

```text
Multi-return geometry without intensity or width.
```

This tests whether geometry alone explains the downstream task.

---

## Baseline 3: Multi-echo baseline

Input:

```text
{(t_i, a_i)}_{i=1}^K
```

Recommended practical version:

```text
[t_i, delta_t_i, a_i, m_i]
```

Purpose:

```text
Conventional multi-echo LiDAR-like representation.
```

This is the most important baseline.

The proposed representation must be compared against this to show that waveform-derived width matters.

---

## Proposed: Event Tensor Net

Input:

```text
{(t_i, delta_t_i, a_i, w_i, m_i)}_{i=1}^K
```

Purpose:

```text
Sparse marked transport-event representation.
```

Main claim:

```text
Width/FWHM carries information beyond conventional multi-echo range-intensity returns.
```

The key comparison is:

```text
(t, delta_t, a, w, m)  vs.  (t, delta_t, a, m)
```

If the proposed representation improves especially on glass and ghost classes, it supports the hypothesis that full-waveform LiDAR provides useful temporal-shape cues beyond multi-echo LiDAR.

---

# Ablation Table

Run the following input variants with the same architecture as much as possible.

| Name     | Event feature           | Meaning                               |
| -------- | ----------------------- | ------------------------------------- |
| `t_only` | `[t, m]`                | single cue: range/depth               |
| `t_dt`   | `[t, delta_t, m]`       | multi-event geometry                  |
| `ta`     | `[t, a, m]`             | simple multi-echo                     |
| `tdta`   | `[t, delta_t, a, m]`    | practical multi-echo                  |
| `taw`    | `[t, a, w, m]`          | waveform width without relative delay |
| `tdtaw`  | `[t, delta_t, a, w, m]` | proposed full event feature           |

Primary comparison:

```text
tdtaw > tdta
```

Secondary comparisons:

```text
tdta > t_dt
taw > ta
tdtaw > taw
```

Interpretation:

* `tdta > t_dt`: intensity helps beyond geometry
* `taw > ta`: width helps beyond multi-echo
* `tdtaw > taw`: relative ray structure helps
* `tdtaw > tdta`: full proposed event representation beats multi-echo baseline

---

# K Sweep

Run:

```text
K = 1, 2, 4, 8
```

Expected behavior:

* `K=1`: close to first-return LiDAR
* `K=2`: should improve ghost/glass
* `K=4`: likely best tradeoff
* `K=8`: may include too many noisy peaks

Report for each K:

```text
object F1
glass F1
ghost F1
F1-mean
```

Recommended first pass:

```text
K=4
```

Then run K sweep after confirming the pipeline works.

---

# Training Script

Create:

```text
eventnet/train.py
```

Required CLI arguments:

```bash
python eventnet/train.py \
  --data_root /path/to/ghost_dataset \
  --split train \
  --K 4 \
  --feature_mode tdtaw \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --save_dir outputs/eventnet/tdtaw_K4
```

Recommended arguments:

```text
--feature_mode
    one of: t_only, t_dt, ta, tdta, taw, tdtaw

--K
    number of events per ray

--ignore_invalid
    whether to ignore padded events in the loss

--class_weights
    optional manually specified class weights

--peak_prominence
--peak_height
--smoothing_sigma
    event extraction parameters
```

---

# Evaluation Script

Create:

```text
eventnet/evaluate.py
```

Required CLI:

```bash
python eventnet/evaluate.py \
  --checkpoint outputs/eventnet/tdtaw_K4/best.pth \
  --data_root /path/to/ghost_dataset \
  --split test \
  --K 4 \
  --feature_mode tdtaw \
  --output_json outputs/eventnet/tdtaw_K4/eval.json
```

Output JSON should include:

```json
{
  "feature_mode": "tdtaw",
  "K": 4,
  "object_f1": 0.0,
  "glass_f1": 0.0,
  "ghost_f1": 0.0,
  "f1_mean": 0.0,
  "per_class_precision": {},
  "per_class_recall": {},
  "confusion_matrix": []
}
```

---

# Sweep Script

Create:

```text
eventnet/run_sweep.py
```

Run all feature modes and K values.

Example:

```bash
python eventnet/run_sweep.py \
  --data_root /path/to/ghost_dataset \
  --Ks 1 2 4 8 \
  --feature_modes t_only t_dt ta tdta taw tdtaw \
  --epochs 50 \
  --save_root outputs/eventnet/sweep
```

Final output table:

| feature_mode |  K | object F1 | glass F1 | ghost F1 | F1-mean |
| ------------ | -: | --------: | -------: | -------: | ------: |
| t_only       |  1 |           |          |          |         |
| tdta         |  4 |           |          |          |         |
| tdtaw        |  4 |           |          |          |         |

---

# Plotting

Create:

```text
eventnet/plot_results.py
```

Required plots:

1. `F1-mean vs K`
2. `glass F1 vs K`
3. `ghost F1 vs K`
4. Bar chart comparing feature modes at best K

Main plot should compare:

```text
t_dt
tdta
tdtaw
full waveform
```

---


# Notes on Fair Comparison

To ensure fairness:

1. Use the same train/val/test split across all feature modes.
2. Use the same architecture capacity as much as possible.
3. Keep K fixed when comparing feature modes.
4. Report parameter count and inference speed.
5. Compare against full waveform Ghost-FWL as an upper bound.
6. Compare against multi-echo `(t, a)` as the most important baseline.
7. Report results per class because glass and ghost are expected to benefit most from waveform-derived width.

---

# Minimal First Milestone

Implement only the following first:

```text
K = 4
feature modes = tdta, tdtaw
model = EventTensorNet
metric = event-level macro-F1 excluding noise
```

This answers the first key question:

```text
Does width w improve over multi-echo-like (t, a)?
```

Once this works, add:

```text
t_only
t_dt
ta
taw
K sweep
voxel-level reconstruction metric
speed/FLOPs
```

---

# File Checklist

Create or modify the following files:

```text
compression/event_extraction.py
eventnet/data.py
eventnet/model.py
eventnet/losses.py
eventnet/metrics.py
eventnet/train.py
eventnet/evaluate.py
eventnet/run_sweep.py
eventnet/plot_results.py
```

Optional:

```text
eventnet/reconstruct_dense.py
eventnet/configs/tdtaw_K4.yaml
eventnet/configs/sweep.yaml
```

---

# One-Sentence Research Claim

The intended research claim is:

```text
Full-waveform LiDAR perception can be reformulated as sparse marked transport-event segmentation, where each return event is represented by range, relative delay, intensity, and temporal width rather than dense waveform samples.
```

The main empirical test is:

```text
Does [t, delta_t, a, w, mask] outperform multi-echo-like [t, delta_t, a, mask], especially on glass and ghost classes?
```
