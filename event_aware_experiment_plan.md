# Top-K Transport Event Representation Experiment

## Goal

Implement an experiment to test whether a dense full-waveform LiDAR signal is truly necessary for downstream Ghost-FWL ghost detection, or whether a sparse list of top-K transport events is sufficient.

The central hypothesis is:

> Ghost-FWL may not require the full dense waveform `x[t]`.
> It may primarily rely on sparse transport events represented by peak position, intensity, and width:
>
> `{(t_i, a_i, w_i)}_{i=1..K}`

where:

* `t_i`: peak position / depth bin
* `a_i`: peak intensity / area / return strength
* `w_i`: FWHM / peak width / temporal spread

This experiment should compare dense full-waveform input, autoencoder-compressed waveform reconstruction, and sparse top-K event reconstruction under a frozen downstream Ghost-FWL evaluator.

---

## Background

We already have a downstream compression evaluation pipeline:

```text
full waveform x [T=700]
  → compression / reconstruction
  → pseudo waveform x_hat [T=700]
  → frozen Ghost-FWL segmentation model
  → object / glass / ghost F1
```

Existing results show:

1. Very high compression can preserve much of the downstream Ghost-FWL F1.
2. Spatial 4×4 compression is more robust than per-pixel compression at high compression ratios.
3. Anti-hallucination loss improves downstream F1 even when reconstruction MSE worsens.
4. This suggests that waveform compression should preserve transport event structure, not just minimize MSE.

Now we want to test a stronger hypothesis:

> Maybe the downstream model mostly needs sparse peak/event parameters rather than a dense reconstructed waveform.

---

## New Experiment Overview

Implement a top-K event pipeline:

```text
original waveform x [T=700]
        ↓
event extractor
        ↓
top-K events: [(t_i, a_i, w_i)] for i=1..K
        ↓
parametric waveform synthesis
        ↓
pseudo waveform x_event [T=700]
        ↓
frozen Ghost-FWL model
        ↓
object / glass / ghost F1
```

The downstream Ghost-FWL model must remain frozen.

The event representation is evaluated by synthesizing a pseudo-waveform from the extracted events, then passing it into the same downstream evaluation harness used for autoencoder compression.

---

## Key Research Questions

1. Is peak position alone enough?
2. Does adding intensity improve ghost/glass/object F1?
3. Does adding FWHM/width improve F1 further?
4. How many events K are needed?
5. Can top-K event reconstruction match or outperform AE-based compressed reconstruction at similar or smaller dimensionality?
6. Does sparse event reconstruction fail specifically on glass, weak ghosts, or overlapping peaks?

---

## Representations to Compare

Implement the following event representations:

### 1. Position only

Only peak positions are retained.

```text
events = [(t_i)] for i=1..K
```

Synthesis uses fixed amplitude and fixed width:

```text
a_i = 1.0
w_i = fixed_width
```

Purpose:

> Test whether Ghost-FWL mostly relies on multi-return geometry / peak positions.

---

### 2. Position + intensity

Retain position and intensity.

```text
events = [(t_i, a_i)] for i=1..K
```

Synthesis uses measured intensity and fixed width:

```text
w_i = fixed_width
```

Purpose:

> Test whether return strength / reflectance / ghost brightness is important.

---

### 3. Position + width

Retain position and FWHM, but use fixed amplitude.

```text
events = [(t_i, w_i)] for i=1..K
```

Synthesis uses:

```text
a_i = 1.0
```

Purpose:

> Test whether width / temporal spread itself is a useful transport cue.

---

### 4. Position + intensity + width

Retain all three parameters.

```text
events = [(t_i, a_i, w_i)] for i=1..K
```

Purpose:

> Test whether sparse transport events can approximate the useful information in full waveforms.

---

### 5. Optional: Position + intensity + width + background statistics

Add simple background or tail statistics:

```text
events = [(t_i, a_i, w_i)] for i=1..K
stats = {
    "background": b,
    "tail_energy": e_tail,
    "total_energy": e_total
}
```

Purpose:

> Test whether non-peak residual information, such as background floor or broad scattering tail, helps glass/ghost detection.

This optional representation can be implemented after the main experiment works.

---

## K Sweep

Evaluate:

```text
K ∈ {1, 2, 3, 4, 6, 8}
```

The effective dimensionality is:

|  K | position only | position + intensity | position + intensity + width |
| -: | ------------: | -------------------: | ---------------------------: |
|  1 |             1 |                    2 |                            3 |
|  2 |             2 |                    4 |                            6 |
|  3 |             3 |                    6 |                            9 |
|  4 |             4 |                    8 |                           12 |
|  6 |             6 |                   12 |                           18 |
|  8 |             8 |                   16 |                           24 |

Compression ratio relative to full waveform `T=700`:

```text
ratio = 700 / dim
```

For `(t,a,w)`:

|  K | dim | ratio |
| -: | --: | ----: |
|  1 |   3 |  233× |
|  2 |   6 |  117× |
|  4 |  12 |   58× |
|  8 |  24 |   29× |

Compare against existing AE baselines:

* full waveform, no compression
* 1D learnable AE at 88×, 44×, 22×, 11×, 6×
* spatial 4×4 AE at 88×, 44×, 22×, 11×, 6×
* anti-hallucination AE models if available

---

## Event Extraction

Create a new file:

```text
compression/event_extraction.py
```

Implement:

```python
def extract_topk_events(
    wave,
    k: int,
    smooth_sigma: float = 1.5,
    min_prominence: float = 0.03,
    min_distance: int = 3,
    rank_by: str = "prominence",
    intensity_mode: str = "area",
):
    """
    Extract top-K transport events from a 1D waveform.

    Args:
        wave:
            1D numpy array or torch tensor of shape [T].
            Expected to be max-normalized before extraction.
        k:
            Number of events to keep.
        smooth_sigma:
            Gaussian smoothing sigma before peak detection.
        min_prominence:
            Minimum peak prominence for scipy.signal.find_peaks.
        min_distance:
            Minimum distance between detected peaks.
        rank_by:
            Ranking criterion. Options:
            - "prominence"
            - "height"
            - "area"
        intensity_mode:
            How to measure a_i. Options:
            - "height": peak height
            - "area": local area around the peak
            - "prominence": peak prominence

    Returns:
        events:
            numpy array of shape [K, 3], columns [t, a, w].
            Missing events are padded with zeros.
        valid_mask:
            boolean array of shape [K], True for valid events.
    """
```

### Extraction details

1. Smooth waveform with `scipy.ndimage.gaussian_filter1d`.
2. Use `scipy.signal.find_peaks`.
3. Use `scipy.signal.peak_widths` to estimate FWHM.
4. Compute each event:

   * `t`: peak index, optionally refined later by sub-bin fitting
   * `a`: height, prominence, or local area
   * `w`: FWHM in bins
5. Rank detected peaks by:

   * primary: prominence
   * optional ablation: area
6. Keep top-K peaks.
7. Sort selected events by time before synthesis.
8. Pad missing events with zeros.

---

## Event Synthesis

Create a new file:

```text
compression/event_synthesis.py
```

Implement:

```python
def synthesize_waveform_from_events(
    events,
    valid_mask=None,
    T: int = 700,
    representation: str = "taw",
    fixed_amplitude: float = 1.0,
    fixed_width: float = 4.0,
    background: float = 0.0,
    normalize: bool = True,
):
    """
    Convert top-K event parameters into a pseudo-waveform.

    Args:
        events:
            array of shape [K, 3], columns [t, a, w].
        valid_mask:
            optional boolean array of shape [K].
        T:
            waveform length.
        representation:
            Options:
            - "t"
            - "ta"
            - "tw"
            - "taw"
            - "taw_bg"
        fixed_amplitude:
            amplitude used when representation does not include intensity.
        fixed_width:
            FWHM used when representation does not include width.
        background:
            optional background floor.
        normalize:
            whether to max-normalize the synthesized waveform.

    Returns:
        wave_hat:
            array of shape [T].
    """
```

Use Gaussian pulse synthesis:

```text
x_hat[t] = Σ_i a_i exp(-(t - t_i)^2 / (2 σ_i^2))
```

Convert FWHM to Gaussian sigma:

```text
sigma = fwhm / (2 * sqrt(2 * log(2)))
```

For stability:

* clamp minimum width, e.g. `w >= 1.0`
* clamp maximum width, e.g. `w <= 80`
* skip invalid padded events
* optionally normalize `x_hat` by max value

---

## Downstream Evaluation Hook

Create:

```text
downstream/run_eval_events.py
```

or add event mode to the existing `downstream/run_eval.py`.

The event hook should be inserted at the same point as the compression autoencoder hook:

```text
original waveform
    → per-pixel max-normalize
    → extract top-K events
    → synthesize pseudo-waveform
    → de-normalize by original max
    → downstream model preprocessing
    → frozen Ghost-FWL model
```

Important:

* Background pixels with `max(wave) <= eps` should be passed through or set to zero.
* Use the same crop/normalize pipeline as the current downstream evaluator.
* Do not retrain the downstream Ghost-FWL model.

---

## Sweep Script

Create:

```text
downstream/run_sweep_events.py
```

It should sweep:

```text
K ∈ {1, 2, 3, 4, 6, 8}
representation ∈ {"t", "ta", "tw", "taw"}
rank_by ∈ {"prominence"}
```

Optional later:

```text
rank_by ∈ {"prominence", "area", "height"}
```

Use the same divide setting as current sweeps:

```text
--divide 3
```

for fast evaluation.

After interesting configs are identified, rerun them on the full test set.

---

## Plotting and Tables

Create:

```text
downstream/run_plot_events.py
```

Generate:

1. F1-mean vs compression ratio
2. per-class F1 vs K
3. comparison table against AE baselines
4. example original vs event-synthesized waveforms

Required output table:

| representation |  K | dim | ratio | object F1 | glass F1 | ghost F1 | F1-mean |
| -------------- | -: | --: | ----: | --------: | -------: | -------: | ------: |

Include rows for:

* full waveform baseline
* AE spatial 4×4 + AH if available
* AE 1D + AH if available
* top-K `t`
* top-K `ta`
* top-K `tw`
* top-K `taw`

---

## Expected Interpretations

### Case A: Top-K `(t,a,w)` matches AE or full waveform

This supports:

> Ghost-FWL downstream perception is mostly governed by sparse transport events, not dense waveform fidelity.

This would be a strong research insight.

---

### Case B: Position-only performs poorly, but `(t,a,w)` performs well

This supports:

> Depth / multi-return position alone is insufficient. Intensity and width are essential transport cues.

This is especially important for differentiating full-waveform LiDAR from ordinary multi-echo LiDAR.

---

### Case C: Top-K events perform much worse than AE reconstruction

This supports:

> Dense waveform residuals, tails, asymmetry, non-Gaussian pulse shape, or background structure contain downstream-relevant information beyond explicit peaks.

Then we should extend the representation with:

* background floor
* tail energy
* skewness
* local residual energy
* low-frequency residual token

---

### Case D: Glass collapses first

This likely means:

> Transparent-object cues are not captured well by simple top-K peaks and may require subtle waveform residuals or spatial context.

In this case, glass should be used as a stress-test class rather than the main success criterion.

---

## Suggested Minimal First Run

Run only:

```text
K = {1, 2, 4, 8}
representation = {"t", "ta", "taw"}
rank_by = "prominence"
divide = 3
```

Then compare against:

* full waveform baseline
* spatial 4×4 AE + AH at 88× and 22×
* 1D AE + AH at 88× and 22×

If the results are promising, rerun:

```text
best top-K configs
full test set, no divide
```

---

## File Structure

Add:

```text
compression/event_extraction.py
compression/event_synthesis.py
downstream/run_eval_events.py
downstream/run_sweep_events.py
downstream/run_plot_events.py
```

Optionally add tests:

```text
tests/test_event_extraction.py
tests/test_event_synthesis.py
```

---

## Implementation Notes

* Keep the Ghost-FWL repo read-only.
* Use the existing monkey-patch approach for `VoxelDataset._load_voxel_grid`.
* Reuse the existing confusion-matrix macro-F1 computation.
* Skip slow per-pixel `scipy.find_peaks` downstream peak metric unless explicitly needed.
* Save fixed example plots for each config:

  * original waveform
  * top-K selected events
  * synthesized pseudo-waveform
* Save JSON outputs under:

```text
downstream/outputs/events/
```

Recommended output file naming:

```text
events_K{K}_{representation}_rank-{rank_by}.json
```

---

## Main Success Criterion

The experiment is successful if it answers:

> Can sparse top-K transport events explain most of the downstream Ghost-FWL performance?

The most important comparison is:

```text
Top-K (t,a,w) vs AE reconstruction vs full waveform
```

at comparable or lower dimensionality.

If top-K `(t,a,w)` is strong, the next research direction should be:

> Event-faithful full-waveform LiDAR compression.

If top-K `(t,a,w)` is weak, the next direction should be:

> Dense residual / tail-aware transport tokenization.

```
```
