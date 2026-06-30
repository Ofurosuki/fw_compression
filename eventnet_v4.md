# Instruction: Implement Sparse Event-ToPM style network for FW LiDAR event tokens

## Background

Current experiments show that `taw K=4` contains almost sufficient information for the Ghost-FWL downstream task when ToPM is retrained on Gaussian pseudo-waveforms.

However, the current retraining path is indirect:

```text
raw waveform
→ top-K events (t, a, w)
→ Gaussian pseudo-waveform T=700
→ ToPM retrain

This verifies that event tokens retain downstream information, but it is not a true token-native architecture.

The next goal is to implement a Sparse Event-ToPM style network:

raw waveform
→ top-K events (t_i, a_i, w_i)
→ sparse event tokens at coordinates (h, w, t_i)
→ ToPM-like spatial-range network
→ per-event / per-peak class prediction

The motivation is aligned with prior full-waveform LiDAR work: LiDAR point clouds are compressed outputs of richer waveform/transient signals, and upstream DSP/representation choices affect downstream task quality . Recent neural DSP work also treats waveform interpretation as a spatial-temporal contextual problem rather than independent per-ray peak finding .

High-level goal

Implement a prototype token-native model that consumes top-K LiDAR event tokens directly:

events[h, w, k] = (t_i, a_i, w_i, rank_i, valid_i)

and predicts semantic labels for each event:

noise / object / glass / ghost

without reconstructing a dense T=700 pseudo-waveform.

This model should test the hypothesis:

taw tokens contain sufficient information, but EventNet underuses them because it flattens event/range structure into 2D channels. A sparse ToPM-like model should preserve event coordinates (h,w,t) and process spatial-range structure more naturally.

Repository / environment assumptions

Use the working repository:

/home/yoshida/fw_compression

Branch:

feature/remove_falsepositive

Do not modify the original Ghost-FWL / ToPM repository:

/data3/user/yoshida/fwl_mae/neurips2026

Use the existing fw_compression venv, not the minimal ai-task-folder venv:

/home/yoshida/fw_compression/.venv/bin/python

Set PYTHONPATH when using Ghost-FWL repo utilities:

export PYTHONPATH=/data3/user/yoshida/fwl_mae/neurips2026/src

Avoid uv run for repo-dependent training code unless it is already known to use the correct cu128 torch environment.

Existing code to reuse

Reuse or inspect the following existing files:

downstream/run_retrain.py
downstream/cache_repr.py
downstream/run_eval_peak.py
eventnet/
FW_Event_Net/
downstream/RETRAIN_RESULTS.md

Important existing mechanisms:

cache_repr.py already precomputes event / pseudo-waveform representations.
run_eval_peak.py already implements paper-metric peak-level F1.
EventNet already has logic for top-K event extraction, labels, rank embeddings, and event-level segmentation.
ToPM retrain proved that taw K=4 is information-sufficient, so the goal is not representation validation but architecture validation.
New implementation target

Create a new module, for example:

eventnet/sparse_event_topm.py
eventnet/train_sparse_event_topm.py
eventnet/eval_sparse_event_topm.py

or similar names consistent with the repo style.

The implementation should include:

Dataset / dataloader for top-K events.
Sparse Event-ToPM model.
Training loop.
Peak-level evaluation.
Minimal ablation config for ta and taw.
1. Input representation
Input shape

The model should consume batched top-K event tensors:

events: FloatTensor[B, H, W, K, F]
valid:  BoolTensor[B, H, W, K]
labels: LongTensor[B, H, W, K]

For taw, features should include:

t_norm  = t_i / T
a       = peak amplitude, per-ray max-normalized
w_norm  = FWHM / T or FWHM / reasonable_width_scale
rank    = k / K or learned rank embedding
valid   = whether this event exists

For ta, features should include:

t_norm, a, rank, valid

Do not synthesize Gaussian waveforms.

Do not create dense [B,H,W,T] waveform input except optionally for debugging.

2. Label definition

Use the same event labeling rule as EventNet:

label_i = annotation at peak bin t_i

Classes:

0: noise/background
1: object
2: glass
3: ghost

Training loss may include noise as a competing class, but headline macro F1 should average over:

object, glass, ghost

consistent with previous evaluation.

3. Architecture: Sparse Event-ToPM style

Implement a model that preserves event/range structure better than EventNet V2.

Baseline EventNet V2 structure

Current V2 roughly does:

event MLP
→ per-pixel cross-event attention
→ flatten K into channels
→ 2D U-Net

This collapses range/event structure into channels before spatial processing.

New desired structure

The new model should treat each event as a token with coordinate:

(h, w, t_i)

and feature:

(a_i, w_i, rank_i, optional alpha_i)

Conceptually:

event tokens at sparse 3D coordinates
→ local spatial-range attention / sparse 3D blocks
→ hierarchical encoder
→ per-event classifier

The first prototype does not need to use MinkowskiEngine or external sparse convolution libraries unless already available. Prefer a pure PyTorch implementation.

Recommended first prototype: Windowed spatial-event attention

Instead of building a true sparse 3D engine, implement a practical approximation:

Input: [B,H,W,K,C]
For each local spatial window, e.g. 4×4 or 8×8 pixels:
    collect all K events inside the window
    tokens = window_H * window_W * K
    apply self-attention over these tokens
Repeat with shifted windows or multiple blocks
Then classify each event token

This preserves:

spatial neighborhood
event rank
range/time embedding
event-to-event interactions across nearby rays

Unlike V2, this lets attention compare:

event at pixel (h,w), depth t_i
with event at nearby pixel (h+1,w), depth t_j

which is closer to ToPM’s spatial-range inductive bias.

Model sketch

Implement something like:

class SparseEventToPM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int = 8,
        k_events: int = 4,
        num_classes: int = 4,
    ):
        ...

Forward:

def forward(self, events, valid):
    # events: [B,H,W,K,F]
    # valid:  [B,H,W,K]
    x = event_mlp(events)                         # [B,H,W,K,C]
    x += rank_embedding                           # [K,C]
    x += time_embedding(t_norm)                   # [B,H,W,K,C]
    x += spatial_position_embedding(h,w)          # optional

    x = windowed_spatial_event_attention(x, valid)
    x = optional_hierarchical_down_up_blocks(x)
    logits = classifier(x)                        # [B,H,W,K,num_classes]
    return logits

Start simple. The first working version can be:

event MLP
→ 4 blocks of windowed spatial-event attention
→ FFN
→ per-event classifier

Then add hierarchy if needed.

4. Attention details
Tokenization inside each window

For each spatial window:

window tokens = H_win × W_win × K events

Example:

window_size = 8, K=4
tokens per window = 8×8×4 = 256

This is manageable.

Attention mask should ignore invalid events.

Use standard multi-head self-attention:

nn.MultiheadAttention(batch_first=True)

or a custom efficient implementation.

Positional encoding

Each token must know:

relative h position inside window
relative w position inside window
t_norm or discretized t bin
rank k

Use a combination of:

learned rank embedding
MLP(t_norm)
learned 2D spatial embedding per window coordinate
optional sinusoidal t embedding

Do not rely only on raw t as a scalar.

5. Loss

Use event-level cross entropy or focal loss.

Because class imbalance is severe, start with the same class treatment as EventNet if available.

Recommended initial loss:

weighted CE + Dice over signal classes

or reuse EventNet’s existing loss implementation.

Important:

Include noise/background in softmax.
Macro F1 headline excludes noise.
Mask invalid events out of the loss.
Do not ignore noise entirely, because previous experiments showed ignoring noise inflates F1.
6. Evaluation

Implement evaluation comparable to EventNet:

object / glass / ghost per-class F1
macro F1 over signal classes

Report both:

event-level F1
peak-level-compatible F1 if possible

The most important comparison is against:

EventNet V2 taw K=4
ToPM retrain taw K=4
ToPM retrain full waveform
ToPM retrain ta K=4

Expected reference numbers:

ToPM retrain full peak F1 ≈ 0.595
ToPM retrain taw K=4 peak F1 ≈ 0.582
ToPM retrain ta K=4 peak F1 ≈ 0.574

EventNet V2 taw K=4 peak F1 ≈ 0.555

The goal of Sparse Event-ToPM is to close the gap between:

EventNet V2 taw K=4 ≈ 0.555
and
ToPM retrain taw K=4 ≈ 0.582

A meaningful target is:

Sparse Event-ToPM taw K=4 peak F1 >= 0.57

A strong target is:

Sparse Event-ToPM taw K=4 peak F1 ≈ 0.58
7. Initial experiments

Run the following minimum experiments.

Experiment 1: taw K=4
input = (t,a,w)
K = 4
model = SparseEventToPM
seed = 42

Compare against EventNet V2 and ToPM retrain.

Experiment 2: ta K=4
input = (t,a)
K = 4
model = SparseEventToPM
seed = 42

This checks whether the architecture preserves the same class tradeoff seen in ToPM retrain:

taw helps ghost
ta helps glass
Experiment 3: per-class breakdown

Always report:

object F1
glass F1
ghost F1
macro F1

Do not report only macro F1.

Important expected pattern from ToPM retrain:

full: 0.770 / 0.300 / 0.715
taw:  0.754 / 0.271 / 0.722
ta:   0.738 / 0.339 / 0.645
       object / glass / ghost

Check whether Sparse Event-ToPM reproduces this pattern.

8. Diagnostics to include

Add optional analysis scripts or logging for:

Confusion matrix.
Per-scene F1.
Per-class F1.
taw correct / ta wrong cases.
ta correct / taw wrong cases.
Performance as a function of event rank.
Performance as a function of amplitude bin.
Performance as a function of width bin.

Especially inspect:

bright ghost events
glass events where taw fails but ta succeeds

These are important because distribution analysis showed:

width helps ghost when amplitude cannot flag it
width hurts glass due to scene-dependent non-transferable cues
9. Implementation constraints
Do not overwrite existing results.
Save all new checkpoints under a new directory, e.g.:
outputs/sparse_event_topm/
Save configs with every run.
Save metrics as JSON and Markdown.
Use deterministic seed where possible.
Keep model small enough for one GPU.
Avoid adding heavyweight dependencies unless necessary.
If adding a dependency, document it clearly.
10. Suggested file structure

Create:

eventnet/sparse_event_topm.py
eventnet/train_sparse_event_topm.py
eventnet/eval_sparse_event_topm.py
eventnet/configs/sparse_event_topm_taw_k4.yaml
eventnet/configs/sparse_event_topm_ta_k4.yaml

Optional:

eventnet/diag_sparse_event_topm_errors.py
11. Success criteria

The implementation is successful if:

It trains end-to-end on cached top-K events.
It predicts [B,H,W,K,num_classes] logits.
It evaluates object/glass/ghost F1 correctly.
It reproduces or exceeds EventNet V2.
It reduces the gap toward ToPM retrain on pseudo-waveform.

Minimum success:

Sparse Event-ToPM taw K=4 > EventNet V2 taw K=4

Strong success:

Sparse Event-ToPM taw K=4 ≈ ToPM retrain taw K=4

Scientific success:

The model clarifies whether the remaining EventNet gap is architectural,
i.e. due to flattening event/range structure too early.
12. Important interpretation

Do not claim this proves full waveform is unnecessary for all LiDAR tasks.

The current claim is narrower:

For Ghost-FWL downstream segmentation, top-K event tokens retain most of the information needed by ToPM. A token-native sparse spatial-range architecture should be able to exploit those tokens without reconstructing dense pseudo-waveforms.

Future conditions such as fog, pileup, overlapping peaks, low-SNR, and K>4 may require richer tokens or residual waveform information.


