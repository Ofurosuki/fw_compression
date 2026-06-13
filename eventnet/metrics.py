"""Metrics: event-level F1 (fast, for model selection) and the PAPER-COMPLIANT
peak-level F1 (for the headline number).

Both reuse the Ghost-FWL repo's own scoring so numbers are comparable:
* ``calculate_metrics_from_confusion_matrix`` — F1 = 2TP/(2TP+FP+FN) per class.
* ``detect_peaks_in_voxel`` + ``evaluate_peaks`` — the repo's peak-level scoring
  (find_peaks(height=max*0.1, width=3) on the raw waveform, then a confusion
  matrix of pred-vs-annotation at those peak bins). This is exactly the
  population behind the paper's "F1-mean ~0.592" (see SCORE_DISCREPANCY.md).

F1-mean is the mean of per-class F1 over the SIGNAL classes {object, glass,
ghost}; Noise is kept in the confusion matrix as a competing class but excluded
from the average (``ignore_visualize_labels=[]``).
"""
from __future__ import annotations

import numpy as np

from hist_lidar.training.test_ViT3D import (
    calculate_metrics_from_confusion_matrix,
    detect_peaks_in_voxel,
    evaluate_peaks,
)

from eventnet.paths import NUM_CLASSES, SIGNAL_CLASSES, LABEL_MAP


# --------------------------------------------------------------------------- #
# Event-level (scored at the model's own extracted events; for val/model-select)
# --------------------------------------------------------------------------- #
def event_confusion(pred, labels, valid, num_classes=NUM_CLASSES):
    """pred/labels/valid: flat arrays (only valid entries scored). cm[true,pred]."""
    m = valid
    t = labels[m].astype(np.int64)
    p = pred[m].astype(np.int64)
    ok = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    idx = t[ok] * num_classes + p[ok]
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def f1_from_cm(cm):
    met = calculate_metrics_from_confusion_matrix(cm, ignore_labels=[])
    per = {LABEL_MAP[i]: float(met["f1"][i]) for i in range(NUM_CLASSES)}
    f1_mean = float(np.mean([met["f1"][i] for i in SIGNAL_CLASSES]))
    return f1_mean, per, met


# --------------------------------------------------------------------------- #
# Paper-compliant peak-level (scored at find_peaks positions on the raw wave)
# --------------------------------------------------------------------------- #
def paint_pred_dense(t_bin, w_bin, pred, valid, amp, T):
    """Reconstruct a dense (T, X, Y) predicted-label volume from per-event preds.

    For each valid event at (x, y, t_i) with width w_i and class c_i, fill
    ``pred_dense[t_i-r : t_i+r+1, x, y] = c_i`` with ``r = max(1, w_i/2)``
    (``initial_plan.md`` reconstruction). Overlaps resolved strongest-last so the
    highest-amplitude event wins. Background stays 0 (noise).

    All inputs are (X, Y, K). Returns uint8 (T, X, Y).
    """
    X, Y, K = t_bin.shape
    R = 40
    out = np.zeros((T, X * Y), dtype=np.uint8)

    v = valid.reshape(-1)
    t = t_bin.reshape(-1).astype(np.int64)
    r = np.maximum(1, (w_bin.reshape(-1) / 2.0)).astype(np.int64)
    c = pred.reshape(-1).astype(np.uint8)
    a = amp.reshape(-1)
    px = (np.repeat(np.arange(X), Y * K).reshape(X, Y, K).reshape(-1))
    py = (np.tile(np.repeat(np.arange(Y), K), X))
    p = px * Y + py

    sel = v & (c > 0)                                   # only paint signal classes
    t, r, c, a, p = t[sel], r[sel], c[sel], a[sel], p[sel]
    order = np.argsort(a)                               # strongest painted last
    t, r, c, p = t[order], r[order], c[order], p[order]

    off = np.arange(-R, R + 1)
    bins = t[:, None] + off[None, :]                    # (E, 2R+1)
    mask = (np.abs(off)[None, :] <= r[:, None]) & (bins >= 0) & (bins < T)
    bb = bins[mask]
    pp = np.broadcast_to(p[:, None], bins.shape)[mask]
    cc = np.broadcast_to(c[:, None], bins.shape)[mask]
    out[bb, pp] = cc                                    # last write (strongest) wins
    return out.reshape(T, X, Y)


def peak_eval_frame(pred_dense, ann_TXY, raw_TXY):
    """Run the repo's peak detection + scoring for one frame. Returns peak CM."""
    peaks = detect_peaks_in_voxel(raw_TXY)
    res = evaluate_peaks(pred_dense, ann_TXY, raw_TXY, peaks,
                         ignore_labels=[], num_classes=NUM_CLASSES)
    return res["peak_confusion_matrix"]


def peak_cm_from_cache(pred_dense, peaks, ann_at_peak, num_classes=NUM_CLASSES):
    """Fast peak CM using precomputed peak coords + annotation labels (identical
    population to ``evaluate_peaks`` with ignore_labels=[]). ``peaks`` (M,3) are
    (d, i, j) indices into the (T,X,Y) volume; ``pred_dense`` is (T,X,Y)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if len(peaks) == 0:
        return cm
    pk = peaks.astype(np.int64)
    pred_at = pred_dense[pk[:, 0], pk[:, 1], pk[:, 2]].astype(np.int64)
    ann = ann_at_peak.astype(np.int64)
    ok = (ann >= 0) & (ann < num_classes) & (pred_at >= 0) & (pred_at < num_classes)
    idx = ann[ok] * num_classes + pred_at[ok]
    cm += np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm
