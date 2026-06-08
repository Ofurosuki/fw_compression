"""Downstream Ghost-FWL evaluation hook (interface + stub).

The full Ghost-FWL pipeline operates on 3D voxel grids ``(H, W, T)`` with a
Transformer-MAE and requires the (currently unavailable) dataset and trained
checkpoints. Per the experiment plan, we *defer* running the real downstream eval
but provide a stable interface so it can be wired in later with no changes to the
training / evaluation code.

To enable real downstream evaluation:
  1. Implement ``GhostFWLEvaluator.evaluate`` to assemble reconstructed waveforms
     ``x_hat`` back into a voxel grid, run the Ghost-FWL model, and return the
     ghost detection/removal scores.
  2. Point ``ghost_fwl_repo`` at the Ghost-FWL checkout and load its model.

Until then, ``evaluate`` raises ``NotImplementedError`` and the evaluation script
falls back to a *proxy* ghost-detection score computed from peaks (see
``proxy_ghost_score``), which approximates the downstream signal on synthetic data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from compression.utils.metrics import detect_peaks, match_peaks


class GhostFWLEvaluator:
    """Stable interface to the (deferred) Ghost-FWL downstream pipeline."""

    def __init__(self, ghost_fwl_repo: Optional[str] = None, checkpoint: Optional[str] = None):
        self.ghost_fwl_repo = ghost_fwl_repo
        self.checkpoint = checkpoint
        self.available = False  # flip to True once real model wiring is added

    def evaluate(self, x_hat: np.ndarray, labels: Optional[List[Dict]] = None) -> Dict[str, float]:
        raise NotImplementedError(
            "Real Ghost-FWL downstream eval is not wired up (dataset/checkpoints "
            "unavailable). Use proxy_ghost_score for now."
        )


def proxy_ghost_score(
    x: np.ndarray,
    x_hat: np.ndarray,
    labels: List[Dict],
    tol: float = 10.0,
    **detect_kwargs,
) -> Dict[str, float]:
    """A lightweight stand-in for the Ghost-FWL ghost-detection score.

    Treats "is there a secondary (ghost) return preserved in x_hat?" as a binary
    detection problem per waveform, using ground-truth ghost labels. This is NOT
    the real Ghost-FWL metric -- it is a proxy that tracks whether transport /
    multi-peak structure survives compression, which is what the real pipeline
    ultimately keys on.

    Returns precision/recall/F1 of ghost-return detection on reconstructions,
    plus the same on the originals as an upper-bound reference.
    """

    def ghost_detected(wave, label):
        pred = detect_peaks(wave, **detect_kwargs).astype(float)
        ref = np.asarray(label["peak_positions"], dtype=float)
        if not bool(label.get("ghost", False)) or len(ref) < 2:
            # no ghost present: detection = did we (wrongly) find a 2nd strong peak?
            return len(pred) >= 2, False  # (predicted_ghost, has_ghost)
        inten = np.asarray(label["peak_intensities"], dtype=float)
        primary = int(np.argmax(inten))
        ghost_idx = [k for k in range(len(ref)) if k != primary]
        matches, _, _ = match_peaks(ref, pred, tol=tol)
        matched_true = {ti for ti, _ in matches}
        found = any(k in matched_true for k in ghost_idx)
        return found, True

    def prf(waves):
        tp = fp = fn = 0
        for i in range(len(waves)):
            pred_g, has_g = ghost_detected(waves[i], labels[i])
            if has_g and pred_g:
                tp += 1
            elif has_g and not pred_g:
                fn += 1
            elif (not has_g) and pred_g:
                fp += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        return {"precision": prec, "recall": rec, "f1": f1}

    recon = prf(x_hat)
    orig = prf(x)
    return {
        "ghost_proxy_precision": recon["precision"],
        "ghost_proxy_recall": recon["recall"],
        "ghost_proxy_f1": recon["f1"],
        "ghost_proxy_f1_upperbound": orig["f1"],
        "ghost_proxy_f1_retention": recon["f1"] / max(orig["f1"], 1e-8),
    }
