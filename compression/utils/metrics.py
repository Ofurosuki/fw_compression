"""Evaluation metrics for waveform reconstruction.

All functions operate on numpy arrays. Batched helpers aggregate over a dataset.
The key research-relevant metrics are *peak/transport* metrics (localization
error, peak-count preservation) -- not just waveform MSE -- because the
hypothesis is that depth-oriented compression loses ghost/multipath structure.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def waveform_mse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-waveform MSE. x, x_hat: [N, T] -> [N]."""
    return np.mean((x - x_hat) ** 2, axis=1)


def energy_error(x: np.ndarray, x_hat: np.ndarray) -> Dict[str, np.ndarray]:
    """Absolute and relative energy (sum) error, per waveform."""
    e = x.sum(axis=1)
    e_hat = x_hat.sum(axis=1)
    abs_err = np.abs(e_hat - e)
    rel_err = abs_err / np.clip(np.abs(e), 1e-8, None)
    return {"energy_abs": abs_err, "energy_rel": rel_err}


def detect_peaks(
    wave: np.ndarray,
    smooth: float = 2.0,
    prominence: float = 0.05,
    distance: int = 20,
    rel_height: float = 0.1,
) -> np.ndarray:
    """Detect peak positions in a single waveform.

    Light Gaussian smoothing suppresses Poisson/read noise spikes, then
    prominence- and relative-height-thresholded peak finding recovers the true
    returns. Defaults are calibrated on the synthetic generator so that detection
    on a *clean original* recovers the ground-truth peaks (count_acc ~0.98,
    recall ~0.996), making compression-induced degradation interpretable.

    Returns an array of integer peak indices (in the original, unsmoothed frame).
    """
    s = gaussian_filter1d(wave, smooth) if smooth and smooth > 0 else wave
    height = rel_height * float(s.max()) if rel_height > 0 else None
    peaks, _ = find_peaks(s, prominence=prominence, distance=distance, height=height)
    return peaks


def detect_peaks_with_prominence(
    wave: np.ndarray,
    smooth: float = 2.0,
    prominence: float = 0.05,
    distance: int = 20,
    rel_height: float = 0.1,
):
    """Like :func:`detect_peaks` but also return each peak's prominence.

    Prominence is the natural severity weight for a *spurious* peak: a tall fake
    return misleads the downstream detector far more than a low ripple does.
    Returns ``(positions, prominences)`` aligned by index.
    """
    s = gaussian_filter1d(wave, smooth) if smooth and smooth > 0 else wave
    height = rel_height * float(s.max()) if rel_height > 0 else None
    peaks, props = find_peaks(s, prominence=prominence, distance=distance, height=height)
    proms = props.get("prominences", np.zeros(len(peaks), dtype=float))
    return peaks, np.asarray(proms, dtype=float)


def false_peak_metrics(
    x: np.ndarray,
    x_hat: np.ndarray,
    labels: Optional[List[Dict]] = None,
    tol: float = 10.0,
    **detect_kwargs,
) -> Dict[str, float]:
    """Waveform-level *spurious* (nonexistent) peak metrics.

    A peak detected on the reconstruction ``x_hat`` that matches **no** reference
    peak within ``tol`` bins is *spurious* -- it did not exist in the original and
    will create false detections downstream. This is the false-positive complement
    of :func:`peak_metrics`' recall, but reported as headline numbers and weighted
    by severity. Reference peaks come from ground-truth ``labels`` when available,
    else from :func:`detect_peaks` on the clean original ``x``.

    Returns:
      - spurious_per_wave   : mean # spurious peaks per waveform (the headline count)
      - spurious_rate       : spurious / all detected peaks  (== 1 - peak_precision)
      - spurious_wave_frac  : fraction of waveforms with >=1 spurious peak
      - spurious_prom_ratio : mean over waveforms of (sum spurious prominence /
                              sum reference prominence) -- severity-weighted, so a
                              tall hallucinated peak counts more than a ripple
      - false_ghost_rate    : fraction of single-/non-ghost waveforms that gain a
                              spurious extra return (the downstream-relevant case:
                              hallucinating a ghost where none exists). NaN without labels.
    """
    N = x.shape[0]
    spur_count: List[int] = []
    spur_prom_ratio: List[float] = []
    waves_with_spur = 0
    total_pred = 0
    total_spur = 0
    nonghost_n = 0
    false_ghost = 0

    for i in range(N):
        if labels is not None:
            ref = np.asarray(labels[i]["peak_positions"], dtype=float)
            is_ghost = bool(labels[i].get("ghost", False))
        else:
            ref = detect_peaks(x[i], **detect_kwargs).astype(float)
            is_ghost = False

        pred, proms = detect_peaks_with_prominence(x_hat[i], **detect_kwargs)
        pred = pred.astype(float)
        # self-consistent normaliser: prominence scale of the reference returns,
        # measured the same way on the original waveform
        _, ref_proms = detect_peaks_with_prominence(x[i], **detect_kwargs)
        ref_prom_sum = float(ref_proms.sum())

        # a predicted peak is spurious if no reference peak lies within tol bins
        if len(ref) == 0:
            spurious = list(range(len(pred)))
        else:
            spurious = [j for j, p in enumerate(pred) if np.min(np.abs(ref - p)) > tol]
        ns = len(spurious)

        spur_count.append(ns)
        total_pred += len(pred)
        total_spur += ns
        if ns > 0:
            waves_with_spur += 1
        sp = float(proms[spurious].sum()) if ns else 0.0
        spur_prom_ratio.append(sp / max(ref_prom_sum, 1e-8))

        # false-ghost: a non-ghost waveform that gains a hallucinated extra return
        if labels is not None and not is_ghost:
            nonghost_n += 1
            if ns >= 1:
                false_ghost += 1

    return {
        "spurious_per_wave": float(np.mean(spur_count)) if spur_count else 0.0,
        "spurious_rate": float(total_spur / max(total_pred, 1)),
        "spurious_wave_frac": float(waves_with_spur / max(N, 1)),
        "spurious_prom_ratio": float(np.mean(spur_prom_ratio)) if spur_prom_ratio else 0.0,
        "false_ghost_rate": float(false_ghost / nonghost_n) if nonghost_n else float("nan"),
    }


def match_peaks(true_pos: np.ndarray, pred_pos: np.ndarray, tol: float = 10.0):
    """Greedy nearest-neighbour matching between true and predicted peak positions.

    Returns (matches, n_matched, errors) where errors are |true - pred| for matched
    pairs within ``tol`` bins.
    """
    if len(true_pos) == 0 or len(pred_pos) == 0:
        return [], 0, np.array([])
    true_pos = np.asarray(true_pos, dtype=float)
    pred_pos = np.asarray(pred_pos, dtype=float)
    used_pred = set()
    matches = []
    errors = []
    # match each true peak to the closest unused predicted peak within tol
    order = np.argsort([-1] * len(true_pos))  # stable order
    for ti in order:
        t = true_pos[ti]
        best_j, best_d = -1, np.inf
        for j, p in enumerate(pred_pos):
            if j in used_pred:
                continue
            d = abs(t - p)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= tol:
            used_pred.add(best_j)
            matches.append((ti, best_j))
            errors.append(best_d)
    return matches, len(matches), np.array(errors)


def peak_metrics(
    x: np.ndarray,
    x_hat: np.ndarray,
    labels: Optional[List[Dict]] = None,
    tol: float = 10.0,
    **detect_kwargs,
) -> Dict[str, float]:
    """Aggregate peak/transport metrics over a batch.

    For each waveform we obtain *reference* peaks (from ground-truth ``labels`` if
    available, else detected on the clean-ish original ``x``) and *reconstructed*
    peaks (detected on ``x_hat``), then compute:

    - peak_loc_error_mean : mean localization error over matched peaks (bins)
    - peak_count_mae       : mean |#ref - #recon|
    - peak_count_acc       : fraction with exactly matching peak count
    - peak_recall          : matched / total reference peaks
    - peak_precision       : matched / total reconstructed peaks
    - ghost_recall         : recall restricted to waveforms flagged ghost (if labels)
    """
    N = x.shape[0]
    loc_errors = []
    count_abs = []
    count_exact = []
    total_ref = 0
    total_pred = 0
    total_matched = 0
    ghost_ref = 0
    ghost_matched = 0

    for i in range(N):
        if labels is not None:
            ref = np.asarray(labels[i]["peak_positions"], dtype=float)
            is_ghost = bool(labels[i].get("ghost", False))
        else:
            ref = detect_peaks(x[i], **detect_kwargs).astype(float)
            is_ghost = False
        pred = detect_peaks(x_hat[i], **detect_kwargs).astype(float)

        _, n_matched, errs = match_peaks(ref, pred, tol=tol)
        if len(errs) > 0:
            loc_errors.append(errs)
        count_abs.append(abs(len(ref) - len(pred)))
        count_exact.append(1.0 if len(ref) == len(pred) else 0.0)
        total_ref += len(ref)
        total_pred += len(pred)
        total_matched += n_matched

        if is_ghost and len(ref) >= 2:
            # ghost peaks = all reference peaks except the strongest (primary)
            inten = np.asarray(labels[i]["peak_intensities"], dtype=float)
            primary = int(np.argmax(inten))
            ghost_idx = [k for k in range(len(ref)) if k != primary]
            ghost_ref += len(ghost_idx)
            matches, _, _ = match_peaks(ref, pred, tol=tol)
            matched_true = {ti for ti, _ in matches}
            ghost_matched += sum(1 for k in ghost_idx if k in matched_true)

    loc_all = np.concatenate(loc_errors) if loc_errors else np.array([0.0])
    return {
        "peak_loc_error_mean": float(loc_all.mean()),
        "peak_loc_error_median": float(np.median(loc_all)),
        "peak_count_mae": float(np.mean(count_abs)),
        "peak_count_acc": float(np.mean(count_exact)),
        "peak_recall": float(total_matched / max(total_ref, 1)),
        "peak_precision": float(total_matched / max(total_pred, 1)),
        "ghost_recall": float(ghost_matched / ghost_ref) if ghost_ref > 0 else float("nan"),
    }


def measure_peak(wave: np.ndarray, pos: float, half_window: int = 25, refine: int = 3) -> Dict:
    """Model-free per-peak measurement (no known IRF).

    Around ``pos`` recover (height, area, FWHM) by local analysis -- used for real
    data where the system IRF is unknown so a parametric (EMG) fit is not available.
    The same routine measures the *reference* params on the original waveform and the
    *reconstructed* params on ``x_hat`` so the comparison is self-consistent.

    The peak location is refined only within ``±refine`` bins of ``pos`` (so a faint
    return is NOT snapped onto a brighter neighbour), while the local baseline is
    estimated over the wider ``±half_window``. Area/FWHM are confined to the
    half-maximum region around the peak, avoiding contamination from nearby peaks.

    Returns dict: position (refined local max), height (above local baseline),
    area (sum above baseline over the half-max region), fwhm (half-max width, bins), ok.
    """
    T = len(wave)
    pos = int(round(float(pos)))
    lo, hi = max(0, pos - half_window), min(T, pos + half_window + 1)
    seg = wave[lo:hi].astype(float)
    if seg.size == 0:
        return dict(position=float(pos), height=0.0, area=0.0, fwhm=0.0, ok=False)
    baseline = float(np.percentile(seg, 10))
    rlo, rhi = max(0, pos - refine), min(T, pos + refine + 1)
    local = int(np.argmax(wave[rlo:rhi])) + rlo
    height = float(wave[local] - baseline)
    if height <= 1e-9:
        return dict(position=float(local), height=0.0, area=0.0, fwhm=0.0, ok=False)
    half = baseline + 0.5 * height
    l = local
    while l > lo and wave[l] > half:
        l -= 1
    r = local
    while r < hi - 1 and wave[r] > half:
        r += 1
    fwhm = float(max(r - l, 1))
    area = float(np.clip(wave[l : r + 1].astype(float) - baseline, 0.0, None).sum())
    return dict(position=float(local), height=height, area=area, fwhm=fwhm, ok=True)


def _greedy_match_by_pos(gt_pos, pred_pos, tol):
    """Greedy nearest matching; returns dict gt_idx -> pred_idx for matched pairs."""
    matched = {}
    used = set()
    for gi, g in enumerate(gt_pos):
        best_j, best_d = -1, np.inf
        for j, p in enumerate(pred_pos):
            if j in used:
                continue
            d = abs(g - p)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= tol:
            used.add(best_j)
            matched[gi] = best_j
    return matched


def per_peak_param_metrics(
    x_hat: np.ndarray,
    labels: List[Dict],
    tau: float,
    width_threshold: float,
    tol: float = 12.0,
    n_max: Optional[int] = None,
    **detect_kwargs,
) -> Dict[str, float]:
    """Per-peak (position / intensity / FWHM) preservation, split narrow vs wide.

    For each waveform: detect peaks on ``x_hat``, fit the known system IRF (EMG,
    tail ``tau`` fixed) to recover (position, intensity=area, FWHM), then match to
    ground-truth peaks. Peaks are grouped by GT FWHM into *narrow* (high-freq) and
    *wide* (low-freq) to expose each encoder's frequency-dependent behaviour.

    Returns, for each group in {all, narrow, wide}:
      <grp>_pos_err          mean |pos_hat - pos_gt|            (bins)
      <grp>_intensity_relerr mean |I_hat - I_gt| / I_gt
      <grp>_fwhm_relerr      mean |w_hat - w_gt| / w_gt
      <grp>_recall           matched GT peaks / GT peaks
      <grp>_n                number of GT peaks
    plus <grp>_precision (matched / detected), width_threshold.
    """
    from compression.utils.peak_fit import fit_peaks

    N = x_hat.shape[0] if n_max is None else min(n_max, x_hat.shape[0])
    G = {g: {"pos": [], "int": [], "fwhm": [], "n_gt": 0, "matched": 0} for g in ("all", "narrow", "wide")}
    n_pred_total = 0

    for i in range(N):
        gt = labels[i]
        gt_pos = np.asarray(gt["peak_positions"], dtype=float)
        gt_int = np.asarray(gt["peak_intensities"], dtype=float)
        gt_fwhm = np.asarray(gt["peak_fwhm"], dtype=float)
        pred_pos = detect_peaks(x_hat[i], **detect_kwargs).astype(float)
        n_pred_total += len(pred_pos)
        fits = fit_peaks(x_hat[i], pred_pos, tau) if len(pred_pos) else []
        fit_pos = np.array([f["position"] for f in fits], dtype=float)
        matched = _greedy_match_by_pos(gt_pos, fit_pos, tol)

        for gi in range(len(gt_pos)):
            grp = "narrow" if gt_fwhm[gi] < width_threshold else "wide"
            for key in (grp, "all"):
                G[key]["n_gt"] += 1
            if gi in matched:
                f = fits[matched[gi]]
                pe = abs(f["position"] - gt_pos[gi])
                ie = abs(f["intensity"] - gt_int[gi]) / max(gt_int[gi], 1e-8)
                we = abs(f["fwhm"] - gt_fwhm[gi]) / max(gt_fwhm[gi], 1e-8)
                for key in (grp, "all"):
                    G[key]["matched"] += 1
                    G[key]["pos"].append(pe)
                    G[key]["int"].append(ie)
                    G[key]["fwhm"].append(we)

    out = {"width_threshold": float(width_threshold)}
    for g, d in G.items():
        m = lambda lst: float(np.mean(lst)) if lst else float("nan")
        out[f"{g}_pos_err"] = m(d["pos"])
        out[f"{g}_intensity_relerr"] = m(d["int"])
        out[f"{g}_fwhm_relerr"] = m(d["fwhm"])
        out[f"{g}_recall"] = float(d["matched"] / max(d["n_gt"], 1))
        out[f"{g}_n"] = int(d["n_gt"])
    out["all_precision"] = float(G["all"]["matched"] / max(n_pred_total, 1))
    return out


def real_peak_metrics(
    x_hat: np.ndarray,
    labels: List[Dict],
    width_threshold: float,
    tol: float = 12.0,
    half_window: int = 25,
    n_max: Optional[int] = None,
    **detect_kwargs,
) -> Dict[str, float]:
    """Real-data per-peak preservation + per-class survival (no known IRF).

    Reference peak params (position / intensity=area / FWHM) were measured on the
    *original* (uncompressed) waveform and stored in ``labels`` -- so this reports
    how well compression preserves the original return, NOT accuracy vs a parametric
    truth (which does not exist for real data). For each waveform we detect peaks on
    ``x_hat``, measure them model-free (``measure_peak``), match to the reference
    peaks, and compute:

    - narrow/wide param fidelity (split by reference FWHM measured on the original):
      ``<grp>_pos_err`` (bins), ``<grp>_intensity_relerr``, ``<grp>_fwhm_relerr``,
      ``<grp>_recall``, ``<grp>_n``  for grp in {all, narrow, wide}.
    - per-class peak survival recall: ``<cls>_recall`` / ``<cls>_n`` for cls in
      {object, glass, ghost} -- the **ghost** recall is the headline transport metric
      (ghosts are faint secondary returns, so the first to vanish under compression).
    """
    N = x_hat.shape[0] if n_max is None else min(n_max, x_hat.shape[0])
    G = {g: {"pos": [], "int": [], "fwhm": [], "n_gt": 0, "matched": 0} for g in ("all", "narrow", "wide")}
    C = {c: {"n_gt": 0, "matched": 0} for c in ("object", "glass", "ghost")}
    n_pred_total = 0

    for i in range(N):
        gt = labels[i]
        gt_pos = np.asarray(gt["peak_positions"], dtype=float)
        gt_int = np.asarray(gt["peak_intensities"], dtype=float)
        gt_fwhm = np.asarray(gt["peak_fwhm"], dtype=float)
        gt_cls = list(gt["peak_classes"])

        pred_pos = detect_peaks(x_hat[i], **detect_kwargs).astype(float)
        n_pred_total += len(pred_pos)
        meas = [measure_peak(x_hat[i], p, half_window=half_window) for p in pred_pos]
        fit_pos = np.array([m["position"] for m in meas], dtype=float)
        matched = _greedy_match_by_pos(gt_pos, fit_pos, tol)

        for gi in range(len(gt_pos)):
            grp = "narrow" if gt_fwhm[gi] < width_threshold else "wide"
            for key in (grp, "all"):
                G[key]["n_gt"] += 1
            cls = gt_cls[gi]
            if cls in C:
                C[cls]["n_gt"] += 1
            if gi in matched:
                m = meas[matched[gi]]
                pe = abs(m["position"] - gt_pos[gi])
                ie = abs(m["area"] - gt_int[gi]) / max(gt_int[gi], 1e-8)
                we = abs(m["fwhm"] - gt_fwhm[gi]) / max(gt_fwhm[gi], 1e-8)
                for key in (grp, "all"):
                    G[key]["matched"] += 1
                    G[key]["pos"].append(pe)
                    G[key]["int"].append(ie)
                    G[key]["fwhm"].append(we)
                if cls in C:
                    C[cls]["matched"] += 1

    out = {"width_threshold": float(width_threshold)}
    mean = lambda lst: float(np.mean(lst)) if lst else float("nan")
    for g, d in G.items():
        out[f"{g}_pos_err"] = mean(d["pos"])
        out[f"{g}_intensity_relerr"] = mean(d["int"])
        out[f"{g}_fwhm_relerr"] = mean(d["fwhm"])
        out[f"{g}_recall"] = float(d["matched"] / max(d["n_gt"], 1))
        out[f"{g}_n"] = int(d["n_gt"])
    for c, d in C.items():
        out[f"{c}_recall"] = float(d["matched"] / max(d["n_gt"], 1)) if d["n_gt"] else float("nan")
        out[f"{c}_n"] = int(d["n_gt"])
    out["all_precision"] = float(G["all"]["matched"] / max(n_pred_total, 1))
    return out


def gt_width_threshold(labels: List[Dict]) -> float:
    """Median GT FWHM over all peaks -- the narrow/wide split point (shared across encoders)."""
    all_fwhm = np.concatenate([np.asarray(l["peak_fwhm"], dtype=float) for l in labels if len(l["peak_fwhm"])])
    return float(np.median(all_fwhm))


def aggregate_metrics(
    x: np.ndarray,
    x_hat: np.ndarray,
    labels: Optional[List[Dict]] = None,
    T: Optional[int] = None,
    K: Optional[int] = None,
    tol: float = 10.0,
    **detect_kwargs,
) -> Dict[str, float]:
    """Full metric bundle for a (x, x_hat) pair set."""
    mse = waveform_mse(x, x_hat)
    en = energy_error(x, x_hat)
    out = {
        "waveform_mse_mean": float(mse.mean()),
        "waveform_mse_std": float(mse.std()),
        "energy_abs_mean": float(en["energy_abs"].mean()),
        "energy_rel_mean": float(en["energy_rel"].mean()),
    }
    out.update(peak_metrics(x, x_hat, labels=labels, tol=tol, **detect_kwargs))
    out.update(false_peak_metrics(x, x_hat, labels=labels, tol=tol, **detect_kwargs))
    if T is not None and K is not None:
        out["compression_ratio"] = float(T) / float(K)
        out["T"] = int(T)
        out["K"] = int(K)
    return out
