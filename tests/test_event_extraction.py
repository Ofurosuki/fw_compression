"""Tests for top-K event extraction (scipy reference + GPU-vectorised batch)."""
import numpy as np
import torch

from compression.event_extraction import (extract_topk_events,
                                           extract_topk_events_batch)


def _two_peak_wave(T=700, t1=200, t2=400, a1=1.0, a2=0.6, s1=5.0, s2=8.0):
    t = np.arange(T)
    w = a1 * np.exp(-((t - t1) ** 2) / (2 * s1 ** 2))
    w += a2 * np.exp(-((t - t2) ** 2) / (2 * s2 ** 2))
    return w.astype(np.float32)


def test_reference_finds_two_peaks_sorted_by_time():
    w = _two_peak_wave()
    ev, valid = extract_topk_events(w, k=4)
    assert valid[:2].all() and not valid[2:].any()
    # sorted by time, positions close to the planted peaks
    assert abs(ev[0, 0] - 200) <= 2 and abs(ev[1, 0] - 400) <= 2
    # FWHM of a sigma=5 gaussian ~ 11.7 bins; sigma=8 ~ 18.8 bins
    assert 8 < ev[0, 2] < 16 and 14 < ev[1, 2] < 24


def test_reference_topk_keeps_strongest():
    w = _two_peak_wave(a1=1.0, a2=0.6)
    ev, valid = extract_topk_events(w, k=1, rank_by="height")
    assert valid.sum() == 1
    assert abs(ev[0, 0] - 200) <= 2          # the taller peak is kept


def test_reference_padding_for_flat_wave():
    ev, valid = extract_topk_events(np.zeros(700, np.float32), k=3)
    assert not valid.any() and np.allclose(ev, 0)


def test_batch_matches_reference_positions():
    waves = np.stack([_two_peak_wave(t1=150, t2=350),
                      _two_peak_wave(t1=250, t2=500, a1=0.9, a2=0.7)])
    wn = torch.from_numpy(waves)
    ev, valid = extract_topk_events_batch(wn, k=4)
    assert valid[:, :2].all() and not valid[:, 2:].any()
    assert abs(ev[0, 0, 0].item() - 150) <= 3
    assert abs(ev[0, 1, 0].item() - 350) <= 3
    assert abs(ev[1, 0, 0].item() - 250) <= 3
    assert abs(ev[1, 1, 0].item() - 500) <= 3


def test_batch_min_distance_nms():
    # two peaks closer than min_distance collapse to one event
    w = _two_peak_wave(t1=300, t2=304, a1=1.0, a2=0.9, s1=4, s2=4)
    ev, valid = extract_topk_events_batch(torch.from_numpy(w[None]), k=4, min_distance=10)
    assert valid[0].sum() == 1
