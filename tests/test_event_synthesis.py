"""Tests for event -> pseudo-waveform synthesis (single + batch)."""
import numpy as np
import torch

from compression.event_synthesis import (synthesize_batch,
                                          synthesize_waveform_from_events)


def test_single_peak_position_and_norm():
    ev = np.array([[300.0, 1.0, 6.0]], np.float32)
    w = synthesize_waveform_from_events(ev, T=700, representation="taw")
    assert np.isclose(w.max(), 1.0, atol=1e-5)
    assert int(np.argmax(w)) == 300


def test_representation_t_ignores_amplitude_and_width():
    # two valid events with different a/w; "t" must use fixed values for both
    ev = np.array([[200.0, 0.3, 20.0], [400.0, 1.0, 4.0]], np.float32)
    w = synthesize_waveform_from_events(ev, T=700, representation="t",
                                        fixed_amplitude=1.0, fixed_width=4.0,
                                        normalize=False)
    # equal fixed amplitude -> both peaks reach the same height
    assert np.isclose(w[200], w[400], atol=1e-3)


def test_ta_uses_intensity_not_width():
    ev = np.array([[200.0, 0.4, 30.0], [400.0, 1.0, 30.0]], np.float32)
    w = synthesize_waveform_from_events(ev, T=700, representation="ta",
                                        fixed_width=4.0, normalize=False)
    assert w[400] > w[200]                      # honours intensity
    # width is fixed (4 bins) -> narrow, not the planted 30
    assert w[200 + 15] < 0.5 * w[200]


def test_invalid_events_skipped():
    ev = np.array([[300.0, 1.0, 6.0], [0.0, 0.0, 0.0]], np.float32)
    mask = np.array([True, False])
    w = synthesize_waveform_from_events(ev, valid_mask=mask, T=700, normalize=False)
    assert w[0] < 1e-6                          # the padded event at t=0 is ignored


def test_batch_matches_single():
    ev = np.array([[150.0, 1.0, 6.0], [450.0, 0.5, 10.0]], np.float32)
    single = synthesize_waveform_from_events(ev, T=700, representation="taw")
    batch = synthesize_batch(torch.from_numpy(ev[None]),
                             torch.ones(1, 2, dtype=torch.bool),
                             T=700, representation="taw")[0].numpy()
    assert np.allclose(single, batch, atol=1e-4)
