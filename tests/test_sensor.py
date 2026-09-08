"""The sensor calibration, and the claim it rests on.

``compression/sensor.py`` replaces 29 copies of four literals with one
description of the ghost dataset's LiDAR. Two of those literals were wrong,
not merely duplicated: the synthesis width and the NMS spacing were both set
as if the pulse were about 4 bins wide, and it is 11. So these tests do two
jobs -- pin the derived windows so they cannot drift back, and re-measure the
pulse off the dataset so the claim behind them stays checkable.
"""
import os

import numpy as np
import pytest
import torch

from compression.event_extraction import extract_topk_events_batch
from compression.event_synthesis import synthesize_batch
from compression.sensor import GHOST


class TestGhostCalibration:
    def test_pulse_is_eleven_bins_fwhm(self):
        assert GHOST.pulse_fwhm_bins == pytest.approx(11.0)
        assert GHOST.pulse_sigma_bins == pytest.approx(4.671270, abs=1e-6)

    def test_windows_follow_from_the_pulse(self):
        assert GHOST.min_distance_bins == pytest.approx(9.34254, abs=1e-5)
        assert GHOST.nms_bins == 9
        assert GHOST.width_search_bins == 10
        assert GHOST.max_fwhm_bins == 21

    def test_bin_is_one_nanosecond_of_round_trip(self):
        assert GHOST.bin_width_range_m == pytest.approx(0.1498962, abs=1e-7)
        assert GHOST.start_range_m == 0.0
        assert GHOST.bin_to_range_m(0) == pytest.approx(0.5 * GHOST.bin_width_range_m)
        # 700 bins of histogram is 105 m of range.
        assert GHOST.bin_to_range_m(699) == pytest.approx(104.8, abs=0.1)

    def test_a_metre_tolerance_is_only_three_bins_here(self):
        """Fog's 0.5 m recall threshold is 10 bins on that sensor and 3.3 on
        this one -- 0.42 of a pulse width there, 0.30 here. Copying the number
        across changes what is being asked for.
        """
        assert GHOST.range_m_to_bins(0.5) == pytest.approx(3.3356, abs=1e-4)
        assert GHOST.range_m_to_bins(0.5) / GHOST.pulse_fwhm_bins == pytest.approx(
            0.3032, abs=1e-4
        )


class TestSynthesisMatchesTheSensor:
    def test_fixed_width_synthesis_reproduces_the_pulse_width(self):
        """``ta`` carries no width, so it synthesises at the sensor's impulse
        response. The pulse that comes back must be that wide -- this is the
        property the old fixed_width=4.0 violated by a factor of 2.75.
        """
        events = torch.tensor([[[350.0, 1.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        wave = synthesize_batch(
            events, valid, T=700, representation="ta",
            fixed_amplitude=1.0, fixed_width=GHOST.pulse_fwhm_bins, normalize=False,
        )[0].numpy()

        above_half = np.flatnonzero(wave >= wave.max() / 2.0)
        fwhm = above_half[-1] - above_half[0] + 1
        assert fwhm == pytest.approx(GHOST.pulse_fwhm_bins, abs=1.0)

    def test_the_old_fixed_width_was_far_too_narrow(self):
        """Kept as a regression guard with the number in it: if anything ever
        sets 4.0 again, it is reconstructing a pulse this sensor cannot emit.
        """
        events = torch.tensor([[[350.0, 1.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        narrow = synthesize_batch(
            events, valid, T=700, representation="ta",
            fixed_amplitude=1.0, fixed_width=4.0, normalize=False,
        )[0].numpy()

        above_half = np.flatnonzero(narrow >= narrow.max() / 2.0)
        fwhm = above_half[-1] - above_half[0] + 1
        assert fwhm <= 5
        assert GHOST.pulse_fwhm_bins / fwhm > 2.0


class TestNmsSpacingSeparatesReturns:
    def _pulse_pair(self, separation, T=700):
        t = np.arange(T, dtype=np.float32)
        sigma = GHOST.pulse_sigma_bins
        w = np.exp(-((t - 300.0) ** 2) / (2 * sigma ** 2))
        w += 0.8 * np.exp(-((t - (300.0 + separation)) ** 2) / (2 * sigma ** 2))
        return torch.from_numpy(w[None])

    def test_returns_further_apart_than_the_window_are_kept_separate(self):
        events, valid = extract_topk_events_batch(
            self._pulse_pair(3.0 * GHOST.pulse_sigma_bins), k=4,
            min_distance=GHOST.nms_bins,
        )
        assert int(valid[0].sum()) == 2
        separation = (events[0, 1, 0] - events[0, 0, 0]).item()
        assert separation > GHOST.nms_bins

    def test_returns_inside_the_window_collapse_to_one(self):
        _events, valid = extract_topk_events_batch(
            self._pulse_pair(1.0 * GHOST.pulse_sigma_bins), k=4,
            min_distance=GHOST.nms_bins,
        )
        assert int(valid[0].sum()) == 1

    def test_the_old_spacing_split_one_return_into_several(self):
        """min_distance=3 is inside a single 11-bin pulse. On a clean pair it
        merely under-suppresses, but the budget it wastes is the point: with
        K=4 events per ray, slots spent on one return's shoulders are slots a
        real second return does not get.
        """
        assert GHOST.nms_bins > 3
        assert 3 < GHOST.pulse_fwhm_bins / 2  # inside the pulse's half-width


DATA_ROOT = "/data3/user/ikeda/ghost_dataset"


@pytest.mark.skipif(
    not os.path.isdir(DATA_ROOT), reason=f"ghost dataset not present at {DATA_ROOT}"
)
class TestPulseWidthAgainstTheDataset:
    """Re-derives the calibration from the data rather than trusting the
    constant, so the claim stays falsifiable as the dataset changes.
    """

    @staticmethod
    def _load_first_frame():
        import glob
        import sys

        sys.path.insert(0, "/data3/user/yoshida/fwl_mae/neurips2026/src")
        from hist_lidar.preprocess.custom_blosc2 import load_blosc2

        path = sorted(glob.glob(f"{DATA_ROOT}/gym_build/data/hist001/*_voxel.b2"))[0]
        return load_blosc2(path).astype(np.float32)

    @staticmethod
    def _median_fwhm(rays, limit=3000):
        widths = []
        for ray in rays[:limit]:
            peak = int(ray.argmax())
            half = ray[peak] / 2.0
            left = peak
            while left > 0 and ray[left - 1] >= half:
                left -= 1
            right = peak
            while right < len(ray) - 1 and ray[right + 1] >= half:
                right += 1
            widths.append(right - left + 1)
        return float(np.median(widths))

    def test_measured_pulse_matches_the_constant(self):
        voxel = self._load_first_frame()
        # Deepest accumulation band (rows 252-315, 144 shots) -- best SNR.
        rays = voxel[:, 252:316, :].reshape(-1, voxel.shape[-1])
        strong = rays[rays.max(axis=1) >= 40]
        assert strong.shape[0] > 1000, "not enough strong returns to measure"
        assert self._median_fwhm(strong) == pytest.approx(GHOST.pulse_fwhm_bins, abs=1.0)

    def test_pulse_width_does_not_depend_on_the_accumulation_band(self):
        """If it did, the width would be a property of the shot count rather
        than of the sensor, and a single constant would be wrong.
        """
        voxel = self._load_first_frame()
        shallow = voxel[:, 0:124, :].reshape(-1, voxel.shape[-1])  # 18 shots
        shallow = shallow[(shallow.max(axis=1) >= 8)]
        assert self._median_fwhm(shallow) == pytest.approx(
            GHOST.pulse_fwhm_bins, abs=1.5
        )
