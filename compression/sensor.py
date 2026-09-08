"""The ghost dataset's LiDAR, and the extraction windows that follow from it.

The extractor's windows were written as bare bin counts spread over the
downstream scripts -- an NMS spacing of 3, a synthesis width of 4, a half-max
search capped at 40 -- repeated at 29 call sites. Every one of them describes
this sensor, and every one of them was set as if its pulse were about 4 bins
wide.

It is not. Measured off the data (half-maximum crossings of isolated strong
returns in the N=144 accumulation band, then checked in the N=18 band and
across six buildings), the impulse response is **FWHM 11 bins**, median, with
p10-p90 of 10-11. So the previous fixed width was 2.75x too narrow: `ta`
reconstructed every return as a pulse the sensor cannot produce, and the NMS
spacing of 3 sat well inside a single return, where one pulse's shoulders can
be taken for separate echoes.

A sensor here is the same two measured numbers as in the fog simulation's
`fwl_mitsuba.sensor`: the impulse response width in bins, and how far one bin
is. The window multiples (2 sigma for NMS, and searching for a half-max
crossing no further than the next candidate could be) are policy shared with
that module, deliberately -- if the two ever disagree, that is a finding about
the sensors, not a difference in how the extractor is configured.

Not derived from this module, and still open:

- ``min_height`` / ``min_prominence`` = 0.03 is a *relative* threshold applied
  after per-ray max normalisation. This sensor accumulates a different number
  of shots per row band (18/36/90/144/36/18 for rows 0-123/124-187/188-251/
  252-315/316-379/380-511, a hard ceiling on the counts), so 0.03 falls below
  one count in the shallow bands and sits at ~4.3 counts in the deepest one.
  Fixing that means detecting on raw counts against a Poisson background
  rather than on a max-normalised trace, which is a change of algorithm.
- ``smooth_sigma`` = 1.5 is much narrower than this pulse; a filter matched to
  the impulse response would smooth with sigma close to ``pulse_sigma_bins``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SPEED_OF_LIGHT_M_PER_S = 299792458.0

_FWHM_PER_STDDEV = 2.3548200450309493  # 2*sqrt(2*ln(2))


@dataclass(frozen=True)
class SensorCalibration:
    """A sensor's impulse response and time axis, plus the derived windows.

    ``nms_sigmas`` is policy shared across sensors; only the two measured
    fields differ between them.
    """

    name: str
    pulse_sigma_bins: float
    bin_width_range_m: float
    start_range_m: float = 0.0
    nms_sigmas: float = 2.0

    @property
    def pulse_fwhm_bins(self) -> float:
        """The width to synthesise a return at when the representation does
        not carry one (the ``ta`` case). Anything else reconstructs a pulse
        this sensor could not have emitted.
        """
        return _FWHM_PER_STDDEV * self.pulse_sigma_bins

    @property
    def min_distance_bins(self) -> float:
        """Two returns closer than one pulse width are not separable, so a
        peak is suppressed within 2 sigma of a stronger one.
        """
        return self.nms_sigmas * self.pulse_sigma_bins

    @property
    def nms_bins(self) -> int:
        return int(round(self.min_distance_bins))

    @property
    def width_search_bins(self) -> int:
        """How far the half-max crossing search may walk from a peak: out to
        where the next candidate could be, never past it.
        """
        return int(math.ceil(self.min_distance_bins))

    @property
    def max_fwhm_bins(self) -> int:
        """The widest FWHM a search bounded by ``width_search_bins`` can
        report, and therefore the only meaningful clamp on it.
        """
        return 2 * self.width_search_bins + 1

    def bin_to_range_m(self, bin_index: float) -> float:
        return self.start_range_m + (bin_index + 0.5) * self.bin_width_range_m

    def range_m_to_bins(self, distance_m: float) -> float:
        """A distance *interval* in bins -- a tolerance, not a position."""
        return distance_m / self.bin_width_range_m

    @classmethod
    def from_bin_duration_ns(
        cls, name: str, pulse_sigma_bins: float, bin_ns: float, **kw
    ) -> "SensorCalibration":
        return cls(
            name=name,
            pulse_sigma_bins=pulse_sigma_bins,
            bin_width_range_m=SPEED_OF_LIGHT_M_PER_S * bin_ns * 1e-9 / 2.0,
            **kw,
        )


#: 1 ns per bin (0.1499 m of range), 700 bins = 105 m, bin 0 at range 0, and
#: an impulse response of FWHM 11 bins measured off the data.
#:
#: The returns are asymmetric -- a rise over ~4 bins and a decay over ~10 --
#: so a symmetric Gaussian of this sigma approximates a shape the synthesis
#: can also model as an exponentially-modified Gaussian (``kernel="emg"``).
GHOST = SensorCalibration.from_bin_duration_ns(
    name="ghost_dataset",
    pulse_sigma_bins=11.0 / _FWHM_PER_STDDEV,
    bin_ns=1.0,
)
