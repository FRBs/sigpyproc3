from __future__ import annotations
import numpy as np

from typing import Optional, Tuple
from numpy import typing as npt

from sigpyproc.Header import Header


class PulseProfile(np.ndarray):
    """An array class to handle a 1-D pulse profile. Pulse should be centered.

    Parameters
    ----------
    input_array : npt.ArrayLike
        1-D array of timeseries
    header : Header
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        1 dimensional array of shape (nsamples) with header metadata

    Notes
    -----
    Data is converted to 32 bits regardless of original type.
    """

    def __new__(cls, input_array: npt.ArrayLike, header: Header) -> PulseProfile:
        """Create a new 1D Pulse profile."""
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

    @property
    def nbins(self) -> int:
        """Number of bins in the profile(`int`, read-only)."""
        return self.shape[0]

    @property
    def tstart(self):
        """Start time of the profile in milliseconds (`float`, read-only)."""
        return -self.nbins * self.header.tsamp * 1e3 / 2

    @property
    def tstop(self):
        """End time of the profile in milliseconds (`float`, read-only)."""
        return self.nbins * self.header.tsamp * 1e3 / 2

    @property
    def mu(self):
        """Mean (estimated by median) of the input timeseries (`float`, read-only)."""
        return np.median(self)

    @property
    def std(self):
        """Standard deviation (estimated by inter-quartile range) of the input timeseries (`float`, read-only).

        Notes
        -----
        stats.norm.ppf(.75) - stats.norm.ppf(.25) -> 1.3489795003921634
        """
        quantiles = np.quantile(self, [0.25, 0.75])
        return np.diff(quantiles)[0] / 1.3489795003921634  # noqa:WPS432

    @property
    def on_pulse(self) -> Tuple[int, int]:
        """Best match boxcar pulse region (`Tuple[int, int]`, read-only)."""
        return (self._boxcar_start, self._boxcar_start + self._boxcar_width)

    @property
    def best_width(self) -> int:
        """Best match boxcar width in bins (`int`, read-only)."""
        return self._boxcar_width

    @property
    def snr(self) -> float:
        """Signal-to-noise based on best match boxcar on pulse."""
        return get_box_snr(self, self.on_pulse)

    def _boxcar_match(self):
        prof_ar = self.copy()
        prof_ar = (prof_ar - self.mu) / self.std
        box_widths = np.arange(1, self.nbins)
        templates = []
        for iwidth in box_widths:
            temp = np.ones(iwidth) / np.sqrt(iwidth)
            temp_pad = np.pad(temp, (0, self.nbins - temp.size))
            templates.append(temp_pad)

        convs = np.fft.irfft(np.fft.rfft(prof_ar) * np.fft.rfft(templates))
        itemp, ibin = np.unravel_index(convs.argmax(), convs.shape)
        self._boxcar_width = box_widths[itemp]
        self._boxcar_start = ibin


def get_box_snr(
    ts: npt.ArrayLike,
    on_pulse: Tuple[int, int],
    mu_baseline: Optional[float] = None,
    std_baseline: Optional[float] = None,
) -> float:
    """Calculates the S/N of the given pulse region.

    Parameters
    ----------
    ts : npt.ArrayLike
        time series array to calculate the SNR of.
    on_pulse : Tuple[int, int]
        [start, end] of on-pulse region in the time series.

    Returns
    -------
    float
        S/N of the given pulse region.
    """
    assert len(on_pulse) == 2, "on_pulse should only contain [start, end]"
    ts = np.asarray(ts).astype(np.float32, copy=False)
    mask = np.ones(len(ts), dtype=bool)
    mask[range(*on_pulse)] = 0
    if mu_baseline is None:
        mu_baseline = np.median(ts[mask])
    if std_baseline is None:
        # Using the inter-quartile range of the data
        # stats.norm.ppf(.75) - stats.norm.ppf(.25) -> 1.3489795003921634
        quantiles = np.quantile(ts[mask], [0.25, 0.75])
        std_baseline = np.diff(quantiles)[0] / 1.3489795003921634  # noqa:WPS432
    nbin_onpulse = len(ts[~mask])
    snr = (ts[~mask].sum() - (nbin_onpulse * mu_baseline)) / std_baseline
    return snr / np.sqrt(nbin_onpulse)
