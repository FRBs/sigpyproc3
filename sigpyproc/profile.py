from __future__ import annotations
import numpy as np

from astropy.modeling.models import Box1D, Gaussian1D, Lorentz1D
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Model1DKernel

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

    def normalize(self) -> PulseProfile:
        norm_prof = (self - self.mu) / self.std
        return norm_prof.view(PulseProfile)

    def box_snr(self, on_pulse: Tuple[int, int]) -> float:
        """Calculate Signal-to-noise ratio for the given pulse region.

        Parameters
        ----------
        on_pulse : Tuple[int, int]
            start and end bins of pulse

        Returns
        -------
        float
            Signal-to-noise ratio
        """
        return get_box_snr(self, on_pulse, mu_baseline=self.mu, std_baseline=self.std)

    def fit_template(self, widths, kind="boxcar"):
        return MatchedFilter(self, widths, kind=kind)


class MatchedFilter(object):
    def __init__(
        self, profile: npt.ArrayLike, widths: npt.ArrayLike, kind="boxcar"
    ) -> None:
        self._profile = np.array(profile, dtype=np.float32)
        self._widths = np.array(widths)
        self._kind = kind

        # Generate a list of pulse templates
        temps, temp_ref_bins = zip(
            *[get_template(iwidth, kind=self._kind) for iwidth in self._widths]
        )

        # templates are already normalized
        convs = np.array(
            [convolve_fft(self._profile, temp, normalize_kernel=False) for temp in temps]
        )

        itemp, ibin = np.unravel_index(convs.argmax(), convs.shape)
        self._peak_bin = ibin
        self._best_temp = temps[itemp]
        self._best_snr = convs[itemp, ibin]
        self._best_width = self._widths[itemp]

        temp_ref_bin = temp_ref_bins[itemp]
        best_temp_padded = np.pad(
            self._best_temp.array, (0, self._profile.size - self._best_temp.array.size)
        )
        self._best_model = (
            np.roll(best_temp_padded, self._peak_bin - temp_ref_bin) * self._best_snr
        )

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio based on best match template on pulse."""
        return self._best_snr

    @property
    def best_model(self) -> np.ndarray:
        """Best match template fit (`np.ndarray`, read-only)."""
        return self._best_model

    @property
    def best_width(self):
        """Best match template width in bins (`int`, read-only)."""
        return self._best_width

    @property
    def peak_bin(self) -> int:
        """Best match template peak bin (`int`, read-only)."""
        return self._peak_bin

    @property
    def on_pulse(self):
        """Best match template pulse region (`Tuple[int, int]`, read-only)."""
        return (
            self._peak_bin - round(self._best_width // 2),
            self._peak_bin + round(self._best_width // 2),
        )


def get_template(width, kind="boxcar"):
    if kind == "boxcar":
        norm = 1 / np.sqrt(width)
        temp = Model1DKernel(Box1D(norm, 0, width), x_size=width)
        ref_bin = temp.center
    elif kind == "gaussian":
        stddev = width / (2 * np.sqrt(2 * np.log(2)))
        norm = 1 / (np.sqrt(np.sqrt(np.pi) * stddev))
        size = np.ceil(8 * stddev) // 2 * 2 + 1  # Round up to odd_integer
        temp = Model1DKernel(Gaussian1D(norm, 0, stddev), x_size=int(size))
        ref_bin = temp.center
    elif kind == "lorentzian":
        stddev = width / (2 * np.sqrt(2 * np.log(2)))
        norm = 1 / (np.sqrt((np.pi * width) / 4))
        size = np.ceil(8 * stddev) // 2 * 2 + 1  # Round up to odd_integer
        temp = Model1DKernel(Lorentz1D(norm, 0, width), x_size=int(size))
        ref_bin = temp.center
    else:
        raise ValueError(f"{kind} not implemented yet.")
    return temp, ref_bin


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
