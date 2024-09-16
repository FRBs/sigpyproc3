from __future__ import annotations

from typing import Literal

import attrs
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from matplotlib import pyplot as plt
from numba import typed

from sigpyproc.core import kernels
from sigpyproc.core.stats import (
    LocMethodType,
    ScaleMethodType,
    ZScoreResult,
    estimate_zscore,
)


class MatchedFilter:
    """
    Matched filter class for pulse detection in 1D time series data.

    This class implements a matched filter algorithm to detect pulses of varying
    durations in 1D time series data. It uses a set of pulse templates with
    varying widths and selects the template that produces the highest
    signal-to-noise ratio (SNR) as the best match.

    Parameters
    ----------
    data : ndarray
        Input data array for matched filtering (1D).
    loc_method : {"median", "mean", "norm"}, optional
        Method to estimate location, by default "median".
    scale_method : str, optional
        Method to estimate scale, by default "iqr".
    temp_kind : {"boxcar", "gaussian", "lorentzian"}, optional
        Type of the pulse template, by default "boxcar".
    nbins_max : int, optional
        Maximum number of bins for template width, by default 32.
    spacing_factor : float, optional
        Factor for spacing between template widths, by default 1.5.

    Raises
    ------
    ValueError
        If the input ``data`` dimension is not 1.

    See Also
    --------
    sigpyproc.core.stats.estimate_zscore : Estimate Z-score of input data.
    sigpyproc.core.stats.estimate_loc : Estimate location of input data.
    sigpyproc.core.stats.estimate_scale: Estimate scale of input data.

    Notes
    -----
    The matched filter is the optimal linear filter for maximizing the signal-to-noise
    ratio (SNR) of a known pulse template in the presence of additive white noise.

    For input data :math:`x(t)` and a template :math:`h(t)`, the matched
    filter output :math:`y(t)` is:

    .. math:: y(t) = (x \\star h)(t) = \\sum_{\\tau} x(\\tau) h(t - \\tau)

    This is computed efficiently using the FFT-based methods. As per the circular
    convolution theorem:

    .. math:: Y(f) = X(f) H(f)
    .. math:: y(t) = \\mathcal{F}^{-1}(Y(f))

    where :math:`\\mathcal{F}^{-1}` is the inverse Fourier transform, :math:`X(f)`
    and :math:`H(f)` are the Fourier transforms of :math:`x(t)` and :math:`h(t)`
    respectively.

    References
    ----------
    .. [1] Wikipedia, "Matched filter",
        https://en.wikipedia.org/wiki/Matched_filter
    .. [2] Wikipedia, "Circular convolution",
        https://en.wikipedia.org/wiki/Circular_convolution

    """

    def __init__(
        self,
        data: np.ndarray,
        loc_method: LocMethodType | Literal["norm"] = "median",
        scale_method: ScaleMethodType | Literal["norm"] = "iqr",
        temp_kind: Literal["boxcar", "gaussian", "lorentzian"] = "boxcar",
        nbins_max: int = 32,
        spacing_factor: float = 1.5,
    ) -> None:
        if data.ndim != 1:
            msg = f"Data dimension {data.ndim} is not supported."
            raise ValueError(msg)
        self._temp_kind = temp_kind
        self._data = np.asarray(data, dtype=np.float32)
        self._zscores = estimate_zscore(
            self.data,
            loc_method=loc_method,
            scale_method=scale_method,
        )
        self._setup_templates(nbins_max, spacing_factor)
        self._compute()

    @property
    def data(self) -> np.ndarray:
        """:obj:`~numpy.ndarray`: Input data array for matched filtering."""
        return self._data

    @property
    def zscores(self) -> ZScoreResult:
        """:class:`~sigpyproc.core.stats.ZScoreResult`: Z-score of the input data."""
        return self._zscores

    @property
    def temp_kind(self) -> str:
        """:obj:`str`: Type of the pulse template."""
        return self._temp_kind

    @property
    def temp_widths(self) -> np.ndarray:
        """:obj:`~numpy.ndarray`: Template widths used for matched filtering."""
        return self._temp_widths

    @property
    def temp_bank(self) -> list[Template]:
        """:obj:`list[Template]`: List of pulse templates used for matched filtering."""
        return self._temp_bank

    @property
    def convs(self) -> np.ndarray:
        """:obj:`~numpy.ndarray`: Convolution results for all templates."""
        return self._convs

    @property
    def peak_bin(self) -> int:
        """:obj:`int`: Best match template peak bin."""
        return int(self._peak_bin)

    @property
    def best_temp(self) -> Template:
        """:class:`~sigpyproc.core.filters.Template`: Best match template."""
        return self._best_temp

    @property
    def snr(self) -> float:
        """:obj:`float`: Signal-to-noise ratio based on best match template."""
        return self._best_snr

    @property
    def best_model(self) -> np.ndarray:
        """:obj:`~numpy.ndarray`: Best match template fit."""
        return (
            self.best_temp.get_model(self.peak_bin, self.data.size)
            * self.snr
            * self.zscores.scale
            + self.zscores.loc
        )

    @property
    def on_pulse(self) -> tuple[int, int]:
        """:obj:`tuple[int, int]`: Best match template pulse region."""
        return self.best_temp.get_on_pulse(self.peak_bin, self.data.size)

    def plot(
        self,
        figsize: tuple[float, float] = (12, 6),
        dpi: int = 100,
    ) -> plt.Figure:
        """
        Plot the pulse template.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (12, 6)
        dpi : int, optional
            Dots per inch, by default 100

        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        title = (
            f"Matched Filter Result (Temp Kind: {self.temp_kind}, "
            f"Best width: {self.best_temp.width:.2f}, "
            f"SNR: {self.snr:.2f})"
        )
        stats_box = f"loc: {self.zscores.loc:.2f}, scale: {self.zscores.scale:.2f}"
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(self.data, label="Data", lw=2)
        ax.plot(self.best_model, label="Best Model", lw=2)
        ax.axvline(self.peak_bin, color="r", linestyle="--", label="Peak", lw=2)
        ax.axvspan(*self.on_pulse, alpha=0.2, color="g", label="On Pulse")
        ax.text(
            0.05,
            0.95,
            stats_box,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={
                "fc": "white",
                "ec": "gray",
                "alpha": 0.8,
                "boxstyle": "round, pad=0.5",
            },
        )

        ax.set(xlabel="Bin", ylabel="Amplitude", title=title, xlim=(0, len(self.data)))
        ax.legend()
        fig.tight_layout()
        return fig

    def _setup_templates(self, nbins_max: int, spacing_factor: float) -> None:
        if self.temp_kind == "boxcar":
            self._temp_widths = self.get_box_width_spacing(nbins_max, spacing_factor)
        else:
            if spacing_factor <= 1:
                msg = "Spacing factor must be greater than 1 for non-boxcar templates."
                raise ValueError(msg)
            npoints = int(np.ceil(np.log(nbins_max) / np.log(spacing_factor))) + 1
            self._temp_widths = np.geomspace(1, nbins_max, npoints)
        temp_bank = []
        for width in self.temp_widths:
            temp = getattr(Template, f"gen_{self.temp_kind}")(width)
            if temp.data.size > self.data.size:
                msg = (
                    f"Template size ({temp.data.size}) is larger than the data size"
                    f"({self.data.size})."
                )
                raise ValueError(msg)
            temp_bank.append(temp)
        self._temp_bank = temp_bank

    def _compute(self) -> None:
        temp_kernels = typed.List([temp.data for temp in self.temp_bank])
        ref_bins = typed.List([temp.ref_bin for temp in self.temp_bank])
        self._convs = kernels.convolve_fft(self.zscores.data, temp_kernels, ref_bins)
        self._itemp, self._peak_bin = np.unravel_index(
            self._convs.argmax(),
            self._convs.shape,
        )
        self._best_temp = self.temp_bank[self._itemp]
        self._best_snr = self._convs[self._itemp, self._peak_bin]

    @staticmethod
    def get_box_width_spacing(
        size_max: int,
        spacing_factor: float = 1.5,
    ) -> np.ndarray:
        """
        Get box width spacing for matched filtering.

        Parameters
        ----------
        size_max : int
            Maximum number of bins for box template width.
        spacing_factor : float, optional
            Spacing factor for width, by default 1.5

        Returns
        -------
        ndarray
            Width spacing for matched filtering.
        """
        widths = [1]
        while widths[-1] < size_max:
            next_width = int(max(widths[-1] + 1, spacing_factor * widths[-1]))
            if next_width > size_max:
                break
            widths.append(next_width)
        return np.array(widths, dtype=np.float32)


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class Template:
    """
    1D pulse template class for matched filtering.

    This class represents various pulse shapes as templates for matched filtering
    and provides methods to generate and visualize them.

    Parameters
    ----------
    data : ndarray
        Pulse template data array (1D).
    width : float
        Width of the pulse template in bins.
    ref_bin : int, optional
        Reference bin for the pulse template, by default 0
    ref : {"start", "peak"}, optional
        Reference type for the pulse template, by default "start"
    kind : str, optional
        Type of the pulse template, by default "custom"
    """

    data: np.ndarray
    width: float
    ref_bin: int = attrs.field(
        default=0,
        validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)],
    )
    ref: str = attrs.field(
        default="start",
        validator=attrs.validators.in_({"start", "peak"}),
    )
    kind: str = attrs.field(
        default="custom",
        validator=attrs.validators.instance_of(str),
    )

    def __attrs_post_init__(self) -> None:
        if not self.data.size:
            msg = "Empty data array is not supported."
            raise ValueError(msg)
        if self.data.ndim != 1:
            msg = f"Only 1D data is supported, got {self.data.ndim}."
            raise ValueError(msg)
        if self.ref_bin >= self.data.size:
            msg = f"Reference bin {self.ref_bin} is out of bounds."
            raise ValueError(msg)

    def get_model(self, peak_bin: int, nbins: int) -> np.ndarray:
        """
        Get profile model for the pulse template.

        Parameters
        ----------
        peak_bin : int
            Peak bin in the profile
        nbins : int
            Profile size

        Returns
        -------
        ndarray
            Profile model for the pulse template
        """
        padded = np.pad(self.data, (0, nbins - self.data.size))
        padded_norm = kernels.normalize_template(padded)
        return np.roll(padded_norm, peak_bin - self.ref_bin)

    def get_on_pulse(self, peak_bin: int, nbins: int) -> tuple[int, int]:
        """
        Get on pulse region in the profile model for the pulse template.

        Parameters
        ----------
        peak_bin : int
            Peak bin in the model
        nbins : int
            Profile size

        Returns
        -------
        tuple[int, int]
            Start and end bin of the on pulse region
        """
        if self.ref == "start":
            pulse_left = peak_bin
            pulse_right = peak_bin + self.width
        else:
            pulse_left = peak_bin - round(self.width)
            pulse_right = peak_bin + round(self.width)
        start = max(0, pulse_left)
        end = min(nbins, pulse_right)
        return (start, int(end))

    def plot(
        self,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 100,
    ) -> plt.Figure:
        """
        Plot the pulse template.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (10, 5)
        dpi : int, optional
            Dots per inch, by default 100

        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(range(self.data.size), self.data, ec="k", fc="#a6cee3")
        ax.axvline(self.ref_bin, ls="--", lw=2, color="k", label="Ref Bin")
        ax.legend()
        ax.set(
            xlim=(-0.5, self.data.size - 0.5),
            xlabel="Bin",
            ylabel="Amplitude",
            title=str(self),
        )
        fig.tight_layout()
        return fig

    @classmethod
    def gen_boxcar(cls, width: int) -> Template:
        """
        Generate a boxcar pulse template.

        Parameters
        ----------
        width : int
            Width of the box in bins.

        Returns
        -------
        Template
            Boxcar pulse template with the reference bin at the start.
        """
        width = int(width)
        if width <= 0:
            msg = f"Width {width} must be greater than 0."
            raise ValueError(msg)
        arr = np.ones(width, dtype=np.float32)
        return cls(arr, width, ref_bin=0, ref="start", kind="boxcar")

    @classmethod
    def gen_gaussian(cls, width: float, extent: float = 3.5) -> Template:
        """
        Generate a Gaussian pulse template.

        Parameters
        ----------
        width : float
            FWHM of the Gaussian pulse in bins.
        extent : float, optional
            Extent of the Gaussian pulse in sigma units, by default 3.5.

        Returns
        -------
        Template
            Gaussian pulse template with the reference bin at the peak.
        """
        if width <= 0:
            msg = f"Width {width} must be greater than 0."
            raise ValueError(msg)
        stddev = gaussian_fwhm_to_sigma * width
        size = int(np.ceil(extent * stddev))
        x = np.arange(-size, size + 1)
        ref_bin = len(x) // 2
        arr = np.exp(-0.5 * x**2 / stddev**2)
        return cls(arr, width, ref_bin=ref_bin, ref="peak", kind="gaussian")

    @classmethod
    def gen_lorentzian(cls, width: float, extent: float = 3.5) -> Template:
        """
        Generate a Lorentzian pulse template.

        Parameters
        ----------
        width : float
            FWHM of the Lorentzian pulse in bins.
        extent : float, optional
            Extent of the Lorentzian pulse in sigma units, by default 3.5.

        Returns
        -------
        Template
            Lorentzian pulse template.
        """
        if width <= 0:
            msg = f"Width {width} must be greater than 0."
            raise ValueError(msg)
        stddev = gaussian_fwhm_to_sigma * width
        size = int(np.ceil(extent * stddev))
        x = np.arange(-size, size + 1)
        ref_bin = len(x) // 2
        arr = 1 / (1 + (x / stddev) ** 2)
        return cls(arr, width, ref_bin=ref_bin, ref="peak", kind="lorentzian")

    def __str__(self) -> str:
        return (
            f"Template(size={self.data.size}, kind={self.kind}, width={self.width:.3f})"
        )

    def __repr__(self) -> str:
        return str(self)
