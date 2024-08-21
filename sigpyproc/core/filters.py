from __future__ import annotations

import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Model1DKernel
from astropy.modeling.models import Box1D, Gaussian1D, Lorentz1D
from astropy.stats import gaussian_fwhm_to_sigma
from matplotlib import pyplot as plt

from sigpyproc.core.stats import estimate_scale


class MatchedFilter:
    """Matched filter class for pulse detection.

    This class implements a matched filter algorithm to detect pulses in 1D data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    temp_kind : str, optional
        Type of the pulse template, by default "boxcar"
    nbins_max : int, optional
        Maximum number of bins for template width, by default 32
    spacing_factor : float, optional
        Factor for spacing between template widths, by default 1.5

    Raises
    ------
    ValueError
        _description_
    """

    def __init__(
        self,
        data: np.ndarray,
        noise_method: str = "iqr",
        temp_kind: str = "boxcar",
        nbins_max: int = 32,
        spacing_factor: float = 1.5,
    ) -> None:
        if data.ndim != 1:
            msg = f"Data dimension {data.ndim} is not supported."
            raise ValueError(msg)
        self._temp_kind = temp_kind
        self._noise_method = noise_method
        self._data = self._get_norm_data(data)
        self._temp_widths = self.get_width_spacing(nbins_max, spacing_factor)
        self._temp_bank = [
            getattr(Template, f"gen_{self.temp_kind}")(iwidth)
            for iwidth in self.temp_widths
        ]

        self._convs = np.array(
            [
                convolve_fft(self.data, temp.kernel, normalize_kernel=False)
                for temp in self.temp_bank
            ],
        )
        self._itemp, self._peak_bin = np.unravel_index(
            self._convs.argmax(),
            self._convs.shape,
        )
        self._best_temp = self.temp_bank[self._itemp]
        self._best_snr = self._convs[self._itemp, self._peak_bin]

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def noise_method(self) -> str:
        return self._noise_method

    @property
    def temp_kind(self) -> str:
        return self._temp_kind

    @property
    def temp_widths(self) -> np.ndarray:
        return self._temp_widths

    @property
    def temp_bank(self) -> list[Template]:
        return self._temp_bank

    @property
    def convs(self) -> np.ndarray:
        return self._convs

    @property
    def peak_bin(self) -> int:
        """Best match template peak bin (`int`, read-only)."""
        return int(self._peak_bin)

    @property
    def best_temp(self) -> Template:
        return self._best_temp

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio based on best match template on pulse."""
        return self._best_snr

    @property
    def best_model(self) -> np.ndarray:
        """Best match template fit (`np.ndarray`, read-only)."""
        return self.snr * np.roll(
            self.best_temp.get_padded(self.data.size),
            self.peak_bin - self.best_temp.ref_bin,
        )

    @property
    def on_pulse(self) -> tuple[int, int]:
        """Best match template pulse region (`Tuple[int, int]`, read-only)."""
        start = max(0, self.peak_bin - round(self.best_temp.width))
        end = min(self.data.size, self.peak_bin + round(self.best_temp.width))
        return (start, end)

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data, label="Data")
        ax.plot(self.best_model, label="Best Model")
        ax.axvline(self.peak_bin, color="r", linestyle="--", label="Peak")
        ax.axvspan(*self.on_pulse, alpha=0.2, color="g", label="On Pulse")
        ax.set(
            xlabel="Bin",
            ylabel="Amplitude",
            title=f"Matched Filter Result (SNR: {self.snr:.2f})",
        )
        ax.legend()
        fig.tight_layout()
        return fig

    def _get_norm_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        median = np.median(data)
        noise_std = estimate_scale(data, self.noise_method)
        return (data - median) / noise_std

    @staticmethod
    def get_width_spacing(
        nbins_max: int,
        spacing_factor: float = 1.5,
    ) -> np.ndarray:
        """Get width spacing for matched filtering.

        Parameters
        ----------
        nbins_max : int
            Maximum number of bins.
        spacing_factor : float, optional
            Spacing factor for width, by default 1.5

        Returns
        -------
        np.ndarray
            Width spacing for matched filtering.
        """
        widths = [1]
        while widths[-1] < nbins_max:
            next_width = int(max(widths[-1] + 1, spacing_factor * widths[-1]))
            if next_width > nbins_max:
                break
            widths.append(next_width)
        return np.array(widths, dtype=np.float32)


class Template:
    """1D pulse template class for matched filtering.

    This class represents various pulse shapes as templates for matched filtering
    and provides methods to generate and visualize them.

    Parameters
    ----------
    kernel : Model1DKernel
        Astropy 1D model kernel.
    width : float
        Width of the pulse template in bins.
    kind : str, optional
        Type of the pulse template, by default "custom"
    """

    def __init__(
        self,
        kernel: Model1DKernel,
        width: float,
        kind: str = "custom",
    ) -> None:
        self._kernel = kernel
        self._width = width
        self._kind = kind

    @property
    def kernel(self) -> Model1DKernel:
        """Astropy 1D model kernel (`Model1DKernel`, read-only)."""
        return self._kernel

    @property
    def width(self) -> float:
        """Width of the pulse template in bins (`float`, read-only)."""
        return self._width

    @property
    def kind(self) -> str:
        """Type of the pulse template (`str`, read-only)."""
        return self._kind

    @property
    def ref_bin(self) -> int:
        """Reference bin of the pulse template (`int`, read-only)."""
        return self.kernel.center[0]

    @property
    def size(self) -> int:
        """Size of the pulse template (`int`, read-only)."""
        return self.kernel.shape[0]

    def get_padded(self, size: int) -> np.ndarray:
        """
        Pad template to desired size.

        Parameters
        ----------
        size: int
            Size of the padded pulse template.
        """
        if self.size >= size:
            msg = f"Template size {self.size} is larger than {size}."
            raise ValueError(msg)
        return np.pad(self.kernel.array, (0, size - self.size))

    def plot(
        self,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 100,
    ) -> plt.Figure:
        """Plot the pulse template.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (10, 5)
        dpi : int, optional
            Dots per inch, by default 100

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(range(self.size), self.kernel.array, ec="k", fc="#a6cee3")
        ax.axvline(self.ref_bin, ls="--", lw=2, color="k", label="Ref Bin")
        ax.legend()
        ax.set(
            xlim=(-0.5, self.size - 0.5),
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
        width: int
            Width of the box in bins.
        """
        norm = 1 / np.sqrt(width)
        temp = Model1DKernel(Box1D(norm, 0, width), x_size=width)
        return cls(temp, width, kind="boxcar")

    @classmethod
    def gen_gaussian(cls, width: float, extent: float = 3.5) -> Template:
        """
        Generate a Gaussian pulse template.

        Parameters
        ----------
        width: float
            FWHM of the Gaussian pulse in bins.

        extent: float
            Extent of the Gaussian pulse in sigma units, by default 3.5.
        """
        stddev = gaussian_fwhm_to_sigma * width
        norm = 1 / (np.sqrt(np.sqrt(np.pi) * stddev))
        size = int(np.ceil(extent * stddev) * 2 + 1)
        temp = Model1DKernel(Gaussian1D(norm, 0, stddev), x_size=size)
        return cls(temp, width, kind="gaussian")

    @classmethod
    def gen_lorentzian(cls, width: float, extent: float = 3.5) -> Template:
        """
        Generate a Lorentzian pulse template for given pulse FWHM (bins).

        Parameters
        ----------
        width: float
            FWHM of the Lorentzian pulse in bins.

        extent: float
            Extent of the Lorentzian pulse in sigma units, by default 3.5.
        """
        stddev = gaussian_fwhm_to_sigma * width
        norm = 1 / (np.sqrt((np.pi * width) / 4))
        size = int(np.ceil(extent * stddev) * 2 + 1)
        temp = Model1DKernel(Lorentz1D(norm, 0, width), x_size=size)
        return cls(temp, width, kind="lorentzian")

    def __str__(self) -> str:
        return f"Template(size={self.size}, kind={self.kind}, width={self.width:.3f})"

    def __repr__(self) -> str:
        return str(self)
