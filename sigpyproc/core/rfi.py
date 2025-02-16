from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import h5py
import numpy as np
from matplotlib import pyplot as plt

from sigpyproc.core import stats
from sigpyproc.header import Header
from sigpyproc.viz.styles import PlotTable

if TYPE_CHECKING:
    from collections.abc import Callable

    from sigpyproc.core.custom_types import MaskMethods


def double_mad_mask(array: np.ndarray, threshold: float = 3) -> np.ndarray:
    """Calculate the mask of an array using the double MAD (Modified z-score).

    Parameters
    ----------
    array : ndarray
        The input array to calculate the mask of.
    threshold : float, optional
        Threshold in sigmas, by default 3.0.

    Returns
    -------
    ndarray
        The mask for the array.

    Raises
    ------
    ValueError
        If the ``threshold`` is not positive.
    """
    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise ValueError(msg)
    zscore = stats.estimate_zscore(array, scale_method="doublemad")
    return np.abs(zscore.data) > threshold


def iqrm_mask(array: np.ndarray, threshold: float = 3, radius: int = 5) -> np.ndarray:
    """Calculate the mask of an array using the IQRM (Interquartile Range Method).

    Parameters
    ----------
    array : ndarray
        The input array to calculate the mask of.
    threshold : float, optional
        Threshold in sigmas, by default 3.0.
    radius : int, optional
        Radius to calculate the IQRM, by default 5.

    Returns
    -------
    ndarray
        The mask for the array.

    Raises
    ------
    ValueError
        If the ``threshold`` is not positive.
    """
    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise ValueError(msg)
    mask = np.zeros_like(array, dtype="bool")
    lags = np.concatenate([np.arange(-radius, 0), np.arange(1, radius + 1)])
    shifted_x = np.lib.stride_tricks.as_strided(
        np.pad(array, radius, mode="edge"),
        shape=(len(array), 2 * radius + 1),
        strides=array.strides * 2,
    )
    lagged_diffs = array[:, np.newaxis] - shifted_x[:, lags + radius]
    lagged_diffs = lagged_diffs.T
    for lagged_diff in lagged_diffs:
        zscore = stats.estimate_zscore(lagged_diff, scale_method="iqr")
        mask = np.logical_or(mask, np.abs(zscore.data) > threshold)
    return mask


@attrs.define(auto_attribs=True, slots=True)
class RFIMask:
    """Class to handle RFI masking.

    Parameters
    ----------
    threshold : float
        Threshold for the mask.
    header : :class:`~sigpyproc.header.Header`
        Header object containing the observation metadata.
    chan_mean : ndarray
        Mean of each channel.
    chan_var : ndarray
        Variance of each channel.
    chan_skew : ndarray
        Skewness of each channel.
    chan_kurt : ndarray
        Kurtosis of each channel.
    chan_maxima : ndarray
        Maximum of each channel.
    chan_minima : ndarray
        Minimum of each channel.
    chan_mask : ndarray, optional
        Final mask of the channels, by default None.
    user_mask : ndarray, optional
        User-defined mask, by default None.
    stats_mask : ndarray, optional
        Mask calculated using channel statistics, by default None.
    custom_mask : ndarray, optional
        Custom mask, by default None.

    Attributes
    ----------
    threshold
    header
    chan_mean
    chan_var
    chan_skew
    chan_kurt
    chan_maxima
    chan_minima
    chan_mask
    user_mask
    stats_mask
    custom_mask
    num_masked
    masked_fraction
    """

    threshold: float
    header: Header
    chan_mean: np.ndarray
    chan_var: np.ndarray
    chan_skew: np.ndarray
    chan_kurt: np.ndarray
    chan_maxima: np.ndarray
    chan_minima: np.ndarray

    chan_mask: np.ndarray = attrs.field()
    user_mask: np.ndarray = attrs.field()
    stats_mask: np.ndarray = attrs.field()
    custom_mask: np.ndarray = attrs.field()

    @chan_mask.default
    def _set_chan_mask(self) -> np.ndarray:
        return np.zeros(self.header.nchans, dtype="bool")

    @user_mask.default
    def _set_user_mask(self) -> np.ndarray:
        return np.zeros(self.header.nchans, dtype="bool")

    @stats_mask.default
    def _set_stats_mask(self) -> np.ndarray:
        return np.zeros(self.header.nchans, dtype="bool")

    @custom_mask.default
    def _set_custom_mask(self) -> np.ndarray:
        return np.zeros(self.header.nchans, dtype="bool")

    @property
    def num_masked(self) -> int:
        """Get the number of masked channels.

        Returns
        -------
        int
            Number of masked channels.
        """
        return np.sum(self.chan_mask)

    @property
    def masked_fraction(self) -> float:
        """Get the fraction of channels masked.

        Returns
        -------
        float
            Fraction of channels masked.
        """
        return self.num_masked * 100 / self.header.nchans

    def apply_mask(self, freq_mask: list[tuple[float, float]]) -> None:
        """Apply a frequency range mask.

        Parameters
        ----------
        freq_mask : list[tuple[float, float]]
            List of frequency ranges to mask.
        """
        user_mask = np.zeros(self.header.nchans, dtype="bool")
        for freq_range in freq_mask:
            mask = np.logical_and(
                self.header.chan_freqs >= freq_range[0],
                self.header.chan_freqs <= freq_range[1],
            )
            user_mask = np.logical_or(user_mask, mask)
        self.user_mask = user_mask
        self.chan_mask = np.logical_or(self.chan_mask, user_mask)

    def apply_method(self, method: MaskMethods = "mad") -> None:
        """Apply a mask method using channel statistics.

        Parameters
        ----------
        method : {'iqrm', 'mad'}, optional
            Method to apply, by default 'mad'.

        Raises
        ------
        ValueError
            If the ``method`` is not supported.
        """
        if method == "mad":
            method_funcn = double_mad_mask
        elif method == "iqrm":
            method_funcn = iqrm_mask
        else:
            msg = f"method {method} not supported"
            raise ValueError(msg)
        mask_var = method_funcn(self.chan_var, self.threshold)
        mask_skew = method_funcn(self.chan_skew, self.threshold)
        mask_kurtosis = method_funcn(self.chan_kurt, self.threshold)
        self.stats_mask = np.logical_or.reduce((mask_var, mask_skew, mask_kurtosis))
        self.chan_mask = np.logical_or(self.chan_mask, self.stats_mask)

    def apply_funcn(self, custom_funcn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Apply a custom function to the existing mask.

        Parameters
        ----------
        custom_funcn : Callable[[ndarray], ndarray]
            Custom function to apply to the mask. The function should take the existing
            mask as input and return additional channels to mask.

        Raises
        ------
        ValueError
            If the ``custom_funcn`` is not callable.
        """
        if not callable(custom_funcn):
            msg = f"{custom_funcn} is not callable"
            raise TypeError(msg)
        self.custom_mask = custom_funcn(self.chan_mask)
        self.chan_mask = np.logical_or(self.chan_mask, self.custom_mask)

    def to_file(self, filename: str | None = None) -> str:
        """Write the mask to a HDF5 file.

        Parameters
        ----------
        filename : str, optional
            Filename to write the mask, by default None.

        Returns
        -------
        str
            Filename written to.
        """
        if filename is None:
            filename = f"{self.header.basename}_mask.h5"
        with h5py.File(filename, "w") as fp:
            fp.attrs["threshold"] = self.threshold
            for key, value in attrs.asdict(self.header).items():
                if isinstance(value, np.integer | np.floating | int | float | str):
                    fp.attrs[key] = value
            for key, value in attrs.asdict(self).items():
                if isinstance(value, np.ndarray):
                    fp.create_dataset(key, data=value)
        return filename

    def plot(
        self,
        figsize: tuple[float, float] = (12, 6.5),
        dpi: int = 100,
    ) -> plt.Figure:
        """Plot the mask.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches, by default (12, 6).
        dpi : int, optional
            Dots per inch, by default 100.

        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        mask_colors = {
            "stats": "#ff7f7f",
            "user": "#e377c2",
            "cust": "#17becf",
        }
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        ax = fig.subplot_mosaic(
            [
                ["kurt", "table"],
                ["skew", "table"],
                ["var", "table"],
                ["mean", "table"],
            ],
            sharex=True,
            width_ratios=[3, 1],
            gridspec_kw={"hspace": 0, "wspace": 0},
            per_subplot_kw={
                "kurt": {"ylabel": "Kurtosis"},
                "skew": {"ylabel": "Skewness"},
                "var": {"ylabel": "Variance"},
                "mean": {"ylabel": "Mean", "xlabel": "Frequency (MHz)"},
            },
        )
        lc = "#3F5D7D"
        ax["kurt"].plot(self.header.chan_freqs, self.chan_kurt, c=lc, lw=1.5)
        ax["skew"].plot(self.header.chan_freqs, self.chan_skew, c=lc, lw=1.5)
        ax["var"].plot(self.header.chan_freqs, self.chan_var, c=lc, lw=1.5)
        ax["mean"].plot(self.header.chan_freqs, self.chan_mean, c=lc, lw=1.5)

        for mask_type, mask in [
            ("stats", self.stats_mask),
            ("user", self.user_mask),
            ("cust", self.custom_mask),
        ]:
            channels = np.where(mask)[0]
            for chan in channels:
                ax["mean"].axvspan(
                    self.header.chan_freqs[chan] - self.header.foff / 2,
                    self.header.chan_freqs[chan] + self.header.foff / 2,
                    facecolor=mask_colors[mask_type],
                    alpha=0.7,
                )
        table = PlotTable()
        table.add_entry("Source", self.header.source)
        table.add_entry("Tobs", f"{self.header.tobs:.3f}", "s")
        table.add_entry("Tsamp", f"{self.header.tsamp * 1e6:.3f}", r"$\mu$s")
        table.add_entry("Nchans", self.header.nchans)
        table.add_entry("Nsamples", self.header.nsamples)
        table.skip_line()
        table.add_entry("Threshold", self.threshold, r"$\sigma$")
        table.add_entry(
            "Mask (total)",
            self.num_masked,
            f"({self.masked_fraction:.2f}%)",
        )
        table.add_entry(
            "Mask (stats)",
            np.sum(self.stats_mask),
            f"({np.sum(self.stats_mask) * 100 / self.header.nchans:.2f}%)",
            mask_colors["stats"],
        )
        table.add_entry(
            "Mask (user)",
            np.sum(self.user_mask),
            f"({np.sum(self.user_mask) * 100 / self.header.nchans:.2f}%)",
            mask_colors["user"],
        )
        table.add_entry(
            "Mask (custom)",
            np.sum(self.custom_mask),
            f"({np.sum(self.custom_mask) * 100 / self.header.nchans:.2f}%)",
            mask_colors["cust"],
        )
        table.plot(ax["table"])
        fig.suptitle(f"RFI Mask for {self.header.filename}")
        return fig

    @classmethod
    def from_file(cls, filename: str) -> RFIMask:
        """Load a mask from a HDF5 file.

        Parameters
        ----------
        filename : str
            Filename to load the mask from.

        Returns
        -------
        RFIMask
            The loaded mask.
        """
        with h5py.File(filename, "r") as fp:
            fp_attrs = dict(fp.attrs.items())
            fp_stats = {key: np.array(val) for key, val in fp.items()}
        hdr_checked = {
            key: value
            for key, value in fp_attrs.items()
            if key in attrs.fields_dict(Header)
        }
        kws = {
            "header": Header(**hdr_checked),
            "threshold": fp_attrs["threshold"],
            **fp_stats,
        }
        return cls(**kws)
