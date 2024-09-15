from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import attrs
import h5py
import numpy as np

from sigpyproc.core import stats
from sigpyproc.header import Header

if TYPE_CHECKING:
    from typing import Callable


def double_mad_mask(array: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Calculate the mask of an array using the double MAD (Modified z-score).

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
    """
    Calculate the mask of an array using the IQRM (Interquartile Range Method).

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
    threshold: float
    header: Header
    chan_mean: np.ndarray
    chan_var: np.ndarray
    chan_skew: np.ndarray
    chan_kurtosis: np.ndarray
    chan_maxima: np.ndarray
    chan_minima: np.ndarray

    chan_mask: np.ndarray = attrs.field()

    @chan_mask.default
    def _set_chan_mask(self) -> np.ndarray:
        return np.zeros(self.header.nchans, dtype="bool")

    @property
    def num_masked(self) -> int:
        """:obj:`int`: Number of masked channels."""
        return np.sum(self.chan_mask)

    @property
    def masked_fraction(self) -> float:
        """:obj:`float`: Fraction of channels masked."""
        return self.num_masked * 100 / self.header.nchans

    def apply_mask(self, chanmask: np.ndarray) -> None:
        """
        Apply a channel mask to the current mask.

        Parameters
        ----------
        chanmask : ndarray
            User channel mask to apply.

        Raises
        ------
        ValueError
            If the ``chanmask`` is not the same size as the current mask.
        """
        chanmask = np.asarray(chanmask, dtype="bool")
        if chanmask.size != self.header.nchans:
            msg = f"chanmask ({chanmask.size}) not equal nchans ({self.header.nchans})"
            raise ValueError(msg)
        self.chan_mask = np.logical_or(self.chan_mask, chanmask)

    def apply_method(self, method: Literal["iqrm", "mad"] = "mad") -> None:
        """
        Apply a mask method using channel statistics.

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
        mask_kurtosis = method_funcn(self.chan_kurtosis, self.threshold)
        mask_stats = np.logical_or.reduce((mask_var, mask_skew, mask_kurtosis))
        self.chan_mask = np.logical_or(self.chan_mask, mask_stats)

    def apply_funcn(self, custom_funcn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Apply a custom function to the channel mask.

        Parameters
        ----------
        custom_funcn : Callable[[ndarray], ndarray]
            Custom function to apply to the mask.

        Raises
        ------
        ValueError
            If the ``custom_funcn`` is not callable.
        """
        if not callable(custom_funcn):
            msg = f"{custom_funcn} is not callable"
            raise TypeError(msg)
        self.chan_mask = custom_funcn(self.chan_mask)

    def to_file(self, filename: str | None = None) -> str:
        """
        Write the mask to a HDF5 file.

        Parameters
        ----------
        filename : str, optional
            Filename to write the mask, by default None

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
                if isinstance(value, (np.integer, np.floating, str)):
                    fp.attrs[key] = value
            for key, value in attrs.asdict(self).items():
                if isinstance(value, np.ndarray):
                    fp.create_dataset(key, data=value)
        return filename

    @classmethod
    def from_file(cls, filename: str) -> RFIMask:
        """
        Load a mask from a HDF5 file.

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
