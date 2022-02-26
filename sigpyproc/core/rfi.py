from __future__ import annotations
import numpy as np
import attrs
import h5py

from numpy import typing as npt
from typing import Callable

from iqrm import iqrm_mask
from sigpyproc.header import Header
from sigpyproc.core.stats import zscore_double_mad


def double_mad_mask(array: npt.ArrayLike, threshold: float = 3) -> np.ndarray:
    """Calculate the mask of an array using the double MAD (Modified z-score).

    Parameters
    ----------
    array : :py:obj:`~numpy.typing.ArrayLike`
        The array to calculate the mask of.
    threshold : float, optional
        Threshold in sigmas, by default 3.0

    Returns
    -------
    numpy.ndarray
        The mask for the array.

    Raises
    ------
    ValueError
        If the threshold is not positive.
    """
    array = np.asarray(array)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return np.abs(zscore_double_mad(array)) > threshold


@attrs.define(auto_attribs=True, slots=True)
class RFIMask(object):
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
    def _set_chan_mask(self):
        return np.zeros(self.header.nchans, dtype="bool")

    @property
    def num_masked(self) -> int:
        """int: Number of masked channels."""
        return np.sum(self.chan_mask)

    @property
    def masked_fraction(self) -> float:
        """float: Fraction of channels masked."""
        return self.num_masked * 100 / self.header.nchans

    def apply_mask(self, chanmask: npt.ArrayLike) -> None:
        """Apply a channel mask to the current mask.

        Parameters
        ----------
        chanmask : :py:obj:`~numpy.typing.ArrayLike`
            User channel mask to apply.

        Raises
        ------
        ValueError
            If the channel mask is not the same size as the current mask.
        """
        chanmask = np.asarray(chanmask, dtype="bool")
        if chanmask.size != self.header.nchans:
            raise ValueError(
                f"chanmask len {chanmask.size} does not match nchans {self.header.nchans}"
            )
        self.chan_mask = np.logical_or(self.chan_mask, chanmask)

    def apply_method(self, method: str) -> None:
        """Apply a mask method using channel statistics.

        Parameters
        ----------
        method : str
            Mask method to apply (`iqrm`, `mad`).

        Raises
        ------
        ValueError
            If the method is not supported.
        """
        if method == "iqrm":
            method_funcn = lambda arr, thres: iqrm_mask(  # noqa: E731
                arr, radius=0.1 * self.header.nchans, threshold=thres
            )
        elif method == "mad":
            method_funcn = double_mad_mask
        else:
            raise ValueError(f"Unknown method {method}")
        mask_var = method_funcn(self.chan_var, self.threshold)
        mask_skew = method_funcn(self.chan_skew, self.threshold)
        mask_kurtosis = method_funcn(self.chan_kurtosis, self.threshold)
        mask_stats = np.logical_or.reduce((mask_var, mask_skew, mask_kurtosis))
        self.chan_mask = np.logical_or(self.chan_mask, mask_stats)

    def apply_funcn(self, custom_funcn: Callable[[npt.ArrayLike], np.ndarray]) -> None:
        """Apply a custom function to the channel mask.

        Parameters
        ----------
        custom_funcn : :py:obj:`~typing.Callable`
            Custom function to apply to the mask.

        Raises
        ------
        ValueError
            If the custom_funcn is not callable.
        """
        if not callable(custom_funcn):
            raise ValueError(f"{custom_funcn} is not callable")
        self.chan_mask = custom_funcn(self.chan_mask)

    def to_file(self, filename: str | None = None) -> str:
        """Write the mask to a HDF5 file.

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
                if isinstance(value, (int, float, str)):
                    fp.attrs[key] = value
            for key, value in attrs.asdict(self).items():
                if isinstance(value, np.ndarray):
                    fp.create_dataset(key, data=value)
        return filename

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
            fp_attrs = {key: val for key, val in fp.attrs.items()}
            fp_stats = {key: np.array(val) for key, val in fp.items()}
        hdr_checked = {
            key: value
            for key, value in fp_attrs.items()
            if key in attrs.fields_dict(Header).keys()
        }
        kws = {
            "header": Header(**hdr_checked),
            "threshold": fp_attrs["threshold"],
            **fp_stats,
        }
        return cls(**kws)
