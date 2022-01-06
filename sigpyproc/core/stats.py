from __future__ import annotations
import numpy as np
import bottleneck as bn

from sigpyproc.core import kernels


def running_median(array, window):
    """
    Calculate the running median of an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to calculate the running median of.

    Returns
    -------
    numpy.ndarray
        The running median of the array.

    """
    pad_size = (
        (window // 2, window // 2) if window % 2 else (window // 2, window // 2 - 1)
    )
    padded = np.pad(array, pad_size, "symmetric")

    median = bn.move_median(padded, window)
    return median[window - 1 :]


def running_mean(array, window):
    """
    Calculate the running mean of an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to calculate the running mean of.

    Returns
    -------
    numpy.ndarray
        The running mean of the array.

    """
    pad_size = (
        (window // 2, window // 2) if window % 2 else (window // 2, window // 2 - 1)
    )
    padded = np.pad(array, pad_size, "symmetric")

    mean = bn.move_mean(padded, window)
    return mean[window - 1 :]


class ChannelStats(object):
    def __init__(self, nchans: int, nsamps: int):
        """Central central moments for filterbank channels in one pass.

        Parameters
        ----------
        nchans : int
            Number of channels in the data.
        nsamps : int
            Number of samples in the data.

        Notes
        -----
        The algorithm is numerically stable and accurate:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        https://www.johndcook.com/blog/skewness_kurtosis/
        https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
        """
        self._nchans = nchans
        self._nsamps = nsamps

        self._mbag = kernels.MomentsBag(nchans)

    @property
    def mbag(self) -> kernels.MomentsBag:
        """kernels.MomentsBag: The central moments of the data."""
        return self._mbag

    @property
    def nchans(self) -> int:
        """int: Get the number of channels."""
        return self._nchans

    @property
    def nsamps(self) -> int:
        """int: Get the number of samples."""
        return self._nsamps

    @property
    def maxima(self) -> np.ndarray:
        """np.ndarray: Get the maximum value of each channel."""
        return self._mbag.max

    @property
    def minima(self) -> np.ndarray:
        """np.ndarray: Get the minimum value of each channel."""
        return self._mbag.min

    @property
    def mean(self) -> np.ndarray:
        """np.ndarray: Get the mean of each channel."""
        return self._mbag.m1

    @property
    def var(self) -> np.ndarray:
        """np.ndarray: Get the variance of each channel."""
        return self._mbag.m2 / self.nsamps

    @property
    def std(self) -> np.ndarray:
        """np.ndarray: Get the standard deviation of each channel."""
        return np.sqrt(self._mbag.m2 / self.nsamps)

    @property
    def skew(self) -> np.ndarray:
        """np.ndarray: Get the skewness of each channel."""
        return (
            np.divide(
                self._mbag.m3,
                np.power(self._mbag.m2, 1.5),
                out=np.zeros_like(self._mbag.m3),
                where=self._mbag.m2 != 0,
            )
            * np.sqrt(self.nsamps)
        )

    @property
    def kurtosis(self) -> np.ndarray:
        """np.ndarray: Get the kurtosis of each channel."""
        return (
            np.divide(
                self._mbag.m4,
                np.power(self._mbag.m2, 2.0),
                out=np.zeros_like(self._mbag.m4),
                where=self._mbag.m2 != 0,
            )
            * self.nsamps
            - 3.0
        )

    def push_data(
        self, array: np.ndarray, gulp_size: int, start_index: int, mode: str = "basic"
    ):
        if mode == "basic":
            kernels.compute_online_moments_basic(array, self.mbag, gulp_size, start_index)
        else:
            kernels.compute_online_moments(array, self.mbag, gulp_size, start_index)

    def __add__(self, other: type[ChannelStats]) -> type[ChannelStats]:
        """Add two ChannelStats objects together as if all the data belonged to one.

        Parameters
        ----------
        other : type[ChannelStats]
            The other ChannelStats object to add.

        Returns
        -------
        type[ChannelStats]
            The sum of the two ChannelStats objects.

        Raises
        ------
        TypeError
            If the other object is not a ChannelStats object.
        """
        if not isinstance(other, ChannelStats):
            raise TypeError("ChannelStats can only be added to other ChannelStats object")

        combined = ChannelStats(self.nchans, self.nsamps)
        kernels.add_online_moments(self.mbag, other.mbag, combined.mbag)
        return combined
