from __future__ import annotations

from typing import Callable

import attrs
import bottleneck as bn
import numpy as np
from astropy import stats as astrostats

from sigpyproc.core import kernels


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class ZScoreResult:
    """Result of a Z-score calculation.

    Attributes
    ----------
    zscores: np.ndarray
        Robust Z-scores of the array.
    loc: float
        Estimated location used for the Z-score calculation.
    scale: float | np.ndarray
        Estimated scale used for the Z-score calculation.
    """

    zscores: np.ndarray
    loc: float
    scale: float | np.ndarray


def running_filter(
    array: np.ndarray,
    window: int,
    filter_func: str = "mean",
) -> np.ndarray:
    """
    Calculate the running filter of an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to calculate the running filter of.
    window : int
        The window size of the filter.
    filter_func : str, optional
        The filter function to use, by default "mean".

    Returns
    -------
    numpy.ndarray
        The running filter of the array.

    Raises
    ------
    ValueError
        If the filter function is not "mean" or "median".

    Notes
    -----
    Window edges are handled by reflecting about the edges.

    """
    pad_size = (
        (window // 2, window // 2) if window % 2 else (window // 2, window // 2 - 1)
    )
    padded_ar = np.pad(array, pad_size, "symmetric")
    if filter_func == "mean":
        filtered_ar = bn.move_mean(padded_ar, window)
    elif filter_func == "median":
        filtered_ar = bn.move_median(padded_ar, window)
    else:
        msg = f"Filter function not recognized: {filter_func}"
        raise ValueError(msg)
    return filtered_ar[window - 1 :]


def estimate_loc(array: np.ndarray, method: str = "median") -> float:
    """Estimate the location of an array.

    Parameters
    ----------
    array : np.ndarray
        The array to estimate the location of.
    method : str, optional
        The method to use for estimating the location, by default "median".

    Returns
    -------
    float
        The estimated location of the array.

    Raises
    ------
    ValueError
        If the method is not supported
    """
    if method == "median":
        loc = np.median(array)
    elif method == "mean":
        loc = np.mean(array)
    else:
        msg = f"Method {method} is not supported for estimating location."
        raise ValueError(msg)
    return loc


def _scale_iqr(array: np.ndarray) -> float:
    """Calculate the normalized Inter-quartile Range (IQR) scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The normalized IQR scale of the array.
    """
    # \scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25)
    norm = 1.3489795003921634
    q1, q3 = np.percentile(array, [25, 75])
    return (q3 - q1) / norm


def _scale_mad(array: np.ndarray) -> float:
    """Calculate the Median Absolute Deviation (MAD) scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The MAD scale of the array.

    Notes
    -----
    https://www.ibm.com/docs/en/cognos-analytics/12.0.0?topic=terms-modified-z-score
    """
    norm = 0.6744897501960817  # scipy.stats.norm.ppf(0.75)
    norm_aad = np.sqrt(2 / np.pi)
    diff = array - np.median(array)
    mad = np.median(np.abs(diff)) / norm
    if mad == 0:
        return np.mean(np.abs(diff)) / norm_aad
    return mad


def _scale_doublemad(array: np.ndarray) -> np.ndarray:
    """Calculate the Double MAD scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        The Double MAD scale of the array.

    Notes
    -----
    The Double MAD is defined as:
    https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    https://aakinshin.net/posts/harrell-davis-double-mad-outlier-detector/

    """
    norm = 0.6744897501960817  # scipy.stats.norm.ppf(0.75)
    norm_aad = np.sqrt(2 / np.pi)
    med = np.median(array)
    diff = array - med
    mad_left = np.median(np.abs(diff[array <= med])) / norm
    mad_right = np.median(np.abs(diff[array >= med])) / norm
    if mad_left == 0:
        scale_left = np.mean(np.abs(diff[array <= med])) / norm_aad
    else:
        scale_left = mad_left

    if mad_right == 0:
        scale_right = np.mean(np.abs(diff[array >= med])) / norm_aad
    else:
        scale_right = mad_right

    return np.where(array < med, scale_left, scale_right)


def _scale_diffcov(array: np.ndarray) -> float:
    """Calculate the Difference Covariance scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The Difference Covariance scale of the array.
    """
    diff = np.diff(array)
    return np.sqrt(-np.cov(diff[:-1], diff[1:])[0, 1])


def _scale_biweight(array: np.ndarray) -> float:
    """Calculate the Biweight Midvariance scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The Biweight Midvariance scale of the array.
    """
    return astrostats.biweight_scale(array)


def _scale_qn(array: np.ndarray) -> float:
    """Calculate the Normalized Qn scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The Normalized Qn scale of the array.
    """
    # \np.sqrt(2) * stats.norm.ppf(5/8)
    norm = 0.4506241100243562
    h = len(array) // 2 + 1
    k = h * (h - 1) // 2
    diffs = np.abs(array[:, None] - array)
    return np.partition(diffs.ravel(), k - 1)[k - 1] / norm


def _scale_sn(array: np.ndarray) -> float:
    """Calculate the Normalized Sn scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The Normalized Sn scale of the array.
    """
    diffs = np.abs(array[:, None] - array)
    return 1.1926 * np.median(np.median(diffs, axis=1))


def _scale_gapper(array: np.ndarray) -> float:
    """Calculate the Gapper Estimator scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The Gapper Estimator scale of the array.
    """
    n = len(array)
    gaps = np.diff(np.sort(array))
    weights = np.arange(1, n) * np.arange(n - 1, 0, -1)
    return np.dot(weights, gaps) * np.sqrt(np.pi) / (n * (n - 1))


def estimate_scale(array: np.ndarray, method: str = "mad") -> float | np.ndarray:
    """Estimate the scale or standard deviation of an array.

    Parameters
    ----------
    array : np.ndarray
        The array to estimate the scale of.
    method : str, optional
        The method to use for estimating the scale, by default "mad".

    Returns
    -------
    float | np.ndarray
        The estimated scale of the array.

    Raises
    ------
    ValueError
        If the method is not supported or if the array is empty.

    Notes
    -----
    https://en.wikipedia.org/wiki/Robust_measures_of_scale

    Following methods are supported:
    - "iqr": Normalized Inter-quartile Range
    - "mad": Median Absolute Deviation
    - "diffcov": Difference Covariance
    - "biweight": Biweight Midvariance
    - "qn": Normalized Qn scale
    - "sn": Normalized Sn scale
    - "gapper": Gapper Estimator
    """
    scale_methods: dict[str, Callable[[np.ndarray], float | np.ndarray]] = {
        "iqr": _scale_iqr,
        "mad": _scale_mad,
        "doublemad": _scale_doublemad,
        "diffcov": _scale_diffcov,
        "biweight": _scale_biweight,
        "qn": _scale_qn,
        "sn": _scale_sn,
        "gapper": _scale_gapper,
    }
    array = np.asarray(array)
    n = len(array)
    if n == 0:
        msg = "Cannot estimate noise from an empty array."
        raise ValueError(msg)

    scale_func = scale_methods.get(method)
    if scale_func is None:
        msg = f"Method {method} is not supported for estimating scale."
        raise ValueError(msg)
    return scale_func(array)


def zscore(
    array: np.ndarray,
    loc_method: str = "median",
    scale_method: str = "mad",
) -> ZScoreResult:
    """Calculate robust Z-scores of an array.

    Parameters
    ----------
    array : np.ndarray
        The array to calculate the Z-score of.
    loc_method : str, optional
        The method to use for estimating the location, by default "median".
    scale_method : str, optional
        The method to use for estimating the scale, by default "mad".

    Returns
    -------
    ZScoreResult
        The robust Z-scores of the array.

    Raises
    ------
    ValueError
        If the location or scale method is not supported.
    """
    loc = estimate_loc(array, loc_method)
    scale = estimate_scale(array, scale_method)
    diff = array - loc
    zscores = np.divide(diff, scale, out=np.zeros_like(diff), where=scale != 0)
    return ZScoreResult(zscores=zscores, loc=loc, scale=scale)


class ChannelStats:
    def __init__(self, nchans: int, nsamps: int) -> None:
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
        https://doi.org/10.2172/1028931
        """
        self._nchans = nchans
        self._nsamps = nsamps

        self._mbag = kernels.MomentsBag(nchans)

    @property
    def mbag(self) -> kernels.MomentsBag:
        """:class:`~sigpyproc.core.kernels.MomentsBag`: Central moments of the data."""
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
        """numpy.ndarray: Get the maximum value of each channel."""
        return self._mbag.max

    @property
    def minima(self) -> np.ndarray:
        """numpy.ndarray: Get the minimum value of each channel."""
        return self._mbag.min

    @property
    def mean(self) -> np.ndarray:
        """numpy.ndarray: Get the mean of each channel."""
        return self._mbag.m1

    @property
    def var(self) -> np.ndarray:
        """numpy.ndarray: Get the variance of each channel."""
        return self._mbag.m2 / self.nsamps

    @property
    def std(self) -> np.ndarray:
        """numpy.ndarray: Get the standard deviation of each channel."""
        return np.sqrt(self._mbag.m2 / self.nsamps)

    @property
    def skew(self) -> np.ndarray:
        """numpy.ndarray: Get the skewness of each channel."""
        return np.divide(
            self._mbag.m3,
            np.power(self._mbag.m2, 1.5),
            out=np.zeros_like(self._mbag.m3),
            where=self._mbag.m2 != 0,
        ) * np.sqrt(self.nsamps)

    @property
    def kurtosis(self) -> np.ndarray:
        """numpy.ndarray: Get the kurtosis of each channel."""
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
        self,
        array: np.ndarray,
        gulp_size: int,
        start_index: int,
        mode: str = "basic",
    ) -> None:
        if mode == "basic":
            kernels.compute_online_moments_basic(
                array,
                self.mbag,
                gulp_size,
                start_index,
            )
        else:
            kernels.compute_online_moments(array, self.mbag, gulp_size, start_index)

    def __add__(self, other: ChannelStats) -> ChannelStats:
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
            msg = f"Only ChannelStats can be added together, not {type(other)}"
            raise TypeError(msg)

        combined = ChannelStats(self.nchans, self.nsamps)
        kernels.add_online_moments(self.mbag, other.mbag, combined.mbag)
        return combined
