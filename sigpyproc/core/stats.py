from __future__ import annotations

from typing import Callable, Literal

import attrs
import bottleneck as bn
import numpy as np
from astropy import stats as astrostats

from sigpyproc.core import kernels

LocMethodType = Literal["median", "mean"]
ScaleMethodType = Literal[
    "iqr",
    "mad",
    "doublemad",
    "diffcov",
    "biweight",
    "qn",
    "sn",
    "gapper",
]


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class ZScoreResult:
    """
    Container for Z-score calculation results.

    Parameters
    ----------
    data: ndarray
        Robust Z-scores of the input array (normalized data).
    loc: float
        Estimated location (central tendency) used for the Z-score calculation.
    scale: float | ndarray
        Estimated scale (variability) used for the Z-score calculation.
        Can be a scalar or an array matching the shape of `data`.
    """

    data: np.ndarray
    loc: float
    scale: float | np.ndarray


def running_filter(
    array: np.ndarray,
    window: int,
    method: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """
    Calculate the running filter of an array.

    Applies a sliding window filter to the input array using the specified method.

    Parameters
    ----------
    array : ndarray
        The input array to filter.
    window : int
        The size of the sliding window.
    method : {"mean", "median"}, optional
        The filtering method to use, by default "mean".

    Returns
    -------
    ndarray
        The filtered array with the same shape as the input array.

    Raises
    ------
    ValueError
        If the ``method`` is not supported.

    Notes
    -----
    Window edges are handled by reflecting about the edges of the input array.

    """
    filter_methods: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "mean": bn.move_mean,
        "median": bn.move_median,
    }
    array = np.asarray(array)
    filter_func = filter_methods.get(method)
    if filter_func is None:
        msg = f"Filter function not recognized: {method}"
        raise ValueError(msg)
    pad_size = (
        (window // 2, window // 2) if window % 2 else (window // 2, window // 2 - 1)
    )
    padded_ar = np.pad(array, pad_size, "symmetric")
    filtered_ar = filter_func(padded_ar, window)
    return filtered_ar[window - 1 :]


def estimate_loc(
    array: np.ndarray,
    method: LocMethodType = "median",
) -> float:
    """
    Estimate the location (central tendency) of an array.

    Parameters
    ----------
    array : ndarray
        The input array to estimate the location of.
    method : {"median", "mean"}, optional
        The method to use for estimating the location, by default "median".

    Returns
    -------
    float
        The estimated location of the array.

    Raises
    ------
    ValueError
        If the ``array`` is empty or if the ``method`` is not supported.
    """
    loc_methods: dict[str, Callable[[np.ndarray], float]] = {
        "median": np.median,
        "mean": np.mean,
    }
    array = np.asarray(array)
    if len(array) == 0:
        msg = "Cannot estimate loc from an empty array."
        raise ValueError(msg)

    loc_func = loc_methods.get(method)
    if loc_func is None:
        msg = f"Method {method} is not supported for estimating location."
        raise ValueError(msg)
    return loc_func(array)


def estimate_scale(
    array: np.ndarray,
    method: ScaleMethodType = "mad",
) -> float | np.ndarray:
    """
    Estimate the scale (variability) or standard deviation of an array.

    Parameters
    ----------
    array : ndarray
        The input array to estimate the scale of.
    method : {"iqr", "mad", "doublemad", "diffcov", "biweight", "qn",
        "sn", "gapper"}, optional

        The method to use for estimating the scale, by default "mad".

        - `iqr`: Normalized Inter-quartile Range.
        - `mad`: Median Absolute Deviation.
        - `doublemad`: Double MAD.
        - `diffcov`: Difference Covariance
        - `biweight`: Biweight Midvariance
        - `qn`: Normalized Qn scale
        - `sn`: Normalized Sn scale
        - `gapper`: Gapper Estimator

    Returns
    -------
    float | ndarray
        The estimated scale of the array. If the method is "doublemad", the
        output is an array of the same shape as the input array.

    Raises
    ------
    ValueError
        If the ``array`` is empty or if the ``method`` is not supported.

    References
    ----------
    .. [1] Wikipedia, "Robust measures of scale",
        https://en.wikipedia.org/wiki/Robust_measures_of_scale

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
    if len(array) == 0:
        msg = "Cannot estimate noise from an empty array."
        raise ValueError(msg)

    scale_func = scale_methods.get(method)
    if scale_func is None:
        msg = f"Method {method} is not supported for estimating scale."
        raise ValueError(msg)
    return scale_func(array)


def estimate_zscore(
    array: np.ndarray,
    loc_method: LocMethodType | Literal["norm"] = "median",
    scale_method: ScaleMethodType | Literal["norm"] = "mad",
) -> ZScoreResult:
    """
    Calculate robust Z-scores of an array.

    Parameters
    ----------
    array : ndarray
        The input array to calculate the Z-score of.
    loc_method : {"median", "mean", "norm"}, optional
        The method to use for estimating the location, by default "median".

        Use "norm" to set the location to 0.
    scale_method : {"mad", "iqr", "doublemad", "diffcov", "biweight", "qn", "sn",
        "gapper", "norm"}, optional

        The method to use for estimating the scale, by default "mad".

        Use "norm" to set the scale to 1.

    Returns
    -------
    ZScoreResult
        A container with the Z-scores, estimated location, and scale.

    Raises
    ------
    ValueError
        If the ``loc_method`` or ``scale_method`` is not supported.

    See Also
    --------
    estimate_loc, estimate_scale
    """
    loc = 0 if loc_method == "norm" else estimate_loc(array, loc_method)
    scale = 1 if scale_method == "norm" else estimate_scale(array, scale_method)
    diff = array - loc
    zscores = np.divide(diff, scale, out=np.zeros_like(diff), where=scale != 0)
    return ZScoreResult(data=zscores, loc=loc, scale=scale)


def _scale_iqr(array: np.ndarray) -> float:
    """
    Calculate the normalized Inter-quartile Range (IQR) scale of an array.

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
    """
    Calculate the Median Absolute Deviation (MAD) scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    float
        The MAD scale of the array.

    References
    ----------
    .. [1] IBM, "Modified Z-score",
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
    """
    Calculate the Double MAD scale of an array.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        The Double MAD scale of the array.

    References
    ----------
    .. [1] Eureka Statistics, "Using the Median Absolute Deviation to Find Outliers",
        https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    .. [2] A. Akinshin, "Harrell-Davis Double MAD Outlier Detector",
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
    """
    Calculate the Difference Covariance scale of an array.

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
    """
    Calculate the Biweight Midvariance scale of an array.

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
    """
    Calculate the Normalized Qn scale of an array.

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
    n = len(array)
    h = n // 2 + 1
    k = h * (h - 1) // 2
    diffs = np.abs(array[:, None] - array)
    return np.partition(diffs[np.triu_indices(n, k=1)].ravel(), k - 1)[k - 1] / norm


def _scale_sn(array: np.ndarray) -> float:
    """
    Calculate the Normalized Sn scale of an array.

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
    """
    Calculate the Gapper Estimator scale of an array.

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


class ChannelStats:
    """
    A class to compute the central moments of filterbank data in one pass.

    Parameters
    ----------
    nchans : int
        Number of channels in the data.
    nsamps : int
        Number of samples in the data.

    References
    ----------
    .. [1] Wikipedia, "Algorithms for calculating variance",
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    .. [2] John D. Cook, "Skewness and kurtosis formulas for normal distributions",
        https://www.johndcook.com/blog/skewness_kurtosis/
    .. [3] Pebay, Philippe P., "One-Pass covariances and Statistical Moments",
        https://doi.org/10.2172/1028931
    """

    def __init__(self, nchans: int, nsamps: int) -> None:
        self._nchans = nchans
        self._nsamps = nsamps
        self._moments = np.zeros(nchans, dtype=kernels.moments_dtype)

    @property
    def moments(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the central moments of the data."""
        return self._moments

    @property
    def nchans(self) -> int:
        """:obj:`int`: Get the number of channels."""
        return self._nchans

    @property
    def nsamps(self) -> int:
        """:obj:`int`: Get the number of samples."""
        return self._nsamps

    @property
    def maxima(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the maximum value of each channel."""
        return self._moments["max"]

    @property
    def minima(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the minimum value of each channel."""
        return self._moments["min"]

    @property
    def mean(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the mean of each channel."""
        return self._moments["m1"]

    @property
    def var(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the variance of each channel."""
        return self._moments["m2"] / self.nsamps

    @property
    def std(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the standard deviation of each channel."""
        return np.sqrt(self.var)

    @property
    def skew(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the skewness of each channel."""
        return np.divide(
            self._moments["m3"],
            np.power(self._moments["m2"], 1.5),
            out=np.zeros_like(self._moments["m3"]),
            where=self._moments["m2"] != 0,
        ) * np.sqrt(self.nsamps)

    @property
    def kurtosis(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Get the kurtosis of each channel."""
        return (
            np.divide(
                self._moments["m4"],
                np.power(self._moments["m2"], 2.0),
                out=np.zeros_like(self._moments["m4"]),
                where=self._moments["m2"] != 0,
            )
            * self.nsamps
            - 3.0
        )

    def push_data(
        self,
        array: np.ndarray,
        start_index: int,
        mode: Literal["basic", "full"] = "basic",
    ) -> None:
        """
        Update the central moments of the data with new samples.

        Parameters
        ----------
        array : ndarray
            The input array to update the moments with.
        start_index : int
            The starting time (sample) index of the data.
        mode : {"basic", "full"}, optional
            The mode to use for computing the moments, by default "basic".

            - "basic": Compute the moments upto 2nd order (variance).
            - "full": Compute the moments upto 4th order (kurtosis).
        """
        if mode == "basic":
            kernels.compute_online_moments_basic(
                array,
                self._moments,
                start_index,
            )
        else:
            kernels.compute_online_moments(array, self._moments, start_index)

    def __add__(self, other: ChannelStats) -> ChannelStats:
        """
        Add two ChannelStats objects together as if all the data belonged to one.

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

        combined = ChannelStats(self.nchans, self.nsamps + other.nsamps)
        kernels.add_online_moments(self._moments, other._moments, combined._moments)
        return combined
