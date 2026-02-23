from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import attrs
import bottleneck as bn
import numpy as np
from astropy import stats as astrostats

from sigpyproc.core import kernels
from sigpyproc.utils import apply_along_axes

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    from sigpyproc.core.custom_types import FilterMethods, LocMethods, ScaleMethods


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class ZScoreResult:
    """Container for Z-score calculation results.

    Parameters
    ----------
    data : ndarray
        Robust Z-scores of the input array (normalized data).
    loc : float
        Estimated location (central tendency) used for the Z-score calculation.
    scale : float | ndarray
        Estimated scale (variability) used for the Z-score calculation.
        Can be a scalar or an array matching the shape of ``data``.

    Attributes
    ----------
    data
    loc
    scale
    """

    data: NDArray[np.floating]
    loc: NDArray[np.floating]
    scale: NDArray[np.floating]


def downsample_1d(
    array: np.ndarray,
    factor: int,
    method: FilterMethods = "mean",
) -> np.ndarray:
    """Downsample a 1D array by reduction factor using a specified method.

    Remainder samples are dropped.

    Parameters
    ----------
    array : ndarray
        The input 1D array to downsample. Supports 1D numeric dtype.
    factor : int
        The downsampling factor.
    method : {"mean", "median"}, optional
        The method to use for downsampling, by default "mean".

    Returns
    -------
    ndarray
        The downsampled array.

    Raises
    ------
    ValueError
        If the ``method`` is not supported.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        msg = "Input array must be a 1D numpy array."
        raise ValueError(msg)
    if factor <= 0 or not isinstance(factor, int):
        msg = "Factor must be a positive integer."
        raise ValueError(msg)
    if factor > array.size:
        msg = "Factor must be less than the size of the array."
        raise ValueError(msg)
    if method == "mean":
        return kernels.downsample_1d_mean(array, factor)
    if method == "median":
        nsamps_new = (array.size // factor) * factor
        return np.median(array[:nsamps_new].reshape(-1, factor), axis=1)
    msg = f"Method {method} is not supported for downsampling."
    raise ValueError(msg)


def downsample_2d(
    array: np.ndarray,
    factors: tuple[int, int],
    method: FilterMethods = "mean",
) -> np.ndarray:
    """Downsample a 2D array by averaging over bins in both dimensions.

    Parameters
    ----------
    array : np.ndarray
        Input 2D array to be downsampled.
    factors : tuple[int, int]
        Downsampling factors (factor1, factor2) for each dimension.
        Must be positive integers.
    method : {"mean", "median"}, optional
        Downsampling method, by default "mean".

    Returns
    -------
    np.ndarray
        Downsampled array with same layout as input.
    """
    factor1, factor2 = factors
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        msg = "Input array must be a 2D numpy array."
        raise ValueError(msg)
    if not all(isinstance(f, int) and f > 0 for f in (factor1, factor2)):
        msg = "Factors must be positive integers."
        raise ValueError(msg)
    if method not in {"mean", "median"}:
        msg = f"method must be either 'mean' or 'median', not {method}."
        raise ValueError(msg)
    np_op = getattr(np, method)
    dim1, dim2 = array.shape
    new_dim1 = dim1 // factor1
    new_dim2 = dim2 // factor2
    new_shape = (new_dim1, factor1, new_dim2, factor2)
    return np_op(
        array[: new_dim1 * factor1, : new_dim2 * factor2].reshape(new_shape),
        axis=(1, 3),
    )


def downsample_2d_flat(
    array: np.ndarray,
    factor1: int,
    factor2: int,
    dim1: int,
    dim2: int,
    method: FilterMethods = "mean",
) -> np.ndarray:
    """Downsample a flattened 2D array by averaging over bins in both dimensions.

    Parameters
    ----------
    array : np.ndarray
        Input flattened 2D array to be downsampled.
    factor1 : int
        Downsampling factor for the first dimension. Must be a positive integer.
    factor2 : int
        Downsampling factor for the second dimension. Must be a positive integer.
    dim1 : int
        Number of bins in the first dimension.
    dim2 : int
        Number of bins in the second dimension.
    method : {"mean", "median"}, optional
        Downsampling method, by default "mean".

    Returns
    -------
    np.ndarray
        Downsampled flattened 2D array

    Notes
    -----
    dim2 must ve the fastest varying dimension.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        msg = "Input array must be a 1D numpy array."
        raise ValueError(msg)
    if factor1 <= 0 or not isinstance(factor1, int):
        msg = "Factor1 must be a positive integer."
        raise ValueError(msg)
    if factor2 <= 0 or not isinstance(factor2, int):
        msg = "Factor2 must be a positive integer."
        raise ValueError(msg)
    if len(array) != dim1 * dim2:
        msg = "Array length must be equal to dim1 * dim2."
        raise ValueError(msg)
    if method == "mean":
        return kernels.downsample_2d_mean_flat(array, factor1, factor2, dim1, dim2)
    if method == "median":
        new_dim1 = dim1 // factor1
        new_dim2 = dim2 // factor2
        new_shape = (new_dim1, factor1, new_dim2, factor2)
        arr_2d = array.reshape(dim1, dim2)[: new_dim1 * factor1, : new_dim2 * factor2]
        result = np.median(arr_2d.reshape(new_shape), axis=(1, 3))
        return result.ravel()
    msg = f"Method {method} is not supported for downsampling."
    raise ValueError(msg)


def running_filter(
    array: np.ndarray,
    window: int,
    method: FilterMethods = "mean",
) -> np.ndarray:
    """Calculate the running filter of an array.

    Applies a sliding window filter to the input array using the specified method.
    Window edges are handled by reflecting about the edges of the input array.

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


def running_filter_fast(
    array: np.ndarray,
    window: int,
    method: FilterMethods = "mean",
    min_points: int = 101,
) -> np.ndarray:
    """Calculate an approximate running filter of an array.

    Downsamples the input array, apply the running filter, and then interpolate
    back to the original size. This is faster than the regular running filter for
    large arrays.

    Parameters
    ----------
    array : ndarray
        The input array to filter.
    window : int
        The size of the sliding window.
    method : {"mean", "median"}, optional
        The filtering method to use, by default "mean".
    min_points : int, optional
        The minimum number of nsamples for the downsampled array, by default 101.
        Lower values will result in faster processing but less accurate results.

    Returns
    -------
    ndarray
        The filtered array with the same shape as the input array.
    """
    ds_factor = int(max(1, window / min_points))
    if ds_factor == 1:
        return running_filter(array, window, method)
    ds = downsample_1d(array, ds_factor, "mean")
    filtered_ds = running_filter(ds, min_points, method)
    x_ds = np.arange(ds.size) * ds_factor + 0.5 * (ds_factor - 1)
    return np.interp(np.arange(array.size), x_ds, filtered_ds)


def estimate_loc(
    data: ArrayLike,
    method: LocMethods = "median",
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> float | NDArray[np.floating]:
    """Estimate the location (central tendency) of an array.

    Parameters
    ----------
    data : ArrayLike
        Input array or object that can be converted to an array.
    method : {"median", "mean"}, optional
        The method to use for estimating the location, by default "median".
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the location, by default None.
    keepdims : bool, optional
        If True, the reduced axis is kept in the result, by default True.

    Returns
    -------
    float | ndarray
        The estimated location of the array.

    Raises
    ------
    ValueError
        If the ``array`` is empty or if the ``method`` is not supported.
    """
    data = np.asanyarray(data)
    if len(data) == 0:
        msg = "Cannot estimate loc from an empty array."
        raise ValueError(msg)

    if method == "mean":
        return np.mean(data, axis=axis, keepdims=keepdims, dtype=np.float64)
    if method == "median":
        return np.median(data, axis=axis, keepdims=keepdims)
    msg = f"Method {method} is not supported for estimating location."
    raise ValueError(msg)


def estimate_scale(
    data: ArrayLike,
    method: ScaleMethods = "mad",
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> np.float64 | NDArray[np.float64]:
    """
    Estimate the scale (variability) or standard deviation of an array.

    Parameters
    ----------
    data : ArrayLike
        Input array or object that can be converted to an array.
    method : {"std", "iqr", "mad", "doublemad", "diffcov", "biweight", "qn", "sn", "gapper"}, optional
        The method to use for estimating the scale, by default "mad".

        - ``std`` : Standard Deviation.
        - ``iqr`` : Normalized Inter-quartile Range.
        - ``mad`` : Median Absolute Deviation.
        - ``doublemad`` : Double MAD.
        - ``diffcov`` : Difference Covariance
        - ``biweight`` : Biweight Midvariance
        - ``qn`` : Normalized Qn scale
        - ``sn`` : Normalized Sn scale
        - ``gapper`` : Gapper Estimator
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.
    keepdims : bool, optional
        If True, the reduced axis is kept in the result, by default True.

    Returns
    -------
    float64 | NDArray[float64]
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
    """  # noqa: E501
    scale_methods: dict[
        str,
        Callable[
            [NDArray[np.float64], int | tuple[int, ...] | None],
            np.float64 | NDArray[np.float64],
        ],
    ] = {
        "iqr": _scale_iqr,
        "mad": _scale_mad,
        "doublemad": _scale_doublemad,
        "diffcov": _scale_diffcov,
        "biweight": _scale_biweight,
        "qn": _scale_qn,
        "sn": _scale_sn,
        "gapper": _scale_gapper,
    }
    data = np.asanyarray(data, dtype=np.float64)
    if len(data) == 0:
        msg = "Cannot estimate scale from an empty array."
        raise ValueError(msg)

    if method == "std":
        return np.std(data, axis=axis, keepdims=keepdims, dtype=np.float64)
    scale_func = scale_methods.get(method)
    if scale_func is None:
        msg = f"Method {method} is not supported for estimating scale."
        raise ValueError(msg)
    result = scale_func(data, axis)
    # First: handle keepdims (regardless of scalar or ndarray)
    if keepdims and method != "doublemad":
        if axis is None:
            axis_tuple = tuple(range(data.ndim))
        elif isinstance(axis, int):
            axis_tuple = (axis,)
        else:
            axis_tuple = axis
        result = np.expand_dims(result, axis=axis_tuple)

    # Then: collapse to scalar only if not keepdims
    if isinstance(result, np.ndarray) and result.size == 1 and not keepdims:
        result = result.flat[0]
    return result


def estimate_zscore(
    data: ArrayLike,
    loc_method: LocMethods | Literal["norm"] = "median",
    scale_method: ScaleMethods | Literal["norm"] = "mad",
    axis: int | None = 0,
) -> ZScoreResult:
    """Calculate robust Z-scores of an array.

    Parameters
    ----------
    data : ArrayLike
        Input array or object that can be converted to an array.
    loc_method : {"median", "mean", "norm"}, optional
        The method to use for estimating the location, by default "median".

        Use "norm" to set the location to 0.
    scale_method : {"std", "iqr", "mad", "doublemad", "diffcov", "biweight", "qn", "sn", "gapper", "norm"}, optional
        The method to use for estimating the scale, by default "mad".

        Use "norm" to set the scale to 1.
    axis : int | None, optional
        Axis along which to compute the Z-scores, by default 0.

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
    scipy.stats.zscore
    """  # noqa: E501
    # use np.float64 for loc and scale to avoid overflow in calculations
    data = np.asanyarray(data, dtype=np.float32)
    if data.size == 0:
        msg = "Cannot estimate Z-scores from an empty array."
        raise ValueError(msg)
    loc = (
        np.zeros(1, dtype=data.dtype)
        if loc_method == "norm"
        else estimate_loc(data, loc_method, axis, keepdims=True)
    )
    scale = (
        np.ones(1, dtype=data.dtype)
        if scale_method == "norm"
        else estimate_scale(data, scale_method, axis, keepdims=True)
    )
    zero_scales = np.isclose(scale, 0)
    if np.any(zero_scales):
        scale = np.where(zero_scales, 1, scale)

    zscores = np.subtract(data, loc, dtype=np.float32)
    np.divide(zscores, scale, out=zscores)
    return ZScoreResult(data=zscores, loc=np.asarray(loc), scale=np.asarray(scale))


def _scale_iqr(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the normalized Inter-quartile Range (IQR) scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.


    Returns
    -------
    np.float64 | NDArray[np.float64]
        The normalized IQR scale of the array.
    """
    data = np.asanyarray(data, dtype=np.float64)
    norm = 1.3489795003921634  # scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25)
    percentiles = np.percentile(data, [25, 75], axis=axis, keepdims=True)
    return np.squeeze(np.diff(percentiles, axis=0) / norm)


def _scale_mad(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the Median Absolute Deviation (MAD) scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.


    Returns
    -------
    np.float64 | NDArray[np.float64]
        The MAD scale of the array.

    References
    ----------
    .. [1] IBM, "Modified Z-score",
        https://www.ibm.com/docs/en/cognos-analytics/12.0.0?topic=terms-modified-z-score
    """
    data = np.asanyarray(data, dtype=np.float64)
    norm = 0.6744897501960817  # scipy.stats.norm.ppf(0.75)
    norm_aad = np.sqrt(2 / np.pi)
    loc = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.abs(data - loc), axis=axis, keepdims=True) / norm
    # Handle zero MAD case using np.isclose for stability
    is_zero_mad = np.isclose(mad, 0)
    if np.any(is_zero_mad):
        aad = np.mean(np.abs(data - loc), axis=axis, keepdims=True) / norm_aad
        mad = np.where(is_zero_mad, aad, mad)
    return np.squeeze(mad)


def _scale_doublemad(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    """Calculate the Double MAD scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.


    Returns
    -------
    np.float64 | NDArray[np.float64]
        The Double MAD scale of the array.

    References
    ----------
    .. [1] Eureka Statistics, "Using the Median Absolute Deviation to Find Outliers",
        https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    .. [2] A. Akinshin, "Harrell-Davis Double MAD Outlier Detector",
        https://aakinshin.net/posts/harrell-davis-double-mad-outlier-detector/
    """
    data = np.asanyarray(data, dtype=np.float64)
    norm = 0.6744897501960817  # scipy.stats.norm.ppf(0.75)
    norm_aad = np.sqrt(2 / np.pi)
    loc = np.median(data, axis=axis, keepdims=True)
    diff = data - loc

    # Split into left and right deviations
    data_left = np.where(data <= loc, np.abs(diff), np.nan)
    data_right = np.where(data >= loc, np.abs(diff), np.nan)
    mad_left = np.nanmedian(data_left, axis=axis, keepdims=True) / norm
    mad_right = np.nanmedian(data_right, axis=axis, keepdims=True) / norm

    # Replace zero MADs with mean absolute deviation
    mad_left = np.where(
        np.isclose(mad_left, 0),
        np.nanmean(data_left, axis=axis, keepdims=True) / norm_aad,
        mad_left,
    )
    mad_right = np.where(
        np.isclose(mad_right, 0),
        np.nanmean(data_right, axis=axis, keepdims=True) / norm_aad,
        mad_right,
    )
    return np.where(data < loc, mad_left, mad_right)


def _scale_diffcov(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the Difference Covariance scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.

    Returns
    -------
    np.float64 | NDArray[np.float64]
        The Difference Covariance scale of the array.
    """
    data = np.asanyarray(data, dtype=np.float64)
    return apply_along_axes(_scale_diffcov_1d, data, axis)


def _scale_diffcov_1d(data: NDArray[np.float64]) -> np.float64:
    """Calculate the Difference Covariance scale of a 1D array."""
    diff = np.diff(data)
    cov = np.cov(diff[:-1], diff[1:])
    return np.sqrt(np.abs(cov[0, 1]))


def _scale_biweight(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the Biweight Midvariance scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.

    Returns
    -------
    np.float64 | NDArray[np.float64]
        The Biweight Midvariance scale of the array.
    """
    return astrostats.biweight_scale(data, axis=axis)  # type: ignore[return-value]


def _scale_qn(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the Normalized Qn scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.

    Returns
    -------
    np.float64 | NDArray[np.float64]
        The Normalized Qn scale of the array.
    """
    data = np.asanyarray(data, dtype=np.float64)
    return apply_along_axes(_scale_qn_1d, data, axis)


def _scale_qn_1d(data: NDArray[np.float64]) -> np.float64:
    """Calculate the Normalized Qn scale of an array."""
    norm = 0.4506241100243562  # np.sqrt(2) * stats.norm.ppf(5/8)
    n = len(data)
    h = n // 2 + 1
    k = h * (h - 1) // 2
    diffs = np.abs(data[:, None] - data)
    return np.partition(diffs[np.triu_indices(n, k=1)].ravel(), k - 1)[k - 1] / norm


def _scale_sn(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the Normalized Sn scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.

    Returns
    -------
    np.float64 | NDArray[np.float64]
        The Normalized Sn scale of the array.
    """
    norm = 1.1926
    data = np.asanyarray(data, dtype=np.float64)
    diffs = np.abs(data[..., None] - data[..., None, :])
    median_diffs = np.median(diffs, axis=-1)
    return norm * np.median(median_diffs, axis=axis)


def _scale_gapper(
    data: NDArray[np.float64],
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | NDArray[np.float64]:
    """Calculate the Gapper Estimator scale of an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        Input array or object that can be converted to an array.
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to compute the scale, by default None.

    Returns
    -------
    np.float64 | NDArray[np.float64]
        The Gapper Estimator scale of the array.
    """
    data = np.asanyarray(data, dtype=np.float64)
    return apply_along_axes(_scale_gapper_1d, data, axis)


def _scale_gapper_1d(data: NDArray[np.float64]) -> np.float64:
    """Calculate the Gapper Estimator scale of an 1D array."""
    n = len(data)
    gaps = np.diff(np.sort(data))
    weights = np.arange(1, n) * np.arange(n - 1, 0, -1)
    return np.dot(weights, gaps) * np.sqrt(np.pi) / (n * (n - 1))


class ChannelStats:
    """A class to compute the central moments of filterbank data in one pass.

    Parameters
    ----------
    nchans : int
        Number of channels in the data.
    nsamps : int
        Number of samples in the data.

    Attributes
    ----------
    nchans
    nsamps
    moments
    maxima
    minima
    mean
    var
    std
    skew
    kurtosis

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
    def nchans(self) -> int:
        """Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return self._nchans

    @property
    def nsamps(self) -> int:
        """Get the number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return self._nsamps

    @property
    def moments(self) -> np.ndarray:
        """Get the central moments of the data.

        Returns
        -------
        ndarray
            The central moments of the data.
        """
        return self._moments

    @property
    def maxima(self) -> np.ndarray:
        """Get the maximum value of each channel.

        Returns
        -------
        ndarray
            The maximum value of each channel.
        """
        return self._moments["max"]

    @property
    def minima(self) -> np.ndarray:
        """Get the minimum value of each channel.

        Returns
        -------
        ndarray
            The minimum value of each channel.
        """
        return self._moments["min"]

    @property
    def mean(self) -> np.ndarray:
        """Get the mean of each channel.

        Returns
        -------
        ndarray
            The mean of each channel.
        """
        return self._moments["m1"]

    @property
    def var(self) -> np.ndarray:
        """Get the variance of each channel.

        Returns
        -------
        ndarray
            The variance of each channel.
        """
        return self._moments["m2"] / self.nsamps

    @property
    def std(self) -> np.ndarray:
        """Get the standard deviation of each channel.

        Returns
        -------
        ndarray
            The standard deviation of each channel.
        """
        return np.sqrt(self.var)

    @property
    def skew(self) -> np.ndarray:
        """Get the skewness of each channel.

        Returns
        -------
        ndarray
            The skewness of each channel.
        """
        return np.divide(
            self._moments["m3"],
            np.power(self._moments["m2"], 1.5),
            out=np.zeros_like(self._moments["m3"]),
            where=self._moments["m2"] != 0,
        ) * np.sqrt(self.nsamps)

    @property
    def kurtosis(self) -> np.ndarray:
        """Get the kurtosis of each channel.

        Returns
        -------
        ndarray
            The kurtosis of each channel.
        """
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
        """Update the central moments of the data with new samples.

        Parameters
        ----------
        array : ndarray
            The input array to update the moments with.
        start_index : int
            The starting time (sample) index of the data.
        mode : {"basic", "full"}, optional
            The mode to use for computing the moments, by default "basic".

            - "basic" : Compute the moments upto 2nd order (variance).
            - "full" : Compute the moments upto 4th order (kurtosis).
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
        """Add two ChannelStats objects together as if all the data belonged to one.

        Parameters
        ----------
        other : ChannelStats
            The other `ChannelStats` object to add.

        Returns
        -------
        ChannelStats
            The sum of the two `ChannelStats` objects.

        Raises
        ------
        TypeError
            If the other object is not a `ChannelStats` object.
        """
        if not isinstance(other, ChannelStats):
            msg = f"Only ChannelStats can be added together, not {type(other)}"
            raise TypeError(msg)

        combined = ChannelStats(self.nchans, self.nsamps + other.nsamps)
        kernels.add_online_moments(self._moments, other._moments, combined._moments)
        return combined
