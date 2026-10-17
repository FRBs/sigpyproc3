"""Numba kernels for Welford online moments and detrending."""

from __future__ import annotations

import numpy as np
from numba import njit, prange, types

moments_dtype = np.dtype(
    [
        ("count", np.int64),
        ("m1", np.float32),
        ("m2", np.float32),
        ("m3", np.float32),
        ("m4", np.float32),
        ("min", np.float32),
        ("max", np.float32),
    ],
    align=True,
)


@njit(cache=True, fastmath=True)
def update_moments(
    val: float,
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    n: int,
) -> tuple[float, float, float, float, int]:
    n += 1
    delta = val - m1
    delta_n = delta / n
    delta_n2 = delta_n * delta_n
    term = delta * delta_n * (n - 1)

    m1 += delta_n
    m4 += term * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
    m3 += term * delta_n * (n - 2) - 3 * delta_n * m2
    m2 += term
    return m1, m2, m3, m4, n


@njit(cache=True, fastmath=True)
def update_moments_basic(
    val: float,
    m1: float,
    m2: float,
    n: int,
) -> tuple[float, float, int]:
    n += 1
    delta = val - m1
    delta_n = delta / n

    m1 += delta_n
    m2 += delta * delta_n * (n - 1)
    return m1, m2, n


@njit(cache=True, parallel=True, fastmath=True, locals={"val": types.f4})
def compute_online_moments(
    array: np.ndarray,
    moments: np.ndarray,
    startflag: int = 0,
) -> None:
    """Compute central moments in one pass through the data."""
    nchans = moments.shape[0]
    nsamps = array.shape[0] // nchans

    if startflag == 0:
        for ichan in range(nchans):
            moments[ichan]["min"] = array[ichan]
            moments[ichan]["max"] = array[ichan]

    for ichan in prange(nchans):
        m1, m2, m3, m4 = (
            moments[ichan]["m1"],
            moments[ichan]["m2"],
            moments[ichan]["m3"],
            moments[ichan]["m4"],
        )
        count = moments[ichan]["count"]
        min_val, max_val = moments[ichan]["min"], moments[ichan]["max"]

        for isamp in range(nsamps):
            val = array[isamp * nchans + ichan]
            m1, m2, m3, m4, count = update_moments(val, m1, m2, m3, m4, count)
            min_val = min(min_val, val)
            max_val = max(max_val, val)
        (
            moments[ichan]["m1"],
            moments[ichan]["m2"],
            moments[ichan]["m3"],
            moments[ichan]["m4"],
        ) = m1, m2, m3, m4
        moments[ichan]["count"] = count
        moments[ichan]["min"], moments[ichan]["max"] = min_val, max_val


@njit(cache=True, parallel=True, fastmath=True, locals={"val": types.f4})
def compute_online_moments_basic(
    array: np.ndarray,
    moments: np.ndarray,
    startflag: int = 0,
) -> None:
    """Compute central moments in one pass through the data."""
    nchans = moments.shape[0]
    nsamps = array.shape[0] // nchans

    if startflag == 0:
        for ichan in range(nchans):
            moments[ichan]["min"] = array[ichan]
            moments[ichan]["max"] = array[ichan]

    for ichan in prange(nchans):
        m1, m2 = moments[ichan]["m1"], moments[ichan]["m2"]
        count = moments[ichan]["count"]
        min_val, max_val = moments[ichan]["min"], moments[ichan]["max"]

        for isamp in range(nsamps):
            val = array[isamp * nchans + ichan]
            m1, m2, count = update_moments_basic(val, m1, m2, count)
            min_val = min(min_val, val)
            max_val = max(max_val, val)
        moments[ichan]["m1"], moments[ichan]["m2"] = m1, m2
        moments[ichan]["count"] = count
        moments[ichan]["min"], moments[ichan]["max"] = min_val, max_val


@njit(cache=True, fastmath=True)
def add_online_moments(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    c["count"][:] = a["count"] + b["count"]
    delta = b["m1"] - a["m1"]
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta2 * delta2

    c["m1"][:] = (a["count"] * a["m1"] + b["count"] * b["m1"]) / c["count"]
    c["m2"][:] = a["m2"] + b["m2"] + delta2 * a["count"] * b["count"] / c["count"]
    c["m3"][:] = (
        a["m3"]
        + b["m3"]
        + delta3
        * a["count"]
        * b["count"]
        * (a["count"] - b["count"])
        / (c["count"] ** 2)
    )
    c["m3"][:] += 3 * delta * (a["count"] * b["m2"] - b["count"] * a["m2"]) / c["count"]
    c["m4"][:] = (
        a["m4"]
        + b["m4"]
        + delta4
        * a["count"]
        * b["count"]
        * (a["count"] ** 2 - a["count"] * b["count"] + b["count"] ** 2)
        / (c["count"] ** 3)
    )
    c["m4"][:] += (
        6
        * delta2
        * (a["count"] ** 2 * b["m2"] + b["count"] ** 2 * a["m2"])
        / (c["count"] ** 2)
    )
    c["m4"][:] += 4 * delta * (a["count"] * b["m3"] - b["count"] * a["m3"]) / c["count"]
    c["max"][:] = np.maximum(a["max"], b["max"])
    c["min"][:] = np.minimum(a["min"], b["min"])


@njit("f4[::1](f4[::1])", cache=True, fastmath=True)
def detrend_1d(arr: np.ndarray) -> np.ndarray:
    """Detrend a 1D array using a linear fit.

    Accumulate centered sums in float64 in one pass, using the closed-form
    coordinate variance. A second pass writes the float32 residuals.
    """
    m = arr.shape[0]
    if m == 0:
        msg = "Input array must be non-empty."
        raise ValueError(msg)
    if m == 1:
        return np.zeros(1, dtype=arr.dtype)
    # Center coordinates and remove a reference level before accumulating in
    # float64. Since sum(x) = 0, covariance needs no separate mean-removal pass.
    m_f = np.float64(m)
    center = (m_f - 1.0) / 2.0
    reference = np.float64(arr[0])
    total = 0.0
    covariance = 0.0
    for i in range(m):
        value = np.float64(arr[i]) - reference
        total += value
        covariance += (i - center) * value
    mean = total / m_f
    variance = m_f * (m_f * m_f - 1.0) / 12.0
    slope = covariance / variance
    result = np.empty(m, dtype=np.float32)
    for i in range(m):
        result[i] = (np.float64(arr[i]) - reference) - mean - slope * (i - center)
    return result


@njit("f4[:, ::1](f4[:, ::1])", cache=True, fastmath=True)
def detrend_2d(arr: np.ndarray) -> np.ndarray:
    """Detrend a 2D array using a linear fit."""
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        result[i] = detrend_1d(arr[i])
    return result
