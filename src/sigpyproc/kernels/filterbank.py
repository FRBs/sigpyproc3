"""Numba kernels for filterbank operations and downsampling."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def downsample_1d_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 1D array by averaging over bins."""
    nsamps_new = len(array) // factor
    result = np.empty(nsamps_new, dtype=array.dtype)
    for isamp in prange(nsamps_new):
        temp = np.float64(0.0)
        start = isamp * factor
        for ifactor in range(factor):
            temp += array[start + ifactor]
        result[isamp] = temp / factor
    return result


@njit(cache=True, fastmath=True)
def downsample_2d_mean_flat(
    array: np.ndarray,
    factor1: int,
    factor2: int,
    dim1: int,
    dim2: int,
) -> np.ndarray:
    """Downsample a flattened 2D array by averaging over bins in both dimensions."""
    new_dim1 = dim1 // factor1
    new_dim2 = dim2 // factor2
    totfactor = factor1 * factor2
    result = np.empty(new_dim1 * new_dim2, dtype=array.dtype)
    for i in prange(new_dim1):
        for j in range(new_dim2):
            temp = np.float64(0.0)
            pos = dim2 * i * factor1 + j * factor2
            for ifactor in range(factor1):
                ipos = pos + ifactor * dim2
                for ifactor2 in range(factor2):
                    temp += array[ipos + ifactor2]
            result[new_dim2 * i + j] = temp / totfactor
    return result


@njit(
    ["void(u1[::1], f4[::1], i8, i8, i8)", "void(f4[::1], f4[::1], i8, i8, i8)"],
    cache=True,
    fastmath=True,
)
def extract_tim(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    nchans: int,
    nsamps: int,
    out_start_index: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    for isamp in prange(nsamps):
        summ = np.float64(0.0)
        for ichan in range(nchans):
            summ += in_arr_2d[isamp, ichan]
        out_arr[out_start_index + isamp] = summ


@njit(
    ["void(u1[::1], f4[::1], i8, i8)", "void(f4[::1], f4[::1], i8, i8)"],
    cache=True,
    fastmath=True,
)
def extract_bpass(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    for isamp in range(nsamps):
        for ichan in range(nchans):
            out_arr[ichan] += in_arr_2d[isamp, ichan]


@njit(
    ["void(u1[::1], f4[::1], i8, i8)", "void(f4[::1], f4[::1], i8, i8)"],
    cache=True,
    fastmath=True,
    parallel=True,
)
def extract_bpass_par(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans).T
    for ichan in prange(nchans):
        summ = np.float64(0.0)
        for isamp in range(nsamps):
            summ += in_arr_2d[ichan, isamp]
        out_arr[ichan] += summ


@njit(
    ["void(u1[::1], b1[::1], u1, i8, i8)", "void(f4[::1], b1[::1], f4, i8, i8)"],
    cache=True,
    fastmath=True,
)
def mask_channels(
    in_arr: np.ndarray,
    chan_mask: np.ndarray,
    mask_value: float,
    nchans: int,
    nsamps: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    masked_chans = np.empty(nchans, dtype=np.int64)
    n_masked = 0
    for ichan in range(nchans):
        if chan_mask[ichan]:
            masked_chans[n_masked] = ichan
            n_masked += 1
    for isamp in prange(nsamps):
        for i in range(n_masked):
            ichan = masked_chans[i]
            in_arr_2d[isamp, ichan] = mask_value


@njit(
    ["u1[::1](u1[::1], i8, i8)", "f4[::1](f4[::1], i8, i8)"],
    cache=True,
    fastmath=True,
    parallel=True,
)
def invert_freq(in_arr: np.ndarray, nchans: int, nsamps: int) -> np.ndarray:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    out_arr_2d = np.empty((nsamps, nchans), dtype=in_arr.dtype)
    for isamp in prange(nsamps):
        out_arr_2d[isamp, :] = in_arr_2d[isamp, ::-1]
    return out_arr_2d.ravel()


@njit(
    [
        "void(u1[::1], u1[::1], f4[::1], i8, i8)",
        "void(f4[::1], f4[::1], f4[::1], i8, i8)",
    ],
    cache=True,
    fastmath=True,
)
def remove_zerodm(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    bpass: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    out_arr_2d = out_arr.reshape(nsamps, nchans)
    chanwts = bpass / bpass.sum()
    for isamp in prange(nsamps):
        zerodm = np.float64(0.0)
        for ichan in range(nchans):
            zerodm += in_arr_2d[isamp, ichan]
        for ichan in range(nchans):
            out_arr_2d[isamp, ichan] = (
                in_arr_2d[isamp, ichan] - zerodm * chanwts[ichan] + bpass[ichan]
            )


downsample_1d_mean_par = njit(downsample_1d_mean.py_func, parallel=True, fastmath=True)
downsample_2d_mean_par = njit(
    downsample_2d_mean_flat.py_func,
    parallel=True,
    fastmath=True,
)
extract_tim_par = njit(extract_tim.py_func, parallel=True, fastmath=True)
mask_channels_par = njit(mask_channels.py_func, parallel=True, fastmath=True)
invert_freq_par = njit(invert_freq.py_func, parallel=True, fastmath=True)
remove_zerodm_par = njit(remove_zerodm.py_func, parallel=True, fastmath=True)
