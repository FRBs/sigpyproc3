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
        "void(u1[::1], u1[::1], f4[::1], i8, i8, i8)",
        "void(f4[::1], f4[::1], f4[::1], i8, i8, i8)",
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
    clip_max: int,
) -> None:
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    out_arr_2d = out_arr.reshape(nsamps, nchans)
    total = np.float64(0.0)
    for ichan in range(nchans):
        total += bpass[ichan]
    chanwts = np.empty(nchans, dtype=np.float64)
    if abs(total) < np.float64(1e-8):
        share = np.float64(1.0) / nchans
        for ichan in range(nchans):
            chanwts[ichan] = share
    else:
        for ichan in range(nchans):
            chanwts[ichan] = bpass[ichan] / total
    for isamp in prange(nsamps):
        zerodm = np.float64(0.0)
        for ichan in range(nchans):
            zerodm += in_arr_2d[isamp, ichan]
        for ichan in range(nchans):
            written = in_arr_2d[isamp, ichan] - zerodm * chanwts[ichan] + bpass[ichan]
            if clip_max > 0:
                if written < 0:
                    written = 0
                elif written > clip_max:
                    written = clip_max
            out_arr_2d[isamp, ichan] = written


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def decimate_2d(
    data: np.ndarray,
    out_rows: int,
    out_cols: int,
    mode: int = 0,
) -> np.ndarray:
    """Decimate a 2D float array to (out_rows, out_cols) using variable-width binning.

    Parameters
    ----------
    data : np.ndarray
        2D input array of shape (nrows, ncols), float32 or float64.
    out_rows : int
        Desired number of output rows.
    out_cols : int
        Desired number of output columns.
    mode : int, optional
        Decimation mode: 0=mean, 1=max, 2=absmax. Default is 0 (mean).

    Returns
    -------
    np.ndarray
        Decimated 2D array of shape (out_rows, out_cols), float32.
    """
    in_rows, in_cols = data.shape
    out = np.full((out_rows, out_cols), np.nan, dtype=np.float32)

    if in_rows == 0 or in_cols == 0 or out_rows == 0 or out_cols == 0:
        return out

    # Compute row boundaries
    row_starts = np.empty(out_rows, dtype=np.int64)
    row_ends = np.empty(out_rows, dtype=np.int64)
    for r in range(out_rows):
        r_start = int(r * in_rows / out_rows)
        r_end = int((r + 1) * in_rows / out_rows)
        if r_end <= r_start:
            r_end = r_start + 1
        r_end = min(r_end, in_rows)
        row_starts[r] = r_start
        row_ends[r] = r_end

    # Compute col boundaries
    col_starts = np.empty(out_cols, dtype=np.int64)
    col_ends = np.empty(out_cols, dtype=np.int64)
    for c in range(out_cols):
        c_start = int(c * in_cols / out_cols)
        c_end = int((c + 1) * in_cols / out_cols)
        if c_end <= c_start:
            c_end = c_start + 1
        c_end = min(c_end, in_cols)
        col_starts[c] = c_start
        col_ends[c] = c_end

    for r in prange(out_rows):
        r_start = row_starts[r]
        r_end = row_ends[r]
        for c in range(out_cols):
            c_start = col_starts[c]
            c_end = col_ends[c]

            count = 0
            val_sum = 0.0
            val_max = -1e30
            val_absmax = -1.0
            val_absmax_orig = 0.0

            for ir in range(r_start, r_end):
                for ic in range(c_start, c_end):
                    v = float(data[ir, ic])
                    if not np.isnan(v):
                        count += 1
                        val_sum += v
                        val_max = max(val_max, v)
                        abs_v = abs(v)
                        if abs_v > val_absmax:
                            val_absmax = abs_v
                            val_absmax_orig = v

            if count > 0:
                if mode == 0:
                    out[r, c] = np.float32(val_sum / count)
                elif mode == 1:
                    out[r, c] = np.float32(val_max)
                elif mode == 2:
                    out[r, c] = np.float32(val_absmax_orig)

    return out


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def minmax_envelope(
    x: np.ndarray,
    start: int,
    stop: int,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute min and max envelopes for a 1D slice across nbins.

    Parameters
    ----------
    x : np.ndarray
        1D array.
    start : int
        Start index (inclusive).
    stop : int
        Stop index (exclusive).
    nbins : int
        Number of output bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mins, maxs) each of length nbins (float32).
    """
    mins = np.full(nbins, np.nan, dtype=np.float32)
    maxs = np.full(nbins, np.nan, dtype=np.float32)

    total_len = stop - start
    if total_len <= 0 or nbins <= 0:
        return mins, maxs

    for b in prange(nbins):
        b_start = start + int(b * total_len / nbins)
        b_end = start + int((b + 1) * total_len / nbins)
        if b_end <= b_start:
            b_end = b_start + 1
        b_end = min(b_end, stop)

        v_min = 1e30
        v_max = -1e30
        count = 0
        for i in range(b_start, b_end):
            val = float(x[i])
            if not np.isnan(val):
                count += 1
                v_min = min(v_min, val)
                v_max = max(v_max, val)

        if count > 0:
            mins[b] = np.float32(v_min)
            maxs[b] = np.float32(v_max)

    return mins, maxs


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def robust_chan_norm(
    data: np.ndarray,
    out: np.ndarray,
    subsample: int = 8,
) -> None:
    """Normalize each channel by a strided median and MAD.

    The scale is ``1.4826 * MAD``, so a Gaussian channel has unit variance.
    A MAD of zero is floored so a constant channel stays finite.

    Parameters
    ----------
    data : np.ndarray
        2D input array of shape (nchans, nsamps), float32.
    out : np.ndarray
        2D output array of shape (nchans, nsamps), float32.
    subsample : int, optional
        Stride used while estimating the median and MAD, by default 8.
    """
    nchans, nsamps = data.shape
    stride = max(1, subsample)

    for ch in prange(nchans):
        n_est = (nsamps + stride - 1) // stride
        buf = np.empty(n_est, dtype=np.float64)
        count = 0
        for i in range(0, nsamps, stride):
            v = float(data[ch, i])
            if not np.isnan(v):
                buf[count] = v
                count += 1

        if count < 2:
            for s in range(nsamps):
                out[ch, s] = np.nan
            continue

        work = buf[:count].copy()
        work.sort()
        mid = count // 2
        med = 0.5 * (work[mid - 1] + work[mid]) if count % 2 == 0 else work[mid]

        for i in range(count):
            work[i] = abs(work[i] - med)
        work.sort()
        mad = 0.5 * (work[mid - 1] + work[mid]) if count % 2 == 0 else work[mid]
        scale = max(1.4826 * mad, 1e-12)
        inv = 1.0 / scale

        for s in range(nsamps):
            val = float(data[ch, s])
            if np.isnan(val):
                out[ch, s] = np.nan
            else:
                out[ch, s] = np.float32((val - med) * inv)


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
