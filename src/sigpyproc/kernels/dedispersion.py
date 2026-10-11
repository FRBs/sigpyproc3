"""Numba kernels for dedispersion, Taylor tree, and time/frequency roll."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(
    [
        "void(u1[::1], f4[::1], i4[::1], i8, i8, i8)",
        "void(f4[::1], f4[::1], i4[::1], i8, i8, i8)",
    ],
    cache=True,
    fastmath=True,
)
def dedisperse(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    delays: np.ndarray,
    nchans: int,
    nsamps: int,
    out_start_index: int,
) -> None:
    max_delay = int(np.max(np.abs(delays)))
    in_arr_2d = in_arr.reshape(nsamps, nchans)
    for isamp in prange(nsamps - max_delay):
        summ = np.float64(0.0)
        for ichan in range(nchans):
            summ += in_arr_2d[isamp + delays[ichan], ichan]
        out_arr[out_start_index + isamp] = summ


dedisperse_par = njit(dedisperse.py_func, parallel=True, fastmath=True)


@njit("i8(i8, i8)", cache=True)
def bitrev(inval: int, nbits: int) -> int:
    """Bit-reverse ``inval`` in ``nbits`` bits (Ramachandran tree)."""
    if nbits <= 1:
        return inval
    ifact = 1
    for _ in range(1, nbits):
        ifact *= 2
    k = inval
    ibitr = (k & 1) * ifact
    for _ in range(2, nbits + 1):
        k //= 2
        ifact //= 2
        ibitr += (k & 1) * ifact
    return ibitr


@njit("void(f4[::1], i8, i8)", cache=True, fastmath=True)
def taylor_tree(buf: np.ndarray, nchans: int, nsamps: int) -> None:
    """In-place Taylor/Ramachandran tree."""
    nsamp = nsamps - 2 * nchans
    npts = nsamp + nchans
    nstages = int(np.log2(nchans) + 0.5)
    nmem = 1
    for istages in range(nstages):
        nmem *= 2
        nsec1 = nchans // nmem
        nmem2 = nmem - 2
        for isec in range(nsec1):
            ndelay = -1
            koff = isec * nmem
            for ipair in range(0, nmem2 + 1, 2):
                ioff1 = (bitrev(ipair, istages + 1) + koff) * nsamps
                i2 = (bitrev(ipair + 1, istages + 1) + koff) * nsamps
                ndelay += 1
                ndelay2 = ndelay + 1
                nfin = npts + ioff1
                for i1 in range(ioff1, nfin):
                    itemp = buf[i1] + buf[i2 + ndelay]
                    buf[i2] = buf[i1] + buf[i2 + ndelay2]
                    buf[i1] = itemp
                    i2 += 1


@njit(
    [
        "void(u1[::1], f4[:, ::1], i4[:, ::1], i8, i8)",
        "void(f4[::1], f4[:, ::1], i4[:, ::1], i8, i8)",
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def dedisperse_many(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    delays: np.ndarray,
    nchans: int,
    nsamps: int,
) -> None:
    """Dedisperse many DMs in one gulp."""
    ndm = delays.shape[0]
    max_delay = 0
    for idm in range(ndm):
        for ichan in range(nchans):
            max_delay = max(max_delay, int(delays[idm, ichan]))
    in_2d = in_arr.reshape(nsamps, nchans)
    for idm in prange(ndm):
        for isamp in range(nsamps - max_delay):
            summ = np.float64(0.0)
            for ichan in range(nchans):
                summ += in_2d[isamp + delays[idm, ichan], ichan]
            out_arr[idm, isamp] = summ


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4[:], i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4)",
    ],
    cache=True,
    parallel=True,
)
def subband(
    inarray: np.ndarray,
    outarray: np.ndarray,
    delays: np.ndarray,
    chan_to_sub: np.ndarray,
    maxdelay: int,
    nchans: int,
    nsubs: int,
    nsamps: int,
) -> None:
    for isamp in prange(nsamps - maxdelay):
        for ichan in range(nchans):
            outarray[nsubs * isamp + chan_to_sub[ichan]] += inarray[
                nchans * (isamp + delays[ichan]) + ichan
            ]


@njit(cache=True, nogil=True, fastmath=True)
def nb_roll(
    arr: np.ndarray,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    return np.roll(arr, shift, axis)


@njit(cache=True, fastmath=True)
def roll_block(arr: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Roll each row of a 2D array along columns by per-row shifts."""
    if arr.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if len(shifts) != arr.shape[0]:
        msg = "Number of shifts must be equal to the number of rows."
        raise ValueError(msg)
    nrows, ncols = arr.shape
    res = np.empty_like(arr)
    for irow in range(nrows):
        shift = shifts[irow] % ncols
        if shift == 0:
            res[irow] = arr[irow]
        else:
            res[irow, shift:] = arr[irow, : ncols - shift]
            res[irow, :shift] = arr[irow, ncols - shift :]
    return res


@njit(cache=True, fastmath=True)
def roll_block_valid(arr: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Roll each row of a 2D array by per-row shifts, keeping only valid columns."""
    if arr.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if len(shifts) != arr.shape[0]:
        msg = "Number of shifts must be equal to the number of rows."
        raise ValueError(msg)
    nrows, ncols = arr.shape
    max_pos_shift = max(0, np.max(shifts))
    min_neg_shift = min(0, np.min(shifts))

    start_col = max_pos_shift
    end_col = ncols + min_neg_shift
    valid_cols = end_col - start_col
    if valid_cols <= 0:
        msg = (
            f"Not enough samples. Required at least {max_pos_shift - min_neg_shift} "
            f"samples, given {ncols}."
        )
        raise ValueError(msg)
    res = np.empty((nrows, valid_cols), dtype=arr.dtype)
    for irow in range(nrows):
        shift = shifts[irow]
        res[irow, :] = arr[irow, start_col - shift : end_col - shift]
    return res


@njit(cache=True, fastmath=True)
def dmt_block(arr: np.ndarray, dm_delays: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or dm_delays.ndim != 2:
        msg = "Input array and delays must be 2D."
        raise ValueError(msg)
    if arr.shape[0] != dm_delays.shape[1]:
        msg = "Number of chans must be same in both arrays."
        raise ValueError(msg)
    _, nsamps = arr.shape
    ndms, _ = dm_delays.shape
    res = np.empty((ndms, nsamps), dtype=arr.dtype)
    for idm in range(ndms):
        res[idm] = np.sum(roll_block(arr, dm_delays[idm]), axis=0)
    return res


@njit(cache=True, fastmath=True)
def dmt_block_valid(arr: np.ndarray, dm_delays: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or dm_delays.ndim != 2:
        msg = "Input array and delays must be 2D."
        raise ValueError(msg)
    if arr.shape[0] != dm_delays.shape[1]:
        msg = "Number of chans must be same in both arrays."
        raise ValueError(msg)
    _, nsamps = arr.shape
    ndms, _ = dm_delays.shape
    max_pos_shift = max(0, np.max(dm_delays))
    min_neg_shift = min(0, np.min(dm_delays))
    valid_samples = nsamps + min_neg_shift - max_pos_shift
    if valid_samples <= 0:
        msg = (
            f"Not enough samples. Required at least {max_pos_shift - min_neg_shift} "
            f"samples, given {nsamps}."
        )
        raise ValueError(msg)
    res = np.empty((ndms, valid_samples), dtype=arr.dtype)
    for idm in range(ndms):
        res[idm] = np.sum(roll_block_valid(arr, dm_delays[idm]), axis=0)
    return res


@njit(cache=True, fastmath=True)
def disperse_block(
    arr: np.ndarray,
    shifts: np.ndarray,
    nsamps_out: int = 1,
    tfactor: int = 1,
) -> np.ndarray:
    """Disperse a 2D block with per-channel delays."""
    if arr.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if len(shifts) != arr.shape[0]:
        msg = "Number of shifts must be equal to the number of rows."
        raise ValueError(msg)
    nrows, ncols = arr.shape
    valid_samps = max(2 * (ncols + np.abs(shifts).max()), nsamps_out)
    res = np.zeros((nrows, valid_samps), dtype=arr.dtype)
    start = valid_samps // 2 - ncols // 2
    end = start + ncols
    for irow in range(nrows):
        res[irow, start + shifts[irow] : end + shifts[irow]] = arr[irow]
    return res
