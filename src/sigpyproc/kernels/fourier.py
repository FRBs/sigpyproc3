"""Numba kernels for Fourier transforms and frequency domain operations."""

from __future__ import annotations

import numpy as np
import rocket_fft
from numba import njit


@njit("f4[::1](c8[::1])", cache=True, fastmath=True)
def form_mspec(fspec: np.ndarray) -> np.ndarray:
    nfreq = len(fspec)
    mspec = np.zeros(nfreq, dtype=np.float32)
    for i in range(nfreq):
        mspec[i] = np.sqrt(fspec[i].real ** 2 + fspec[i].imag ** 2)
    return mspec


@njit("f4[::1](c8[::1])", cache=True, fastmath=True)
def form_interp_mspec(fspec: np.ndarray) -> np.ndarray:
    nfreq = len(fspec)
    mspec = np.zeros(nfreq, dtype=np.float32)
    re_prev = 0
    im_prev = 0
    for i in range(nfreq):
        re, im = fspec[i].real, fspec[i].imag
        ampsq = re**2 + im**2
        ampsq_diff = 0.5 * ((re - re_prev) ** 2 + (im - im_prev) ** 2)
        mspec[i] = np.sqrt(max(ampsq, ampsq_diff))
        re_prev, im_prev = re, im
    return mspec


@njit("c8[:](c8[:], i8, i8, i8)", cache=True, fastmath=True)
def fs_running_median(
    spec_arr: np.ndarray,
    start_width: int,
    end_width: int,
    end_freq_bin: int,
) -> np.ndarray:
    nspecs = len(spec_arr)
    out_arr = np.zeros_like(spec_arr)
    powbuf = np.empty(nspecs, dtype=np.float32)
    for ii in range(nspecs):
        powbuf[ii] = spec_arr[ii].real ** 2 + spec_arr[ii].imag ** 2

    # A zero local noise estimate carries no scale information. Leave that
    # segment's amplitudes unchanged rather than divide by zero.

    out_arr[0] = 1.0 + 0j
    rindex, windex, binnum = 1, 1, 1
    buflen = start_width

    numread_old = buflen
    mean_old = np.median(powbuf[rindex : rindex + numread_old]) / np.log(2)
    rindex += numread_old
    binnum += numread_old
    buflen = round(start_width * np.log(binnum))

    while rindex < nspecs:
        numread_new = min(buflen, nspecs - rindex)
        mean_new = np.median(powbuf[rindex : rindex + numread_new]) / np.log(2)
        rindex += numread_new

        slope = (mean_new - mean_old) / (numread_old + numread_new)
        for i in range(numread_old):
            power = mean_old + slope * ((numread_old + numread_new) / 2 - i)
            norm = 1 / np.sqrt(power) if power > 0 else 1.0
            out_arr[windex + i] = spec_arr[windex + i] * norm

        windex += numread_old
        binnum += numread_new
        if binnum < end_freq_bin:
            buflen = round(start_width * np.log(binnum))
        else:
            buflen = end_width
        numread_old, mean_old = numread_new, mean_new

    norm = 1 / np.sqrt(mean_old) if mean_old > 0 else 1.0
    out_arr[windex : windex + numread_old] = (
        spec_arr[windex : windex + numread_old] * norm
    )
    return out_arr


@njit(cache=True, fastmath=True, inline="always")
def _harmonic_count(ifold: int) -> int:
    """Power-of-two harmonic count for fold ``ifold`` (0 → 2, 1 → 4, …)."""
    return 2 ** (ifold + 1)


@njit(cache=True, fastmath=True)
def _lyne_tables(nharm: int, nfoldi: int) -> tuple[np.ndarray, np.ndarray]:
    """SIGPROC ``sumarray`` and ``factarray`` for one stretch."""
    nptr = nharm // 2
    offsets = np.empty(nharm * nptr, dtype=np.int32)
    slot = 0
    for block in range(nharm):
        step = 1
        while step < nharm:
            offsets[slot] = step * block // nharm
            slot += 1
            step += 2
    pointers = np.empty(nptr, dtype=np.int32)
    slot = 0
    step = 1
    while step < nharm:
        pointers[slot] = (step * nfoldi + nharm // 2) // nharm
        slot += 1
        step += 2
    return offsets, pointers


@njit(cache=True, fastmath=True)
def sum_harmonics_lyne(pow_spec: np.ndarray, nfolds: int) -> np.ndarray:
    """Lyne-Ashworth stretch used by SIGPROC ``sumHarms``.

    Fold ``i`` stretches by ``H = 2**(i+1)``. Each stage reads the original
    spectrum with ``H/2`` pointers and adds them into ``H`` neighbouring
    bins of a running sum. Pointer ``k`` then steps by ``2k+1``. The next
    stage starts where the previous stretch ended.
    """
    nfreqs = len(pow_spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float32)
    folded = pow_spec.astype(np.float32).copy()
    nfold_prev = 0
    for ifold in range(nfolds):
        nharm = _harmonic_count(ifold)
        nfoldi = min(nfreqs, max(1, nharm * nfold_prev - nharm // 2))
        nptr = nharm // 2
        offsets, pointers = _lyne_tables(nharm, nfoldi)
        origin = nfoldi
        limit = nfreqs - (nharm - 1)
        while origin < limit:
            for neighbour in range(nharm):
                dest = origin + neighbour
                for ptr in range(nptr):
                    src = pointers[ptr] + offsets[neighbour * nptr + ptr]
                    if 0 <= src < nfreqs:
                        folded[dest] += pow_spec[src]
            for ptr in range(nptr):
                pointers[ptr] += 2 * ptr + 1
            origin += nharm
        out[ifold, :] = folded
        nfold_prev = nfoldi
    return out


@njit(cache=True, fastmath=True)
def sum_harmonics_simple(pow_spec: np.ndarray, nfolds: int) -> np.ndarray:
    """Sum ``spec[h*n]`` for ``h = 1 … H`` with no peak drift.

    ``H = 2**(i+1)`` on fold ``i``. Bins that fall past the end of the
    spectrum are left out. There is no ``1/sqrt(H)`` scale.
    """
    nfreqs = len(pow_spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float32)
    for ifold in range(nfolds):
        nharm = _harmonic_count(ifold)
        for fund in range(1, nfreqs):
            total = np.float64(0.0)
            for harm in range(1, nharm + 1):
                bin_h = harm * fund
                if bin_h >= nfreqs:
                    break
                total += pow_spec[bin_h]
            out[ifold, fund] = total
    return out


@njit(cache=True, fastmath=True)
def sum_harmonics_presto(pow_spec: np.ndarray, nfolds: int) -> np.ndarray:
    """Stock PRESTO harmonic sum: even harmonics, nearest-bin drift.

    For fold ``H``, end-bin ``n`` sums ``spec[round(h/H*n)]`` at the
    fundamental and at even ``h``. The sum is stored on fundamental bin
    ``round(n/H)``. When several end-bins share a fundamental, the largest
    sum is kept.
    """
    nfreqs = len(pow_spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float32)
    for ifold in range(nfolds):
        nharm = _harmonic_count(ifold)
        partial = np.zeros(nfreqs, dtype=np.float64)
        for end in range(nfreqs):
            fund = int(end / nharm + 0.5)
            if fund <= 0 or fund >= nfreqs:
                continue
            total = np.float64(pow_spec[fund])
            for harm in range(2, nharm + 1, 2):
                bin_h = int(harm * end / nharm + 0.5)
                if bin_h >= nfreqs:
                    break
                total += pow_spec[bin_h]
            partial[end] = total
        for end in range(nfreqs):
            fund = int(end / nharm + 0.5)
            if 0 < fund < nfreqs and partial[end] > out[ifold, fund]:
                out[ifold, fund] = partial[end]
    return out


@njit(cache=True, fastmath=True)
def sum_harmonics_greedy(pow_spec: np.ndarray, nfolds: int) -> np.ndarray:
    """Greedy harmonic sum (Adámek, Roy & Armour 2022).

    At harmonic ``h`` the sum takes the louder of bins ``h*n+d`` and
    ``h*n+d+1``, and the drift ``d`` grows by one when the upper bin wins.
    Fold ``i`` is that path summed through ``H = 2**(i+1)`` harmonics.
    """
    nfreqs = len(pow_spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float32)
    for ifold in range(nfolds):
        nharm = _harmonic_count(ifold)
        for fund in range(1, nfreqs):
            drift = 0
            if fund + 1 < nfreqs and pow_spec[fund + 1] > pow_spec[fund]:
                drift = 1
                total = np.float64(pow_spec[fund + 1])
            else:
                total = np.float64(pow_spec[fund])
            for harm in range(2, nharm + 1):
                base = harm * fund + drift
                if base >= nfreqs:
                    break
                chosen = base
                if base + 1 < nfreqs and pow_spec[base + 1] > pow_spec[base]:
                    drift += 1
                    chosen = base + 1
                total += pow_spec[chosen]
            out[ifold, fund] = total
    return out


@njit(cache=True, fastmath=True)
def nb_fft_good_size(n: int, real: bool = False) -> int:  # noqa: FBT001, FBT002
    """Get the good size for FFT."""
    return int(rocket_fft.good_size(np.int64(n), real=np.bool_(real)))


@njit(cache=True, fastmath=True)
def nb_rfft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.rfft(arr, n)


@njit(cache=True, fastmath=True)
def nb_irfft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.irfft(arr, n)


@njit(cache=True, fastmath=True)
def nb_fft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.fft(arr, n)


@njit(cache=True, fastmath=True)
def nb_ifft(arr: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.ifft(arr, n)


@njit(cache=True, fastmath=True)
def fftconvolve(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """Convolve two 1D arrays using FFT in mode 'full'."""
    if in1.ndim != 1 or in2.ndim != 1:
        msg = "Input arrays must be 1D."
        raise ValueError(msg)
    n1 = len(in1)
    n2 = len(in2)
    if n1 == 0 or n2 == 0:
        return np.zeros(0, dtype=in1.dtype)
    n = n1 + n2 - 1
    n_good = nb_fft_good_size(n, real=True)
    sp1 = np.fft.rfft(in1, n_good)
    sp2 = np.fft.rfft(in2, n_good)
    ret = np.fft.irfft(sp1 * sp2, n_good)
    return ret[:n]


@njit(cache=True, fastmath=True)
def circular_pad_goodsize(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    n_good = nb_fft_good_size(n, real=True)
    result = np.empty(n_good, dtype=arr.dtype)
    for i in range(n_good):
        result[i] = arr[i % n]
    return result


@njit(cache=True, fastmath=True)
def normalize_template(arr: np.ndarray) -> np.ndarray:
    """Normalize the template to have zero mean and unit power."""
    mean = np.mean(arr)
    arr_norm = arr - mean
    norm = np.sqrt(np.sum(arr_norm**2))
    if norm == 0:
        return arr_norm
    return arr_norm / norm


@njit(cache=True, fastmath=True)
def convolve_templates(
    data: np.ndarray,
    temp_bank: list[np.ndarray],
    ref_bin: list[int],
) -> np.ndarray:
    """Convolve the data with the templates in the template bank."""
    nbins = len(data)
    ntemps = len(temp_bank)
    convs = np.empty((ntemps, nbins), dtype=data.dtype)
    data_pad = circular_pad_goodsize(data)
    n_pad = len(data_pad)
    data_fft = np.fft.rfft(data_pad)
    for itemp in range(ntemps):
        temp_kernel = temp_bank[itemp]
        temp_pad = np.zeros_like(data_pad)
        temp_pad[: len(temp_kernel)] = temp_kernel
        temp_pad = np.roll(temp_pad, -ref_bin[itemp])
        temp_pad = np.roll(temp_pad[::-1], 1)
        temp_norm = normalize_template(temp_pad)
        conv = np.fft.irfft(data_fft * np.fft.rfft(temp_norm), n_pad)
        convs[itemp, :] = conv[:nbins]
    return convs
