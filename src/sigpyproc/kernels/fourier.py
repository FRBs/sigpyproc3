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
            norm = 1 / np.sqrt(mean_old + slope * ((numread_old + numread_new) / 2 - i))
            out_arr[windex + i] = spec_arr[windex + i] * norm

        windex += numread_old
        binnum += numread_new
        if binnum < end_freq_bin:
            buflen = round(start_width * np.log(binnum))
        else:
            buflen = end_width
        numread_old, mean_old = numread_new, mean_new

    norm = 1 / np.sqrt(mean_old)
    out_arr[windex : windex + numread_old] = (
        spec_arr[windex : windex + numread_old] * norm
    )
    return out_arr


@njit("f4[:,:](f4[:], i8)", cache=True, fastmath=True)
def sum_harmonics(pow_spec: np.ndarray, nfolds: int) -> np.ndarray:
    nfreqs = len(pow_spec)
    sum_arr = np.zeros((nfolds, nfreqs), dtype=np.float32)
    harm_sum = pow_spec.copy()
    nfold1 = 0
    for iff in range(nfolds):
        nharm = 2 ** (iff + 1)
        nfoldi = int(max(1, min(nharm * nfold1 - nharm // 2, nfreqs)))
        harm_arr = np.array(
            [kk * ll // nharm for ll in range(nharm) for kk in range(1, nharm, 2)],
            dtype=np.int32,
        )

        facts_ar = np.array(
            [(kk * nfoldi + nharm // 2) // nharm for kk in range(1, nharm, 2)],
            dtype=np.int32,
        )
        for ifold in range(nfoldi, nfreqs - (nharm - 1), nharm):
            for iharm in range(nharm):
                for kk in range(nharm // 2):
                    harm_sum[ifold + iharm] += pow_spec[
                        facts_ar[kk] + harm_arr[iharm * nharm // 2 + kk]
                    ]
            for kk in range(nharm // 2):
                facts_ar[kk] += 2 * kk + 1
        sum_arr[iff] = harm_sum
    return sum_arr


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
    data_fft = np.fft.rfft(data_pad)
    for itemp in range(ntemps):
        temp_kernel = temp_bank[itemp]
        temp_pad = np.zeros_like(data_pad)
        temp_pad[: len(temp_kernel)] = temp_kernel
        temp_pad = np.roll(temp_pad, -ref_bin[itemp])
        temp_pad = np.roll(temp_pad[::-1], 1)
        temp_norm = normalize_template(temp_pad)
        conv = np.fft.irfft(data_fft * np.fft.rfft(temp_norm))
        convs[itemp, :] = conv[:nbins]
    return convs
