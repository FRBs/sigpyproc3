"""Numba kernels for pulsar folding, resampling, and Fast Folding Algorithm (FFA)."""

from __future__ import annotations

import numpy as np
from numba import njit

CONST_C_VAL = 299792458.0  # Speed of light in m/s (astropy.constants.c.value)


@njit(
    [
        "void(u1[:], f4[:], i4[:], i4[:], i4, f4, f4, f4, i4, i4, i4, i4, i4, i4, i4)",
        "void(f4[:], f4[:], i4[:], i4[:], i4, f4, f4, f4, i4, i4, i4, i4, i4, i4, i4)",
    ],
    cache=True,
)
def fold(
    inarray: np.ndarray,
    fold_ar: np.ndarray,
    count_ar: np.ndarray,
    delays: np.ndarray,
    maxdelay: int,
    tsamp: float,
    period: float,
    accel: float,
    total_nsamps: int,
    nsamps: int,
    nchans: int,
    nbins: int,
    nints: int,
    nsubs: int,
    index: int,
) -> None:
    factor1 = total_nsamps / nints
    factor2 = nchans / nsubs
    tobs = total_nsamps * tsamp
    for isamp in range(nsamps - maxdelay):
        tj = (isamp + index) * tsamp
        phase = (
            nbins * tj * (1 + accel * (tj - tobs) / (2 * CONST_C_VAL)) / period + 0.5
        )
        phasebin = abs(int(phase)) % nbins
        subint = (isamp + index) // factor1
        pos1 = (subint * nbins * nsubs) + phasebin

        for ichan in range(nchans):
            sub_band = ichan // factor2
            pos2 = int(pos1 + (sub_band * nbins))
            val = inarray[nchans * (isamp + delays[ichan]) + ichan]
            fold_ar[pos2] += val
            count_ar[pos2] += 1


@njit("f4[:](f4[:], f4, f4)", cache=True)
def resample_tim(array: np.ndarray, accel: float, tsamp: float) -> np.ndarray:
    nsamps = len(array) - 1 if accel > 0 else len(array)
    resampled = np.zeros(nsamps, dtype=array.dtype)

    partial_calc = (accel * tsamp) / (2 * CONST_C_VAL)
    tot_drift = partial_calc * (nsamps // 2) ** 2
    last_bin = 0
    for ii in range(nsamps):
        index = int(ii + partial_calc * (ii - nsamps // 2) ** 2 - tot_drift)
        resampled[index] = array[ii]
        if index - last_bin > 1:
            resampled[index - 1] = array[ii]
        last_bin = index
    return resampled


@njit(cache=True, fastmath=True)
def ffa_stage(block: np.ndarray) -> np.ndarray:
    """Run one FFA binary tree on a ``(n_rows, period)`` fold block."""
    n_rows, nbin = block.shape
    n_pad = 1
    k = 0
    while n_pad < n_rows:
        n_pad *= 2
        k += 1
    src = np.zeros((n_pad, nbin), dtype=np.float32)
    src[:n_rows, :] = block
    if k == 0:
        return src
    dst = np.empty((n_pad, nbin), dtype=np.float32)
    for stage in range(k):
        step = 1 << stage
        group = step << 1
        for i in range(0, n_pad, group):
            for shift in range(step):
                a = src[i + shift]
                b = src[i + step + shift]
                out0 = dst[i + 2 * shift]
                out1 = dst[i + 2 * shift + 1]
                for j in range(nbin):
                    out0[j] = a[j] + b[j]
                out1[nbin - 1] = a[nbin - 1] + b[0]
                for j in range(nbin - 1):
                    out1[j] = a[j] + b[j + 1]
        src, dst = dst, src
    return src


@njit(cache=True, fastmath=True)
def ffa_profile_snr(profiles: np.ndarray) -> np.ndarray:
    """Robust SNR of each FFA profile: ``(max - median) / (1.4826 * MAD)``."""
    nprof = profiles.shape[0]
    snrs = np.empty(nprof, dtype=np.float32)
    mad_norm = np.float32(1.4826)
    for i in range(nprof):
        row = profiles[i]
        med = np.median(row)
        mad = np.median(np.abs(row - med))
        if mad <= 0:
            snrs[i] = 0.0
        else:
            snrs[i] = (np.max(row) - med) / (mad_norm * mad)
    return snrs
