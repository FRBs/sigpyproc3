"""Numba kernels for filterbank digifil operations, scrunching, and quantisation."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def fscrunch(data: np.ndarray, factor: int) -> np.ndarray:
    """Sum consecutive groups of ``factor`` channels."""
    nsamp, npol, nchan = data.shape
    nout = nchan // factor
    out = np.empty((nsamp, npol, nout), dtype=np.float32)
    for isamp in range(nsamp):
        for ipol in range(npol):
            for ochan in range(nout):
                base = ochan * factor
                acc = data[isamp, ipol, base]
                for k in range(1, factor):
                    acc = np.float32(acc + data[isamp, ipol, base + k])
                out[isamp, ipol, ochan] = acc
    return out


@njit(cache=True, fastmath=True)
def tscrunch(data: np.ndarray, factor: int) -> np.ndarray:
    """Sum consecutive groups of ``factor`` time samples."""
    _, npol, nchan = data.shape
    nout = data.shape[0] // factor
    out = np.empty((nout, npol, nchan), dtype=np.float32)
    for osamp in range(nout):
        base = osamp * factor
        for ipol in range(npol):
            for ichan in range(nchan):
                out[osamp, ipol, ichan] = data[base, ipol, ichan]
        for k in range(1, factor):
            for ipol in range(npol):
                for ichan in range(nchan):
                    out[osamp, ipol, ichan] = np.float32(
                        out[osamp, ipol, ichan] + data[base + k, ipol, ichan],
                    )
    return out


@njit(cache=True, fastmath=True)
def pscrunch_intensity(data: np.ndarray) -> np.ndarray:
    """Sum the first two polarisations (``PP + QQ``) into a single intensity."""
    nsamp, _, nchan = data.shape
    out = np.empty((nsamp, nchan), dtype=np.float32)
    for isamp in range(nsamp):
        for ichan in range(nchan):
            out[isamp, ichan] = np.float32(
                data[isamp, 0, ichan] + data[isamp, 1, ichan]
            )
    return out


@njit(cache=True, fastmath=True)
def accumulate_stats(
    data: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
) -> None:
    """Accumulate per-column sum and sum of squares in double precision."""
    nsamp, ncol = data.shape
    for isamp in range(nsamp):
        for icol in range(ncol):
            val = np.float64(data[isamp, icol])
            sums[icol] += val
            sumsq[icol] += val * val


@njit(cache=True, fastmath=True)
def apply_scale_offset(
    data: np.ndarray,
    offset: np.ndarray,
    scale: np.ndarray,
    out: np.ndarray,
) -> None:
    """Apply per-column offset and scale as ``(x + offset) * scale`` in float32."""
    nsamp, ncol = data.shape
    for isamp in range(nsamp):
        for icol in range(ncol):
            shifted = np.float32(data[isamp, icol] + offset[icol])
            out[isamp, icol] = np.float32(shifted * scale[icol])


@njit(cache=True, fastmath=True, inline="always")
def _quantise_value(
    value: np.float32,
    mean: np.float32,
    scale: np.float32,
    digi_max: int,
) -> int:
    """Quantise one value as dspsr's ``SigProcDigitizer`` does."""
    scaled = np.float32(value * scale)
    shifted = np.float32(scaled + mean)
    rounded = np.float64(shifted) + 0.5
    limit = 2.0**31 - 1.0
    if rounded > limit:
        rounded = limit
    elif rounded < -limit:
        rounded = -limit
    result = int(rounded)
    if result < 0:
        result = 0
    elif result > digi_max:
        result = digi_max
    return result


@njit(cache=True, fastmath=True)
def quantise(
    data: np.ndarray,
    flip: bool,  # noqa: FBT001
    means: np.ndarray,
    scale: np.float32,
    digi_max: int,
    out: np.ndarray,
) -> None:
    """Quantise an ``(nrow, nchan)`` array to unsigned integers."""
    nrow, nchan = data.shape
    for irow in range(nrow):
        for ochan in range(nchan):
            ichan = nchan - 1 - ochan if flip else ochan
            out[irow, ochan] = _quantise_value(
                data[irow, ichan],
                means[irow],
                scale,
                digi_max,
            )


@njit(cache=True, fastmath=True)
def quantise_packed(
    data: np.ndarray,
    flip: bool,  # noqa: FBT001
    means: np.ndarray,
    scale: np.float32,
    digi_max: int,
    nbit: int,
    packed: np.ndarray,
) -> None:
    """Quantise and pack 1, 2 or 4-bit samples least-significant-bit first."""
    nrow, nchan = data.shape
    samp_per_byte = 8 // nbit
    bytes_per_row = nchan // samp_per_byte
    for irow in range(nrow):
        for ochan in range(nchan):
            ichan = nchan - 1 - ochan if flip else ochan
            value = _quantise_value(data[irow, ichan], means[irow], scale, digi_max)
            byte = irow * bytes_per_row + ochan // samp_per_byte
            shift = (ochan % samp_per_byte) * nbit
            packed[byte] |= np.uint8(value << shift)
