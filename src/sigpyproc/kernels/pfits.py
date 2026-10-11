"""Numba kernels that calibrate a PSRFITS search-mode sub-integration.

1-, 2- and 4-bit samples are expanded with
:func:`sigpyproc.kernels.bitpacking.unpack_array` (big-endian, high bits
first) before these kernels run. Two-bit samples then use dspsr's four
levels. 8-bit and 16-bit samples are read here: 16-bit is big-endian, and
8-bit is signed only when ``SIGNINT`` is set. The kernels write float32 and
can apply the PSRFITS scale, offset and weight while reducing to one product.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_TWO_BIT = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)


@njit(cache=True, fastmath=True, inline="always")
def _digit(
    raw: np.ndarray,
    isamp: int,
    ipol: int,
    ichan: int,
    nbits: int,
    signed: bool,  # noqa: FBT001
) -> np.float32:
    """Return one stored sample as float32, before scale and weight.

    ``raw`` is ``(nsamp, npol, nchan)`` for 1-, 2-, 4- and 8-bit data. Sub-byte
    values are already one sample per channel. 16-bit ``raw`` is
    ``(nsamp, npol, 2 * nchan)`` big-endian bytes.
    """
    if nbits == 16:
        hi = np.int32(raw[isamp, ipol, ichan * 2])
        lo = np.int32(raw[isamp, ipol, ichan * 2 + 1])
        value = hi * 256 + lo
        if value >= 32768:
            value = value - 65536
        return np.float32(value)
    sample = raw[isamp, ipol, ichan]
    if nbits == 2:
        return _TWO_BIT[sample]
    if nbits == 8 and signed and sample >= 128:
        return np.float32(np.int16(sample) - 256)
    return np.float32(sample)


@njit(cache=True, fastmath=True, inline="always")
def _calibrated(
    raw: np.ndarray,
    isamp: int,
    ipol: int,
    ichan: int,
    nbits: int,
    signed: bool,  # noqa: FBT001
    zero_off: np.float32,
    do_zero: bool,  # noqa: FBT001
    scales: np.ndarray,
    offsets: np.ndarray,
    do_scale: bool,  # noqa: FBT001
) -> np.float32:
    """Apply the dspsr zero, scale and offset to one sample."""
    value = _digit(raw, isamp, ipol, ichan, nbits, signed)
    if do_zero:
        value = np.float32(value - zero_off)
    if do_scale:
        value = np.float32(value * scales[ipol, ichan] + offsets[ipol, ichan])
    return value


@njit(cache=True, fastmath=True)
def decode_intensity(
    raw: np.ndarray,
    nbits: int,
    signed: bool,  # noqa: FBT001
    zero_off: np.float32,
    do_zero: bool,  # noqa: FBT001
    scales: np.ndarray,
    offsets: np.ndarray,
    do_scale: bool,  # noqa: FBT001
    weights: np.ndarray,
    do_weight: bool,  # noqa: FBT001
    pol: int,
    out: np.ndarray,
) -> None:
    """Decode one sub-integration to a single ``(nsamp, nchan)`` product.

    ``pol < 0`` sums the first two polarisations (Stokes I of ``PPQQ`` and
    coherence data, ``P0 + P1``). ``pol >= 0`` copies that polarisation.
    """
    nsamp = raw.shape[0]
    nchan = out.shape[1]
    for isamp in range(nsamp):
        for ichan in range(nchan):
            if pol < 0:
                acc = _calibrated(
                    raw,
                    isamp,
                    0,
                    ichan,
                    nbits,
                    signed,
                    zero_off,
                    do_zero,
                    scales,
                    offsets,
                    do_scale,
                )
                acc = np.float32(
                    acc
                    + _calibrated(
                        raw,
                        isamp,
                        1,
                        ichan,
                        nbits,
                        signed,
                        zero_off,
                        do_zero,
                        scales,
                        offsets,
                        do_scale,
                    ),
                )
            else:
                acc = _calibrated(
                    raw,
                    isamp,
                    pol,
                    ichan,
                    nbits,
                    signed,
                    zero_off,
                    do_zero,
                    scales,
                    offsets,
                    do_scale,
                )
            if do_weight:
                acc = np.float32(acc * weights[ichan])
            out[isamp, ichan] = acc


@njit(cache=True, fastmath=True)
def decode_cube(
    raw: np.ndarray,
    nbits: int,
    signed: bool,  # noqa: FBT001
    zero_off: np.float32,
    do_zero: bool,  # noqa: FBT001
    scales: np.ndarray,
    offsets: np.ndarray,
    do_scale: bool,  # noqa: FBT001
    weights: np.ndarray,
    do_weight: bool,  # noqa: FBT001
    out: np.ndarray,
) -> None:
    """Decode one sub-integration to ``(nsamp, npol, nchan)`` float32."""
    nsamp = raw.shape[0]
    npol = raw.shape[1]
    nchan = out.shape[2]
    for isamp in range(nsamp):
        for ipol in range(npol):
            for ichan in range(nchan):
                value = _calibrated(
                    raw,
                    isamp,
                    ipol,
                    ichan,
                    nbits,
                    signed,
                    zero_off,
                    do_zero,
                    scales,
                    offsets,
                    do_scale,
                )
                if do_weight:
                    value = np.float32(value * weights[ichan])
                out[isamp, ipol, ichan] = value
