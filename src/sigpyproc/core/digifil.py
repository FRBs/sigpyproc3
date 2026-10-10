"""core for digifil operations: kernels, rescaler, digitiser and bit packing.

Data layout is ``(nsamp, npol, nchan)`` for polarisation-resolved data, so that
the channel axis is contiguous, and ``(nsamp, nchan)`` after polarisation
scrunching. Quantisation works on rows of ``nchan`` values.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# Digitiser mean per output bit depth (SigProcDigitizer.C) and the sigma used
# for the 4, 8 and 16-bit scale (dspsr uses 6 sigma for all of these).
DIGI_MEAN: dict[int, float] = {1: 0.5, 2: 1.5, 4: 7.5, 8: 127.5, 16: 32768.0}
DIGI_SIGMA = 6.0


@njit(cache=True, fastmath=True)
def fscrunch(data: np.ndarray, factor: int) -> np.ndarray:
    """Sum consecutive groups of ``factor`` channels.

    Parameters
    ----------
    data : NDArray[float32]
        Input data with shape ``(nsamp, npol, nchan)``.
    factor : int
        Number of adjacent channels to sum. Trailing channels that do not fill
        a complete group are dropped.

    Returns
    -------
    NDArray[float32]
        Data with shape ``(nsamp, npol, nchan // factor)``.
    """
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
    """Sum consecutive groups of ``factor`` time samples.

    Parameters
    ----------
    data : NDArray[float32]
        Input data with shape ``(nsamp, npol, nchan)``. Trailing samples that
        do not fill a complete group are dropped.
    factor : int
        Number of adjacent samples to sum.

    Returns
    -------
    NDArray[float32]
        Data with shape ``(nsamp // factor, npol, nchan)``.
    """
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
    """Sum the first two polarisations (``PP + QQ``) into a single intensity.

    Parameters
    ----------
    data : NDArray[float32]
        Input data with shape ``(nsamp, npol, nchan)`` where ``npol >= 2``.

    Returns
    -------
    NDArray[float32]
        Intensity with shape ``(nsamp, nchan)``.
    """
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
    """Accumulate per-column sum and sum of squares in double precision.

    The accumulation runs over samples in order, matching dspsr's summation order.

    Parameters
    ----------
    data : NDArray[float32]
        Data with shape ``(nsamp, ncol)``.
    sums : NDArray[float64]
        Running per-column sum, updated in place.
    sumsq : NDArray[float64]
        Running per-column sum of squares, updated in place.
    """
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
    """Apply per-column offset and scale as ``(x + offset) * scale`` in float32.

    Parameters
    ----------
    data : NDArray[float32]
        Input data with shape ``(nsamp, ncol)``.
    offset : NDArray[float32]
        Per-column offset with shape ``(ncol,)``.
    scale : NDArray[float32]
        Per-column scale with shape ``(ncol,)``.
    out : NDArray[float32]
        Output array with shape ``(nsamp, ncol)``.
    """
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
    """Quantise one value as dspsr's ``SigProcDigitizer`` does.

    Computes ``(int)((x * scale + mean) + 0.5)``, with the C cast truncating
    towards zero, and clips the result to ``[0, digi_max]``.
    """
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
    """Quantise an ``(nrow, nchan)`` array to unsigned integers.

    Parameters
    ----------
    data : NDArray[float32]
        Data with shape ``(nrow, nchan)`` in input channel order.
    flip : bool
        Reverse the channel order in the output.
    means : NDArray[float32]
        Digitiser mean applied before the cast, one value per row.
    scale : float32
        Digitiser scale applied to the input.
    digi_max : int
        Maximum quantised value (``2**nbit - 1``).
    out : NDArray[uint8 | uint16]
        Output array with shape ``(nrow, nchan)``.
    """
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
    """Quantise and pack 1, 2 or 4-bit samples least-significant-bit first.

    Matches dspsr's sub-byte layout: within each row, output channel ``c`` is
    stored in bits ``(c % (8 // nbit)) * nbit`` of byte ``c // (8 // nbit)``.

    Parameters
    ----------
    data : NDArray[float32]
        Data with shape ``(nrow, nchan)`` in input channel order. ``nchan`` must
        be a multiple of ``8 // nbit``.
    flip : bool
        Reverse the channel order in the output.
    means : NDArray[float32]
        Digitiser mean applied before the cast, one value per row.
    scale : float32
        Digitiser scale applied to the input.
    digi_max : int
        Maximum quantised value (``2**nbit - 1``).
    nbit : int
        Bits per sample, one of 1, 2 or 4.
    packed : NDArray[uint8]
        Zeroed output array of length ``nrow * nchan // (8 // nbit)``.
    """
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


def digitiser_params(
    nbit: int,
    *,
    rescale_on: bool,
    scale_in: float,
    scale_fac: float,
) -> tuple[np.float32, np.float32]:
    """Return the digitiser ``(mean, scale)`` as dspsr's ``SigProcDigitizer``.

    Parameters
    ----------
    nbit : int
        Output bits per sample (1, 2, 4, 8 or 16).
    rescale_on : bool
        Whether the data were rescaled to unit variance first. Without rescaling
        the nbit-dependent mean and scale are not applied.
    scale_in : float
        Scale factor already carried by the data.
    scale_fac : float
        Additional user scale factor (``-s``).

    Returns
    -------
    tuple[float32, float32]
        Digitiser mean and scale.
    """
    if not rescale_on:
        mean = 0.0
        scale = 1.0
    elif nbit in {4, 8, 16}:
        mean = DIGI_MEAN[nbit]
        scale = mean / DIGI_SIGMA
    else:
        mean = DIGI_MEAN[nbit]
        scale = 1.0
    scale = scale / (scale_in * scale_fac)
    return np.float32(mean), np.float32(scale)


def polarisation_means(npol_out: int, nbit: int, *, rescale_on: bool) -> np.ndarray:
    """Return the digitiser mean for each output polarisation.

    Without rescaling, dspsr adds the bit-depth mean to polarisations 2 and 3 (the
    cross terms, which are signed) so that the unsigned output fits; the others use
    zero. With rescaling every polarisation uses the bit-depth mean.

    Parameters
    ----------
    npol_out : int
        Number of output polarisations.
    nbit : int
        Output bits per sample.
    rescale_on : bool
        Whether the data were rescaled to unit variance first.

    Returns
    -------
    NDArray[float32]
        One mean per output polarisation.
    """
    table_mean = np.float32(DIGI_MEAN[nbit])
    if rescale_on:
        return np.full(npol_out, table_mean, dtype=np.float32)
    means = np.zeros(npol_out, dtype=np.float32)
    means[2:] = table_mean
    return means


def quantise_rows(
    rows: np.ndarray,
    nbit: int,
    *,
    flip: bool,
    means: np.ndarray,
    scale: np.float32,
) -> np.ndarray:
    """Quantise rows of data to the sigproc sample format.

    Parameters
    ----------
    rows : NDArray[float32]
        Data with shape ``(nrow, nchan)`` in input channel order.
    nbit : int
        Output bits per sample: 1, 2, 4, 8 or 16.
    flip : bool
        Reverse the channel order in the output.
    means : NDArray[float32]
        Digitiser mean for each row.
    scale : float32
        Digitiser scale.

    Returns
    -------
    NDArray
        Packed ``uint8`` array for 1, 2 and 4-bit output, ``uint8`` for 8-bit and
        little-endian ``uint16`` for 16-bit output, each in row-major order.
    """
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    digi_max = (1 << nbit) - 1
    if nbit in {1, 2, 4}:
        packed = np.zeros(rows.size // (8 // nbit), dtype=np.uint8)
        quantise_packed(rows, flip, means, scale, digi_max, nbit, packed)
        return packed
    out_dtype = np.dtype("<u2") if nbit == 16 else np.dtype(np.uint8)
    out = np.empty(rows.shape, dtype=out_dtype)
    quantise(rows, flip, means, scale, digi_max, out)
    return out


def float_rows(rows: np.ndarray, *, flip: bool, scale_in: float) -> np.ndarray:
    """Convert rows to little-endian float32 sigproc samples (``-b -32``).

    Parameters
    ----------
    rows : NDArray[float32]
        Data with shape ``(nrow, nchan)`` in input channel order.
    flip : bool
        Reverse the channel order in the output.
    scale_in : float
        Scale factor carried by the data, divided out as in dspsr.

    Returns
    -------
    NDArray[float32]
        Little-endian float32 array with output channel order.
    """
    out = rows[:, ::-1] if flip else rows
    return (out / np.float32(scale_in)).astype("<f4", copy=False)


def dspsr_block_samples(
    nbit_in: int,
    npol: int,
    nchan: int,
    *,
    copies: int,
    resolution: int,
    max_ram_bytes: float,
) -> int:
    """Return the input block length dspsr uses for rescaling.

    dspsr sizes its input blocks from a RAM limit (``-U``, default 256 MB):
    ``floor(max_ram / nbyte_dat)`` samples, where ``nbyte_dat`` is the input sample
    size plus four bytes per float copy, and rounds the result up to the format
    resolution (whole PSRFITS subints, or a multiple of four samples for sigproc).

    Parameters
    ----------
    nbit_in : int
        Input bits per sample.
    npol : int
        Number of input polarisations.
    nchan : int
        Number of input channels.
    copies : int
        Number of float time-series buffers in the pipeline.
    resolution : int
        Format resolution in samples.
    max_ram_bytes : float
        RAM limit in bytes.

    Returns
    -------
    int
        Block length in input samples.
    """
    nbyte_dat = (nbit_in / 8 + 4 * copies) * npol * nchan
    nblock = math.floor(max_ram_bytes / nbyte_dat)
    resolution = max(1, resolution)
    return max(resolution, math.ceil(nblock / resolution) * resolution)


class Rescaler:
    """Rescale a stream to zero mean and unit variance, as dspsr ``Rescale``.

    Statistics are accumulated per column over the samples of an interval of
    ``nsample`` samples, which may span several input blocks. At the end of each
    block during the first interval the statistics are computed from the samples so
    far (dspsr behaviour), so block boundaries affect the first interval. Each
    block is rescaled with the most recent offsets and scales.

    Parameters
    ----------
    ncol : int
        Number of independent columns (channels, or polarisation-channels).
    nsample : int
        Number of samples per statistics interval.
    constant : bool
        Freeze the offset and scale after the first complete interval.
    """

    def __init__(self, ncol: int, nsample: int, *, constant: bool) -> None:
        self.nsample = nsample
        self.constant = constant
        self.first_integration = True
        self.isample = 0
        self.sums = np.zeros(ncol, dtype=np.float64)
        self.sumsq = np.zeros(ncol, dtype=np.float64)
        self.offset = np.zeros(ncol, dtype=np.float32)
        self.scale = np.ones(ncol, dtype=np.float32)

    def _compute(self) -> None:
        nval_recip = 1.0 / self.isample
        mean = self.sums * nval_recip
        meansq = self.sumsq * nval_recip
        variance = meansq - mean * mean
        positive = variance > 0
        safe = np.where(positive, variance, 1.0)
        self.offset = (-mean).astype(np.float32)
        self.scale = np.where(positive, 1.0 / np.sqrt(safe), 1.0).astype(np.float32)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Rescale one block of ``(nsamp, ncol)`` data.

        Parameters
        ----------
        data : NDArray[float32]
            Block data.

        Returns
        -------
        NDArray[float32]
            Rescaled block.
        """
        nsamp = data.shape[0]
        out = np.empty_like(data)
        if nsamp == 0:
            return out
        if self.constant and not self.first_integration:
            apply_scale_offset(data, self.offset, self.scale, out)
            return out

        start = 0
        while start < nsamp:
            step = min(nsamp - start, self.nsample - self.isample)
            end = start + step

            if self.first_integration or not self.constant:
                accumulate_stats(data[start:end], self.sums, self.sumsq)
                self.isample += step

            if self.isample == self.nsample or (
                self.first_integration and end == nsamp
            ):
                self._compute()
                if self.isample == self.nsample:
                    self.first_integration = False
                    self.isample = 0
                    self.sums[:] = 0
                    self.sumsq[:] = 0

            if self.constant and not self.first_integration:
                end = nsamp
            apply_scale_offset(data[start:end], self.offset, self.scale, out[start:end])
            start = end
        return out
