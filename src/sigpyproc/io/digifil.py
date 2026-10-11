"""Sigproc filterbank conversion and redigitisation.

This is the streaming driver for filterbank digifil operations. It accepts
PSRFITS files, sigproc filterbank files, or any sigproc reader object, and
reproduces dspsr's ``digifil`` pipeline.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np

from sigpyproc.header import Header
from sigpyproc.io import sigproc
from sigpyproc.io.fileio import FileWriter
from sigpyproc.io.pfits import PFITSFile
from sigpyproc.kernels.digitizer import (
    accumulate_stats,
    apply_scale_offset,
    fscrunch,
    pscrunch_intensity,
    quantise,
    quantise_packed,
    tscrunch,
)

# Digitiser mean per output bit depth (SigProcDigitizer.C) and the sigma used
# for the 4, 8 and 16-bit scale (dspsr uses 6 sigma for all of these).
DIGI_MEAN: dict[int, float] = {1: 0.5, 2: 1.5, 4: 7.5, 8: 127.5, 16: 32768.0}
DIGI_SIGMA = 6.0


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


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from astropy.units import Quantity
    from numpy.typing import NDArray

    from sigpyproc.base import Filterbank

logger = logging.getLogger(__name__)

MEGABYTE = 1024.0 * 1024.0
VALID_NBITS = (1, 2, 4, 8, 16, -32)
PFITS_SUFFIXES = {".sf", ".fits", ".rf", ".cf"}
# dspsr's resolution for sigproc input (multiple of four samples).
SIGPROC_RESOLUTION = 4


@attrs.define(slots=True, auto_attribs=True)
class DigifilConfig:
    """Configuration of the digifil pipeline.

    Parameters
    ----------
    nbits : int, optional
        Bits per output sample: 1, 2, 4, 8, 16, or -32 for 32-bit float,
        by default 8.
    gulp : int, optional
        Input samples read per block. The rescale statistics depend on these block
        boundaries, so a given ``gulp`` reproduces dspsr's output. By default it is
        the block dspsr uses with its 256 MB RAM limit (see :func:`default_gulp`).
    rescale_seconds : float, optional
        Rescale interval in seconds, 0 disables rescaling, by default 10.
    rescale_constant : bool, optional
        Keep the offset and scale constant after the first estimate,
        by default False.
    tscrunch_factor : int, optional
        Time decimation factor, by default 1.
    fscrunch_factor : int, optional
        Frequency decimation factor, by default 1.
    scale_fac : float, optional
        Additional scale factor applied before digitising, by default 1.
    scloffs : bool, optional
        Denormalise PSRFITS data with DAT_SCL and DAT_OFFS, by default False.
    npol : int, optional
        Output polarisations as in dspsr ``-d``: 1 forms PP+QQ intensity from
        multi-pol input; 2 or 4 keep all input polarisations, by default 1.
    poln_select : int, optional
        Output only this input polarisation (dspsr ``-P``), -1 keeps all,
        by default -1.
    """

    nbits: int = 8
    gulp: int | None = None
    rescale_seconds: float = 10.0
    rescale_constant: bool = False
    tscrunch_factor: int = 1
    fscrunch_factor: int = 1
    scale_fac: float = 1.0
    scloffs: bool = False
    npol: int = 1
    poln_select: int = -1

    def __attrs_post_init__(self) -> None:
        if self.nbits not in VALID_NBITS:
            msg = f"nbits must be one of {VALID_NBITS}, got {self.nbits}"
            raise ValueError(msg)
        if self.npol not in {1, 2, 4}:
            msg = f"npol must be 1, 2 or 4, got {self.npol}"
            raise ValueError(msg)
        if self.poln_select < -1:
            msg = (
                "poln_select must be -1 or a polarisation index, "
                f"got {self.poln_select}"
            )
            raise ValueError(msg)
        if self.gulp is not None and self.gulp <= 0:
            msg = f"gulp must be positive, got {self.gulp}"
            raise ValueError(msg)
        if self.rescale_seconds < 0:
            msg = f"rescale_seconds must be >= 0, got {self.rescale_seconds}"
            raise ValueError(msg)
        if self.tscrunch_factor < 0 or self.fscrunch_factor < 0:
            msg = "tscrunch_factor and fscrunch_factor must be >= 0"
            raise ValueError(msg)
        if self.scale_fac == 0:
            msg = "scale_fac must be non-zero"
            raise ValueError(msg)


@attrs.define(slots=True, auto_attribs=True)
class _Source:
    """Metadata and block reader for a digifil input."""

    header: Header
    filename: str
    nchan: int
    npol: int
    nbit: int
    resolution: int
    nsamples: int
    blocks: Callable[..., Iterator[NDArray[np.float32]]]
    """Yield ``(nsamp, npol, nchan)`` blocks of about ``gulp`` samples."""


def _as_mhz(value: float | Quantity) -> float:
    """Return a frequency in MHz as a plain float (PSRFITS headers use Quantity)."""
    if isinstance(value, float | int):
        return float(value)
    return float(value.to_value("MHz"))


def _trim(
    chunks: Iterable[NDArray[np.float32]],
    start: int,
    nsamps: int,
) -> Iterator[NDArray[np.float32]]:
    """Yield the samples ``[start, start + nsamps)`` from a stream of chunks."""
    skip = start
    remaining = nsamps
    for chunk in chunks:
        if remaining <= 0:
            break
        nchunk = chunk.shape[0]
        if skip >= nchunk:
            skip -= nchunk
            continue
        part = chunk[skip : skip + remaining]
        skip = 0
        remaining -= part.shape[0]
        if part.shape[0]:
            yield part


def _rechunk(
    chunks: Iterable[NDArray[np.float32]],
    nsrc: int,
) -> Iterator[NDArray[np.float32]]:
    """Regroup ``(nsamp, npol, nchan)`` arrays into blocks of ``nsrc`` samples."""
    pending: list[NDArray[np.float32]] = []
    npending = 0
    for chunk in chunks:
        pending.append(chunk)
        npending += chunk.shape[0]
        while npending >= nsrc:
            joined = np.concatenate(pending)
            yield joined[:nsrc]
            rest = joined[nsrc:]
            pending = [rest] if rest.shape[0] else []
            npending = rest.shape[0]
    if npending:
        yield np.concatenate(pending)


def _open_reader(reader: Filterbank, start: int, nsamps: int | None) -> _Source:
    """Wrap a sigproc filterbank reader (``FilReader`` or similar)."""
    hdr = reader.header
    nchan = hdr.nchans
    total = hdr.nsamples - start if nsamps is None else nsamps

    def blocks(gulp: int, *, exact: bool) -> Iterator[NDArray[np.float32]]:  # noqa: ARG001
        for nsamp, _, data in reader.read_plan(
            gulp=gulp,
            start=start,
            nsamps=total,
            quiet=True,
        ):
            yield np.array(data[: nsamp * nchan], dtype=np.float32).reshape(
                nsamp,
                1,
                nchan,
            )

    return _Source(
        header=hdr,
        filename=str(getattr(reader, "filename", "digifil")),
        nchan=nchan,
        npol=1,
        nbit=hdr.nbits,
        resolution=SIGPROC_RESOLUTION,
        nsamples=total,
        blocks=blocks,
    )


def _open_pfits(
    filename: str, *, scloffs: bool, start: int, nsamps: int | None
) -> _Source:
    """Open a PSRFITS file, reading whole SUBINT rows with DAT_SCL/DAT_OFFS if asked."""
    hdr = Header.from_pfits(filename)
    hdr = hdr.new_header(
        {"fch1": _as_mhz(hdr.fch1), "foff": _as_mhz(hdr.foff)},
    )
    with PFITSFile(filename) as fits:
        sub = fits.sub_hdr
        npol = sub.npol
        if npol > 1 and sub.poln_state == "Stokes":
            msg = "digifil does not support Stokes PSRFITS input"
            raise NotImplementedError(msg)
        nchan = sub.nchans
        subint_samples = sub.subint_samples
        nsubint = sub.nsubint
        nbit = sub.nbits
    total = hdr.nsamples - start if nsamps is None else nsamps

    def subints() -> Iterator[NDArray[np.float32]]:
        with PFITSFile(filename) as fits:
            for isub in range(nsubint):
                data = fits.read_subint(isub, scloffs=scloffs, weights=False)
                yield np.asarray(data, dtype=np.float32).reshape(
                    subint_samples,
                    npol,
                    nchan,
                )

    def blocks(gulp: int, *, exact: bool) -> Iterator[NDArray[np.float32]]:
        selected = _trim(subints(), start, total)
        if exact:
            return _rechunk(selected, gulp)
        return selected

    return _Source(
        header=hdr,
        filename=filename,
        nchan=nchan,
        npol=npol,
        nbit=nbit,
        resolution=subint_samples,
        nsamples=total,
        blocks=blocks,
    )


def _open_source(
    source: str | Path | Filterbank,
    *,
    scloffs: bool,
    start: int,
    nsamps: int | None,
) -> _Source:
    """Open a filename or a reader object as a digifil source."""
    if isinstance(source, str | Path):
        path = str(source)
        if Path(path).suffix.lower() in PFITS_SUFFIXES:
            return _open_pfits(path, scloffs=scloffs, start=start, nsamps=nsamps)
        from sigpyproc.readers import FilReader

        return _open_reader(FilReader(path), start, nsamps)

    from sigpyproc.readers import PFITSReader

    if isinstance(source, PFITSReader):
        return _open_pfits(source.filename, scloffs=scloffs, start=start, nsamps=nsamps)
    return _open_reader(source, start, nsamps)


def _polarisation_plan(npol_in: int, config: DigifilConfig) -> tuple[int, bool, int]:
    """Return ``(pol_select, pscrunch, npol_out)`` for the polarisation options.

    ``-P`` selects one input polarisation; ``-d 1`` forms PP+QQ intensity from
    multi-pol input; otherwise all input polarisations are kept.
    """
    if config.poln_select >= npol_in:
        msg = (
            f"poln_select={config.poln_select} out of range for "
            f"{npol_in}-polarisation input"
        )
        raise ValueError(msg)
    if config.poln_select >= 0:
        return config.poln_select, False, 1
    if config.npol == 1 and npol_in > 1:
        return -1, True, 1
    return -1, False, npol_in


def _float_buffers(config: DigifilConfig, *, pscrunch: bool) -> int:
    """Count the float time-series buffers dspsr allocates for a configuration."""
    return (
        1
        + int(config.fscrunch_factor > 1)
        + int(config.tscrunch_factor > 1)
        + int(pscrunch)
    )


class _Digifil:
    """Stateful digifil pipeline acting on successive input blocks."""

    def __init__(
        self,
        config: DigifilConfig,
        source: _Source,
    ) -> None:
        self.config = config
        self.nchan = source.nchan
        if config.fscrunch_factor > 1:
            self.nchan = source.nchan // config.fscrunch_factor
        self.npol_in = source.npol
        self.pol_select, self.pscrunch, self.npol_out = _polarisation_plan(
            source.npol,
            config,
        )
        self.tfactor = max(1, config.tscrunch_factor)
        self.rate_out = 1.0 / (source.header.tsamp * self.tfactor)
        self.flip = source.header.foff > 0

        self.scale_total = np.float32(1.0)
        if config.fscrunch_factor > 1:
            self.scale_total = np.float32(
                self.scale_total * np.float32(config.fscrunch_factor),
            )
        if self.tfactor > 1:
            self.scale_total = np.float32(self.scale_total * np.float32(self.tfactor))

        self.rescale_on = config.rescale_seconds != 0
        self.rescaler: Rescaler | None = None
        if self.rescale_on:
            nsample = int(config.rescale_seconds * self.rate_out)
            if nsample == 0:
                msg = (
                    "Rescale window is zero samples, increase rescale_seconds "
                    f"(got {config.rescale_seconds} s at {self.rate_out} samples/s)"
                )
                raise ValueError(msg)
            self.rescaler = Rescaler(
                self.npol_out * self.nchan,
                nsample,
                constant=config.rescale_constant,
            )

        self.scale_in = np.float32(1.0) if self.rescale_on else self.scale_total
        self.is_float = config.nbits == -32
        self.nbit = abs(config.nbits)
        self.digi_scale = np.float32(1.0)
        self.pol_mean = np.zeros(self.npol_out, dtype=np.float32)
        if not self.is_float:
            _, self.digi_scale = digitiser_params(
                self.nbit,
                rescale_on=self.rescale_on,
                scale_in=float(self.scale_in),
                scale_fac=config.scale_fac,
            )
            self.pol_mean = polarisation_means(
                self.npol_out,
                self.nbit,
                rescale_on=self.rescale_on,
            )
            if self.nbit < 8 and self.nchan % (8 // self.nbit) != 0:
                msg = (
                    f"nchan={self.nchan} must be a multiple of "
                    f"{8 // self.nbit} for {self.nbit}-bit output"
                )
                raise ValueError(msg)

        self.carry = np.empty((0, self.npol_out, self.nchan), dtype=np.float32)

    @property
    def nbits_out(self) -> int:
        return 32 if self.is_float else self.nbit

    def _tscrunch(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Time-scrunch a block, carrying leftover samples to the next block."""
        factor = self.tfactor
        head = None
        if self.carry.shape[0]:
            need = factor - self.carry.shape[0]
            if data.shape[0] < need:
                self.carry = np.concatenate([self.carry, data])
                return data[:0]
            head = tscrunch(np.concatenate([self.carry, data[:need]]), factor)
            data = data[need:]

        nout = data.shape[0] // factor
        body = tscrunch(data[: nout * factor], factor)
        self.carry = np.array(data[nout * factor :], dtype=np.float32)
        if head is None:
            return body
        return np.concatenate([head, body])

    def process(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run the selection, scrunch and rescale stages on one block.

        Parameters
        ----------
        block : NDArray[float32]
            Input block with shape ``(nsamp, npol_in, nchan_in)``.

        Returns
        -------
        NDArray[float32]
            Processed block with shape ``(nsamp_out, npol_out, nchan)``, in input
            channel order.
        """
        data = block
        if self.pol_select >= 0:
            data = np.ascontiguousarray(data[:, self.pol_select : self.pol_select + 1])
        if self.config.fscrunch_factor > 1:
            data = fscrunch(data, self.config.fscrunch_factor)

        if self.tfactor > 1:
            data = self._tscrunch(data)

        if self.pscrunch:
            data = pscrunch_intensity(data)[:, np.newaxis, :]
        else:
            data = np.ascontiguousarray(data)

        if self.rescaler is not None:
            nsamp = data.shape[0]
            flat = self.rescaler.process(data.reshape(nsamp, -1))
            data = flat.reshape(nsamp, self.npol_out, self.nchan)
        return data

    def encode(self, data: NDArray[np.float32]) -> bytes:
        """Quantise (or convert to float) one block and return the packed bytes.

        Parameters
        ----------
        data : NDArray[float32]
            Block with shape ``(nsamp, npol_out, nchan)`` in input channel order.

        Returns
        -------
        bytes
            Sigproc samples in (time, polarisation, channel) order, with channels in
            output order.
        """
        rows = data.reshape(-1, data.shape[-1])
        if self.is_float:
            return float_rows(
                rows, flip=self.flip, scale_in=float(self.scale_in)
            ).tobytes()
        means = np.tile(self.pol_mean, data.shape[0])
        return quantise_rows(
            rows,
            self.nbit,
            flip=self.flip,
            means=means,
            scale=self.digi_scale,
        ).tobytes()


def _output_header(
    hdr_in: Header,
    nchan_in: int,
    nchan_out: int,
    npol_out: int,
    nsamples_out: int,
    tsamp_out: float,
    nbits_out: int,
) -> Header:
    """Build the output sigproc header as dspsr digifil does.

    The output bandwidth is negative (dspsr's convention), so the channels are
    reversed when the input bandwidth is positive. Optional keys of the input, such as
    ``rawdatafile`` and the azimuth and zenith angles, are carried over unchanged.
    """
    bw_abs = abs(hdr_in.foff) * nchan_in
    centre = hdr_in.fch1 + 0.5 * hdr_in.foff * (nchan_in - 1)
    foff_out = -bw_abs / nchan_out
    fch1_out = centre + 0.5 * bw_abs - 0.5 * (bw_abs / nchan_out)
    return hdr_in.new_header(
        {
            "nchans": nchan_out,
            "foff": foff_out,
            "fch1": fch1_out,
            "tsamp": tsamp_out,
            "nbits": nbits_out,
            "nifs": npol_out,
            "nsamples": nsamples_out,
        },
    )


DSPSR_MAX_RAM_MB = 256.0


def _dspsr_default_gulp(
    src: _Source,
    cfg: DigifilConfig,
    *,
    pscrunch: bool,
    max_ram_mb: float = DSPSR_MAX_RAM_MB,
) -> int:
    """Return the block length dspsr uses for this input under a RAM limit."""
    return dspsr_block_samples(
        src.nbit,
        src.npol,
        src.nchan,
        copies=_float_buffers(cfg, pscrunch=pscrunch),
        resolution=src.resolution,
        max_ram_bytes=max_ram_mb * MEGABYTE,
    )


def default_gulp(
    source: str | Path | Filterbank,
    config: DigifilConfig | None = None,
    *,
    max_ram_mb: float = DSPSR_MAX_RAM_MB,
    start: int = 0,
    nsamps: int | None = None,
) -> int:
    """Return the ``gulp`` dspsr digifil would use for an input under a RAM limit.

    dspsr sizes its input blocks from a RAM limit (``-U``, 256 MB by default), so this
    converts that limit into the equivalent number of input samples per block.

    Parameters
    ----------
    source : str | Path | Filterbank
        Input file or reader object.
    config : DigifilConfig, optional
        Pipeline configuration; its polarisation and scrunch options set the
        buffer count, by default the digifil defaults.
    max_ram_mb : float, optional
        RAM limit in megabytes, by default 256 (dspsr's default).
    start : int, optional
        First input sample, by default 0.
    nsamps : int, optional
        Number of input samples, by default all from ``start``.

    Returns
    -------
    int
        Input samples per block.
    """
    cfg = config if config is not None else DigifilConfig()
    src = _open_source(source, scloffs=cfg.scloffs, start=start, nsamps=nsamps)
    _, pscrunch, _ = _polarisation_plan(src.npol, cfg)
    return _dspsr_default_gulp(src, cfg, pscrunch=pscrunch, max_ram_mb=max_ram_mb)


def _default_outfile(infile: str) -> str:
    stem = Path(infile).stem
    outfile = f"{stem}.fil"
    if Path(outfile).resolve() == Path(infile).resolve():
        outfile = f"{stem}_digi.fil"
    return outfile


def digifil(
    source: str | Path | Filterbank,
    outfile: str | None = None,
    config: DigifilConfig | None = None,
    *,
    start: int = 0,
    nsamps: int | None = None,
) -> str:
    """Convert or redigitise sigproc or PSRFITS data, following dspsr digifil.

    This is the Python API for the ``spp_digifil`` command. It accepts a filename or
    an open reader (``FilReader``, ``PFITSReader``, or any object with a sigproc
    ``header`` and ``read_plan``).

    Parameters
    ----------
    source : str | Path | Filterbank
        Input file or reader object.
    outfile : str, optional
        Output sigproc filterbank name, by default ``<stem>.fil`` of the input.
    config : DigifilConfig, optional
        Pipeline configuration, by default the digifil defaults.
    start : int, optional
        First input sample to process, by default 0.
    nsamps : int, optional
        Number of input samples to process, by default all from ``start``.

    Returns
    -------
    str
        Name of the output file.

    Raises
    ------
    ValueError
        If the configuration is inconsistent with the input.
    """
    cfg = config if config is not None else DigifilConfig()
    src = _open_source(source, scloffs=cfg.scloffs, start=start, nsamps=nsamps)
    filename = src.filename
    if outfile is None:
        outfile = _default_outfile(filename)

    _, pscrunch, npol_out = _polarisation_plan(src.npol, cfg)
    pipeline = _Digifil(cfg, src)
    tscr = max(1, cfg.tscrunch_factor)
    hdr_out = _output_header(
        src.header,
        src.nchan,
        pipeline.nchan,
        npol_out,
        src.nsamples // tscr,
        1.0 / pipeline.rate_out,
        pipeline.nbits_out,
    )

    block = (
        cfg.gulp
        if cfg.gulp is not None
        else _dspsr_default_gulp(
            src,
            cfg,
            pscrunch=pscrunch,
        )
    )
    exact = cfg.rescale_seconds != 0
    out_file = FileWriter(outfile, mode="w+", nbits=pipeline.nbits_out)
    out_sig = hdr_out.to_sigproc()
    out_file.write(sigproc.encode_header(out_sig))
    for raw in src.blocks(block, exact=exact):
        processed = pipeline.process(raw)
        if processed.shape[0] == 0:
            continue
        out_file.write(pipeline.encode(processed))
    out_file.close()
    logger.info(f"digifil: wrote {outfile}")
    return outfile
