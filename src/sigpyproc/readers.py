"""Classes for reading filterbank and PSRFITS data.

This module contains the FilReader and PFITSReader classes for reading
filterbank and PSRFITS data.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Self

import attrs
import numpy as np
from rich.progress import track

from sigpyproc.base import Filterbank
from sigpyproc.block import FilterbankBlock
from sigpyproc.core.logging import progress_disabled
from sigpyproc.header import Header
from sigpyproc.io.bits import check_packed_length
from sigpyproc.io.fileio import FileReader, allocate_buffer
from sigpyproc.io.pfits import PFITSFile, PrimaryHdr, SubintHdr
from sigpyproc.kernels.dedispersion import (
    place_dedisp_block,
    place_dedisp_span,
    roll_block_valid,
)
from sigpyproc.utils import detect_file_type, get_callerfunc

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable, Iterator, Sequence

    from sigpyproc.core.custom_types import LocMethods
    from sigpyproc.io.bits import BitsInfo

logger = logging.getLogger(__name__)

# One sequential read is cheaper than a seek per sample while the dispersed
# rectangle still fits in memory. Above this, stream or seek to the channels
# the DM actually uses.
_BULK_BYTES = 32 * 1024 * 1024
_STREAM_GULP_BYTES = 8 * 1024 * 1024


def _check_plan_args(gulp: int, start: int) -> None:
    """Reject a read plan that would reach a kernel with a bad window."""
    if gulp < 1:
        msg = f"gulp must be >= 1, got {gulp}"
        raise ValueError(msg)
    if start < 0:
        msg = f"start must be >= 0, got {start}"
        raise ValueError(msg)


def _chan_start(header: Header, fch1: float, nchans: int) -> int:
    """Return the first channel index of a contiguous sub-band."""
    if nchans < 1 or nchans > header.nchans or header.foff == 0:
        msg = f"requested block is out of range: fch1={fch1}, nchans={nchans}"
        raise ValueError(msg)
    chan_start = round((fch1 - header.fch1) / header.foff)
    if chan_start < 0 or chan_start + nchans > header.nchans:
        msg = f"requested block is out of range: fch1={fch1}, nchans={nchans}"
        raise ValueError(msg)
    return chan_start


def _dedisp_window(
    header: Header,
    start: int,
    nsamps: int,
    dm: float,
) -> tuple[np.ndarray, int, int]:
    """Return aligned delays and the input sample where that window starts.

    Delays are shifted so the earliest channel is zero. Output sample 0 is
    the reference-frequency sample at ``start``.
    """
    if nsamps < 1:
        msg = f"nsamps must be at least 1, got {nsamps}"
        raise ValueError(msg)
    raw = np.asarray(header.get_dmdelays(dm), dtype=np.int32)
    lo = int(raw.min())
    hi = int(raw.max())
    in_start = start + lo
    n_in = nsamps + (hi - lo)
    if start < 0 or in_start < 0 or in_start + n_in > header.nsamples:
        msg = (
            f"requested dedispersed block is out of range: "
            f"start={start}, nsamps={nsamps}"
        )
        raise ValueError(msg)
    aligned = np.ascontiguousarray(raw - lo, dtype=np.int32)
    return aligned, in_start, n_in


def _dedisp_read_mode(
    n_in: int,
    nsamps: int,
    samp_stride: int,
    nbits: int,
    *,
    small_reads: bool | None,
) -> str:
    """Choose how to read a dispersed block from disk.

    ``"bulk"`` is one sequential read of the whole delay window. ``"stream"``
    walks that window in small gulps and keeps only the output block.
    ``"partial"`` seeks to the channels live at each sample. ``small_reads``
    forces partial reads (``True``) or one bulk read (``False``). ``None``
    picks from the DM geometry.
    """
    if nbits < 8 and small_reads is True:
        msg = (
            "small_reads needs at least 8 bits per sample so a channel can be "
            f"sought directly; this file is {nbits}-bit"
        )
        raise ValueError(msg)
    if small_reads is False:
        return "bulk"
    if small_reads is True:
        return "partial"
    bulk_bytes = n_in * samp_stride
    active = nsamps / n_in if n_in else 1.0
    if bulk_bytes <= _BULK_BYTES:
        return "bulk"
    if nbits < 8 or active >= 0.5:
        return "stream"
    return "partial"


def _active_spans(
    delays: np.ndarray,
    nsamps: int,
    n_in: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Channel slice live at each input sample, or None if delays are gapped.

    Delays must be monotonic in channel order. The slice is inclusive of
    ``clo`` and exclusive of ``chi``.
    """
    diff = np.diff(delays)
    if diff.size and not (np.all(diff >= 0) or np.all(diff <= 0)):
        return None
    nchans = int(delays.size)
    decreasing = bool(diff.size and np.all(diff <= 0))
    ordered = delays[::-1] if decreasing else delays
    sample = np.arange(n_in, dtype=np.int32)
    left = np.searchsorted(ordered, sample - nsamps, side="right")
    right = np.searchsorted(ordered, sample, side="right")
    if decreasing:
        clo = nchans - right
        chi = nchans - left
    else:
        clo = left
        chi = right
    return clo.astype(np.int32, copy=False), chi.astype(np.int32, copy=False)


def _gather_dispersed(
    block: np.ndarray,
    aligned: np.ndarray,
    nsamps: int,
) -> np.ndarray:
    """Keep ``nsamps`` valid samples per channel from a dispersed window."""
    shifts = np.ascontiguousarray(-aligned, dtype=np.int64)
    out = roll_block_valid(block, shifts)
    if out.shape[1] != nsamps:
        msg = f"dedispersed length {out.shape[1]} != {nsamps}"
        raise RuntimeError(msg)
    return out


def open_filterbank(filename: str | Path) -> FilReader | PFITSReader:
    """Open a sigproc filterbank or a search-mode PSRFITS file.

    Parameters
    ----------
    filename : str | Path
        ``.fil``, ``.tim``, ``.fits``, or ``.sf`` path.

    Returns
    -------
    FilReader | PFITSReader
        Reader for the file.

    Raises
    ------
    ValueError
        If the file is not sigproc or PSRFITS search data.
    """
    kind = detect_file_type(filename)
    if kind == "sigproc":
        return FilReader(filename)
    if kind == "pfits":
        return PFITSReader(str(filename))
    msg = (
        f"{filename} is {kind} data; filterbank commands read "
        "sigproc and PSRFITS search files"
    )
    raise ValueError(msg)


class FilReader(Filterbank):
    """A class to handle the reading of sigproc format filterbank files.

    Parameters
    ----------
    filenames : str | Path | Sequence[str | Path]
        Filterbank file name(s).
    check_contiguity : bool, optional
        Check if the input files are contiguous, by default True.

    Returns
    -------
    :class:`~sigpyproc.base.Filterbank`
        Base container of filterbank data with observational metadata.

    Notes
    -----
    To be considered as a Sigproc format filterbank file the header must only contain
    keywords found in the :py:obj:`~sigpyproc.io.sigproc.SIGPROC_SCHEMA` dictionary.
    """

    def __init__(
        self,
        filenames: str | Path | Sequence[str | Path],
        *,
        check_contiguity: bool = True,
    ) -> None:
        if isinstance(filenames, str | Path):
            filenames = [filenames]
        self._filenames = filenames
        self._header = Header.from_sigproc(
            self._filenames,
            check_contiguity=check_contiguity,
        )
        check_packed_length(self._header.nchans, self._header.nbits)
        if self._header.nifs != 1:
            msg = f"FilReader supports one IF, got nifs={self._header.nifs}"
            raise NotImplementedError(msg)
        self._file = FileReader(
            self.header.stream_info,
            mode="r",
            nbits=self.header.nbits,
        )
        super().__init__()

    @property
    def header(self) -> Header:
        return self._header

    @property
    def filename(self) -> str | Path:
        """str: Name of the input file (first file in case of multiple input files)."""
        return self._filenames[0]

    @property
    def bitsinfo(self) -> BitsInfo:
        """:class:`~sigpyproc.io.bits.BitsInfo`: Bits info of input file data."""
        return self._file.bitsinfo

    @property
    def chan_stride(self) -> float:
        """int: Channel byte stride in input data."""
        return self.bitsinfo.itemsize / self.bitsinfo.bitfact

    @property
    def samp_stride(self) -> int:
        """int: Sample byte stride in input data."""
        return int(self.header.nchans * self.chan_stride)

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _maybe_signed(self, data: np.ndarray) -> np.ndarray:
        """Return samples in the dtype the streaming kernels accept.

        Signed 8-bit storage is decoded to float32. A 16-bit gulp is copied
        to float32, which holds every ``uint16`` value exactly. Packed and
        unsigned 8-bit samples stay ``uint8``. 32-bit samples are already
        float32.
        """
        if self.header.signed and self.header.nbits == 8:
            return np.ascontiguousarray(data.view(np.int8), dtype=np.float32)
        if self.header.nbits == 16:
            return np.ascontiguousarray(data, dtype=np.float32)
        return data

    def read_block(
        self,
        start: int,
        nsamps: int,
        fch1: float | None = None,
        nchans: int | None = None,
    ) -> FilterbankBlock:
        fch1 = fch1 if fch1 is not None else self.header.fch1
        nchans = nchans if nchans is not None else self.header.nchans
        chan_start = _chan_start(self.header, fch1, nchans)
        if start < 0 or start + nsamps > self.header.nsamples:
            msg = f"requested block is out of range: start={start}, nsamps={nsamps}"
            raise ValueError(msg)

        self._file.seek(start * self.samp_stride)
        data = self._maybe_signed(self._file.cread(self.header.nchans * nsamps))
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()

        data_block = data[chan_start : chan_start + nchans]
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header(
            {
                "tstart": start_mjd,
                "nsamples": nsamps_read,
                "fch1": fch1,
                "nchans": nchans,
            },
        )
        return FilterbankBlock(data_block, new_header)

    def read_dedisp_block(
        self,
        start: int,
        nsamps: int,
        dm: float,
        *,
        small_reads: bool | None = None,
    ) -> FilterbankBlock:
        """Read a dedispersed block straight from disk.

        Parameters
        ----------
        start : int
            First output sample. This is the reference-frequency sample.
        nsamps : int
            Number of output samples.
        dm : float
            Dispersion measure.
        small_reads : bool or None, optional
            How to read the delay window.

            ``None`` (default) picks from the DM. A window that fits in 32 MiB
            is one sequential read. A larger window is streamed in 8 MiB gulps
            when most channels are live, and sought per sample down to the
            live channels when a high DM leaves most of each sample unused.
            ``True`` forces those channel seeks. ``False`` forces the single
            sequential read. Seeks need at least 8 bits per sample. 1-, 2- and
            4-bit data are always read as whole samples.

        Returns
        -------
        FilterbankBlock
            Shape ``(nchans, nsamps)``. A bulk read holds the delay window
            only while it copies into this block. Stream and channel-seek
            reads never build that window.

        Raises
        ------
        ValueError
            If the delay window runs off the file, or ``small_reads`` is forced
            on sub-byte data.
        """
        aligned, in_start, n_in = _dedisp_window(self.header, start, nsamps, dm)
        mode = _dedisp_read_mode(
            n_in,
            nsamps,
            self.samp_stride,
            self.header.nbits,
            small_reads=small_reads,
        )
        if mode == "bulk":
            data = self._dedisp_bulk(in_start, n_in, aligned, nsamps)
        elif mode == "partial":
            data = self._dedisp_partial(in_start, n_in, aligned, nsamps)
        else:
            data = self._dedisp_stream(in_start, n_in, aligned, nsamps)
        new_header = self.header.new_header(
            {
                "tstart": self.header.mjd_after_nsamps(start),
                "nsamples": nsamps,
            },
        )
        return FilterbankBlock(data, new_header, dm=dm)

    def _dedisp_dtype(self) -> np.dtype[np.generic]:
        if self.header.nbits == 16 or (self.header.signed and self.header.nbits == 8):
            return np.dtype(np.float32)
        return self.bitsinfo.dtype

    def _dedisp_bulk(
        self,
        in_start: int,
        n_in: int,
        aligned: np.ndarray,
        nsamps: int,
    ) -> np.ndarray:
        block = self.read_block(in_start, n_in)
        return _gather_dispersed(block.data, aligned, nsamps)

    def _dedisp_stream(
        self,
        in_start: int,
        n_in: int,
        aligned: np.ndarray,
        nsamps: int,
    ) -> np.ndarray:
        nchans = self.header.nchans
        out = np.zeros((nchans, nsamps), dtype=self._dedisp_dtype())
        gulp = max(1, min(n_in, _STREAM_GULP_BYTES // max(self.samp_stride, 1)))
        self._file.seek(in_start * self.samp_stride)
        done = 0
        while done < n_in:
            take = min(gulp, n_in - done)
            raw = self._maybe_signed(self._file.cread(take * nchans))
            place_dedisp_block(out, aligned, raw.reshape(take, nchans), done)
            done += take
        return out

    def _dedisp_partial(
        self,
        in_start: int,
        n_in: int,
        aligned: np.ndarray,
        nsamps: int,
    ) -> np.ndarray:
        spans = _active_spans(aligned, nsamps, n_in)
        if spans is None:
            return self._dedisp_stream(in_start, n_in, aligned, nsamps)
        clo, chi = spans
        nchans = self.header.nchans
        out = np.zeros((nchans, nsamps), dtype=self._dedisp_dtype())
        item = self.bitsinfo.itemsize
        stride = self.samp_stride
        pos: int | None = None
        for isamp in range(n_in):
            lo_c = int(clo[isamp])
            hi_c = int(chi[isamp])
            if hi_c <= lo_c:
                continue
            target = (in_start + isamp) * stride + lo_c * item
            if pos != target:
                gap = -1 if pos is None else target - pos
                useful = (hi_c - lo_c) * item
                if 0 <= gap <= useful:
                    if gap:
                        self._file.cread(gap // item)
                else:
                    self._file.seek(target)
            chunk = self._maybe_signed(self._file.cread(hi_c - lo_c))
            if chunk.size != hi_c - lo_c:
                msg = f"short read at sample {in_start + isamp}"
                raise OSError(msg)
            place_dedisp_span(out, aligned, lo_c, hi_c, isamp, chunk)
            pos = target + (hi_c - lo_c) * item
        return out

    def read_plan(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        skipback: int = 0,
        description: str | None = None,
        *,
        quiet: bool = False,
        allocator: Callable[[int], Buffer] | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        _check_plan_args(gulp, start)
        if nsamps is None:
            nsamps = self.header.nsamples - start
        if description is None:
            description = f"{get_callerfunc(inspect.stack())} : "
        gulp = min(nsamps, gulp)
        skipback = abs(skipback)
        if skipback >= gulp:
            msg = f"readsamps ({gulp}) must be > skipback ({skipback})"
            raise ValueError(msg)

        # Here we set the allocator and allocate the readinto buffer
        allocator = bytearray if allocator is None else allocator
        read_buffer = allocate_buffer(allocator, gulp * self.samp_stride)
        # Here we allocate (if needed) a buffer for unpacking the data
        if self.bitsinfo.unpack:
            # The unpacking always unpacks up to 8-bits, so the size should
            # be nsamps * nchans
            # Is there a way to guarantee that the behaviour is correct here?
            unpack_buffer = allocate_buffer(allocator, gulp * self.header.nchans)
            data = np.frombuffer(unpack_buffer, dtype=self.bitsinfo.dtype)
        else:
            unpack_buffer = None
            data = np.frombuffer(read_buffer, dtype=self.bitsinfo.dtype)

        self._file.seek(start * self.samp_stride)
        nreads, lastread = divmod(nsamps, (gulp - skipback))
        if lastread < skipback:
            nreads -= 1
            lastread = nsamps - (nreads * (gulp - skipback))
        blocks = [
            (ii, gulp * self.header.nchans, -skipback * self.header.nchans)
            for ii in range(nreads)
        ]
        if lastread != 0:
            blocks.append((nreads, lastread * self.header.nchans, 0))

        for ii, block, skip in track(
            blocks,
            description=description,
            disable=quiet or progress_disabled(),
        ):
            logger.debug(
                f"read_plan: Reading block {ii}/{nreads}, {block} elements, "
                f"with skipback={skip}",
            )
            need = int(block * self.chan_stride)
            unpack_view = (
                None if unpack_buffer is None else memoryview(unpack_buffer)[:block]
            )
            nbytes = self._file.creadinto(memoryview(read_buffer)[:need], unpack_view)
            expected_nbytes = need
            if nbytes != expected_nbytes:
                msg = (
                    f"Unexpected number of bytes read from file {nbytes} (actual) "
                    f"!= {expected_nbytes} (expected)"
                )
                raise ValueError(msg)
            if skip != 0:
                self._file.seek(int(skip * self.chan_stride), whence=1)
            # ``data`` is a view of the gulp buffer. It is valid until the next
            # iteration, which overwrites it. Copy before storing it.
            yield block // self.header.nchans, ii, self._maybe_signed(data[:block])


class PFITSReader(Filterbank):
    """
    Reads a filterbank file from a FITS file.

    Parameters
    ----------
    filename : str
        Name of the input file.
    """

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._header = Header.from_pfits(self._filename)
        self._fitsfile = PFITSFile(self._filename)
        super().__init__()

    def close(self) -> None:
        """Close the underlying FITS file."""
        self._fitsfile.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def pri_hdr(self) -> PrimaryHdr:
        """:class:`~sigpyproc.io.pfits.PrimaryHdr`: Primary header of input file."""
        return self._fitsfile.pri_hdr

    @property
    def sub_hdr(self) -> SubintHdr:
        """:class:`~sigpyproc.io.pfits.SubintHdr`: Subint header of input file."""
        return self._fitsfile.sub_hdr

    @property
    def header(self) -> Header:
        return self._header

    @property
    def filename(self) -> str:
        """str: Name of the input file (first file in case of multiple input files)."""
        return self._filename

    @property
    def bitsinfo(self) -> BitsInfo:
        """:class:`~sigpyproc.io.bits.BitsInfo`: Bits info of input file data."""
        return self._fitsfile.bitsinfo

    def read_block(
        self,
        start: int,
        nsamps: int,
        fch1: float | None = None,
        nchans: int | None = None,
    ) -> FilterbankBlock:
        fch1 = fch1 if fch1 is not None else self.header.fch1
        nchans = nchans if nchans is not None else self.header.nchans
        chan_start = _chan_start(self.header, fch1, nchans)
        if start < 0 or start + nsamps > self.header.nsamples:
            msg = f"requested block is out of range: start={start}, nsamps={nsamps}"
            raise ValueError(msg)

        data = self._read_samples(start, nsamps)
        if data.shape[0] != nsamps:
            msg = f"short PSRFITS read: {data.shape[0]} != {nsamps}"
            raise ValueError(msg)
        data = data.reshape(nsamps, self.header.nchans).transpose()
        data_block = data[chan_start : chan_start + nchans]
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header(
            {"tstart": start_mjd, "nsamples": nsamps, "fch1": fch1, "nchans": nchans},
        )
        return FilterbankBlock(data_block, new_header)

    def read_dedisp_block(
        self,
        start: int,
        nsamps: int,
        dm: float,
        *,
        small_reads: bool | None = None,
    ) -> FilterbankBlock:
        """Read a dedispersed block from search-mode PSRFITS.

        ``small_reads`` selects the same modes as
        :meth:`FilReader.read_dedisp_block`. A PSRFITS row has no per-channel
        byte seek, so ``True`` streams whole samples instead of seeking inside
        a sample. ``None`` uses one read while the delay window is at most
        32 MiB, and streams above that.
        """
        aligned, in_start, n_in = _dedisp_window(self.header, start, nsamps, dm)
        mode = _dedisp_read_mode(
            n_in,
            nsamps,
            self.header.nchans * 4,
            32,
            small_reads=small_reads,
        )
        if mode == "partial":
            mode = "stream"
        if mode == "bulk":
            block = self.read_block(in_start, n_in)
            data = _gather_dispersed(block.data, aligned, nsamps)
        else:
            data = self._dedisp_stream(in_start, n_in, aligned, nsamps)
        new_header = self.header.new_header(
            {
                "tstart": self.header.mjd_after_nsamps(start),
                "nsamples": nsamps,
            },
        )
        return FilterbankBlock(data, new_header, dm=dm)

    def _dedisp_stream(
        self,
        in_start: int,
        n_in: int,
        aligned: np.ndarray,
        nsamps: int,
    ) -> np.ndarray:
        nchans = self.header.nchans
        out = np.zeros((nchans, nsamps), dtype=np.float32)
        stride = nchans * 4
        gulp = max(1, min(n_in, _STREAM_GULP_BYTES // max(stride, 1)))
        done = 0
        while done < n_in:
            take = min(gulp, n_in - done)
            raw = self._read_samples(in_start + done, take)
            place_dedisp_block(out, aligned, raw, done)
            done += take
        return out

    def read_plan(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        skipback: int = 0,
        description: str | None = None,
        *,
        quiet: bool = False,
        allocator: Callable[[int], Buffer] | None = None,  # noqa: ARG002
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        _check_plan_args(gulp, start)
        if nsamps is None:
            nsamps = self.header.nsamples - start
        if description is None:
            description = f"{inspect.stack()[1][3]} : "

        gulp = min(nsamps, gulp)
        skipback = abs(skipback)
        if skipback >= gulp:
            msg = f"readsamps ({gulp}) must be > skipback ({skipback})"
            raise ValueError(msg)
        nreads, lastread = divmod(nsamps, (gulp - skipback))
        if lastread < skipback:
            nreads -= 1
            lastread = nsamps - (nreads * (gulp - skipback))
        blocks = [(ii, gulp, -skipback) for ii in range(nreads)]
        if lastread != 0:
            blocks.append((nreads, lastread, 0))

        pos = start
        for ii, block, skip in track(
            blocks,
            description=description,
            disable=quiet or progress_disabled(),
        ):
            data = self._read_samples(pos, block)
            pos += block + skip
            yield block, ii, data.ravel()

    def _read_samples(self, start: int, nsamps: int) -> np.ndarray:
        """Read ``nsamps`` samples from ``start``, reading only the subints needed."""
        subint = self.sub_hdr.subint_samples
        startsub, startsamp = divmod(start, subint)
        nsubs = (startsamp + nsamps + subint - 1) // subint
        data = self._fitsfile.read_subints(startsub, nsubs)
        return data[startsamp : startsamp + nsamps]


@attrs.define(auto_attribs=True, slots=True)
class PulseExtractor:
    """Extracts a data block from a filterbank file centered on a pulse.

    The extracted block is centered on the given pulse toa at the desired
    frequency in the band. The block is padded if the pulse is too close
    to the edge of the filterbank file.

    Parameters
    ----------
    filfile : str
        Name of the filterbank file.
    pulse_toa : int
        Time of arrival of the pulse in samples at the toa_freq.
    pulse_width : int
        Width of the pulse in samples.
    pulse_dm : float
        Dispersion measure of the pulse.
    toa_freq : float, optional
        Frequency at which the pulse toa is given, by default None (highest frequency)
    nchans : int, optional
        Number of channels to extract, by default None (all channels)
    min_nsamps : int, optional
        Minimum number of samples in the extracted block, by default 256
    quiet : bool, optional
        If True, suppresses logging messages, by default False
    """

    filfile: str
    pulse_toa: int
    pulse_width: int
    pulse_dm: float
    min_nsamps: int = 256
    toa_freq: float | None = None
    nchans: int | None = None

    sub_hdr: Header = attrs.field(init=False)
    _disp_delay: int = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        hdr = Header.from_sigproc(self.filfile)
        self.toa_freq = self.toa_freq or hdr.fch1
        self.nchans = self.nchans or hdr.nchans
        self.sub_hdr = hdr.new_header({"fch1": self.toa_freq, "nchans": self.nchans})
        # Just to be safe, add 5 times the pulse width
        self._disp_delay = (
            max(np.abs(self.sub_hdr.get_dmdelays(self.pulse_dm, in_samples=True)))
            + self.pulse_width * 5
        )

    @property
    def t_decimate(self) -> int:
        """int: Decimation factor to consider."""
        return max(1, self.pulse_width // 2)

    @property
    def disp_delay(self) -> int:
        """int: Dispersion delay in samples."""
        return self._disp_delay

    @property
    def block_delay(self) -> int:
        """int: Dispersion Block size in samples."""
        return ((self.disp_delay // self.t_decimate) + 1) * self.t_decimate

    @property
    def nsamps(self) -> int:
        """int: Number of samples in the output block."""
        return max(2 * self.block_delay, self.min_nsamps * self.t_decimate)

    @property
    def nstart(self) -> int:
        """int: Start sample of the output block."""
        return self.pulse_toa - self.nsamps // 2

    @property
    def nstart_file(self) -> int:
        """int: Start sample to read in the file."""
        return max(0, self.nstart)

    @property
    def nsamps_file(self) -> int:
        """int: Number of samples to read in the file."""
        return min(
            self.nsamps + min(0, self.nstart),
            self.sub_hdr.nsamples - max(0, self.nstart),
        )

    @property
    def pulse_toa_block(self) -> int:
        """int: Time of arrival of the pulse in the output block."""
        return self.pulse_toa - self.nstart

    def get_data(self, pad_mode: LocMethods = "median") -> FilterbankBlock:
        """Extract the filterbank block centered on the pulse.

        Parameters
        ----------
        pad_mode : {'median', 'mean'}, optional
            Mode for padding the data, by default "median"

        Returns
        -------
        :class:`~sigpyproc.block.FilterbankBlock`
            Data block.
        """
        logger.info(
            f"Required samples = {2 * self.block_delay}, "
            f"Reading samples = {self.nsamps}",
        )
        logger.debug(f"nstart = {self.nstart}, nsamps = {self.nsamps}")
        logger.debug(
            f"nstart_file = {self.nstart_file}, nsamps_file = {self.nsamps_file}",
        )
        with FilReader(self.filfile) as fil:
            block = fil.read_block(
                start=self.nstart_file,
                nsamps=self.nsamps_file,
                fch1=self.toa_freq,
                nchans=self.nchans,
            )
        if self.nstart < 0 or self.nstart + self.nsamps > self.sub_hdr.nsamples:
            offset = abs(min(0, self.nstart))
            logger.info(f"Padding with {pad_mode}. start offset = {offset}")
            block = block.pad_samples(self.nsamps, offset, pad_mode=pad_mode)
        return block
