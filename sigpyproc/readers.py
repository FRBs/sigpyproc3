from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import track
from typing_extensions import Buffer

from sigpyproc.base import Filterbank
from sigpyproc.block import FilterbankBlock
from sigpyproc.header import Header
from sigpyproc.io.fileio import FileReader, allocate_buffer
from sigpyproc.io.pfits import PFITSFile, PrimaryHdr, SubintHdr
from sigpyproc.utils import get_callerfunc, get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from sigpyproc.io.bits import BitsInfo

logger = get_logger(__name__)


class FilReader(Filterbank):
    """A class to handle the reading of sigproc format filterbank files.

    Parameters
    ----------
    filenames : str or list of str
        filterbank file or list of filterbank files
    check_contiguity : bool, optional
        whether to check if files are contiguous, by default True

    Returns
    -------
    :class:`~sigpyproc.base.Filterbank`
        Base container of filterbank data with observational metadata

    Notes
    -----
    To be considered as a Sigproc format filterbank file the header must only
    contain keywords found in the :data:`~sigpyproc.io.sigproc.header_keys` dictionary.
    """

    def __init__(
        self,
        filenames: str | list[str],
        *,
        check_contiguity: bool = True,
    ) -> None:
        if isinstance(filenames, str):
            filenames = [filenames]
        self._filenames = filenames
        self._header = Header.from_sigproc(
            self._filenames,
            check_contiguity=check_contiguity,
        )
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
    def filename(self) -> str:
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

    def read_block(self, start: int, nsamps: int) -> FilterbankBlock:
        if start < 0 or start + nsamps > self.header.nsamples:
            msg = f"requested block is out of range: start={start}, nsamps={nsamps}"
            raise ValueError(msg)
        self._file.seek(start * self.samp_stride)
        data = self._file.cread(self.header.nchans * nsamps)
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header(
            {"tstart": start_mjd, "nsamples": nsamps_read},
        )
        return FilterbankBlock(data, new_header)

    def read_dedisp_block(self, start: int, nsamps: int, dm: float) -> FilterbankBlock:
        delays = self.header.get_dmdelays(dm)
        min_sample = start + delays
        max_sample = min_sample + nsamps
        if np.any(min_sample < 0) or np.any(max_sample > self.header.nsamples):
            msg = (
                f"requested dedispersed block is out of range: "
                f"start={start}, nsamps={nsamps}"
            )
            raise ValueError(msg)

        self._file.seek(start * self.samp_stride)
        samples_read = np.zeros(self.header.nchans, dtype=int)
        data = np.zeros((self.header.nchans, nsamps), dtype=self._file.bitsinfo.dtype)

        for isamp in track(range(nsamps), description="Reading dedispersed data ..."):
            samples_offset = start + isamp
            relevant_chans = np.argwhere(
                np.logical_and(
                    max_sample > samples_offset,
                    min_sample <= samples_offset,
                ),
            ).flatten()
            chans_slice = np.arange(
                relevant_chans.min(),
                relevant_chans.max() + 1,
                dtype=int,
            )
            # Read channel data for for each sample
            sample_data = self._file.cread(self.header.nchans)
            data[chans_slice, samples_read[chans_slice]] = sample_data[chans_slice]

            # Update sample counts
            samples_read[chans_slice] += 1

        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header({"tstart": start_mjd, "nsamples": nsamps})
        return FilterbankBlock(data, new_header, dm=dm)

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

        for ii, block, skip in track(blocks, description=description, disable=quiet):
            logger.debug(
                f"read_plan: Reading block {ii}/{nreads}, {block} elements, "
                f"with skipback={skip}",
            )
            nbytes = self._file.creadinto(read_buffer, unpack_buffer)
            expected_nbytes = int(block * self.chan_stride)
            if nbytes != expected_nbytes:
                msg = (
                    f"Unexpected number of bytes read from file {nbytes} (actual) "
                    f"!= {expected_nbytes} (expected)"
                )
                raise ValueError(msg)
            if skip != 0:
                self._file.seek(int(skip * self.chan_stride), whence=1)
            yield block // self.header.nchans, ii, data[:block]


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

    def read_block(self, start: int, nsamps: int) -> FilterbankBlock:
        if start < 0 or start + nsamps > self.header.nsamples:
            msg = f"requested block is out of range: start={start}, nsamps={nsamps}"
            raise ValueError(msg)

        startsub, startsamp = divmod(start, self.sub_hdr.subint_samples)
        nsubs = (
            nsamps + self.sub_hdr.subint_samples - 1
        ) // self.sub_hdr.subint_samples

        data = self._fitsfile.read_subints(startsub, nsubs)

        data = data[startsamp : startsamp + nsamps]
        data = data.reshape(nsamps, self.header.nchans).transpose()
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header({"tstart": start_mjd, "nsamples": nsamps})
        return FilterbankBlock(data, new_header)

    def read_dedisp_block(self, start: int, nsamps: int, dm: float) -> FilterbankBlock:  # noqa: ARG002
        msg = "Not implemented for PFITSReader"
        raise NotImplementedError(msg)

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

        for ii, block, skip in track(blocks, description=description, disable=quiet):
            startsub, startsamp = divmod(start, self.sub_hdr.subint_samples)
            nsubs = (
                nsamps + self.sub_hdr.subint_samples - 1
            ) // self.sub_hdr.subint_samples

            data = self._fitsfile.read_subints(startsub, nsubs)
            data = data[startsamp : startsamp + nsamps]
            start += block + skip
            yield block, ii, data.ravel()


class PulseExtractor:
    """Extracts a data block from a filterbank file centered on a pulse.

    The extracted block is centered on the given pulse toa at the highest
    frequency in the band. The block is padded if the pulse is too close
    to the edge of the filterbank file.

    Parameters
    ----------
    filfile : str
        Name of the filterbank file.
    pulse_toa : int
        Time of arrival of the pulse in samples at the highest frequency.
    pulse_width : int
        Width of the pulse in samples.
    pulse_dm : float
        Dispersion measure of the pulse.
    min_nsamps : int, optional
        Minimum number of samples in the extracted block, by default 256
    quiet : bool, optional
        If True, suppresses logging messages, by default False
    """

    def __init__(
        self,
        filfile: str,
        pulse_toa: int,
        pulse_width: int,
        pulse_dm: float,
        *,
        min_nsamps: int = 256,
    ) -> None:
        self.fil = FilReader(filfile)
        self.header = self.fil.header
        self.pulse_toa = pulse_toa
        self.pulse_width = pulse_width
        self.pulse_dm = pulse_dm
        self.min_nsamps = min_nsamps

        # Just to be safe, add 5 times the pulse width
        self._disp_delay = (
            max(np.abs(self.header.get_dmdelays(pulse_dm, in_samples=True)))
            + self.pulse_width * 5
        )

    @property
    def decimation_factor(self) -> int:
        """int: Decimation factor to consider."""
        return max(1, self.pulse_width // 2)

    @property
    def disp_delay(self) -> int:
        """int: Dispersion delay in samples."""
        return self._disp_delay

    @property
    def block_delay(self) -> int:
        """int: Dispersion Block size in samples."""
        return (
            (self.disp_delay // self.decimation_factor) + 1
        ) * self.decimation_factor

    @property
    def nsamps(self) -> int:
        """int: Number of samples in the output block."""
        return max(2 * self.block_delay, self.min_nsamps * self.decimation_factor)

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
            self.header.nsamples - max(0, self.nstart),
        )

    @property
    def pulse_toa_block(self) -> int:
        """int: Time of arrival of the pulse in the output block."""
        return self.pulse_toa - self.nstart

    def get_data(self, pad_mode: str = "median") -> FilterbankBlock:
        """Extract the filterbank block centered on the pulse.

        Parameters
        ----------
        pad_mode : str, optional
            Mode for padding the data, by default "median"

        Returns
        -------
        FilterbankBlock
            Data block.
        """
        logger.info(
            f"PulseExtractor: Required samples = {2 * self.block_delay}, "
            f"Reading samples = {self.nsamps}",
        )
        logger.debug(f"PulseExtractor: nstart = {self.nstart}, nsamps = {self.nsamps}")
        logger.debug(
            f"PulseExtractor: nstart_file = {self.nstart_file}, "
            f"nsamps_file = {self.nsamps_file}",
        )
        data = self.fil.read_block(start=self.nstart_file, nsamps=self.nsamps_file)

        if self.nstart < 0 or self.nstart + self.nsamps > self.header.nsamples:
            data = self._pad_data(data, pad_mode=pad_mode)
        return FilterbankBlock(data, self.header.new_header())

    def _pad_data(
        self,
        data: FilterbankBlock,
        pad_mode: str = "median",
    ) -> FilterbankBlock:
        """Pad the data block with the given mode.

        Parameters
        ----------
        data : FilterbankBlock
            Data block to be padded.
        pad_mode : str, optional
            Mode for padding the data, by default "median"

        Returns
        -------
        FilterbankBlock
            Padded data block.

        Raises
        ------
        ValueError
            If the pad_mode is not "mean" or "median".
        """
        if pad_mode == "mean":
            pad_arr = np.mean(data, axis=1)
        elif pad_mode == "median":
            pad_arr = np.median(data, axis=1)
        else:
            msg = f"pad_mode {pad_mode} not supported."
            raise ValueError(msg)

        data_pad = np.ones((self.header.nchans, self.nsamps), dtype=pad_arr.dtype)
        data_pad *= pad_arr[:, None]

        offset = min(0, self.nstart)
        logger.info(f"PulseExtractor: Padding with {pad_mode}. start offset = {offset}")
        data_pad[:, -offset : -offset + self.nsamps_file] = data
        return FilterbankBlock(data_pad, self.header.new_header())
