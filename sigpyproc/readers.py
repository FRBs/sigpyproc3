from __future__ import annotations
import numpy as np
import inspect

from collections.abc import Iterator
from rich.progress import track

from sigpyproc.io import pfits
from sigpyproc.io.fileio import FileReader
from sigpyproc.io.bits import BitsInfo
from sigpyproc.header import Header
from sigpyproc.base import Filterbank
from sigpyproc.block import FilterbankBlock
from sigpyproc.utils import get_logger


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

    def __init__(self, filenames: str | list[str], check_contiguity: bool = True) -> None:
        if isinstance(filenames, str):
            filenames = [filenames]
        self._filenames = filenames
        self._header = Header.from_sigproc(
            self._filenames, check_contiguity=check_contiguity
        )
        self._file = FileReader(
            self._filenames,
            self.header.hdrlens,
            self.header.datalens,
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
    def sampsize(self) -> int:
        """int: Sample byte stride in input data."""
        return self.header.nchans * self.bitsinfo.itemsize // self.bitsinfo.bitfact

    def read_block(self, start: int, nsamps: int) -> FilterbankBlock:
        if start < 0 or start + nsamps > self.header.nsamples:
            raise ValueError("requested block is out of range")
        self._file.seek(start * self.sampsize)
        data = self._file.cread(self.header.nchans * nsamps)
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header(
            {"tstart": start_mjd, "nsamples": nsamps_read}
        )
        return FilterbankBlock(data, new_header)

    def read_dedisp_block(self, start: int, nsamps: int, dm: float) -> FilterbankBlock:
        delays = self.header.get_dmdelays(dm)
        min_sample = start + delays
        max_sample = min_sample + nsamps
        if np.any(min_sample < 0) or np.any(max_sample > self.header.nsamples):
            raise ValueError("requested dedispersed block is out of range")

        self._file.seek(start * self.sampsize)
        samples_read = np.zeros(self.header.nchans, dtype=int)
        data = np.zeros((self.header.nchans, nsamps), dtype=self._file.bitsinfo.dtype)

        for isamp in track(range(nsamps), description="Reading dedispersed data ..."):
            samples_offset = start + isamp
            relevant_chans = np.argwhere(
                np.logical_and(max_sample > samples_offset, min_sample <= samples_offset)
            ).flatten()
            chans_slice = slice(relevant_chans.min(), relevant_chans.max() + 1)

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
        quiet: bool = False,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        if nsamps is None:
            nsamps = self.header.nsamples - start
        if description is None:
            description = f"{inspect.stack()[1][3]} : "

        gulp = min(nsamps, gulp)
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError(f"readsamps ({gulp}) must be > skipback ({skipback})")
        self._file.seek(start * self.sampsize)
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

        # / self.logger.debug(f"Reading plan: nsamps = {nsamps}, nreads = {nreads}")
        # / self.logger.debug(f"Reading plan: gulp = {gulp}, lastread = {lastread}, skipback = {skipback}")
        for ii, block, skip in track(blocks, description=description, disable=quiet):
            data = self._file.cread(block)
            if skip != 0:
                self._file.seek(
                    skip * self.bitsinfo.itemsize // self.bitsinfo.bitfact, whence=1
                )
            yield block // self.header.nchans, ii, data


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
        self._fitsfile = pfits.PFITSFile(self._filename)
        super().__init__()

    @property
    def pri_hdr(self) -> pfits.PrimaryHdr:
        """:class:`~sigpyproc.io.pfits.PrimaryHdr`: Primary header of input file."""
        return self._fitsfile.pri_hdr

    @property
    def sub_hdr(self) -> pfits.SubintHdr:
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

    @property
    def sampsize(self) -> int:
        """int: Sample byte stride in input data."""
        return self.header.nchans * self.bitsinfo.itemsize // self.bitsinfo.bitfact

    def read_block(self, start: int, nsamps: int) -> FilterbankBlock:
        if start < 0 or start + nsamps > self.header.nsamples:
            raise ValueError("requested block is out of range")

        startsub, startsamp = divmod(start, self.sub_hdr.subint_samples)
        nsubs = (nsamps + self.sub_hdr.subint_samples - 1) // self.sub_hdr.subint_samples

        data = self._fitsfile.read_subints(startsub, nsubs)

        data = data[startsamp : startsamp + nsamps]
        data = data.reshape(nsamps, self.header.nchans).transpose()
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header({"tstart": start_mjd, "nsamples": nsamps})
        return FilterbankBlock(data, new_header)

    def read_dedisp_block(self, start: int, nsamps: int, dm: float) -> FilterbankBlock:
        raise NotImplementedError("Not implemented for PFITSReader")

    def read_plan(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        skipback: int = 0,
        description: str | None = None,
        quiet: bool = False,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        if nsamps is None:
            nsamps = self.header.nsamples - start
        if description is None:
            description = f"{inspect.stack()[1][3]} : "

        gulp = min(nsamps, gulp)
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError(f"readsamps ({gulp}) must be > skipback ({skipback})")
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


class PulseExtractor(object):
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
        min_nsamps: int = 256,
        quiet: bool = False,
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
        self._configure_logger(quiet=quiet)

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
        return ((self.disp_delay // self.decimation_factor) + 1) * self.decimation_factor

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
        """Extracts the data block from the filterbank file.

        Parameters
        ----------
        pad_mode : str, optional
            Mode for padding the data, by default "median"

        Returns
        -------
        FilterbankBlock
            Data block.
        """
        self.logger.info(
            f"Required samples = {2 * self.block_delay}, Reading samples = {self.nsamps}"
        )
        self.logger.debug(f"nstart = {self.nstart}, nsamps = {self.nsamps}")
        self.logger.debug(
            f"nstart_file = {self.nstart_file}, nsamps_file = {self.nsamps_file}"
        )
        data = self.fil.read_block(start=self.nstart_file, nsamps=self.nsamps_file)

        if self.nstart < 0 or self.nstart + self.nsamps > self.header.nsamples:
            data = self._pad_data(data, pad_mode=pad_mode)
        return FilterbankBlock(data, self.header.new_header())

    def _pad_data(
        self, data: FilterbankBlock, pad_mode: str = "median"
    ) -> FilterbankBlock:
        """Pads the data block with the given mode.

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
            raise ValueError(f"pad_mode {pad_mode} not supported.")

        data_pad = np.ones((self.header.nchans, self.nsamps), dtype=pad_arr.dtype)
        data_pad *= pad_arr[:, None]

        offset = min(0, self.nstart)
        self.logger.info(f"Padding with {pad_mode}. start offset = {offset}")
        data_pad[:, -offset : -offset + self.nsamps_file] = data
        return data_pad

    def _configure_logger(self, **kwargs) -> None:
        logger_name = "PulseExtractor"
        self.logger = get_logger(logger_name, **kwargs)
