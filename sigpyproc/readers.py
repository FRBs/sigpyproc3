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
        verbose: bool = False,
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
        for ii, block, skip in track(blocks, description=description):
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
        verbose: bool = False,
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

        for ii, block, skip in track(blocks, description=description):
            startsub, startsamp = divmod(start, self.sub_hdr.subint_samples)
            nsubs = (
                nsamps + self.sub_hdr.subint_samples - 1
            ) // self.sub_hdr.subint_samples

            data = self._fitsfile.read_subints(startsub, nsubs)
            data = data[startsamp : startsamp + nsamps]
            start += block + skip
            yield block, ii, data.ravel()
