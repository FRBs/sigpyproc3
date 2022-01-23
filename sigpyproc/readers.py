from __future__ import annotations
import numpy as np
import inspect

from collections.abc import Iterator
from rich.progress import track, Progress

from sigpyproc.io.fileio import FileReader
from sigpyproc.io.bits import BitsInfo
from sigpyproc.header import Header
from sigpyproc.base import Filterbank
from sigpyproc.block import FilterbankBlock


class FilReader(Filterbank):
    """Class to handle the reading of sigproc format filterbank files.

    Returns
    -------
    :class:`~sigpyproc.base.Filterbank`
        container of filterbank data with observational metadata

    Notes
    -----
    To be considered as a Sigproc format filterbank file the header must only
    contain keywords found in the ``HeaderParams.header_keys`` dictionary.
    """

    def __init__(self, filenames: str | list[str], check_contiguity: bool = True) -> None:
        """Initialize Filterbank reading.

        Parameters
        ----------
        filenames : str or list of str
           filterbank file or list of filterbank files
        check_contiguity : bool, optional
            whether to check if files are contiguous, by default True
        """
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
    def filename(self) -> str:
        """Name of the input file (`str`, read-only)."""
        return self._filenames[0]

    @property
    def header(self) -> Header:
        return self._header

    @property
    def bitsinfo(self) -> BitsInfo:
        """Bits info of input file data (:class:`~sigpyproc.io.bits.BitsInfo`, read-only)."""
        return self._file.bitsinfo

    @property
    def sampsize(self) -> int:
        """Sample size in input data (`int`, read-only)."""
        return self.header.nchans * self.bitsinfo.itemsize // self.bitsinfo.bitfact

    def read_block(self, start: int, nsamps: int) -> FilterbankBlock:
        self._file.seek(start * self.sampsize)
        data = self._file.cread(self.header.nchans * nsamps)
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()
        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header({"tstart": start_mjd})
        return FilterbankBlock(data, new_header)

    def read_dedispersed_block(
        self,
        start: int,
        nsamps: int,
        dm: float,
        small_reads: bool = True,
    ) -> FilterbankBlock:
        """Read a block of dedispersed filterbank data.

        Best used in cases where I/O time dominates reading a block of data.

        Parameters
        ----------
        start : int
            first time sample of the block to be read
        nsamps : int
            number of samples in the block (i.e. block will be nsamps*nchans in size)
        dm : float
            dispersion measure to dedisperse at
        small_reads : bool, optional
            if the datum size is greater than 1 byte, only read the data needed
            instead of every frequency of every sample, by default True

        Returns
        -------
        :class:`~sigpyproc.block.FilterbankBlock`
            2-D array of filterbank data
        """
        data = np.zeros((self.header.nchans, nsamps), dtype=self._file.bitsinfo.dtype)
        min_sample = start + self.header.get_dmdelays(dm)
        max_sample = min_sample + nsamps
        curr_sample = np.zeros(self.header.nchans, dtype=int)

        start_mjd = self.header.mjd_after_nsamps(start)
        new_header = self.header.new_header({"tstart": start_mjd})

        lowest_chan, highest_chan, sample_offset = (0, 0, start)
        with Progress() as progress:
            task = progress.add_task("Reading...", total=nsamps * self.header.nchans)
            while curr_sample[-1] < nsamps:
                relevant_channels = np.argwhere(
                    np.logical_and(
                        max_sample > sample_offset, min_sample <= sample_offset
                    )
                ).flatten()
                lowest_chan = np.min(relevant_channels)
                highest_chan = np.max(relevant_channels)
                sampled_chans = np.arange(lowest_chan, highest_chan + 1, dtype=int)
                read_length = sampled_chans.size

                if self.bitsinfo.bitfact == 1 and small_reads:
                    next_offset = (
                        sample_offset * self.sampsize
                        + lowest_chan * self.bitsinfo.itemsize
                    )
                    # TODO fix for multifile
                    self._file.seek(self.header.hdrlens[0] + next_offset)

                    data[sampled_chans, curr_sample[sampled_chans]] = self._file.cread(
                        read_length
                    )

                else:
                    next_offset = sample_offset * self.sampsize
                    # TODO fix for multifile
                    self._file.seek(self.header.hdrlens[0] + next_offset)

                    sample = self._file.cread(self.sampsize)
                    data[sampled_chans, curr_sample[sampled_chans]] = sample[
                        sampled_chans
                    ]

                curr_sample[sampled_chans] += 1

                if curr_sample[highest_chan] > nsamps:
                    sample_offset = min_sample[highest_chan + 1]
                else:
                    sample_offset += 1

                progress.update(task, advance=read_length)

        data = FilterbankBlock(data, new_header)
        data.dm = dm
        return data

    def read_plan(
        self,
        gulp: int = 16385,
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
            self._file.seek(
                skip * self.bitsinfo.itemsize // self.bitsinfo.bitfact,
                whence=1,
            )
            yield int(block // self.header.nchans), int(ii), data
