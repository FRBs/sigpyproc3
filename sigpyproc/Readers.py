import numpy as np
import inspect
from tqdm import tqdm

from sigpyproc.io import FileReader
from sigpyproc.Header import Header
from sigpyproc.Filterbank import Filterbank, FilterbankBlock


class FilReader(Filterbank):
    """Class to handle the reading of sigproc format filterbank files

    Parameters
    ----------
    filename : str
        name of filterbank file

    Returns
    -------
    :class:`~sigpyproc.Filterbank.Filterbank`
        container of filterbank data with observational metadata

    Notes
    -----
    To be considered as a Sigproc format filterbank file the header must only
    contain keywords found in the ``HeaderParams.header_keys`` dictionary.
    """

    def __init__(self, filenames, check_contiguity=True):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.filename = filenames[0]
        self.header = Header.parseSigprocHeader(
            self.filenames, check_contiguity=check_contiguity
        )
        self._file = FileReader(
            filenames,
            self.header.hdrlens,
            self.header.datalens,
            mode="r",
            nbits=self.header.nbits,
        )
        self.bitsinfo = self._file.bitsinfo
        self.sampsize = (
            self.header.nchans * self.bitsinfo.itemsize // self.bitsinfo.bitfact
        )
        super().__init__()

    def readBlock(self, start, nsamps, as_filterbankBlock=True):
        """Read a block of filterbank data.

        Parameters
        ----------
        start : int
            first time sample of the block to be read
        nsamps : int
            number of samples in the block (i.e. block will be nsamps*nchans in size)
        as_filterbankBlock : bool, optional
            whether to read data as filterbankBlock or numpy array, by default True

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock` or :py:obj:`numpy.ndarray`
            2-D array of filterbank data
        """
        self._file.seek(start * self.sampsize)
        data = self._file.cread(self.header.nchans * nsamps)
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()
        start_mjd = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({"tstart": start_mjd})
        if as_filterbankBlock:
            return FilterbankBlock(data, new_header)
        return data

    def readDedispersedBlock(
        self, start, nsamps, dm, as_filterbankBlock=True, small_reads=True
    ):
        """Read a block of dedispersed filterbank data, best used in cases where
        I/O time dominates reading a block of data.

        Parameters
        ----------
        start : int
            first time sample of the block to be read
        nsamps : int
            number of samples in the block (i.e. block will be nsamps*nchans in size)
        dm : float
            dispersion measure to dedisperse at
        as_filterbankBlock : bool, optional
            whether to read data as filterbankBlock or numpy array, by default True
        small_reads : bool, optional
            if the datum size is greater than 1 byte, only read the data needed
            instead of every frequency of every sample, by default True

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock` or :py:obj:`numpy.ndarray`
            2-D array of filterbank data
        """
        data = np.zeros((self.header.nchans, nsamps), dtype=self._file.dtype)
        min_sample = start + self.header.getDMdelays(dm)
        max_sample = min_sample + nsamps
        curr_sample = np.zeros(self.header.nchans, dtype=int)

        start_mjd = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({"tstart": start_mjd})

        lowest_chan, highest_chan, sample_offset = (0, 0, start)
        with tqdm(total=nsamps * self.header.nchans) as progress:
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
                    self._file.seek(self.header.hdrlen + next_offset)

                    data[sampled_chans, curr_sample[sampled_chans]] = self._file.cread(
                        read_length
                    )

                else:
                    next_offset = sample_offset * self.sampsize
                    self._file.seek(self.header.hdrlen + next_offset)

                    sample = self._file.cread(self.sampsize)
                    data[sampled_chans, curr_sample[sampled_chans]] = sample[
                        sampled_chans
                    ]

                curr_sample[sampled_chans] += 1

                if curr_sample[highest_chan] > nsamps:
                    sample_offset = min_sample[highest_chan + 1]
                else:
                    sample_offset += 1

                progress.update(read_length)

        if as_filterbankBlock:
            data = FilterbankBlock(data, new_header)
            data.dm = dm
            return data
        return data

    def readPlan(
        self, gulp, skipback=0, start=0, nsamps=None, tqdm_desc=None, verbose=True
    ):
        """A generator used to perform filterbank reading.

        Parameters
        ----------
        gulp : int
            number of samples in each read
        skipback : int, optional
            number of samples to skip back after each read, by default 0
        start : int, optional
            first sample to read from filterbank, by default 0 (start of the file)
        nsamps : int, optional
            total number samples to read, by default None (end of the file)
        tqdm_desc : str, optional
            [description], by default None
        verbose : bool, optional
            flag for display of reading plan information, by default True

        Yields
        -------
        int, int, :py:obj:`numpy.ndarray`
            An generator that can read through the file.

        Raises
        ------
        ValueError
            If read samples < ``skipback``.

        Notes
        -----
        For each read, the generator yields a tuple ``x``, where:

            * ``x[0]`` is the number of samples read
            * ``x[1]`` is the index of the read (i.e. ``x[1]=0`` is the first read)
            * ``x[2]`` is a 1-D numpy array containing the data that was read

        Examples
        --------
        The normal calling syntax for this is function is:

        >>> for nsamps, ii, data in self.readPlan(*args,**kwargs):
                # do something
        where data always has contains ``nchans*nsamps`` points.
        """
        if nsamps is None:
            nsamps = self.header.nsamples - start
        gulp = min(nsamps, gulp)
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError("readsamps must be > skipback value")
        self._file.seek(start * self.sampsize)
        nreads = nsamps // (gulp - skipback)
        lastread = nsamps - (nreads * (gulp - skipback))
        if lastread < skipback:
            nreads -= 1
            lastread = nsamps - (nreads * (gulp - skipback))
        blocks = [
            (ii, gulp * self.header.nchans, -skipback * self.header.nchans)
            for ii in range(nreads)
        ]
        if lastread != 0:
            blocks.append((nreads, lastread * self.header.nchans, 0))

        if verbose:
            print("\nFilterbank reading plan:")
            print("------------------------")
            print(f"Called on file:       {self.filename}")
            print(f"Called by:            {inspect.stack()[1][3]}")
            print(f"Number of samps:      {nsamps}")
            print(f"Number of reads:      {nreads}")
            print(f"Nsamps per read:      {blocks[0][1]//self.header.nchans}")
            print(f"Nsamps of final read: {blocks[-1][1]//self.header.nchans}")
            print(f"Nsamps to skip back:  {-1*blocks[0][2]//self.header.nchans}\n")

        if tqdm_desc is None:
            tqdm_desc = f"{inspect.stack()[1][3]} : "
        for ii, block, skip in tqdm(blocks, desc=tqdm_desc):
            data = self._file.cread(block)
            self._file.seek(
                skip * self.bitsinfo.itemsize // self.bitsinfo.bitfact, whence=1,
            )
            yield int(block // self.header.nchans), int(ii), data
