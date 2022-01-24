from __future__ import annotations
import warnings
import numpy as np

from numpy import typing as npt
from abc import ABC, abstractmethod
from collections.abc import Iterator

from sigpyproc.foldedcube import FoldedData
from sigpyproc.timeseries import TimeSeries
from sigpyproc.header import Header
from sigpyproc.block import FilterbankBlock
from sigpyproc.core import stats, kernels


class Filterbank(ABC):
    """Base class exporting methods for the manipulation of frequency-major order pulsar data.

    Notes
    -----
    The Filterbank class should never be instantiated directly. Instead it
    should be inherited by data reading classes.
    """

    def __init__(self) -> None:
        self._chan_stats: stats.ChannelStats | None = None

    @property
    @abstractmethod
    def header(self) -> Header:
        """:class:`~sigpyproc.header.Header`: Header metadata of input file."""
        pass

    @abstractmethod
    def read_block(self, start: int, nsamps: int) -> FilterbankBlock:
        """Read a block of filterbank data.

        Parameters
        ----------
        start : int
            first time sample of the block to be read
        nsamps : int
            number of samples in the block (i.e. block will be nsamps*nchans in size)

        Returns
        -------
        :class:`~sigpyproc.block.FilterbankBlock`
            2-D array of filterbank data
        """
        pass

    @abstractmethod
    def read_plan(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        skipback: int = 0,
        description: str | None = None,
        verbose: bool = False,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """A generator used to perform filterbank reading.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 16384
        start : int, optional
            first sample to read from filterbank, by default 0 (start of file)
        nsamps : int, optional
            total number of samples to read, by default None (end of the file)
        skipback : int, optional
            number of samples to skip back after each read, by default 0
        description : str, optional
            description of task show next to progress bar (rich), by default Calling Stack
        verbose : bool, optional
            whether to display progress bar, by default False

        Yields
        -------
        :py:obj:`~collections.abc.Iterator` (tuple(int, int, :py:obj:`~numpy.ndarray`))
            An iterable of the values in the sequence.

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

        >>> for nsamps, ii, data in self.readPlan(**kwargs):
                # do something
        where data always has contains ``nchans*nsamps`` points.
        """
        pass

    @property
    def chan_stats(self) -> stats.ChannelStats | None:
        """:class:`~sigpyproc.core.stats.ChannelStats`: Channel statistics of input data."""
        return self._chan_stats

    def compute_stats(self, **plan_kwargs) -> None:
        """Compute channelwise statistics of data (upto kurtosis).

        Parameters
        ----------
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.
        """
        bag = stats.ChannelStats(self.header.nchans, self.header.nsamples)
        for nsamps, ii, data in self.read_plan(**plan_kwargs):
            bag.push_data(data, nsamps, ii, mode="full")
        self._chan_stats = bag

    def compute_stats_basic(self, **plan_kwargs) -> None:
        """Compute channelwise statistics of data (only mean and rms).

        Parameters
        ----------
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.
        """
        bag = stats.ChannelStats(self.header.nchans, self.header.nsamples)
        for nsamps, ii, data in self.read_plan(**plan_kwargs):
            bag.push_data(data, nsamps, ii, mode="basic")
        self._chan_stats = bag

    def collapse(
        self, gulp: int = 16384, start: int = 0, nsamps: int | None = None, **plan_kwargs
    ) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 16384
        start : int, optional
            start sample, by default 0
        nsamps : int, optional
            number of samples to read, by default all
        **plan_kwargs : dict
            Additional Keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            A zero-DM time series
        """
        tim_len = (self.header.nsamples - start) if nsamps is None else nsamps
        tim_ar = np.zeros(tim_len, dtype=np.float32)
        for nsamp, ii, data in self.read_plan(
            gulp=gulp, start=start, nsamps=nsamps, **plan_kwargs
        ):
            kernels.extract_tim(data, tim_ar, self.header.nchans, nsamp, ii * gulp)
        return TimeSeries(tim_ar, self.header.new_header({"nchans": 1, "dm": 0}))

    def bandpass(self, **plan_kwargs) -> TimeSeries:
        """Average across each time sample for all frequencies.

        Parameters
        ----------
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            the bandpass of the data
        """
        bpass_ar = np.zeros(self.header.nchans, dtype=np.float32)
        num_samples = 0
        for nsamps, _ii, data in self.read_plan(**plan_kwargs):
            kernels.extract_bpass(data, bpass_ar, self.header.nchans, nsamps)
            num_samples += nsamps
        bpass_ar /= num_samples
        return TimeSeries(bpass_ar, self.header.new_header({"nchans": 1}))

    def dedisperse(self, dm: float, gulp: int = 16384, **plan_kwargs) -> TimeSeries:
        """Dedisperse the data and collapse to a time series.

        Parameters
        ----------
        dm : float
            dispersion measure to dedisperse to
        gulp : int, optional
            number of samples in each read, by default 16384
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            a dedispersed time series

        Notes
        -----
        If gulp < maximum dispersion delay, gulp is taken to be twice the
        maximum dispersion delay.
        """
        chan_delays = self.header.get_dmdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        tim_len = self.header.nsamples - max_delay
        tim_ar = np.zeros(tim_len, dtype=np.float32)
        for nsamps, ii, data in self.read_plan(
            gulp=gulp, skipback=max_delay, **plan_kwargs
        ):
            kernels.dedisperse(
                data,
                tim_ar,
                chan_delays,
                max_delay,
                self.header.nchans,
                nsamps,
                ii * (gulp - max_delay),
            )
        return TimeSeries(tim_ar, self.header.new_header({"nchans": 1, "dm": dm}))

    def read_chan(self, chan: int, gulp: int = 16384, **plan_kwargs) -> TimeSeries:
        """Read a single frequency channel from the data as a time series.

        Parameters
        ----------
        chan : int
            channel to retrieve (0 is the highest frequency channel)
        gulp : int, optional
            number of samples in each read, by default 16384
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            selected channel as a time series

        Raises
        ------
        ValueError
            If chan is out of range (chan < 0 or chan > total channels).
        """
        if chan >= self.header.nchans or chan < 0:
            raise ValueError("Selected channel out of range.")
        tim_ar = np.empty(self.header.nsamples, dtype=np.float32)
        for nsamps, ii, data in self.read_plan(gulp=gulp, **plan_kwargs):
            data = data.reshape(nsamps, self.header.nchans)
            tim_ar[ii * gulp : (ii + 1) * gulp] = data[:, chan]
        return TimeSeries(tim_ar, self.header.new_header({"dm": 0, "nchans": 1}))

    def invert_freq(self, filename: str = None, **plan_kwargs) -> str:
        """Invert the frequency ordering of the data and write to a new file.

        Parameters
        ----------
        filename : str, optional
            name of output file, by default ``basename_inverted.fil``
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            filename = f"{self.header.basename}_inverted.fil"

        changes = {
            "fch1": self.header.fch1 + (self.header.nchans - 1) * self.header.foff,
            "foff": self.header.foff * -1,
        }

        out_file = self.header.prep_outfile(filename, changes, nbits=self.header.nbits)
        for nsamp, _ii, data in self.read_plan(**plan_kwargs):
            out_ar = kernels.invert_freq(data, self.header.nchans, nsamp)
            out_file.cwrite(out_ar)
        out_file.close()
        return out_file.name

    def apply_channel_mask(
        self, chanmask: npt.ArrayLike, filename: str | None = None, **plan_kwargs
    ) -> str:
        """Set the data in the given channels to zero.

        Parameters
        ----------
        chanmask : :py:obj:`~numpy.typing.ArrayLike`
            binary channel mask (0 for bad channel, 1 for good)
        filename : str, optional
            name of the output filterbank file, by default ``basename_masked.fil``
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            filename = f"{self.header.basename}_masked.fil"

        mask = np.array(chanmask).astype("ubyte")
        out_file = self.header.prep_outfile(filename)
        for nsamps, _ii, data in self.read_plan(**plan_kwargs):
            kernels.mask_channels(data, mask, self.header.nchans, nsamps)
            out_file.cwrite(data)
        return out_file.name

    def downsample(
        self,
        tfactor: int = 1,
        ffactor: int = 1,
        gulp: int = 16384,
        filename: str | None = None,
        **plan_kwargs,
    ) -> str:
        """Downsample data in time and/or frequency and write to file.

        Parameters
        ----------
        tfactor : int, optional
            factor by which to downsample in time, by default 1
        ffactor : int, optional
            factor by which to downsample in frequency, by default 1
        gulp : int, optional
            number of samples in each read, by default 16384
        filename : str, optional
            name of file to write to, by default ``basename_tfactor_ffactor.fil``
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            output file name

        Raises
        ------
        ValueError
            If number of channels is not divisible by `ffactor`.
        """
        if filename is None:
            filename = f"{self.header.basename}_f{ffactor:d}_t{tfactor:d}.fil"
        if self.header.nchans % ffactor != 0:
            raise ValueError("Bad frequency factor given")

        # Gulp must be a multiple of tfactor
        gulp = int(np.ceil(gulp / tfactor) * tfactor)

        changes = {
            "tsamp": self.header.tsamp * tfactor,
            "nchans": self.header.nchans // ffactor,
            "foff": self.header.foff * ffactor,
        }
        out_file = self.header.prep_outfile(filename, changes)

        for nsamps, _ii, data in self.read_plan(gulp=gulp, **plan_kwargs):
            write_ar = kernels.downsample_2d(
                data, tfactor, ffactor, self.header.nchans, nsamps
            )
            out_file.cwrite(write_ar)
        return out_file.name

    def extract_samps(
        self, start: int, nsamps: int, filename: str | None = None, **plan_kwargs
    ) -> str:
        """Extract a subset of time samples from the data and write to file.

        Parameters
        ----------
        start : int
            start sample to extract
        nsamps : int
            number of samples to extract
        filename : str, optional
            name of output file, by default ``basename_samps_start_start+nsamps.fil``
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            name of new file

        Raises
        ------
        ValueError
            If `start` or `nsamps` are out of bounds.
        """
        if start < 0 or start + nsamps > self.header.nsamples:
            raise ValueError("Selected samples out of range")
        if filename is None:
            filename = f"{self.header.basename}_samps_{start:d}_{start+nsamps:d}.fil"
        out_file = self.header.prep_outfile(
            filename,
            updates={"tstart": self.header.mjd_after_nsamps(start)},
            nbits=self.header.nbits,
        )
        for _count, _ii, data in self.read_plan(
            start=start, nsamps=nsamps, **plan_kwargs
        ):
            out_file.cwrite(data)
        out_file.close()
        return out_file.name

    def extract_chans(
        self, chans: npt.ArrayLike | None = None, **plan_kwargs
    ) -> list[str]:
        """Extract a subset of channels from the data and write each to file.

        Parameters
        ----------
        chans : :py:obj:`~numpy.typing.ArrayLike`, optional
            channel numbers to extract, by default all channels
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        list of str
            names of all files written to disk

        Raises
        ------
        ValueError
            If chans are out of range (chan < 0 or chan > total channels).

        Notes
        -----
        Time series are written to disk with names based on channel number.
        """
        if chans is None:
            chans = np.arange(self.header.nchans)
        chans = np.array(chans).astype("int")
        if np.all(np.logical_or(chans >= self.header.nchans, chans < 0)):
            raise ValueError("Selected channel out of range.")

        out_files = [
            self.header.prep_outfile(
                f"{self.header.basename}_chan{chan:04d}.tim",
                {"nchans": 1, "nbits": 32, "data_type": "time series"},
                nbits=32,
            )
            for chan in chans
        ]
        for nsamps, _ii, data in self.read_plan(**plan_kwargs):
            data = data.reshape(nsamps, self.header.nchans)
            for ifile, out_file in enumerate(out_files):
                out_file.cwrite(data[:, chans[ifile]])

        for out_file in out_files:
            out_file.close()

        return [out_file.name for out_file in out_files]

    def extract_bands(
        self, chanstart: int, nchans: int, chanpersub: int | None = None, **plan_kwargs
    ) -> list[str]:
        """Extract a subset of Sub-bands from the data and write each to file.

        Parameters
        ----------
        chanstart : int
            start channel to extract
        nchans : int
            number of channel to extract
        chanpersub : int, optional
            number of channels in each sub-band, by default ``nchans``
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        list of str
            names of all files written to disk

        Raises
        ------
        ValueError
            If ``chanpersub`` is less than 1 or greater than ``nchans``.
        ValueError
            If ``nchans`` is not divisible by ``chanpersub``.
        ValueError
            If ``chanstart`` is out of range (``chanstart`` < 0 or ``chanstart`` > total channels).

        Notes
        -----
        Filterbanks are written to disk with names based on sub-band number.
        """
        if chanpersub is None:
            chanpersub = nchans
        if chanpersub <= 1 or chanpersub > nchans:
            raise ValueError("chanpersub must be > 1 and <= nchans")
        if chanstart + nchans > self.header.nchans or chanstart < 0:
            raise ValueError("Selected channel out of range.")
        if nchans % chanpersub != 0:
            raise ValueError("Number of channels must be divisible by sub-band size.")

        nsub = (nchans - chanstart) // chanpersub
        fstart = self.header.fch1 + chanstart * self.header.foff

        out_files = [
            self.header.prep_outfile(
                f"{self.header.basename}_sub{isub:02d}.fil",
                {
                    "nchans": chanpersub,
                    "fch1": fstart + isub * chanpersub * self.header.foff,
                },
                nbits=self.header.nbits,
            )
            for isub in range(nsub)
        ]

        for nsamps, _ii, data in self.read_plan(**plan_kwargs):
            data = data.reshape(nsamps, self.header.nchans)
            for ifile, out_file in enumerate(out_files):
                iband_chanstart = chanstart + ifile * chanpersub
                subband_ar = data[:, iband_chanstart : iband_chanstart + chanpersub]
                out_file.cwrite(subband_ar.ravel())

        for out_file in out_files:
            out_file.close()

        return [out_file.name for out_file in out_files]

    def requantize(
        self,
        nbits_out: int,
        remove_bandpass: bool = False,
        filename: str | None = None,
        **plan_kwargs,
    ) -> str:
        """Eequantize the data and write to a new file.

        Parameters
        ----------
        nbits_out : int
            number of bits into requantize the data
        remove_bandpass : bool, optional
            remove the bandpass from the data, by default False
        filename : str, optional
            name of output file, by default ``basename_digi.fil``
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            name of output file

        Raises
        ------
        ValueError
            If ``nbits_out`` is less than 1 or greater than 32.
        """
        if nbits_out not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("nbits_out must be one of {1, 2, 4, 8, 16, 32}")
        if filename is None:
            filename = f"{self.header.basename}_digi.fil"

        out_file = self.header.prep_outfile(filename, nbits=nbits_out, quantize=True)
        for _nsamps, _ii, data in self.read_plan(**plan_kwargs):
            out_file.cwrite(data)
        out_file.close()
        return out_file.name

    def remove_zerodm(self, filename: str | None = None, **plan_kwargs):
        """Remove the channel-weighted zero-DM from the data and write to disk.

        Parameters
        ----------
        filename : str, optional
            name of output file , by default ``basename_noZeroDM.fil``
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            name of output file

        Notes
        -----
        Based on Presto implementation of Eatough, Keane & Lyne 2009 [1]_.

        References
        ----------
        .. [1] R. P. Eatough, E. F. Keane, A. G. Lyne, An interference removal technique for radio pulsar searches,
            MNRAS, Volume 395, Issue 1, May 2009, Pages 410-415.
        """
        if filename is None:
            filename = f"{self.header.basename}_noZeroDM.fil"

        bpass = self.bandpass(**plan_kwargs)
        chanwts = bpass / bpass.sum()
        out_ar = np.empty(
            self.header.nsamples * self.header.nchans, dtype=self.header.dtype
        )
        out_file = self.header.prep_outfile(filename, nbits=self.header.nbits)
        for nsamps, _ii, data in self.read_plan(**plan_kwargs):
            kernels.remove_zerodm(
                data, out_ar, bpass, chanwts, self.header.nchans, nsamps
            )
            out_file.cwrite(out_ar[: nsamps * self.header.nchans])
        out_file.close()
        return out_file.name

    def subband(
        self, dm: float, nsub: int, filename: str = None, gulp: int = 16384, **plan_kwargs
    ) -> str:
        """Produce a set of dedispersed subbands from the data.

        Parameters
        ----------
        dm : float
            the DM of the subbands
        nsub : int
            the number of subbands to produce
        filename : str, optional
            output file name of subbands, by default ``basename_DM.subbands``
        gulp : int, optional
            number of samples in each read, by default 16384
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            name of output subbands file
        """
        subfactor = self.header.nchans // nsub
        chan_delays = self.header.get_dmdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        # must be memset to zero in c code
        out_ar = np.empty((gulp - max_delay) * nsub, dtype="float32")
        new_foff = self.header.foff * self.header.nchans // nsub
        new_fch1 = self.header.ftop - new_foff / 2
        chan_to_sub = np.arange(self.header.nchans, dtype="int32") // subfactor
        changes = {
            "fch1": new_fch1,
            "foff": new_foff,
            "refdm": dm,
            "nchans": nsub,
            "nbits": 32,
        }
        if filename is None:
            filename = f"{self.header.basename}_DM{dm:06.2f}.subbands"
        out_file = self.header.prep_outfile(filename, changes, nbits=32)

        for nsamps, _ii, data in self.read_plan(
            gulp=gulp, skipback=max_delay, **plan_kwargs
        ):
            kernels.subband(
                data,
                out_ar,
                chan_delays,
                chan_to_sub,
                max_delay,
                self.header.nchans,
                nsub,
                nsamps,
            )
            out_file.cwrite(out_ar[: (nsamps - max_delay) * nsub])
        return filename

    def fold(
        self,
        period: float,
        dm: float,
        accel: float = 0,
        nbins: int = 50,
        nints: int = 32,
        nbands: int = 32,
        gulp: int = 16384,
        **plan_kwargs,
    ) -> FoldedData:
        """Fold data into discrete phase, subintegration and subband bins.

        Parameters
        ----------
        period : float
            period in seconds to fold with
        dm : float
            dispersion measure to dedisperse to
        accel : float, optional
            acceleration in m/s/s to fold with, by default 0
        nbins : int, optional
            number of phase bins in output, by default 50
        nints : int, optional
            number of subintegrations in output, by default 32
        nbands : int, optional
            number of subbands in output, by default 32
        gulp : int, optional
            number of samples in each read, by default 16384
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldedData`
            3 dimensional data cube

        Raises
        ------
        ValueError
            If `nbands * nints * nbins` is too large

        Notes
        -----
        If gulp < maximum dispersion delay, gulp is taken to be twice the
        maximum dispersion delay.
        """
        if np.isclose(np.modf(period / self.header.tsamp)[0], 0.001, atol=0):
            warnings.warn("Foldng interval is an integer multiple of the sampling time")
        if nbins > period / self.header.tsamp:
            warnings.warn("Number of phase bins is greater than period/sampling time")
        if (self.header.nsamples * self.header.nchans) // (nbands * nints * nbins) < 10:
            raise ValueError("nbands x nints x nbins is too large.")
        nbands = min(nbands, self.header.nchans)
        chan_delays = self.header.get_dmdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        fold_ar = np.zeros(nbins * nints * nbands, dtype="float32")
        count_ar = np.zeros(nbins * nints * nbands, dtype="int32")
        for nsamps, ii, data in self.read_plan(gulp, skipback=max_delay, **plan_kwargs):
            kernels.fold(
                data,
                fold_ar,
                count_ar,
                chan_delays,
                max_delay,
                self.header.tsamp,
                period,
                accel,
                self.header.nsamples,
                nsamps,
                self.header.nchans,
                nbins,
                nints,
                nbands,
                ii * (gulp - max_delay),
            )
        fold_ar /= count_ar
        fold_ar = fold_ar.reshape(nints, nbands, nbins)
        return FoldedData(fold_ar, self.header.new_header(), period, dm, accel)
