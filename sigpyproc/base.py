from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.core import kernels
from sigpyproc.core.rfi import RFIMask
from sigpyproc.core.stats import ChannelStats
from sigpyproc.foldedcube import FoldedData
from sigpyproc.timeseries import TimeSeries
from sigpyproc.utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from typing_extensions import Buffer, TypedDict, Unpack

    from sigpyproc.block import FilterbankBlock
    from sigpyproc.core.types import MaskMethods
    from sigpyproc.header import Header

    class PlanKwargs(TypedDict, total=False):
        description: str | None
        quiet: bool
        allocator: Callable[[int], Buffer] | None


class Filterbank(ABC):
    """Base class for manipulating frequency-major order pulsar data.

    The Filterbank class should never be instantiated directly. Instead it
    should be inherited by data reading classes.

    Attributes
    ----------
    header
    chan_stats
    """

    def __init__(self) -> None:
        self._chan_stats: ChannelStats | None = None
        self.logger = get_logger(__name__)

    @property
    @abstractmethod
    def header(self) -> Header:
        """Header metadata of input file.

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            Header object containing metadata of the input file.
        """

    @abstractmethod
    def read_block(
        self,
        start: int,
        nsamps: int,
        fch1: float | None = None,
        nchans: int | None = None,
    ) -> FilterbankBlock:
        """Read a data block from the filterbank file stream.

        Parameters
        ----------
        start : int
            First time sample of the block to be read.
        nsamps : int
            Number of samples in the block (i.e. block will be nsamps*nchans in size).
        fch1 : float, optional
            Frequency of the first channel, by default None (header value).
        nchans : int, optional
            Number of channels in the block, by default None (header value).

        Returns
        -------
        :class:`~sigpyproc.block.FilterbankBlock`
            2-D array of filterbank data with observational metadata.

        Raises
        ------
        ValueError
            if requested ``nsamps`` or ``nchans`` are out of range.
        """

    @abstractmethod
    def read_dedisp_block(self, start: int, nsamps: int, dm: float) -> FilterbankBlock:
        """Read a block of dedispersed filterbank data.

        Best used in cases where I/O time dominates reading a block of data.

        Parameters
        ----------
        start : int
            First time sample of the block to be read.
        nsamps : int
            Number of samples in the block (i.e. block will be nsamps*nchans in size).
        dm : float
            Dispersion measure to dedisperse at.

        Returns
        -------
        :class:`~sigpyproc.block.FilterbankBlock`
            2-D array of filterbank data with observational metadata.

        Raises
        ------
        ValueError
            if requested dedispersed ``nsamps`` are out of range.
        """

    @abstractmethod
    def read_plan(
        self,
        *,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        skipback: int = 0,
        description: str | None = None,
        quiet: bool = False,
        allocator: Callable[[int], Buffer] | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Read sequential filterbank in gulps and yield.

        Parameters
        ----------
        gulp : int, optional
            Number of time samples in each read, by default 16384.
        start : int, optional
            Starting sample to read from, by default 0 (start of file).
        nsamps : int, optional
            Total number of samples to read, by default None (end of the file).
        skipback : int, optional
            Number of samples to skip back after each read, by default 0.
        description : str, optional
            Annotation for progress bar (rich), by default Calling Stack.
        quiet : bool, optional
            Disable progress bar and logging, by default False.
        allocator : Callable[[int], Buffer], optional
            An allocator callback that returns an object implementing
            the Python Buffer Protocol interface (PEP 3118) for the
            data to be read into, by default None.

        Yields
        ------
        Iterator[tuple[int, int, ndarray]]
            Tuple of number of samples read, index of read, and the unpacked data read.

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

        >>> for nsamps_r, ii, data in self.read_plan(**plan_kwargs):
                # do something
        where data always has contains ``nchans*nsamps`` points.
        """

    @property
    def chan_stats(self) -> ChannelStats | None:
        """Channel statistics of the data.

        Returns
        -------
        :class:`~sigpyproc.core.stats.ChannelStats` | None
            Channel statistics object if computed, else None.
        """
        return self._chan_stats

    def compute_stats(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> None:
        """Compute channelwise statistics of data.

        Channel statistics include mean, rms, skewness, kurtosis, maxima, and minima.

        Parameters
        ----------
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.
        """
        bag = ChannelStats(self.header.nchans, self.header.nsamples)
        for _, ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            bag.push_data(data, ii, mode="full")
        self._chan_stats = bag

    def compute_stats_basic(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> None:
        """Compute channelwise statistics of data (basic).

        Channel statistics include mean, rms, maxima, and minima.

        Parameters
        ----------
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.
        """
        bag = ChannelStats(self.header.nchans, self.header.nsamples)
        for _, ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            bag.push_data(data, ii, mode="basic")
        self._chan_stats = bag

    def collapse(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Parameters
        ----------
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional Keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            A zero-DM time series.
        """
        tim_len = (self.header.nsamples - start) if nsamps is None else nsamps
        tim_ar = np.zeros(tim_len, dtype=np.float32)
        for nsamps_r, ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            kernels.extract_tim(data, tim_ar, self.header.nchans, nsamps_r, ii * gulp)
        return TimeSeries(tim_ar, self.header.new_header({"nchans": 1, "dm": 0}))

    def bandpass(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> TimeSeries:
        """Compute the bandpass of the data.

        Average across each time sample for all frequencies.

        Parameters
        ----------
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            Bandpass of the data.
        """
        bpass_ar = np.zeros(self.header.nchans, dtype=np.float32)
        num_samples = 0
        for nsamps_r, _, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            kernels.extract_bpass(data, bpass_ar, self.header.nchans, nsamps_r)
            num_samples += nsamps_r
        bpass_ar /= num_samples
        return TimeSeries(
            bpass_ar,
            self.header.new_header({"nchans": 1, "nsamples": len(bpass_ar)}),
        )

    def dedisperse(
        self,
        dm: float,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> TimeSeries:
        """Dedisperse and collapse to a time series.

        Parameters
        ----------
        dm : float
            Dispersion measure to dedisperse to.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            A dedispersed time series.

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
        for nsamps_r, ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            skipback=max_delay,
            **plan_kwargs,
        ):
            kernels.dedisperse(
                data,
                tim_ar,
                chan_delays,
                max_delay,
                self.header.nchans,
                nsamps_r,
                ii * (gulp - max_delay),
            )
        return TimeSeries(
            tim_ar,
            self.header.new_header({"nchans": 1, "dm": dm, "nsamples": tim_len}),
        )

    def read_chan(
        self,
        ichan: int,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> TimeSeries:
        """Read a single frequency channel as a time series.

        Parameters
        ----------
        ichan : int
            Channel index to retrieve (0 is the highest frequency channel).
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            Selected channel as a time series.

        Raises
        ------
        ValueError
            If ichan is out of range (ichan < 0 or ichan > nchans).
        """
        if ichan >= self.header.nchans or ichan < 0:
            msg = f"Selected channel {ichan} out of range."
            raise ValueError(msg)
        tim_ar = np.empty(self.header.nsamples, dtype=np.float32)
        for nsamps_r, ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            data_2d = data.reshape(nsamps_r, self.header.nchans)
            tim_ar[ii * gulp : (ii + 1) * gulp] = data_2d[:, ichan]
        return TimeSeries(tim_ar, self.header.new_header({"dm": 0, "nchans": 1}))

    def invert_freq(
        self,
        outfile_name: str | None = None,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Invert frequency axis and write to a new file.

        Parameters
        ----------
        outfile_name : str, optional
            Name of output file, by default ``basename_inverted.fil``.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.
        """
        if outfile_name is None:
            outfile_name = f"{self.header.basename}_inverted.fil"

        updates = {
            "fch1": self.header.fch1 + (self.header.nchans - 1) * self.header.foff,
            "foff": self.header.foff * -1,
        }

        out_file = self.header.prep_outfile(
            outfile_name,
            updates=updates,
            nbits=self.header.nbits,
        )
        for nsamps_r, _, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            out_ar = kernels.invert_freq(data, self.header.nchans, nsamps_r)
            out_file.cwrite(out_ar)
        out_file.close()
        return outfile_name

    def apply_channel_mask(
        self,
        chanmask: np.ndarray,
        maskvalue: float = 0,
        outfile_name: str | None = None,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Apply a channel mask and write to a new file.

        Parameters
        ----------
        chanmask : :py:obj:`~numpy.typing.ArrayLike`
            Boolean array of channel mask (1 or True for bad channel).
        maskvalue : float, optional
            Value to set the masked data to, by default 0.
        outfile_name : str, optional
            Name of the output filterbank file, by default ``basename_masked.fil``.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.
        """
        if outfile_name is None:
            outfile_name = f"{self.header.basename}_masked.fil"

        mask = np.array(chanmask).astype("bool")
        maskvalue = np.float32(maskvalue).astype(self.header.dtype)
        out_file = self.header.prep_outfile(outfile_name)
        for nsamps_r, _ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            kernels.mask_channels(data, mask, maskvalue, self.header.nchans, nsamps_r)
            out_file.cwrite(data)
        return outfile_name

    def downsample(
        self,
        tfactor: int = 1,
        ffactor: int = 1,
        outfile_name: str | None = None,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Decimate in time and frequency and write to file.

        Parameters
        ----------
        tfactor : int, optional
            Factor by which to downsample in time, by default 1.
        ffactor : int, optional
            Factor by which to downsample in frequency, by default 1.
        outfile_name : str, optional
            Name of file to write to, by default ``basename_tfactor_ffactor.fil``.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.

        Raises
        ------
        ValueError
            If number of channels is not divisible by ``ffactor``.
        """
        if outfile_name is None:
            outfile_name = f"{self.header.basename}_f{ffactor:d}_t{tfactor:d}.fil"
        if self.header.nchans % ffactor != 0:
            msg = f"Bad frequency factor given: {ffactor:d}"
            raise ValueError(msg)

        # Gulp must be a multiple of tfactor
        gulp = int(np.ceil(gulp / tfactor) * tfactor)

        updates = {
            "tsamp": self.header.tsamp * tfactor,
            "nchans": self.header.nchans // ffactor,
            "foff": self.header.foff * ffactor,
        }
        out_file = self.header.prep_outfile(outfile_name, updates=updates)

        for nsamps_r, _ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            write_ar = kernels.downsample_2d(
                data,
                tfactor,
                ffactor,
                self.header.nchans,
                nsamps_r,
            )
            out_file.cwrite(write_ar)
        return outfile_name

    def extract_samps(
        self,
        start: int,
        nsamps: int,
        outfile_name: str | None = None,
        gulp: int = 16384,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Extract a subset of samples and write to file.

        Parameters
        ----------
        start : int
            Starting time sample to extract.
        nsamps : int
            Number of time samples to extract.
        outfile_name : str, optional
            Output file name, by default ``basename_samps_{start}_{start+nsamps}.fil``.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.

        Raises
        ------
        ValueError
            If ``start`` or ``nsamps`` are out of bounds.
        """
        if start < 0 or start + nsamps > self.header.nsamples:
            msg = f"Selected samples out of range: {start:d} to {start+nsamps:d}"
            raise ValueError(msg)
        if outfile_name is None:
            outfile_name = (
                f"{self.header.basename}_samps_{start:d}_{start+nsamps:d}.fil"
            )
        out_file = self.header.prep_outfile(
            outfile_name,
            updates={"tstart": self.header.mjd_after_nsamps(start)},
            nbits=self.header.nbits,
        )
        for _, _, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            out_file.cwrite(data)
        out_file.close()
        return outfile_name

    def extract_chans(
        self,
        chans: np.ndarray | None = None,
        outfile_base: str | None = None,
        batch_size: int = 200,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> list[str]:
        """Extract a subset of channels and write to file.

        Time series are written to disk with names based on channel number.

        Parameters
        ----------
        chans : :py:obj:`~numpy.typing.ArrayLike`, optional
            Channel numbers to extract, by default all channels.
        outfile_base : str, optional
            Base name of output files, by default ``header.basename``.
        batch_size : int, optional
            Number of channels to extract in each batch, by default 200.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        list[str]
            Names of all files written to disk.

        Raises
        ------
        ValueError
            If chans are out of range (chan < 0 or chan > total channels).
        """
        if chans is None:
            chans = np.arange(self.header.nchans)
        chans = np.array(chans).astype("int")
        nchans_extract = len(chans)
        if np.all(np.logical_or(chans >= self.header.nchans, chans < 0)):
            msg = f"Selected channels out of range: {chans.min()} to {chans.max()}"
            raise ValueError(msg)
        if outfile_base is None:
            outfile_base = self.header.basename
        filenames = [f"{outfile_base}_chan{chan:04d}.tim" for chan in chans]

        # Process in batches to avoid file open/close limits
        for batch_start in range(0, nchans_extract, batch_size):
            batch_end = min(batch_start + batch_size, nchans_extract)
            batch_chans = chans[batch_start:batch_end]
            batch_files = filenames[batch_start:batch_end]

            with ExitStack() as stack:
                out_files = [
                    stack.enter_context(
                        self.header.prep_outfile(
                            filename,
                            updates={
                                "nchans": 1,
                                "nbits": 32,
                                "data_type": "time series",
                            },
                            nbits=32,
                        ),
                    )
                    for filename in batch_files
                ]
                for nsamps_r, _, data in self.read_plan(
                    gulp=gulp,
                    start=start,
                    nsamps=nsamps,
                    **plan_kwargs,
                ):
                    data_2d = data.reshape(nsamps_r, self.header.nchans)
                    for ifile, out_file in enumerate(out_files):
                        out_file.cwrite(data_2d[:, batch_chans[ifile]])
        return filenames

    def extract_bands(
        self,
        chanstart: int,
        nchans: int,
        chanpersub: int | None = None,
        outfile_base: str | None = None,
        batch_size: int = 200,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> list[str]:
        """Extract a subset of Sub-bands and write to file.

        Filterbanks are written to disk with names based on sub-band number.

        Parameters
        ----------
        chanstart : int
            Start channel to extract.
        nchans : int
            Number of channel to extract.
        chanpersub : int, optional
            Number of channels in each sub-band, by default ``nchans``.
        outfile_base : str, optional
            Base name of output files, by default ``header.basename``.
        batch_size : int, optional
            Number of sub-bands to extract in each batch, by default 200.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        list[str]
            Names of all files written to disk.

        Raises
        ------
        ValueError
            If ``chanpersub`` is less than 1 or greater than ``nchans``.
        ValueError
            If ``nchans`` is not divisible by ``chanpersub``.
        ValueError
            If ``chanstart`` is out of range (``chanstart`` < 0
            or ``chanstart`` > total channels).
        """
        if chanpersub is None:
            chanpersub = nchans
        if chanpersub <= 1 or chanpersub > nchans:
            msg = f"chanpersub must be > 1 and <= nchans. Got {chanpersub}"
            raise ValueError(msg)
        if chanstart + nchans > self.header.nchans or chanstart < 0:
            msg = f"Selected channels out of range: {chanstart} to {chanstart+nchans}"
            raise ValueError(msg)
        if nchans % chanpersub != 0:
            msg = f"Number of channels must be divisible by sub-band size. Got {nchans}"
            raise ValueError(msg)

        nsub = (self.header.nchans - chanstart) // chanpersub
        fstart = self.header.fch1 + chanstart * self.header.foff

        if outfile_base is None:
            outfile_base = self.header.basename

        filenames = [f"{outfile_base}_sub{isub:02d}.fil" for isub in range(nsub)]

        # Process in batches to avoid file open/close limits
        for batch_start in range(0, nsub, batch_size):
            batch_end = min(batch_start + batch_size, nsub)
            batch_files = filenames[batch_start:batch_end]

            with ExitStack() as stack:
                out_files = [
                    stack.enter_context(
                        self.header.prep_outfile(
                            filename,
                            updates={
                                "nchans": chanpersub,
                                "fch1": fstart
                                + (batch_start + i) * chanpersub * self.header.foff,
                            },
                            nbits=self.header.nbits,
                        ),
                    )
                    for i, filename in enumerate(batch_files)
                ]

                for nsamps_r, _ii, data in self.read_plan(
                    gulp=gulp,
                    start=start,
                    nsamps=nsamps,
                    **plan_kwargs,
                ):
                    data_2d = data.reshape(nsamps_r, self.header.nchans)
                    for ifile, out_file in enumerate(out_files):
                        iband_chanstart = chanstart + (batch_start + ifile) * chanpersub
                        subband_ar = data_2d[
                            :,
                            iband_chanstart : iband_chanstart + chanpersub,
                        ]
                        out_file.cwrite(subband_ar.ravel())
        return filenames

    def requantize(
        self,
        nbits_out: int,
        outfile_name: str | None = None,
        *,
        remove_bandpass: bool = False,  # noqa: ARG002
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Requantize the data and write to a new file.

        Parameters
        ----------
        nbits_out : int
            Number of bits into requantize the data.
        outfile_name : str, optional
            Name of output file, by default ``basename_digi.fil``.
        remove_bandpass : bool, optional
            Remove the bandpass from the data, by default False.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.

        Raises
        ------
        ValueError
            If ``nbits_out`` is less than 1 or greater than 32.
        """
        if nbits_out not in {1, 2, 4, 8, 16, 32}:
            msg = f"nbits_out must be one of {1, 2, 4, 8, 16, 32}, got {nbits_out}"
            raise ValueError(msg)
        if outfile_name is None:
            outfile_name = f"{self.header.basename}_digi.fil"

        out_file = self.header.prep_outfile(outfile_name, nbits=nbits_out)
        for _, _, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            out_file.cwrite(data)
        out_file.close()
        return outfile_name

    def remove_zerodm(
        self,
        outfile_name: str | None = None,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Remove zero-DM and write to a new file.

        Remove the channel-weighted zero-DM from the data and write to disk.

        Parameters
        ----------
        outfile_name : str, optional
            Name of output file , by default ``basename_noZeroDM.fil``.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.

        Notes
        -----
        Based on Presto implementation of Eatough, Keane & Lyne 2009 [1]_.

        References
        ----------
        .. [1] R. P. Eatough, E. F. Keane, A. G. Lyne, An interference removal
            technique for radio pulsar searches,
            MNRAS, Volume 395, Issue 1, May 2009, Pages 410-415.
        """
        if outfile_name is None:
            outfile_name = f"{self.header.basename}_noZeroDM.fil"

        bpass = self.bandpass(**plan_kwargs).data
        chanwts = bpass / bpass.sum()
        out_ar = np.empty(
            self.header.nsamples * self.header.nchans,
            dtype=self.header.dtype,
        )
        out_file = self.header.prep_outfile(outfile_name, nbits=self.header.nbits)
        for nsamps_r, _, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        ):
            kernels.remove_zerodm(
                data,
                out_ar,
                bpass,
                chanwts,
                self.header.nchans,
                nsamps_r,
            )
            out_file.cwrite(out_ar[: nsamps_r * self.header.nchans])
        out_file.close()
        return outfile_name

    def subband(
        self,
        dm: float,
        nsub: int,
        outfile_name: str | None = None,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> str:
        """Subband the data and write to a new file.

        Produce a set of dedispersed subbands from the data.

        Parameters
        ----------
        dm : float
            The DM of the subbands.
        nsub : int
            The number of subbands to produce.
        outfile_name : str, optional
            Output file name of subbands, by default ``basename_DM.subbands``.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        str
            Name of output file.
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
        updates = {
            "fch1": new_fch1,
            "foff": new_foff,
            "refdm": dm,
            "nchans": nsub,
            "nbits": 32,
        }
        if outfile_name is None:
            outfile_name = f"{self.header.basename}_DM{dm:06.2f}.subbands"
        out_file = self.header.prep_outfile(outfile_name, updates=updates, nbits=32)

        for nsamps_r, _ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            skipback=max_delay,
            **plan_kwargs,
        ):
            kernels.subband(
                data,
                out_ar,
                chan_delays,
                chan_to_sub,
                max_delay,
                self.header.nchans,
                nsub,
                nsamps_r,
            )
            out_file.cwrite(out_ar[: (nsamps_r - max_delay) * nsub])
        return outfile_name

    def fold(
        self,
        period: float,
        dm: float,
        accel: float = 0,
        nbins: int = 50,
        nints: int = 32,
        nbands: int = 32,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> FoldedData:
        """Fold the data and return a 3D data cube.

        Fold data into discrete phase, subintegration and subband bins.

        Parameters
        ----------
        period : float
            Period in seconds to fold with.
        dm : float
            Dispersion measure to dedisperse to.
        accel : float, optional
            Acceleration in m/s/s to fold with, by default 0.
        nbins : int, optional
            Number of phase bins in output, by default 50.
        nints : int, optional
            Number of subintegrations in output, by default 32.
        nbands : int, optional
            Number of subbands in output, by default 32.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldedData`
            3 dimensional data cube.

        Raises
        ------
        ValueError
            If ``nbands * nints * nbins`` is too large.

        Notes
        -----
        If gulp < maximum dispersion delay, gulp is taken to be twice the
        maximum dispersion delay.
        """
        if np.isclose(np.modf(period / self.header.tsamp)[0], 0.001, atol=0):
            self.logger.warning(
                "Foldng interval is an integer multiple of the sampling time",
            )
        if nbins > period / self.header.tsamp:
            self.logger.warning(
                "Number of phase bins is greater than period/sampling time",
            )
        if (self.header.nsamples * self.header.nchans) // (nbands * nints * nbins) < 10:
            msg = f"nbands x nints x nbins is too large: {nbands*nints*nbins}"
            raise ValueError(msg)
        nbands = min(nbands, self.header.nchans)
        chan_delays = self.header.get_dmdelays(dm)
        max_delay = int(chan_delays.max())
        gulp = max(2 * max_delay, gulp)
        fold_ar = np.zeros(nbins * nints * nbands, dtype="float32")
        count_ar = np.zeros(nbins * nints * nbands, dtype="int32")
        for nsamps_r, ii, data in self.read_plan(
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            skipback=max_delay,
            **plan_kwargs,
        ):
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
                nsamps_r,
                self.header.nchans,
                nbins,
                nints,
                nbands,
                ii * (gulp - max_delay),
            )
        fold_ar /= count_ar
        fold_ar = fold_ar.reshape(nints, nbands, nbins)
        return FoldedData(fold_ar, self.header.new_header(), period, dm, accel)

    def clean_rfi(
        self,
        method: MaskMethods = "mad",
        threshold: float = 3,
        chanmask: np.ndarray | None = None,
        custom_funcn: Callable[[np.ndarray], np.ndarray] | None = None,
        outfile_name: str | None = None,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        **plan_kwargs: Unpack[PlanKwargs],
    ) -> tuple[str, RFIMask]:
        """Clean RFI from the data.

        Parameters
        ----------
        method : str, optional
            Method to use for cleaning ("mad", "iqrm"), by default "mad".
        threshold : float, optional
            Threshold for cleaning, by default 3.
        chanmask : :py:obj:`~numpy.typing.ArrayLike`, optional
            User channel mask to use (1 or True for bad channels), by default None.
        custom_funcn : :py:obj:`~typing.Callable`, optional
            Custom function to apply to the mask, by default None.
        outfile_name : str, optional
            Output file name, by default None.
        gulp : int, optional
            Number of samples in each read, by default 16384.
        start : int, optional
            Start sample, by default 0.
        nsamps : int, optional
            Number of samples to read, by default all.
        **plan_kwargs : dict
            Additional keyword arguments for :func:`read_plan`.

        Returns
        -------
        tuple[str, :class:`~sigpyproc.core.rfi.RFIMask`]
            Filename and mask of cleaned data.

        Raises
        ------
        ValueError
            If ``method`` is not "mad" or "iqrm".
        """
        if chanmask is None:
            chanmask = np.zeros(self.header.nchans, dtype="bool")
        if method not in {"mad", "iqrm"}:
            msg = f"Clean method must be 'mad' or 'iqrm', got {method}"
            raise ValueError(msg)

        if self.chan_stats is None:
            # 1st pass to compute channel statistics (upto kurtosis)
            self.compute_stats(gulp=gulp, start=start, nsamps=nsamps, **plan_kwargs)

        if not isinstance(self.chan_stats, ChannelStats):
            msg = "Channel statistics not computed properly"
            raise TypeError(msg)
        # Initialise mask
        rfimask = RFIMask(
            threshold,
            self.header,
            self.chan_stats.mean,
            self.chan_stats.var,
            self.chan_stats.skew,
            self.chan_stats.kurtosis,
            self.chan_stats.maxima,
            self.chan_stats.minima,
        )
        rfimask.apply_mask(chanmask)
        rfimask.apply_method(method)
        if custom_funcn is not None:
            rfimask.apply_funcn(custom_funcn)

        maskvalue = 0
        # Apply the channel mask
        out_file = self.apply_channel_mask(
            rfimask.chan_mask,
            maskvalue,
            outfile_name=outfile_name,
            gulp=gulp,
            start=start,
            nsamps=nsamps,
            **plan_kwargs,
        )
        return out_file, rfimask
