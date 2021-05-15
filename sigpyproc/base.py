from __future__ import annotations
import warnings
import numpy as np

from numpy import typing as npt
from typing import List, Optional, Generator, Tuple
from abc import ABC, abstractmethod

from sigpyproc.foldedcube import FoldedData
from sigpyproc.timeseries import TimeSeries
from sigpyproc.header import Header
from sigpyproc import libcpp  # type: ignore


class Filterbank(ABC):
    """Class exporting methods for the manipulation of frequency-major order pulsar data.

    Attributes
    ----------
    chan_means : :py:obj:`~numpy.ndarray`
        the mean value of each channel
    chan_vars : :py:obj:`~numpy.ndarray`
        the variance of each channel
    chan_stdevs : :py:obj:`~numpy.ndarray`
        the standard deviation of each channel
    chan_skews : :py:obj:`~numpy.ndarray`
        the skewness of each channel
    chan_kurts : :py:obj:`~numpy.ndarray`
        the kurtosis of each channel
    chan_max : :py:obj:`~numpy.ndarray`
        the maximum value of each channel
    chan_min : :py:obj:`~numpy.ndarray`
        the minimum value of each channel

    Notes
    -----
    The Filterbank class should never be instantiated directly. Instead it
    should be inherited by data reading classes.
    """

    def __init__(self) -> None:
        self.chan_means: Optional[np.ndarray] = None
        self.chan_vars: Optional[np.ndarray] = None
        self.chan_stdevs: Optional[np.ndarray] = None
        self.chan_skews: Optional[np.ndarray] = None
        self.chan_kurts: Optional[np.ndarray] = None
        self.chan_maxima: Optional[np.ndarray] = None
        self.chan_minima: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def header(self) -> Header:
        raise NotImplementedError()

    @abstractmethod
    def read_plan(
        self, *args, **kwargs
    ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        raise NotImplementedError()

    @property
    def omp_max_threads(self) -> int:
        """Maximum number of omp threads (`int`)."""
        return libcpp.omp_get_max_threads()

    @property
    def omp_threads(self) -> int:
        """Number of omp threads (`int`)."""
        return libcpp.omp_get_num_threads()

    def set_omp_threads(self, nthreads: Optional[int] = None) -> None:
        """Set the number of threads available to OpenMP.

        Parameters
        ----------
        nthreads : int, optional
            number of threads to use, by default 4
        """
        if nthreads is None:
            nthreads = 4
        libcpp.omp_set_num_threads(nthreads)

    def compute_stats(self, gulp: int = 512, **kwargs) -> None:
        """Compute channelwise statistics of data (upto kurtosis).

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512

        """
        maxima_ar = np.zeros(self.header.nchans, dtype="float32")
        minima_ar = np.zeros(self.header.nchans, dtype="float32")
        count_ar = np.zeros(self.header.nchans, dtype="int64")
        m1_ar = np.zeros(self.header.nchans, dtype="float32")
        m2_ar = np.zeros(self.header.nchans, dtype="float32")
        m3_ar = np.zeros(self.header.nchans, dtype="float32")
        m4_ar = np.zeros(self.header.nchans, dtype="float32")
        for nsamps, ii, data in self.read_plan(gulp, **kwargs):
            libcpp.compute_moments(
                data,
                m1_ar,
                m2_ar,
                m3_ar,
                m4_ar,
                maxima_ar,
                minima_ar,
                count_ar,
                self.header.nchans,
                nsamps,
                ii,
            )

        means_ar = m1_ar
        var_ar = m2_ar / self.header.nsamples
        stdev_ar = np.sqrt(var_ar)

        m2_ar[m2_ar == 0] = np.nan
        skew_ar = m3_ar / np.power(m2_ar, 1.5) * np.sqrt(self.header.nsamples)
        kurt_ar = m4_ar / np.power(m2_ar, 2.0) * self.header.nsamples - 3.0

        stdev_ar[np.isnan(stdev_ar)] = 0
        skew_ar[np.isnan(skew_ar)] = 0
        kurt_ar[np.isnan(kurt_ar)] = -3.0

        self.chan_means = means_ar.astype("float32")
        self.chan_vars = var_ar.astype("float32")
        self.chan_stdevs = stdev_ar.astype("float32")
        self.chan_skews = skew_ar.astype("float32")
        self.chan_kurts = kurt_ar.astype("float32")
        self.chan_maxima = maxima_ar
        self.chan_minima = minima_ar

    def compute_stats_simple(self, gulp: int = 512, **kwargs) -> None:
        """Compute channelwise statistics of data (mean and rms).

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512

        """
        maxima_ar = np.zeros(self.header.nchans, dtype="float32")
        minima_ar = np.zeros(self.header.nchans, dtype="float32")
        count_ar = np.zeros(self.header.nchans, dtype="int64")
        m1_ar = np.zeros(self.header.nchans, dtype="float32")
        m2_ar = np.zeros(self.header.nchans, dtype="float32")
        for nsamps, ii, data in self.read_plan(gulp, **kwargs):
            libcpp.compute_moments_simple(
                data,
                m1_ar,
                m2_ar,
                maxima_ar,
                minima_ar,
                count_ar,
                self.header.nchans,
                nsamps,
                ii,
            )

        means_ar = m1_ar
        var_ar = m2_ar / self.header.nsamples
        stdev_ar = np.sqrt(var_ar)
        m2_ar[m2_ar == 0] = np.nan
        stdev_ar[np.isnan(stdev_ar)] = 0

        self.chan_means = means_ar.astype("float32")
        self.chan_vars = var_ar.astype("float32")
        self.chan_stdevs = stdev_ar.astype("float32")
        self.chan_maxima = maxima_ar
        self.chan_minima = minima_ar

    def collapse(
        self, gulp: int = 512, start: int = 0, nsamps: int = None, **kwargs
    ) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        start : int, optional
            start sample, by default 0
        nsamps : int, optional
            number of samples to read, by default all

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            A zero-DM time series
        """
        if nsamps is None:
            size = self.header.nsamples - start
        else:
            size = nsamps
        timar = np.zeros(size, dtype="float32")
        for nsamp, ii, data in self.read_plan(
            gulp,
            start=start,
            nsamps=nsamps,
            **kwargs,
        ):
            libcpp.get_tim(data, timar, self.header.nchans, nsamp, ii * gulp)
        return TimeSeries(timar, self.header.new_header({"nchans": 1, "dm": 0}))

    def bandpass(self, gulp: int = 512, **kwargs) -> TimeSeries:
        """Average across each time sample for all frequencies.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            the bandpass of the data
        """
        bpass_ar = np.zeros(self.header.nchans, dtype="float64")
        num_samples = 0
        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            libcpp.get_bpass(data, bpass_ar, self.header.nchans, nsamps)
            num_samples += nsamps
        bpass_ar /= num_samples
        return TimeSeries(bpass_ar, self.header.new_header({"nchans": 1}))

    def dedisperse(self, dm: float, gulp: int = 10000, **kwargs) -> TimeSeries:
        """Dedisperse the data to a time series.

        Parameters
        ----------
        dm : float
            dispersion measure to dedisperse to
        gulp : int, optional
            number of samples in each read, by default 10000

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
        tim_ar = np.zeros(tim_len, dtype="float32")
        for nsamps, ii, data in self.read_plan(gulp, skipback=max_delay, **kwargs):
            libcpp.dedisperse(
                data,
                tim_ar,
                chan_delays,
                max_delay,
                self.header.nchans,
                nsamps,
                ii * (gulp - max_delay),
            )
        return TimeSeries(tim_ar, self.header.new_header({"nchans": 1, "dm": dm}))

    def get_chan(self, chan: int, gulp: int = 512, **kwargs) -> TimeSeries:
        """Retrieve a single frequency channel from the data.

        Parameters
        ----------
        chan : int
            channel to retrieve (0 is the highest frequency channel)
        gulp : int, optional
            number of samples in each read, by default 512

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
        tim_ar = np.empty(self.header.nsamples, dtype="float32")
        for nsamps, ii, data in self.read_plan(gulp, **kwargs):
            libcpp.get_chan(data, tim_ar, chan, self.header.nchans, nsamps, ii * gulp)
        return TimeSeries(tim_ar, self.header.new_header({"dm": 0, "nchans": 1}))

    def fold(
        self,
        period: float,
        dm: float,
        accel: float = 0,
        nbins: int = 50,
        nints: int = 32,
        nbands: int = 32,
        gulp: int = 10000,
        **kwargs,
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
            number of samples in each read, by default 10000

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
        for nsamps, ii, data in self.read_plan(gulp, skipback=max_delay, **kwargs):
            libcpp.foldfil(
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

    def invert_freq(
        self,
        gulp: int = 512,
        start: int = 0,
        nsamps: int = None,
        filename: str = None,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Invert the frequency ordering of the data and write new data to a new file.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        start : int, optional
            start sample, by default 0
        nsamps : int, optional
            number of samples to read, by default all
        filename : str, optional
            name of output file, by default ``basename_inverted.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            filename = f"{self.header.basename}_inverted.fil"

        if nsamps is None:
            size = self.header.nsamples - start
        else:
            size = nsamps
        out_ar = np.empty(size * self.header.nchans, dtype=self.header.dtype)

        changes = {
            "fch1": self.header.fch1 + (self.header.nchans - 1) * self.header.foff,
            "foff": self.header.foff * -1,
        }

        out_file = self.header.prep_outfile(
            filename, changes, nbits=self.header.nbits, back_compatible=back_compatible
        )
        for nsamp, _ii, data in self.read_plan(
            gulp,
            start=start,
            nsamps=nsamps,
            **kwargs,
        ):
            libcpp.invert_freq(data, out_ar, self.header.nchans, nsamp)
            out_file.cwrite(out_ar[: nsamp * self.header.nchans])
        out_file.close()
        return out_file.name

    def subband(
        self, dm: float, nsub: int, filename: str = None, gulp: int = 10000, **kwargs
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
            number of samples in each read, by default 10000

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
        out_file = self.header.prep_outfile(
            filename, changes, nbits=32, back_compatible=True
        )

        for nsamps, _ii, data in self.read_plan(gulp, skipback=max_delay, **kwargs):
            libcpp.subband(
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

    def to_8bit(
        self,
        filename: str = None,
        gulp: int = 512,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Convert 1-,2- or 4-bit data to 8-bit data and write to file.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename_8bit.fil``
        gulp : int, optional
            number of samples in each read, by default 512
        back_compatible : bool, optional
            sigproc compatibility flag, by default True

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            filename = f"{self.header.basename}_8bit.fil"

        out_file = self.header.prep_outfile(
            filename, {"nbits": 8}, nbits=8, back_compatible=back_compatible
        )
        for _nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            out_file.cwrite(data)
        return out_file.name

    def apply_channel_mask(
        self,
        chanmask: npt.ArrayLike,
        outfilename: Optional[str] = None,
        gulp: int = 512,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Set the data in the given channels to zero.

        Parameters
        ----------
        chanmask : :py:obj:`~numpy.typing.ArrayLike`
            binary channel mask (0 for bad channel, 1 for good)
        outfilename : str, optional
            name of the output filterbank file, by default ``basename_masked.fil``
        gulp : int, optional
            number of samples in each read, by default 512
        back_compatible : bool, optional
            sigproc compatibility flag, by default True

        Returns
        -------
        str
            name of output file
        """
        if outfilename is None:
            outfilename = f"{self.header.basename}_masked.fil"
        mask = np.array(chanmask).astype("ubyte")
        out_file = self.header.prep_outfile(outfilename, back_compatible=back_compatible)
        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            libcpp.mask_channels(data, mask, self.header.nchans, nsamps)
            out_file.cwrite(data)
        return out_file.name

    def downsample(
        self,
        tfactor: int = 1,
        ffactor: int = 1,
        gulp: int = 512,
        filename: Optional[str] = None,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Downsample data in time and/or frequency and write to file.

        Parameters
        ----------
        tfactor : int, optional
            factor by which to downsample in time, by default 1
        ffactor : int, optional
            factor by which to downsample in frequency, by default 1
        gulp : int, optional
            number of samples in each read, by default 512
        filename : str, optional
            name of file to write to, by default ``basename_tfactor_ffactor.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

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

        out_file = self.header.prep_outfile(
            filename,
            {
                "tsamp": self.header.tsamp * tfactor,
                "nchans": self.header.nchans // ffactor,
                "foff": self.header.foff * ffactor,
            },
            back_compatible=back_compatible,
        )

        write_ar = np.zeros(
            gulp * self.header.nchans // ffactor // tfactor,
            dtype=self.header.dtype,
        )
        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            libcpp.downsample(
                data, write_ar, tfactor, ffactor, self.header.nchans, nsamps
            )
            out_file.cwrite(write_ar[: nsamps * self.header.nchans // ffactor // tfactor])
        return out_file.name

    def split(
        self,
        start: int,
        nsamps: int,
        filename: Optional[str] = None,
        gulp: int = 1024,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Split data in time.

        Parameters
        ----------
        start : int
            start sample of split
        nsamps : int
            number of samples in split
        filename : str, optional
            name of output file, by default ``basename_start_start+nsamps.fil``
        gulp : int, optional
            number of samples in each read, by default 1024
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of new file
        """
        if filename is None:
            filename = f"{self.header.basename}_{start:d}_{start+nsamps:d}.fil"
        out_file = self.header.prep_outfile(
            filename,
            updates={"tstart": self.header.mjd_after_nsamps(start)},
            nbits=self.header.nbits,
        )
        for _count, _ii, data in self.read_plan(
            gulp,
            start=start,
            nsamps=nsamps,
            **kwargs,
        ):
            out_file.cwrite(data)
        out_file.close()
        return out_file.name

    def split_chans(
        self, gulp: int = 1024, back_compatible: bool = True, **kwargs
    ) -> List[str]:
        """Split the data into component channels and write each to file.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 1024
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        list of str
            names of all files written to disk

        Notes
        -----
        Time series are written to disk with names based on channel number.
        """
        tim_ar = np.empty([self.header.nchans, gulp], dtype="float32")
        out_files = [
            self.header.prep_outfile(
                f"{self.header.basename}_chan{ii:04d}.tim",
                {"nchans": 1, "nbits": 32, "data_type": 2},
                back_compatible=back_compatible,
                nbits=32,
            )
            for ii in range(self.header.nchans)
        ]
        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            libcpp.splitToChans(data, tim_ar, self.header.nchans, nsamps, gulp)
            for ifile, out_file in enumerate(out_files):
                out_file.cwrite(tim_ar[ifile][:nsamps])

        for out_file in out_files:
            out_file.close()

        return [out_file.name for out_file in out_files]

    def split_bands(
        self,
        chanpersub: int,
        chanstart: int = 0,
        gulp: int = 1024,
        back_compatible: bool = True,
        **kwargs,
    ) -> List[str]:
        """Split the data into component Sub-bands and write to disk.

        Parameters
        ----------
        chanpersub : int
            number of channels in each sub-band
        chanstart : int, optional
            start channel of split, by default 0
        gulp : int, optional
            number of samples in each read, by default 1024
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        list of str
            names of all files written to disk

        Notes
        -----
        Filterbanks are written to disk with names based on sub-band number.
        """
        nsub = (self.header.nchans - chanstart) // chanpersub
        fstart = self.header.fch1 + chanstart * self.header.foff

        out_files = [
            self.header.prep_outfile(
                f"{self.header.basename}_sub{ii:02d}.fil",
                {
                    "nchans": chanpersub,
                    "fch1": fstart + ii * chanpersub * self.header.foff,
                },
                back_compatible=back_compatible,
                nbits=self.header.nbits,
            )
            for ii in range(nsub)
        ]

        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            for iband, out_band in enumerate(out_files):
                data = data.reshape(nsamps, self.header.nchans)
                subband_ar = data[
                    :,
                    chanstart + chanpersub * iband : chanstart + chanpersub * (iband + 1),
                ]
                out_band.cwrite(subband_ar.ravel())

        for out_file in out_files:
            out_file.close()

        return [out_file.name for out_file in out_files]

    def remove_bandpass(
        self,
        gulp: int = 512,
        filename: Optional[str] = None,
        back_compatible: bool = True,
        **kwargs,
    ) -> str:
        """Remove the bandpass from the data and write new data to a new file.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        filename : str, optional
            name of output file, by default ``basename_bpcorr.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file

        Raises
        ------
        TypeError
            If `nbits` of input file is < 8.
        """
        if self.header.nbits < 8:
            raise TypeError(f"{self.header.nbits}-bit filterbank not supported yet!")
        if filename is None:
            filename = f"{self.header.basename}_bpcorr.fil"

        if self.chan_stdevs is None:
            self.compute_stats(gulp=gulp, **kwargs)

        out_ar = np.empty(
            self.header.nsamples * self.header.nchans,
            dtype=self.header.dtype,
        )
        out_file = self.header.prep_outfile(
            filename, nbits=self.header.nbits, back_compatible=back_compatible
        )
        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            libcpp.remove_bandpass(
                data,
                out_ar,
                self.chan_means,
                self.chan_stdevs,
                self.header.nchans,
                nsamps,
            )
            out_file.cwrite(out_ar[: nsamps * self.header.nchans])
        out_file.close()
        return out_file.name

    def remove_zerodm(
        self,
        gulp: int = 512,
        filename: Optional[str] = None,
        back_compatible: bool = True,
        **kwargs,
    ):
        """Remove the channel-weighted zero-DM from the data and write to disk.

        Parameters
        ----------
        gulp : int, optional
            number of samples in each read, by default 512
        filename : str, optional
            name of output file , by default ``basename_noZeroDM.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

        Returns
        -------
        str
            name of output file

        Notes
        -----
        Based on Presto implementation of Eatough, Keane & Lyne 2009
        """
        if filename is None:
            filename = f"{self.header.basename}_noZeroDM.fil"

        bpass = self.bandpass(gulp=gulp, **kwargs)
        chanwts = bpass / bpass.sum()
        out_ar = np.empty(
            self.header.nsamples * self.header.nchans,
            dtype=self.header.dtype,
        )
        out_file = self.header.prep_outfile(
            filename, nbits=self.header.nbits, back_compatible=back_compatible
        )
        for nsamps, _ii, data in self.read_plan(gulp, **kwargs):
            libcpp.remove_zerodm(data, out_ar, bpass, chanwts, self.header.nchans, nsamps)
            out_file.cwrite(out_ar[: nsamps * self.header.nchans])
        out_file.close()
        return out_file.name
